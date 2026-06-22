from __future__ import annotations

import time

import numpy as np

from examples.atom_real.hardware.robot_control_dds import Control_sim


class AtomUpperControl:
    """Local Atom upper-body controller built into this repository."""

    def __init__(self) -> None:
        self.robot = Control_sim()
        self.robot.start()

    def wait_until_ready(self, timeout_s: float = 3.0) -> bool:
        deadline = time.time() + max(float(timeout_s), 0.1)
        while time.time() < deadline:
            if (
                self.robot.upper_msg is not None
                and self.robot.hand_msg is not None
                and self.robot.main_state_msg is not None
            ):
                return True
            time.sleep(0.02)
        return False

    def has_error(self) -> bool:
        main_nodes_state = self.robot.main_state_msg
        if main_nodes_state is None:
            return True

        joint_groups = [
            (main_nodes_state.left_arm, "left arm"),
            (main_nodes_state.right_arm, "right arm"),
            (main_nodes_state.head, "head"),
        ]

        for joint_group, _group_name in joint_groups:
            for axis in joint_group:
                if axis.pos_err_code != 0 or axis.vel_err_code != 0 or axis.torque_err_code != 0:
                    return True
                if axis.servo_state != 3 or axis.error_code != 0 or axis.node_state != 5:
                    return True

        for slave in main_nodes_state.ecat2can:
            if slave.slave_state != 1 or slave.error_code != 0:
                return True
        return False

    def setUpperControl(self, state: bool):
        result = -1
        for _ in range(10):
            result = self.robot.RPC.CallSwitchUpperLimbControl(state)
            if result == 1:
                return result
            time.sleep(0.05)
        return result

    def get_joint_state(self):
        arm_state = []
        for i in range(1, 15):
            arm_state.append(self.robot.upper_msg.motor_state[i].q)

        head_state = []
        for i in range(15, 17):
            head_state.append(self.robot.upper_msg.motor_state[i].q)

        hand_state = []
        for i in range(6, 12):
            hand_state.append(self.robot.hand_msg.hands[i].q)

        for i in range(6):
            hand_state.append(self.robot.hand_msg.hands[i].q)

        return np.array(arm_state), np.array(hand_state), np.array(head_state)

    def command_joint_state(
        self,
        left_joint_state: np.ndarray,
        right_joint_state: np.ndarray,
        left_hand_state: np.ndarray,
        right_hand_state: np.ndarray,
        head_state: np.ndarray,
    ) -> None:
        q_ref = np.zeros(17, dtype=np.float64)
        torso = self.robot.upper_msg.motor_state[0].q
        q_ref[0] = torso
        q_ref[1:8] = np.asarray(left_joint_state, dtype=np.float64)
        q_ref[8:15] = np.asarray(right_joint_state, dtype=np.float64)
        q_ref[15:17] = np.asarray(head_state, dtype=np.float64)

        q_hand = np.concatenate(
            [
                np.asarray(right_hand_state, dtype=np.float64),
                np.asarray(left_hand_state, dtype=np.float64),
            ]
        )
        self.robot.send_cmd(q_ref, q_hand)

    def close(self) -> None:
        try:
            self.setUpperControl(False)
        except Exception:
            pass
