import contextlib
import logging
import threading
import time
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.atom_real.hardware import AtomUpperControl
from examples.xtrainer_real.hardware.realsense_camera import RealSenseCamera

logger = logging.getLogger(__name__)


class OpenCVCamera:
    """OpenCV camera wrapper for UVC-style top cameras."""

    def __init__(self, device_id: int, *, width: int = 1280, height: int = 720, fps: int = 30):
        self.device_id = int(device_id)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._capture = None
        self._connected = False
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self._connected:
            return

        import cv2

        capture = cv2.VideoCapture(self.device_id)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open OpenCV camera: /dev/video{self.device_id}")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)

        for _ in range(5):
            capture.read()

        self._capture = capture
        self._connected = True

    def async_read(self, timeout_ms: int = 50) -> np.ndarray | None:
        if not self._connected or self._capture is None:
            return None

        with self._lock:
            ok, bgr = self._capture.read()
        if not ok or bgr is None:
            return None

        return np.ascontiguousarray(bgr[..., ::-1])

    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._connected = False


def _pack_state(arm_state: np.ndarray, hand_state: np.ndarray, head_state: np.ndarray) -> np.ndarray:
    # Dataset/action order for Atom:
    # [left_arm(7), left_hand(6), right_arm(7), right_hand(6), head(2)]
    return np.concatenate([arm_state[:7], hand_state[:6], arm_state[7:14], hand_state[6:12], head_state]).astype(
        np.float64
    )


class AtomRealEnvironment(_environment.Environment):
    """Atom real-robot environment for openpi-client runtime."""

    def __init__(
        self,
        *,
        camera_top_serial: str = "",
        camera_top_id: int = -1,
        camera_top_width: int = 1280,
        camera_top_height: int = 720,
        camera_top_fps: float = 60.0,
        camera_left_wrist_serial: str = "",
        camera_right_wrist_serial: str = "",
        camera_fps: float = 30.0,
        render_height: int = 224,
        render_width: int = 224,
        prompt: str | None = None,
        reset_pose: Optional[list[float]] = None,  # noqa: UP007
        max_joint_delta: float = 0.17,
        ramp_step: float = 0.01,
        ramp_max_steps: int = 100,
        enable_upper_control_on_connect: bool = True,
        disable_upper_control_on_close: bool = True,
    ) -> None:
        self._robot = None

        if camera_top_id >= 0:
            self._cam_top = OpenCVCamera(
                camera_top_id,
                width=camera_top_width,
                height=camera_top_height,
                fps=round(camera_top_fps),
            )
        elif camera_top_serial:
            self._cam_top = RealSenseCamera(camera_top_serial, fps=round(camera_fps))
        else:
            raise ValueError("Either camera_top_id or camera_top_serial is required for Atom inference.")

        self._cam_left = (
            RealSenseCamera(camera_left_wrist_serial, fps=round(camera_fps)) if camera_left_wrist_serial else None
        )
        self._cam_right = (
            RealSenseCamera(camera_right_wrist_serial, fps=round(camera_fps)) if camera_right_wrist_serial else None
        )

        self._render_height = render_height
        self._render_width = render_width
        self._prompt = prompt

        self._max_joint_delta = max_joint_delta
        self._ramp_step = ramp_step
        self._ramp_max_steps = max(ramp_max_steps, 1)

        self._enable_upper_control_on_connect = enable_upper_control_on_connect
        self._disable_upper_control_on_close = disable_upper_control_on_close

        self._connected = False
        self._last_action: np.ndarray | None = None

        self._reset_pose = None
        if reset_pose is not None:
            reset = np.asarray(reset_pose, dtype=np.float64).reshape(-1)
            if reset.shape[0] != 28:
                raise ValueError(f"Expected reset_pose length 28, got {reset.shape[0]}")
            self._reset_pose = reset

    @override
    def reset(self) -> None:
        self._ensure_connected()
        if self._reset_pose is not None:
            self._move_smooth(self._get_atom_qpos(), self._reset_pose)
            time.sleep(0.2)
        self._last_action = self._get_atom_qpos()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        self._ensure_connected()
        top_frame = self._read_camera_frame(self._cam_top, "top")
        top_frame = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(top_frame, self._render_height, self._render_width)
        )

        observation = {
            "observation.state": self._get_atom_qpos().astype(np.float32),
            "observation.images.top": top_frame,
        }

        if self._cam_left is not None:
            left_frame = self._read_camera_frame(self._cam_left, "left_wrist")
            left_frame = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(left_frame, self._render_height, self._render_width)
            )
            observation["observation.images.left_wrist"] = left_frame

        if self._cam_right is not None:
            right_frame = self._read_camera_frame(self._cam_right, "right_wrist")
            right_frame = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_frame, self._render_height, self._render_width)
            )
            observation["observation.images.right_wrist"] = right_frame

        if self._prompt is not None:
            observation["prompt"] = self._prompt
        return observation

    @override
    def apply_action(self, action: dict) -> None:
        self._ensure_connected()
        if "actions" not in action:
            raise KeyError(f"Missing 'actions' in action dict: {tuple(action.keys())}")

        target = np.asarray(action["actions"], dtype=np.float64).reshape(-1).copy()
        if target.shape[0] != 28:
            raise ValueError(f"Expected action length 28, got {target.shape[0]}")
        if not np.all(np.isfinite(target)):
            raise ValueError("Action contains non-finite values.")

        if self._last_action is None:
            self._last_action = self._get_atom_qpos()

        # Apply the jump check to arm dimensions only.
        arm_delta = max(
            float(np.max(np.abs(target[:7] - self._last_action[:7]))),
            float(np.max(np.abs(target[13:20] - self._last_action[13:20]))),
        )
        if arm_delta > self._max_joint_delta:
            self._move_smooth(self._last_action, target)
        else:
            self._send_atom_action(target)

        self._last_action = target.copy()

    def close(self) -> None:
        if not self._connected:
            return

        for camera in (self._cam_top, self._cam_left, self._cam_right):
            if camera is not None:
                with contextlib.suppress(Exception):
                    camera.disconnect()

        if self._robot is not None and self._disable_upper_control_on_close:
            with contextlib.suppress(Exception):
                self._robot.setUpperControl(False)

        self._robot = None
        self._connected = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return

        self._robot = AtomUpperControl()
        if not self._robot.wait_until_ready():
            raise RuntimeError("Atom robot state did not become ready in time.")
        if getattr(self._robot, "has_error", None) is not None and self._robot.has_error() != 0:
            raise RuntimeError("Atom robot reports active alarm state. Please clear alarms before running policy.")

        if self._enable_upper_control_on_connect:
            try:
                self._robot.setUpperControl(True)
            except Exception:
                logger.exception("Failed to enable upper-limb control automatically.")

        for camera in (self._cam_top, self._cam_left, self._cam_right):
            if camera is not None:
                camera.connect()

        self._connected = True
        self._last_action = self._get_atom_qpos()

    def _read_camera_frame(self, camera, name: str, retries: int = 5) -> np.ndarray:
        for _ in range(max(retries, 1)):
            try:
                frame = camera.async_read(timeout_ms=50)
            except TypeError:
                frame = camera.async_read()
            except Exception:
                frame = None

            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                return frame
            time.sleep(0.005)
        raise RuntimeError(f"Failed to read camera frame: {name}")

    def _get_atom_qpos(self) -> np.ndarray:
        if self._robot is None:
            raise RuntimeError("Robot is not connected")

        arm_state, hand_state, head_state = self._robot.get_joint_state()
        arm_state = np.asarray(arm_state, dtype=np.float64).reshape(-1)
        hand_state = np.asarray(hand_state, dtype=np.float64).reshape(-1)
        head_state = np.asarray(head_state, dtype=np.float64).reshape(-1)

        if arm_state.shape[0] != 14 or hand_state.shape[0] != 12 or head_state.shape[0] != 2:
            raise RuntimeError(
                f"Unexpected state shapes from robot: arm={arm_state.shape}, hand={hand_state.shape}, head={head_state.shape}"
            )

        return _pack_state(arm_state, hand_state, head_state)

    def _send_atom_action(self, action: np.ndarray) -> None:
        if self._robot is None:
            raise RuntimeError("Robot is not connected")

        left_arm = action[:7]
        left_hand = action[7:13]
        right_arm = action[13:20]
        right_hand = action[20:26]
        head = action[26:28]

        self._robot.command_joint_state(left_arm, right_arm, left_hand, right_hand, head)

    def _move_smooth(self, start_action: np.ndarray, goal_action: np.ndarray) -> None:
        max_delta = float(np.max(np.abs(goal_action - start_action)))
        steps = min(int(np.ceil(max_delta / max(self._ramp_step, 1e-6))), self._ramp_max_steps)
        if steps <= 1:
            self._send_atom_action(goal_action)
            return

        for joints in np.linspace(start_action, goal_action, steps):
            self._send_atom_action(joints)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
