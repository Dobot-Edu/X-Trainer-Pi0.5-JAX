from dobot_atom.msg import dds_

from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from cyclonedds.domain import DomainParticipant

import struct


def get_imu_state():
    return dds_.IMUState_(
        quaternion=[1, 0, 0, 0],
        rpy=[0, 0, 0],
        accelerometer=[0, 0, 0],
        gyroscope=[0, 0, 0],
        temperature=0
    )


def get_motor_cmd():
    return dds_.MotorCmd_(
        mode=1,
        q=0.,
        dq=0.,
        tau=0.,
        kp=0,
        kd=0,
    )


def get_bms_cmd():
    return dds_.BmsCmd_(
        clearErrors=0
    )


def set_upper_state():
    return dds_.UpperCmd_(
        motor_cmd=[get_motor_cmd() for _ in range(17)],
    )


def set_lower_state():
    return dds_.LowerCmd_(
        motor_cmd=[get_motor_cmd() for _ in range(12)],
    )


def get_hand_cmd_state():
    return dds_.HandsCmd_(
        hands=[get_motor_cmd() for _ in range(12)],
    )


def get_amr_cmd_state():
    return dds_.AMRCommand_(
        command_type=0,
        target_id=0,
        linear_vel=0.0,
        angular_vel=0.0,
        command_id=0,
        timestamp=0,
        theta=0.,
    )


def get_fsm():
    return dds_.SetFsmId_(
        id=0,
        current_action='\0'
    )


def get_main_state():
    return dds_.MainNodesState_(
        left_leg=[get_axis_state_info() for _ in range(6)],
        right_leg=[get_axis_state_info() for _ in range(6)],
        waist=get_axis_state_info(),
        left_arm=[get_axis_state_info() for _ in range(7)],
        right_arm=[get_axis_state_info() for _ in range(7)],
        head=[get_axis_state_info() for _ in range(2)],
        ecat2can=[get_ecat_slave_info() for _ in range(2)],
    )


def get_axis_state_info():
    return dds_.AxisStateInfo_(
        servo_state=0,
        error_code=0,
        warn_code=0,
        pos_err_code=0,
        vel_err_code=0,
        torque_err_code=0,
        node_state=0,
        display_op_mode=0,
        is_virtual=0,
        mcu_temp=0,
        mos_temp=0,
        motor_temp=0,
        bus_voltage=0,
        software_version=0,
    )


def get_ecat_slave_info():
    return dds_.EcatSlaveInfo_(
        is_virtual=0,
        slave_state=0,
        error_code=0,
        software_version=0,
    )


def get_upper_cmd_state():
    return dds_.UpperBodyCmd_(
        upperBody=[get_motor_cmd() for _ in range(16)],
    )


def get_bms_state():
    return dds_.BmsState_(
        version_high=0,
        version_low=0,
        status=0,
        soc=0,
        current=0,
        cycle=0,
        bq_ntc=[0, 0],
        mcu_ntc=[0, 0],
        cell_vol=[0 for _ in range(15)]
    )


def get_motor_state():
    return dds_.MotorState_(
        mode=1,
        q=0.,
        dq=0.,
        ddq=0.,
        tau_est=0.,
        q_raw=0.,
        dq_raw=0.,
        ddq_raw=0.,
        mcuTemp=0,
        mosTemp=0,
        motorTemp=0,
        busVoltage=0,
    )


def get_arm_state():
    return dds_.UpperBodyState_(
        motor_state=[get_motor_state() for _ in range(30)],
    )


def get_low_state():
    return dds_.LowState_(
        head=(0, 0),
        level_flag=0,
        frame_reserve=0,
        sn=(0, 0),
        version=(0, 0),
        bandwidth=0,
        imu_state=get_imu_state(),
        motor_state=[get_motor_state() for _ in range(30)],
        bms_state=get_bms_state(),
        foot_force=(0, 0, 0, 0),
        foot_force_est=(0, 0, 0, 0),
        tick=0,
        wireless_remote=[0 for _ in range(40)],
        bit_flag=0,
        adc_reel=0.,
        temperature_ntc1=0,
        temperature_ntc2=0,
        power_v=0.,
        power_a=0.,
        fan_frequency=(0, 0, 0, 0),
        reserve=0,
        crc=0
    )


def get_wireless_remote_state(keys):
    # prepare an empty list
    wireless_remote_b = [0 for _ in range(40)]

    btn_order = ['A', 'A', 'LT', 'RT', 'SELECT', 'START', 'LB', 'RB',
                 'LEFT', 'DOWN', 'RIGHT', 'UP', 'Y', 'X', 'B', 'A']
    btn = ''.join([f'{keys[k]}' for k in btn_order])

    wireless_remote_b[2] = int(btn[:8], 2)
    wireless_remote_b[3] = int(btn[8:], 2)

    # process sticks
    sticks = ['LX', 'RX', 'RY', 'LY']
    packs = list(map(
        lambda stick: struct.pack('<f', -keys[stick] if stick in ['LY'] else keys[stick]),
        sticks
    ))
    wireless_remote_b[4:8] = packs[0]
    wireless_remote_b[8:12] = packs[1]
    wireless_remote_b[12:16] = packs[2]
    wireless_remote_b[20:24] = packs[3]

    # print(wireless_remote_b)

    return wireless_remote_b


if __name__ == '__main__':
    keys = {'A': 0,
            'B': 0,
            'X': 0,
            'Y': 0,
            'LB': 0,
            'RB': 0,
            'LT': 0,
            'RT': 0,
            'SELECT': 0,
            'START': 0,
            'UP': 0,
            'DOWN': 0,
            'LEFT': 0,
            'RIGHT': 0,
            'LY': 0,
            'LX': 0,
            'RY': 0,
            'RX': 0}
