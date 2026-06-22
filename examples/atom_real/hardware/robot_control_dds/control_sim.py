import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)
sys.path.append(BASE_DIR)

from dataclasses import dataclass
from cyclonedds.domain import DomainParticipant, Domain
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
# from atom.msg import dds_
from .rpc.callSwitchUpperLimbControl import JsonRpcClient

import threading
import logging
import subprocess
import re

from .dobot_atom.msg import dds_

from .DDSHelper import *

import numpy as np
import time

import json

logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s',
    datefmt='%m%d %H:%M:%S'
)

log = logging.getLogger()


class Control_sim:
    def __init__(self):

        # self.domain = Domain(
        # domain_id=0, config="<CycloneDDS><Domain><General><Interface><NetworkInterface address='192.168.8.234'/></Interface></General></Domain></CycloneDDS>")
        self.participant = DomainParticipant()

        self.upper_state_topic = Topic(self.participant, "rt/upper/state", dds_.UpperState_)
        self.upper_cmd_topic = Topic(self.participant, "rt/upper/cmd", dds_.UpperCmd_)
        self.upper_state_reader = DataReader(self.participant, self.upper_state_topic)
        self.upper_cmd_writer = DataWriter(self.participant, self.upper_cmd_topic)
        
        # self.lower_state_topic = Topic(self.participant, "rt/lower/state", dds_.LowerState_)
        # self.lower_cmd_topic = Topic(self.participant, "rt/lower/cmd", dds_.LowerCmd_)
        # self.lower_state_reader = DataReader(self.participant, self.lower_state_topic)
        # self.lower_cmd_writer = DataWriter(self.participant, self.lower_cmd_topic)
        
        self.inspire_state_topic = Topic(self.participant, "rt/hands/state", dds_.HandsState_)
        self.inspire_cmd_topic = Topic(self.participant, "rt/hands/cmd", dds_.HandsCmd_)
        self.inspire_state_reader = DataReader(self.participant, self.inspire_state_topic)
        self.inspire_cmd_writer = DataWriter(self.participant, self.inspire_cmd_topic)
        
        self.amr_state_topic = Topic(self.participant, "rt/amr/state", dds_.AMRState_)
        self.amr_cmd_topic = Topic(self.participant, "rt/amr/cmd", dds_.AMRCommand_)
        self.amr_state_reader = DataReader(self.participant, self.amr_state_topic)
        self.amr_cmd_writer = DataWriter(self.participant, self.amr_cmd_topic)

        self.fsm_topic = Topic(self.participant, "rt/set/fsm/id", dds_.SetFsmId_)
        self.fsm_cmd_writer = DataWriter(self.participant, self.fsm_topic)
        self.fsm_reader = DataReader(self.participant, self.fsm_topic)

        self.main_state_topic = Topic(self.participant, "rt/main/nodes/state", dds_.MainNodesState_)
        self.main_state_reader = DataReader(self.participant, self.main_state_topic)

        self.upper_msg = None
        self.read_upper_state_thread = threading.Thread(target=self.read_upper_state)
        self.read_upper_state_thread.setDaemon(True)
        self.read_upper_lock = threading.Lock()
        
        # self.lower_msg = None
        # self.read_lower_state_thread = threading.Thread(target=self.read_lower_state)
        # self.read_lower_state_thread.setDaemon(True)
        # self.read_lower_lock = threading.Lock()

        self.amr_msg = None
        self.read_amr_state_thread = threading.Thread(target=self.read_amr_state)
        self.read_amr_state_thread.setDaemon(True)
        self.read_amr_lock = threading.Lock()
        

        self.read_hand_thread = threading.Thread(target=self.read_hand_state)
        self.read_hand_thread.setDaemon(True)
        self.read_hand_lock = threading.Lock()

        self.fsm_msg = None
        self.fsm_read_cmd_thread = threading.Thread(target=self.listen_fsm)
        self.fsm_read_cmd_thread.setDaemon(True)
        self.fsm_cmd_lock = threading.Lock()
        
        self.main_state_msg = None
        self.main_state_thread = threading.Thread(target=self.listen_main_state)
        self.main_state_thread.setDaemon(True)
        self.main_state_lock = threading.Lock()

        self.hand_msg = None
        self.cmd_upper_msg = set_upper_state()
        self.cmd_lower_msg = set_lower_state()
        self.cmd_hand_msg = get_hand_cmd_state()
        self.cmd_amr_msg = get_amr_cmd_state()
        self.cmd_fsm = get_fsm()
        self.main_state = get_main_state()
        self.send_fsm_cmd()

        self.RPC = JsonRpcClient()

        self.controlParamsFile = BASE_DIR + '/controlParams_P2.json'
        # print("self.controlParamsFile", self.controlParamsFile)
        with open(self.controlParamsFile, 'r') as file:
            self.controlParams = json.load(file)


    def read_upper_state(self):
        for msg in self.upper_state_reader.read_iter(timeout=duration(minutes=0.01)):
            with self.read_upper_lock:
                self.upper_msg = msg
                time.sleep(0.001)
                # print(msg)

    # def read_lower_state(self):
    #     for msg in self.lower_state_reader.read_iter(timeout=duration(minutes=0.01)):
    #         with self.read_lower_lock:
    #             self.lower_msg = msg
    #             time.sleep(0.001)
    #             # print(msg)
                # time.sleep(0.001)
                # print(msg)

    def read_lower_state(self):
        for msg in self.lower_state_reader.read_iter(timeout=duration(minutes=0.01)):
            with self.read_lower_lock:
                self.lower_msg = msg
                time.sleep(0.001)
                # print(msg)

    def read_amr_state(self):
        for msg in self.amr_state_reader.read_iter(timeout=duration(minutes=0.01)):
            with self.read_amr_lock:
                self.amr_msg = msg
                time.sleep(0.001)

    def read_hand_state(self):
        for msg in self.inspire_state_reader.read_iter(timeout=duration(minutes=0.01)):
            with self.read_hand_lock:
                self.hand_msg = msg
                time.sleep(0.001)
                # print(msg)

    def listen_fsm(self):
        for msg in self.fsm_reader.take_iter(timeout=duration(minutes=1)):
            with self.fsm_cmd_lock:
                self.fsm_msg = msg
                time.sleep(0.001)

    def listen_main_state(self):
        for msg in self.main_state_reader.take_iter(timeout=duration(minutes=1)):
            with self.main_state_lock:
                self.main_state_msg = msg
                time.sleep(0.001)

    def send_cmd(self, q, q_hand):
        for i in range(17):
            self.cmd_upper_msg.motor_cmd[i].q = q[i]
            self.cmd_upper_msg.motor_cmd[i].dq = 0.
            self.cmd_upper_msg.motor_cmd[i].tau = 0.
            self.cmd_upper_msg.motor_cmd[i].kp = self.controlParams[i]["kp"]
            self.cmd_upper_msg.motor_cmd[i].kd = self.controlParams[i]["kd"]

        for i in range(12):
            self.cmd_hand_msg.hands[i].q = q_hand[i]
            self.cmd_hand_msg.hands[i].dq = 0.
            self.cmd_hand_msg.hands[i].tau = 0.
            self.cmd_hand_msg.hands[i].kp = 10
            self.cmd_hand_msg.hands[i].kd = 0

        # 下肢控制
        # for i in range(12):
        #     self.cmd_lower_msg.motor_cmd[i].q = 0.5
        #     self.cmd_lower_msg.motor_cmd[i].dq = 0.
        #     self.cmd_lower_msg.motor_cmd[i].tau = 0.
        #     self.cmd_lower_msg.motor_cmd[i].kp = 10
        #     self.cmd_lower_msg.motor_cmd[i].kd = 1

        # self.lower_cmd_writer.write(self.cmd_lower_msg)
        self.upper_cmd_writer.write(self.cmd_upper_msg)
        self.inspire_cmd_writer.write(self.cmd_hand_msg)

    def send_amr_cmd(self, vx, vy, vw):
        self.cmd_amr_msg.linear_vel = vx
        self.cmd_amr_msg.angular_vel = vw
        self.amr_cmd_writer.write(self.cmd_amr_msg)

    def send_vel_cmd(self, q, q_hand, dq,send_finger):
        for i in range(17):
            self.cmd_upper_msg.motor_cmd[i].q = q[i]
            self.cmd_upper_msg.motor_cmd[i].dq = dq[i]
            self.cmd_upper_msg.motor_cmd[i].tau = 0.
            self.cmd_upper_msg.motor_cmd[i].kp = self.controlParams[i]["kp"]
            self.cmd_upper_msg.motor_cmd[i].kd = self.controlParams[i]["kd"]
        self.upper_cmd_writer.write(self.cmd_upper_msg)

        if send_finger:
            for i in range(12):
                self.cmd_hand_msg.hands[i].q = q_hand[i]
                self.cmd_hand_msg.hands[i].dq = 0.
                self.cmd_hand_msg.hands[i].tau = 0.
                self.cmd_hand_msg.hands[i].kp = 10
                self.cmd_hand_msg.hands[i].kd = 0
            self.inspire_cmd_writer.write(self.cmd_hand_msg)

    def send_fsm_cmd(self, fsm_id=2):
        self.cmd_fsm.id = fsm_id
        self.fsm_cmd_writer.write(self.cmd_fsm)

    def start(self):
        # print('Start simulation!')
        self.fsm_read_cmd_thread.start()
        print("Start dds")
        self.read_upper_state_thread.start()
        # self.read_lower_state_thread.start()
        self.read_hand_thread.start()
        self.main_state_thread.start()
        # self.read_amr_state_thread.start()
        time.sleep(1)
        t = 0.
        T_tol = 5
        q_ref = {}
        q_hand = {}


    # 在 Control_sim 类中添加以下方法:
    '''
    def send_lift_cmd(self, q_lift, dq_lift=None, kp=10, kd=1):
        """
        控制升降轴关节
        
        Args:
            q_lift: 包含1-3个升降轴关节目标位置的列表或字典
            dq_lift: 包含1-3个升降轴关节目标速度的列表或字典(可选)
            kp: 位置控制比例增益
            kd: 位置控制微分增益
        """
        # 确保 q_lift 是列表格式且长度不超过3
        if isinstance(q_lift, dict):
            q_list = [q_lift.get(i, 0) for i in range(min(3, len(q_lift)))]
        else:
            q_list = list(q_lift)[:3]  # 最多取前3个值
        
        # 如果未提供速度指令，则默认为0
        if dq_lift is None:
            dq_list = [0.0] * len(q_list)
        else:
            if isinstance(dq_lift, dict):
                dq_list = [dq_lift.get(i, 0) for i in range(len(q_list))]
            else:
                dq_list = list(dq_lift)[:len(q_list)]
        
        # 设置升降轴关节控制指令
        for i in range(len(q_list)):
            self.cmd_lower_msg.motor_cmd[i].q = q_list[i]
            self.cmd_lower_msg.motor_cmd[i].dq = dq_list[i] if i < len(dq_list) else 0.0
            self.cmd_lower_msg.motor_cmd[i].tau = 0.
            self.cmd_lower_msg.motor_cmd[i].kp = kp
            self.cmd_lower_msg.motor_cmd[i].kd = kd

        # 发送指令到 rt/lower/cmd 话题
        self.lower_cmd_writer.write(self.cmd_lower_msg)
    '''

if __name__ == '__main__':
    main = Control_sim()
    # main.RPC.CallSwitchUpperLimbControl(True)
    main.start()
    
    # example : 读取main_state_msg状态
    # print("右臂伺服报错:",main.main_state_msg.left_arm[0].error_code)
    # print("报错:", main.main_state_msg.ecat2can[0].error_code)
    # print("报错:", main.main_state_msg.ecat2can[1].error_code)
    
    # example: 发送AMR速度指令
    # while 1:
    #     main.send_amr_cmd(-0.1, 0.0, 0.0)
    #     time.sleep(0.01)
