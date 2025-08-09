import math
from typing import Optional, Tuple, Union

import numpy as np
import pygame

from pygame import gfxdraw
import gym
from gym import spaces
from gym.utils import seeding
#import adrc
import LADRC
from landing_gear import mode
import matplotlib.pyplot as plt
import pandas as pd


class Adrc_control_landing_env(gym.Env):
    # 物理模型
    cfg = LADRC.ADRCConfig()
    ADRC_control = LADRC.ADRC(cfg)
    landing_model = mode.landing_gear()

    environment_name = "Landing gear"
    """
    describe    Adrc控制单飞机起落架
    学习参数     2 wc【0.5-1.5】  w0【0-2】
    """
    simea = 1 * 10 ** -8  # 充当极小化因子，接近于0；以防计算影响

    wc_range = [simea, 2]
    w0_range = [simea, 2]

    low_list = np.array([ wc_range[0], w0_range[0], ], dtype=np.float32)
    high_list = np.array([ wc_range[1], w0_range[1]], dtype=np.float32)
    def __init__(self):
        self.dt = self.ADRC_control.dt
        self.action_space = spaces.Box(low=self.low_list,high=self.high_list,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100,high=100,shape=(4,), dtype=np.float32)
        self.v = 0
        #111   设定do_action的条件存在性
        self.do_action = True
        self.ADRC_control.v1 = self.landing_model.v

    def step(self, action) :
        global done
        done = 0
        if self.do_action:
            self.ADRC_control.get_parameter(action)

        #状态方程
        self.u = self.ADRC_control.con_transport(0, y=self.landing_model.z1_1).__float__()

        self.landing_model.u = self.u
        self.landing_model.env_transport()

        self.state = np.array(
            [self.landing_model.z1, self.landing_model.z1_1, self.landing_model.z2, self.landing_model.z2_1],
            dtype=np.float64)

        # 奖励设置
        #self.e = self.v[0] - self.landing_model.z1_1
        self.es1 = 0 - self.landing_model.z1_1
        reward = -np.clip(abs(self.es1), 0,3)

        return np.array(self.state, dtype=np.float32), reward, done, {}


    def reset(self) :
        self.v = 0
        # 将adrc和环境重置
        self.landing_model.reset()
        self.ADRC_control.reset()
        self.ADRC_control.v1 = self.landing_model.v
        # 重设环境的情况
        self.state = np.array([self.landing_model.z1,self.landing_model.z1_1,
                               self.landing_model.z2,self.landing_model.z2_1 ], dtype=np.float64)

        return np.array(self.state, dtype=np.float32)

    def save(self, ep, action, re, rt):
        Date_ADRC = pd.DataFrame(self.ADRC_control.logger.__dict__)

        A = 'Data/LADRC_' + str(ep) + '.xlsx'
        Date_ADRC.to_excel(A)
        Date_mode = pd.DataFrame(self.landing_model.logger.__dict__)
        B = 'Data/mode_' + str(ep) + '.xlsx'
        Date_mode.to_excel(B)
        Data = pd.DataFrame()
        Data['action'] = action
        Data['Reward'] = re
        Data['Return'] = rt
        C = 'Data/data_' + str(ep) + '.xlsx'
        Data.to_excel(C)
        print(Data)


'debug'
if __name__ == '__main__':
    enverment = Adrc_control_landing_env()
    print(enverment.ADRC_control.logger.__dict__)
    print(enverment.landing_model.z1_1)
    enverment.reset()
    list_t = np.arange(0.0, 8.0, enverment.dt)
    # a为0，初始无主动控制从被动控制
    a = np.zeros(2)
    enverment.do_action = False
    for _ in list_t:
        state,reward,done,_=enverment.step(action=a)
        print(enverment.ADRC_control.v2)
    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list_t, np.array(enverment.landing_model.logger.z1, dtype=object), label = 'm1位移' )
    plt.plot(list_t, np.array(enverment.landing_model.logger.z1_1, dtype=object), label = 'm1速度')
    plt.legend()
    plt.show()
    enverment.ADRC_control.show()
    enverment.save()

