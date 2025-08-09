import math
from typing import Optional, Tuple, Union

import numpy as np
import pygame

from pygame import gfxdraw
import gym
from gym import spaces
from gym.utils import seeding
#import adrc
import LADRC2
from landing_gear import mode
import matplotlib.pyplot as plt
import pandas as pd


class Adrc_control_landing_env(gym.Env):
    # 物理模型
    cfg = LADRC2.ADRCConfig()
    ADRC_control = LADRC2.ADRC(cfg)
    landing_model = mode.landing_gear()

    environment_name = "Landing gear"
    """
    describe    Adrc控制单飞机起落架
    学习参数     ADRC控制器的各类指标有
                    TD跟踪器：跟踪因子、滤波因子=dt
                    ESO观测器： b0、delat、反馈增益1,2,3   5个
                    非线性反馈：增益beta、alpha（误差e、de） 4个
    """
    simea = 1 * 10 ** -8  # 充当极小化因子，接近于0；以防计算影响

    h_range = [simea, 0.002]
    r_range = [simea, 2.]

    b0_range = [1., 15.]
    delat_range = [simea, 1.]
    eso_beta01_range = [100., 100.]
    eso_beta02_range = [100., 1000.]
    eso_beta03_range = [simea, 500.]

    nlsef_beta1_range = [0.001, 20.]  # 跟踪输入信号增益 (float or list)
    nlsef_beta2_range = [0.001, 50.]  # 跟踪微分信号增益 (float or list)
    nlsef_alpha1_range = [simea, 1 - simea]  # 0 < alpha1 < 1  (float or list)
    nlsef_alpha2_range = [1. + simea, 10.]  # alpha2 > 1      (float or list)

    low_list0 = np.array([h_range[0], r_range[0], b0_range[0], delat_range[0], eso_beta01_range[0], eso_beta02_range[0],
                    eso_beta03_range[0], nlsef_beta1_range[0], nlsef_beta2_range[0], nlsef_alpha1_range[0],
                    nlsef_alpha2_range[0]], dtype=np.float32)
    high_list0 = np.array([h_range[1], r_range[1], b0_range[1], delat_range[1], eso_beta01_range[1], eso_beta02_range[1],
                     eso_beta03_range[1], nlsef_beta1_range[1], nlsef_beta2_range[1], nlsef_alpha1_range[1],
                     nlsef_alpha2_range[1]], dtype=np.float32)
    # ladrc的参数2个

    wc_range = [10, 40]
    w0_range = [simea, 5]

    low_list = np.array([r_range[0], wc_range[0], w0_range[0], b0_range[0]], dtype=np.float32)
    high_list = np.array([r_range[1], wc_range[1], w0_range[1], b0_range[1]], dtype=np.float32)
    def __init__(self):
        self.dt = self.ADRC_control.dt
        self.action_space = spaces.Box(low=self.low_list,high=self.high_list,shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10000,high=10000,shape=(9,), dtype=np.float32)
        self.v = np.zeros(int(2/self.ADRC_control.dt+1))
        #111   设定do_action的条件存在性
        self.do_action = True
        self.ADRC_control.v1[0] = self.landing_model.v

    def step(self, action) :
        global done
        done = 0
        if self.do_action:
            a_ = self.high_list - self.low_list
            b_ = a_*action+self.low_list
            self.ADRC_control.get_parameter(b_)

        #状态方程
        self.u = self.ADRC_control.con_transport(np.array(0), y=self.landing_model.z1_1).__float__()
        self.v = self.v[1:]
        self.landing_model.u = self.u
        self.landing_model.env_transport()
        self.state = np.array([self.landing_model.z1,self.landing_model.z1_1,self.landing_model.z2,self.landing_model.z2_1,
                            self.landing_model.s1, self.landing_model.s1_1,self.landing_model.Fd, self.landing_model.Ft,
                            self.landing_model.u])
        # 奖励设置和是否停止 done
        #self.e = self.v[0] - self.landing_model.z1_1
        self.e = 0 - self.landing_model.z1_1
        reward = abs(self.e)
        if len(self.landing_model.logger.z1_1)>100:
            if all(map(lambda x: x < 1, self.landing_model.logger.z1_1[-100:])):
                done =1
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def get_reward(self):
        self.r = 1
    def reset(self) :
        self.v = np.zeros(int(2 / self.ADRC_control.dt+1))
        # 将adrc和环境重置
        self.landing_model.reset()
        self.ADRC_control.reset()
        self.ADRC_control.v1 = np.ones(self.ADRC_control.dim) * self.landing_model.v
        # 重设环境的情况
        self.state = np.array([self.landing_model.z1,self.landing_model.z1_1,self.landing_model.z2,self.landing_model.z2_1,
                            self.landing_model.s1, self.landing_model.s1_1,self.landing_model.Fd, self.landing_model.Ft,
                            self.landing_model.u])

        return np.array(self.state, dtype=np.float32)
    def save(self):
        Date_ADRC = pd.DataFrame(self.ADRC_control.logger.__dict__)
        print(Date_ADRC)
        Date_ADRC.to_excel('Data/ADRC.xlsx')
        Date_mode = pd.DataFrame(self.landing_model.logger.__dict__)
        print(Date_mode)
        Date_ADRC.to_excel('Data/mode.xlsx')
        Data = pd.merge(Date_ADRC,Date_mode)
        Data.to_excel('Data/data.xlsx')


'debug'
if __name__ == '__main__':
    enverment = Adrc_control_landing_env()
    print(enverment.ADRC_control.logger.__dict__)
    print(enverment.landing_model.z1_1)
    enverment.reset()
    list_t = np.arange(0.0, 8.0, enverment.dt)
    a = np.zeros(11)
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

