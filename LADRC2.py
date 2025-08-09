# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:27:34 2022

@author: HJ
"""

''' ADRC '''
# model free controller
from typing import Union
from dataclasses import dataclass
import numpy as np
import pylab as pl
from landing_gear import mode

import matplotlib.pyplot as plt
from common_control import BaseController, SignalLike, ListLike, NdArray
import pandas as pd
__all__ = ['ADRCConfig', 'ADRC']


# ADRC控制器参数
@dataclass
class ADRCConfig:
    """LADRC自抗扰控制算法参数
    :param dt: float, 控制器步长
    :param b0: SignalLike, 扩张状态观测器(LESO)被控系统系数
    :param w0: float, 扩张状态观测器(LESO)被控系统频率系数
    :param wc: float, 跟踪器(TD)频率系数
    :param r: float,
    """
    dt: float = 0.001  # 控制器步长 (float)

    wc: float = 11.    # LSEF被控系统系数 (float)
    # 扩张状态观测器
    b0: float = 40.  # 被控系统系数 (float)
    w0: float = 0.001  # 被控系统系数 (float)

    r: float = 9 # TD跟踪器 (float)

    # 控制约束
    u_max: SignalLike = float(1)   # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
    u_min: SignalLike = float(-1)  # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)


class Logger:
    pass


# ADRC自抗扰算法
class ADRC(BaseController):
    """ADRC自抗扰控制"""
    def __init__(self, cfg: ADRCConfig):
        super().__init__()
        self.name = 'LADRC'  # 算法名称
        self.dt = cfg.dt  # 仿真步长
        self.dim = 1 # 反馈信号y和跟踪信号v的维度
        # TD超参
        self.wc = cfg.wc
        self.b0 = cfg.b0
        self.w0 = cfg.w0
        self.r = cfg.r
        # TD
        self.dt = cfg.dt
        # 控制器初始化
        self.v1 = np.ones(self.dim) * 3.
        self.v2 = np.zeros(self.dim)
        self.z1 = np.zeros(self.dim)
        self.z2 = np.zeros(self.dim)
        self.z3 = np.zeros(self.dim)
        self.u = np.zeros(self.dim)
        self.t = 0

        # 存储器
        self.logger = Logger()
        self.logger.t = []  # 时间
        self.logger.u = []  # 控制
        self.logger.y = []  # 实际信号
        self.logger.v = []  # 输入信号
        self.logger.v1 = []  # 观测
        self.logger.e1 = []  # 误差1
        self.logger.e2 = []  # 误差2
        self.logger.z3 = []  # 干扰

        # 控制约束
        self.u_max = cfg.u_max
        self.u_min = cfg.u_min

    @staticmethod
    def getConfig():
        return ADRCConfig

    def get_parameter(self, action):
        # TD超参
        self.wc = action[1]
        self.b0 = action[3]
        self.w0 = action[2]
        self.r = action[0]
    def reset(self):
        # 控制器初始化
        self.v1 = np.ones(self.dim) * 3.
        self.v2 = np.zeros(self.dim)
        self.z1 = np.zeros(self.dim)
        self.z2 = np.zeros(self.dim)
        self.z3 = np.zeros(self.dim)
        self.u = np.zeros(self.dim)
        self.t = 0
        # 存储器  清空
        self.logger = Logger()
        self.logger.t = []  # 时间
        self.logger.u = []  # 控制
        self.logger.y = []  # 实际信号
        self.logger.v = []  # 输入信号
        self.logger.v1 = []  # 观测
        self.logger.e1 = []  # 误差1
        self.logger.e2 = []  # 误差2
        self.logger.z3 = []  # 干扰

    # ADRC控制器（v为参考轨迹，y为实际轨迹）
    def con_transport(self, v, y, *, ctrl_method=1) -> NdArray:
        v = np.array(v)
        y = np.array(y)
        # TD
        self._TD(v)
        # ESO
        self._ESO(y)
        self.z1 = np.nan_to_num(self.z1)
        self.z2 = np.nan_to_num(self.z2)
        self.z3 = np.nan_to_num(self.z3)
        # NLSEF
        e1 = self.v1 - self.z1
        e2 = self.v2 - self.z2
        u0 = self._NLSEF(e1, e2, ctrl_method)
        # 控制量
        self.u = u0 - self.z3 / self.b0
        self.u = np.clip(self.u, self.u_min, self.u_max)
        self.t += self.dt

        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u.item())
        self.logger.y.append(y.item())
        self.logger.v.append(v.item())
        self.logger.v1.append(self.v1.item())
        self.logger.e1.append((self.v1 - self.z1).item())
        self.logger.e2.append((self.v2 - self.z2).item())
        self.logger.z3.append(self.z3.item())
        return self.u

    # 跟踪微分器
    def _TD(self, v: NdArray):
        fh = -self.r * self.r * (self.v1 - v) - 2 * self.r * self.v2
        self.v2 += fh * self.dt
        self.v1 += self.v2 * self.dt


    # 扩张状态观测器   
    def _ESO(self, y):
        self.l1 = 3 * self.w0
        self.l2 = 3 * self.w0 * self.w0
        self.l3 = self.w0 * self.w0 * self.w0

        err = self.z1 - y
        self.z1 += (self.z2 - self.l1 * err) * self.dt
        self.z2 += (self.z3 - self.l2 * err + self.b0 * self.u) * self.dt
        self.z3 += -self.l3 * err * self.dt
        return None

    # 非线性状态误差反馈控制律
    def _NLSEF(self, e1: NdArray, e2: NdArray, ctrl_method=1) -> NdArray:
        self.kp = self.wc * self.wc
        self.kd = 2 * self.wc

        self.u0 = self.kp * e1 + self.kd * e2
        self.u = (self.u0 - self.z3) / self.b0
        return self.u0  # (dim, )


    # 输出
    def show(self, interference: ListLike = None, *, save=False, show_img=True):
        """控制器控制效果绘图输出
        :param interference: ListLike, 实际干扰数据, 用于对比ADRC控制器估计的干扰是否准确
        :param save: bool, 是否存储绘图
        :param show_img: bool, 是否CMD输出图像
        """
        # 响应曲线 与 控制曲线
        super().show(save=save)
        # TD曲线
        self._figure(fig_name='Tracking Differentiator (TD)', t=self.logger.t,
                     y1=self.logger.v1, y1_label='td',
                     y2=self.logger.v, y2_label='input',
                     xlabel='time', ylabel='response signal', save=save)
        # 误差曲线
        self._figure(fig_name='Error Curve', t=self.logger.t,
                     y1=self.logger.e1, y1_label='error',
                     xlabel='time', ylabel='error signal', save=save)
        self._figure(fig_name='Differential of Error Curve', t=self.logger.t,
                     y1=self.logger.e2, y1_label='differential estimation of error',
                     xlabel='time', ylabel='error differential signal', save=save)
        # 干扰估计曲线
        if interference is not None:
            interference = interference if len(interference) == len(self.logger.t) else None
        self._figure(fig_name='Interference Estimation', t=self.logger.t,
                     y1=self.logger.e2, y1_label='interference estimation',
                     y2=interference, y2_label='real interference',
                     xlabel='time', ylabel='interference signal', save=save)
        # 显示图像
        if show_img:
            self._show_img()


'debug'
if __name__ == '__main__':
    cfg = ADRCConfig()
    adrc = ADRC(cfg)
    land_gear = mode.landing_gear()
    list_z1 = []
    list_z1_1 = []
    list_t = np.arange(0.0, 2.0, land_gear.dt)
    y = 3
    for i in range(2000):
        u = adrc.con_transport(0, y=y)
        land_gear.u = u
        land_gear.env_transport()

        list_z1.append(float(land_gear.z1))
        y = land_gear.z1_1
        list_z1_1.append(float(land_gear.z1_1))
    #        print(land_gear.s,land_gear.s_1)
    # print(adrc.logger.y)
    #    print(type(land_gear.s),type(list_s))
    data = pd.DataFrame(list_z1_1)
    data.to_excel('控制.xlsx')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(1)
    plt.plot(list_t, np.array(list_z1, dtype=object), label='m1位移')
    plt.plot(list_t, list_z1_1, label='m1速度')
    plt.legend()
    plt.figure(2)
    plt.plot(list_t, adrc.logger.u, label='控制')
    plt.legend()
    plt.show()
    print(len(adrc.logger.t), len(adrc.logger.e2))
    adrc.show()

