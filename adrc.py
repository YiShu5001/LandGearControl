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
from common_control import BaseController,SignalLike, ListLike, NdArray


__all__ = ['ADRCConfig', 'ADRC']


# ADRC控制器参数
@dataclass
class ADRCConfig:
    """ADRC自抗扰控制算法参数
    :param dt: float, 控制器步长
    :param dim: int, 输入信号维度, 即控制器输入v、y的维度, ADRC输出u也为dim维
    :param h: float, 跟踪微分器(TD)滤波因子, 系统调用步长, 默认None设置成dt
    :param r: SignalLike, 跟踪微分器(TD)快速跟踪因子
    :param b0: SignalLike, 扩张状态观测器(ESO)被控系统系数
    :param delta: SignalLike, ESO的fal(e, alpha, delta)函数线性区间宽度
    :param beta01: SignalLike, ESO的反馈增益1
    :param beta02: SignalLike, ESO的反馈增益2
    :param beta03: SignalLike, ESO的反馈增益3
    :param beta1: SignalLike, NLSEF参数, 跟踪输入信号的增益
    :param beta2: SignalLike, NLSEF参数, 跟踪微分信号的增益
    :param alpha1: SignalLike, 非线性反馈控制律(NLSEF)参数, 0 < alpha1 < 1
    :param alpha2: SignalLike, NLSEF参数, alpha2 > 1
    :Type : SignalLike = float (标量) 或 list / ndarray (一维数组即向量)\n
    备注:\n
    dim>1时SignalLike为向量时, 相当于同时设计了dim个不同的ADRC控制器, 必须满足dim==len(SignalLike)\n
    dim>1时SignalLike为标量时, 相当于设计了dim个参数相同的ADRC控制器, 控制效果可能不好\n
    """

    dt: float = 0.001              # 控制器步长 (float)
    dim: int = 1                   # 输入维度 (int)
    # 跟踪微分器
    h: float = 0.001               # 滤波因子，系统调用步长，默认None设置成dt (float)
    r: SignalLike = 100        # 快速跟踪因子 (float or list)
    # 扩张状态观测器
    b0: SignalLike = 30.          # 被控系统系数 (float or list)
    delta: SignalLike = 1.2      # fal(e, alpha, delta)函数线性区间宽度 (float or list)
    eso_beta01: SignalLike = 20.  # ESO反馈增益1 (float or list)
    eso_beta02: SignalLike = 250  # ESO反馈增益2 (float or list)
    eso_beta03: SignalLike = 0.1  # ESO反馈增益3 (float or list)
    # 非线性状态反馈控制率
    nlsef_beta1: SignalLike = 1.       # 跟踪输入信号增益 (float or list)
    nlsef_beta2: SignalLike = 0.5     # 跟踪微分信号增益 (float or list)
    nlsef_alpha1: SignalLike = 0.1   # 0 < alpha1 < 1  (float or list)
    nlsef_alpha2: SignalLike = 0.8   # alpha2 > 1      (float or list)
    # 控制约束
    u_max: SignalLike = float(10)   # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
    u_min: SignalLike = float(-10)  # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)

    def __post_init__(self):
        if self.h is None:
            self.h = self.dt

class Logger:
    pass

# ADRC自抗扰算法
class ADRC(BaseController):
    """ADRC自抗扰控制"""

    @staticmethod
    def _reshape_param(param: Union[float, list[float], NdArray], dim: int) -> NdArray:
        """float | array_like -> ndarray (dim, )"""
        param = pl.array(param).flatten()  # (dim0, ) or (1, )
        if len(param) != dim:
            assert len(param) == 1, "param为float或dim维的ArrayLike"
            return param.repeat(dim)  # (dim, )
        return param

    def __init__(self, cfg: ADRCConfig):
        super().__init__()
        self.name = 'ADRC'       # 算法名称
        self.dt = cfg.dt         # 仿真步长
        self.dim = cfg.dim       # 反馈信号y和跟踪信号v的维度
        # TD超参
        self.r = self._reshape_param(cfg.r, self.dim) # 快速跟踪因子
        self.h = self._reshape_param(cfg.h, self.dim) # 滤波因子，系统调用步长
        # ESO超参
        self.b0 = self._reshape_param(cfg.b0, self.dim)             # 系统系数
        self.delta = self._reshape_param(cfg.delta, self.dim)       # fal(e, alpha, delta)函数线性区间宽度        
        self.beta01 = self._reshape_param(cfg.eso_beta01, self.dim) # ESO反馈增益1
        self.beta02 = self._reshape_param(cfg.eso_beta02, self.dim) # ESO反馈增益2
        self.beta03 = self._reshape_param(cfg.eso_beta03, self.dim) # ESO反馈增益3
        # NLSEF超参
        self.beta1 = self._reshape_param(cfg.nlsef_beta1, self.dim)   # 跟踪输入信号增益
        self.beta2 = self._reshape_param(cfg.nlsef_beta2, self.dim)   # 跟踪微分信号增益
        self.alpha1 = self._reshape_param(cfg.nlsef_alpha1, self.dim) # 0 < alpha1 < 1 < alpha2
        self.alpha2 = self._reshape_param(cfg.nlsef_alpha2, self.dim) # alpha2 > 1
        # 控制约束
        self.u_max = self._reshape_param(cfg.u_max, self.dim)
        self.u_min = self._reshape_param(cfg.u_min, self.dim)
        # 控制器初始化
        self.v1 = np.ones(self.dim) *3.
        self.v2 = np.zeros(self.dim)
        self.z1 = np.zeros(self.dim)
        self.z2 = np.zeros(self.dim)
        self.z3 = np.zeros(self.dim)
        self.u = np.zeros(self.dim)
        self.t = 0
        
        # 存储器
        self.logger = Logger()
        self.logger.t = [] # 时间
        self.logger.u = [] # 控制
        self.logger.y = [] # 实际信号
        self.logger.v = [] # 输入信号
        self.logger.v1 = []    # 观测
        self.logger.e1 = []    # 误差1
        self.logger.e2 = []    # 误差2
        self.logger.z3 = []    # 干扰


    @staticmethod
    def getConfig():
        return ADRCConfig

    def get_parameter(self,action):
        # TD超参
        self.h = self._reshape_param(action[0], self.dim) # 滤波因子，系统调用步长
        self.r = self._reshape_param(action[1], self.dim) # 快速跟踪因子
        # ESO超参
        self.b0 = self._reshape_param(action[2], self.dim)             # 系统系数
        self.delta = self._reshape_param(action[3], self.dim)       # fal(e, alpha, delta)函数线性区间宽度
        self.beta01 = self._reshape_param(action[4], self.dim) # ESO反馈增益1
        self.beta02 = self._reshape_param(action[5], self.dim) # ESO反馈增益2
        self.beta03 = self._reshape_param(action[6], self.dim) # ESO反馈增益3
        # NLSEF超参
        self.beta1 = self._reshape_param(action[7], self.dim)   # 跟踪输入信号增益
        self.beta2 = self._reshape_param(action[8], self.dim)   # 跟踪微分信号增益
        self.alpha1 = self._reshape_param(action[9], self.dim) # 0 < alpha1 < 1 < alpha2
        self.alpha2 = self._reshape_param(action[10], self.dim) # alpha2 > 1

    def reset(self):
        # 控制器初始化
        self.v1 = np.ones(self.dim)*5.
        self.v2 = np.zeros(self.dim)
        self.z1 = np.zeros(self.dim)
        self.z2 = np.zeros(self.dim)
        self.z3 = np.zeros(self.dim)
        self.u = np.zeros(self.dim)
        self.t = 0
        # 存储器  清空
        self.logger = Logger()
        self.logger.t = [] # 时间
        self.logger.u = [] # 控制
        self.logger.y = [] # 实际信号
        self.logger.v = [] # 输入信号
        self.logger.v1 = []    # 观测
        self.logger.e1 = []    # 误差1
        self.logger.e2 = []    # 误差2
        self.logger.z3 = []    # 干扰




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
        fh = self._fhan(self.v1 - v, self.v2, self.r, self.h)
        self.v1 = self.v1 + self.h * self.v2
        self.v2 = self.v2 + self.h * fh
    

    # 扩张状态观测器
    def _ESO(self, y: NdArray):
        e = self.z1 - y
        fe = self._fal(e, 1/2, self.delta)
        fe1 = self._fal(e, 1/4, self.delta)
        self.z1 = self.z1 + self.h * (self.z2 - self.beta01 * e)
        self.z2 = self.z2 + self.h * (self.z3 - self.beta02 * fe + self.b0 * self.u)
        self.z3 = self.z3 + self.h * (- self.beta03 * fe1)
    

    # 非线性状态误差反馈控制律
    def _NLSEF(self, e1: NdArray, e2: NdArray, ctrl_method=1) -> NdArray:
        ctrl_method %= 4
        if ctrl_method == 0:
            u0 = self.beta1 * e1 + self.beta2 * e2
        elif ctrl_method == 1:
            u0 = self.beta1 * self._fal(e1, self.alpha1, self.delta) + self.beta2 * self._fal(e2, self.alpha2, self.delta)
        elif ctrl_method == 2:
            u0 = -self._fhan(e1, e2, self.r, self.h)
        else:
            c = 1.5
            u0 = -self._fhan(e1, c*e2, self.r, self.h)
        return u0 # (dim, )
    

    @staticmethod
    def _fhan(x1: NdArray, x2: NdArray, r: NdArray, h: NdArray) -> NdArray:
        def fsg(x, d):
            return (np.sign(x + d) - np.sign(x - d)) / 2
        d = r * h**2
        a0 = h * x2
        y = x1 + a0
        a1 = np.sqrt(d * (d + 8*abs(y)) + 1e-8)
        a2 = a0 + np.sign(y) * (a1 - d) / 2
        a = (a0 + y) * fsg(y, d) + a2 * (1 - fsg(y, d))
        fh = -r * (a/d) * fsg(y, d) - r * np.sign(a) * (1 - fsg(a, d))
        return fh
    
    @staticmethod
    def _fal(err: NdArray, alpha: Union[NdArray, float], delta: NdArray) -> NdArray:
        if not isinstance(alpha, NdArray):
            alpha = np.ones_like(err) * alpha
        fa = np.zeros_like(err)
        mask = np.abs(err) <= delta
        fa[mask] = err[mask] / delta[mask]**(alpha[mask] - 1)
        fa[~mask] = np.abs(err[~mask])**alpha[~mask] * np.sign(err[~mask])
        return fa
    

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
    long = 6
    list_t = np.arange(0.0, long, land_gear.dt)
    y=5
    for i in range(long*1000):
        u = adrc.con_transport(0,y=y)
        land_gear.u = u
        land_gear.env_transport()

        list_z1.append(float(land_gear.z1))
        y = land_gear.z1_1
        list_z1_1.append(float(land_gear.z1_1))
#        print(land_gear.s,land_gear.s_1)
   # print(adrc.logger.y)
#    print(type(land_gear.s),type(list_s))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(1)
    plt.plot(list_t, np.array(list_z1, dtype=object), label = 'm1位移' )
    plt.plot(list_t, list_z1_1, label = 'm1速度')
    plt.legend()
    plt.figure(2)
    plt.plot(list_t, adrc.logger.u, label = '控制')
    plt.legend()
    plt.show()
    print(len(adrc.logger.t), len(adrc.logger.e2))
    #adrc.show()

