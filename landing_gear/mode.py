import math
import numpy as np
import pydy
import collections as co
import pandas as pd
import scipy
import sympy
from sympy import *
from sympy.physics.mechanics import *
import matplotlib.pyplot as plt
from typing import Union
from dataclasses import dataclass
import numpy as np
import pylab as pl
from landing_gear import mode

import matplotlib.pyplot as plt
from common_control import BaseController,SignalLike, ListLike, NdArray

class Logger:
    pass
class landing_gear_symbols():
    def __init__(self):
        super().__init__()
        # 参量设置
        self.dt = 0.001
        self.t = 0
        #设定作用电压u ,被控量 s,s_1
        self.s0 = 0.165
        self.s1_1 = 0
        self.s2 = 0
        self.u = 0.0
        # 实验变化量 载重m1 和 冲击速度v
        self.m1 = 600
        self.v = 3.5
        # 定值
        self.m2 = 18
        self.Ap = 2.5*10**(-3)
        self.b = 1.13
        self.C = 6.58
        self.g = 9.81
        self.n = 1.3
        self.P0 = 810
        self.Pa = 101.3
        self.kt = 412
        self.V0 = 6.37*10**(-4)

    def func_Fa(self): #s1 上部分速度
        x1 = self.V0/(self.V0 - self.Ap* self.s1)
        x2 = math.copysign(1,x1)* pow( abs(x1),self.n)
        Fa = self.Ap*(float(self.P0 * x2) - self.Pa) *1000
        self.Fa = Fa
        return Fa
    def func_Fmr(self): #s1_1 上部分速度 u
        Fmr = self.u * math.copysign(1,self.s1_1)*10000
        self.Fmr = Fmr
        return Fmr
    def func_Fv(self): #s1_1 上部分速度
        Fv = self.C * self.s1_1 *1000
        self.Fv = Fv
        return Fv
    def func_Ft(self): #s2 = z2+ground
        x1 = math.copysign(1,self.s2)* pow( abs(self.s2),self.b)
        Ft = self.kt * x1 * 1000
        self.Ft = max(Ft,0)
        return self.Ft

class landing_gear(landing_gear_symbols):
    def __init__(self):
        super().__init__()
        self.unstable_change = True
        np.random.seed(2024022268)
        self.z1 = 0.
        self.z1_1 = self.v
        self.z2 = 0.
        self.z2_1 = self.v

        self.s1 = self.z1 -self.z2
        self.s1_1 = self.z1_1 -self.z2_1

        self.ground = 0
        self.s2 = self.z2 +self.ground

        self.t = 0
        # 路面参数
        self.gain = 0.05
        self.Tend = 5
        self.X = [self.z1, self.z2, self.z1_1, self.z2_1]
        # 存储器
        self.logger = Logger()
        self.logger.t = []
        self.logger.u = []
        self.logger.s1 = []
        self.logger.s2 = []
        self.logger.s1_1 = []
        self.logger.z1 = []
        self.logger.z1_1 = []
        self.logger.z2 = []
        self.logger.z2_1 = []
        self.logger.X = []
        self.logger.Fd = []
        self.logger.Ft = []
        self.logger.Fa = []
        self.logger.way = []
        #initial
        self.Ft = 0
        self.Fa = 0
        self.Fv = 0
        self.Fmr = 0
        self.Fd = 0
    def reset(self,v=3.0,u=0.,m1 = 600):
        self.u = u
        self.v = v
        self.z1 = 0.
        self.z1_1 = self.v
        self.z2 = 0.
        self.z2_1 = self.v
        self.s1 = 0
        self.s1_1 = 0
        self.s2 = 0
        self.m1 = m1
        self.t = 0
        # 存储器
        self.logger = Logger()
        self.logger.t = []
        self.logger.u = []
        self.logger.s1 = []
        self.logger.s1_1 = []
        self.logger.s2 = []
        self.logger.z1 = []
        self.logger.z1_1 = []
        self.logger.z2 = []
        self.logger.z2_1 = []
        self.logger.X = []
        self.logger.Fd = []
        self.logger.Ft = []
        self.logger.Fa = []
        self.logger.way = []

    def unstable_way(self, Ts=4 ):

        if self.t >= Ts and self.t <= Ts + np.pi / self.Tend:
            x = self.t - Ts
            p = self.gain * math.sin(self.Tend * x)
        else:
            p = 0
        self.ground = p + np.random.random()/1000

        return self.ground

    def env_transport(self):        #顺序： s-->力(从下往上)-->加速度-->位移

        if self.unstable_change:
            self.ground = self.unstable_way()
        else:
            self.ground = 0



        self.s1 = self.z1 - self.z2
        self.s1_1 = self.z1_1 - self.z2_1
        self.s2 = self.z2 + self.ground

        self.Ft = self.func_Ft()
        self.Fa = self.func_Fa()
        self.Fv = self.func_Fv()
        self.Fmr = self.func_Fmr()
        self.Fd = self.Fmr + self.Fa + self.Fv
        #print(self.Fa,self.Fv,self.Fmr,self.Ft)

        self.z1_1 += (self.g - self.Fd/self.m1) * self.dt
        self.z2_1 += (self.g + (self.Fd - self.Ft) / self.m2) * self.dt
        self.z1 += self.z1_1 * self.dt
        self.z2 += self.z2_1 * self.dt

        self.t += self.dt
        self.X = [self.z1, self.z2, self.z1_1, self.z2_1]
        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u)
        self.logger.s1.append(self.s1)
        self.logger.s2.append(self.s2)
        self.logger.s1_1.append(self.s1_1)
        self.logger.z1.append(self.z1)
        self.logger.z1_1.append(self.z1_1)
        self.logger.z2.append(self.z2)
        self.logger.z2_1.append(self.z2_1)
        self.logger.X.append(self.X)
        self.logger.Fd.append(self.Fd)
        self.logger.Fa.append(self.Fa)
        self.logger.Ft.append(self.Ft)
        self.logger.way.append(self.ground)

'debug'
if __name__ == '__main__':
    sym = landing_gear_symbols()
# 更改仿真时长
#    sym.dt = sym.dt / 10
    land_gear = landing_gear()
    list_s1 = []
    list_s1_1 = []
    list_z1 = []
    list_z1_1 = []
    list_z2 = []
    list_z2_1 = []
    list_t = np.arange(0.0, 8.0, sym.dt)
    land_gear.u = 0
    for i in list_t:
        if land_gear.z1_1 * land_gear.s1_1 > 0:
            u1 = 0.126*land_gear.z1_1
        else:
            u1 = 0
        if land_gear.z2_1 * land_gear.s1_1 < 0:
            u2 = 0.126*land_gear.z1_1
        else:
            u2 = 0
        land_gear.u = u1+u2
        #print(land_gear.u)

        land_gear.u = 0
        land_gear.env_transport()
        list_s1.append(land_gear.s1)
        list_z1.append(land_gear.z1)
        list_z1_1.append(land_gear.z1_1)
        list_s1_1.append((land_gear.s1_1))
        list_z2.append(land_gear.z2)
        list_z2_1.append((land_gear.z2_1))

        #print(land_gear.logger.way)


    #print(land_gear.logger.__dir__())
    #print(land_gear.logger.__dict__.__len__())

    data = pd.DataFrame(land_gear.logger.__dict__,columns=land_gear.logger.__dir__()[0:14])
    print(data)
   # data.to_excel('nocontrol.xlsx')

    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list_t, list_s1, label = 's伸缩位移' )
    plt.plot(list_t, list_s1_1, label = 's伸缩速度')
    plt.plot(list_t, np.zeros(list_t.__len__()), label='0')
    plt.legend()

    plt.figure(2)
    plt.plot(list_t, list_z1, label='m1位移')
    plt.plot(list_t, list_z1_1, label='m1速度')
    plt.plot(list_t, np.zeros(list_t.__len__()), label='0')
    plt.legend()

    plt.figure(3)
    plt.plot(list_t, list_z2, label='m2位移')
    plt.plot(list_t, list_z2_1, label='m2速度')
    plt.plot(list_t, np.zeros(list_t.__len__()), label='0')
    plt.legend()

    plt.figure(4)
    plt.plot(list_t, list_z1, label='m1位移')
    plt.plot(list_t, list_z2, label='m2位移')
    plt.plot(list_t, list_s1, label='s位移')
    plt.plot(list_t, np.zeros(list_t.__len__()), label='0')
    plt.legend()
    plt.figure(5)
    plt.plot(list_t, list_z1_1, label='m1速度')
    plt.plot(list_t, list_z2_1, label='m2速度')
    plt.plot(list_t, list_s1_1, label='s速度')
    plt.plot(list_t, np.zeros(list_t.__len__()), label='0')
    plt.legend()

    plt.figure(6)
    plt.plot(list_z1, land_gear.logger.Fd, label='')
#    plt.plot(list_z1, land_gear.logger.Fa, label='')
    plt.legend()

    plt.figure(7)
    plt.plot(list_t, land_gear.logger.way, label='way')
    plt.ylabel('Height(m)')
    plt.xlabel('Time(s)')
    plt.savefig('way.pdf', format='pdf', dpi=1200, transparent=True, bbox_inches='tight')
    plt.savefig('way.png', format='png', dpi=1200, transparent=True, bbox_inches='tight')
    plt.legend()

    plt.figure(8)
    plt.plot(list_t, land_gear.logger.u, label='路况')
    plt.legend()
    plt.show()
