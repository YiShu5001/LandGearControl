##
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

param1 = np.array(pd.read_excel('ddpg_step.xlsx')['action1'])
param2 = np.array(pd.read_excel('ddpg_step.xlsx')['action2'])

##
# 时间、动作参数1和动作参数2的模拟数据
time = np.linspace(0, 8, 8000)


# 创建3D图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', box_aspect=(2, 1, 1))

# 设置视角
ax.view_init(elev=30, azim=45)

# 绘制3D曲线
ax.plot(time, param1, param2, label='Action Parameters over Time')

# 设置标签
ax.set_xlabel('Time')
ax.set_ylabel('kp')
ax.set_zlabel('ω0')

# 添加图例
ax.legend()

# 显示图形
plt.show()
