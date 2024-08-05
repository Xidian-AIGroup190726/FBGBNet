import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker

# x=[str([2,3]),str([2.5,3.5]),str([2.5,4]),str([2,4.5]),str([2.5,5]),str([3,5.5]),str([3,6])]
x = [3, 3.5, 4, 4.5, 5, 5.5, 6]
x3 = [3.5, 4, 4.5, 5, 5.5, 6]
x35 = [4, 4.5, 5, 5.5, 6]
x4 = [4.5, 5, 5.5, 6]
x45 = [5, 5.5, 6]
x5 = [5.5, 6]
x55 = 6
# p1=[0.659,0.662,0.655,0.653,0.651,0.645,0.636]
r2 = [0.660, 0.660, 0.654, 0.653, 0.650, 0.644, 0.634]
r25 = [0.659, 0.662, 0.655, 0.652, 0.651, 0.642, 0.636]
r3 = [0.658, 0.652, 0.652, 0.648, 0.645, 0.636]
r35 = [0.651, 0.650, 0.647, 0.642, 0.635]
r4 = [0.649, 0.646, 0.640, 0.635]
r45 = [0.644, 0.639, 0.634]
r5 = [0.638, 0.633]
r55 = 0.633

# hrsid
R2 = [0.510, 0.512, 0.511, 0.508, 0.506, 0.501, 0.493]
R25 = [0.510, 0.511, 0.513, 0.510, 0.506, 0.502, 0.494]
R3 = [0.511, 0.512, 0.511, 0.507, 0.503, 0.495]
R35 = [0.511, 0.509, 0.508, 0.502, 0.495]
R4 = [0.508, 0.506, 0.501, 0.496]
R45 = [0.505, 0.499, 0.495]
R5 = [0.498, 0.493]
R55 = 0.491

color_list = ['#17becf', 'tab:blue', 'tab:orange', 'tab:green', 'red', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# label_list=[0.660,0.659,0.662,0.655,0.653,0.651,0.645,0.636]
plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
plt.plot(x, R2, linewidth=1, label="C2=2", color='g', marker='o', linestyle='-')
plt.plot(x, R25, linewidth=1, label="C2=2.5", color='r', marker='*', linestyle='-')
plt.plot(x3, R3, linewidth=1, label="C2=3", color='b', marker='p', linestyle='-')
plt.plot(x35, R35, linewidth=1, label="C2=3.5", color='tab:orange', marker='v', linestyle='-')
plt.plot(x4, R4, linewidth=1, label="C2=4", color='tab:pink', marker='+', linestyle='-')
plt.plot(x45, R45, linewidth=1, label="C2=4.5", color='#17becf', marker='D', linestyle='-')
plt.plot(x5, R5, linewidth=1, label="C2=5", color='k', marker='d', linestyle='-')
plt.plot(x55, R55, linewidth=1, label="C2=5.5", color='tab:olive', marker='1', linestyle='-')

# for i in range(0,len(p1)):
# plt.scatter(x[i],p1[i],c='tab:green',marker='o',s=30)#,label=label_list[i])
plt.legend(frameon=False, fontsize='x-small')
font1 = {'family': 'DejaVu Sans',
         'weight': 'normal',
         'size': 15, }
plt.xlabel('$\t{C1}$', font1)
plt.ylabel('$\t{mAP}$', font1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('/media/ExtDisk/yxt/IMG/HRSID.png', dpi=800)
plt.show()
