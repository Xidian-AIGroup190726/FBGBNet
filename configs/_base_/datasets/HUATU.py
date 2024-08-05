from matplotlib.pyplot import figure, ylabel, scatter, xticks, show, xlabel, rcParams
import numpy as np

# 1. 首先是导入包，创建数据
n = 10
x = [20,33,4.8,34.1,28.4,34.3,36.4,33.1,33.5]  # 随机产生10个0~2之间的x坐标
y = [8.4/1000,31.3/1000,208.6/1000,29.1/1000,34.5/1000,29.2/1000,27.5/1000,30.2/1000,30/1000]  # 随机产生10个0~2之间的y坐标
# 2.创建一张figure
fig = figure(1)
# 3. 设置颜色 color 值【可选参数，即可填可不填】，方式有几种
# colors = np.random.rand(n) # 随机产生10个0~1之间的颜色值，或者
colors = ['r', 'g', 'y', 'b', 'r', 'c', 'g', 'b', 'k']  # 可设置随机数取
# 4. 设置点的面积大小 area 值 【可选参数】
area = 10 * np.arange(1, n + 1)
# 5. 设置点的边界线宽度 【可选参数】
widths = np.arange(5)  # 0-9的数字
# 6. 正式绘制散点图：scatter
scatter(x, y, c=colors, linewidths=1, alpha=0.5, marker='o')
# 7. 设置轴标签：xlabel、ylabel
# 设置X轴标签
xlabel('FPS(img/s)')
# 设置Y轴标签
ylabel('Per img times(s)')
# 8. 设置图标题：title
# 9. 设置轴的上下限显示值：xlim、ylim
# 设置横轴的上下限值
#plt.xlim(25, 120)
# 设置纵轴的上下限值
#plt.ylim(10, 70)
# 10. 设置轴的刻度值：xticks、yticks
# 设置横轴精准刻度
#plt.xticks(np.arange(np.min(x) - 0.2, np.max(x) + 0.2, step=0.3))
# 设置纵轴精准刻度
#plt.yticks(np.arange(np.min(y) - 0.2, np.max(y) + 0.2, step=0.3))
# 也可按照xlim和ylim来设置
# 设置横轴精准刻度
xticks(np.arange(0, 50, step=5))
# 设置纵轴精准刻度
#plt.yticks(np.arange(-0.5, 2.5, step=0.5))

# 11. 在图中某些点上（位置）显示标签：annotate
# plt.annotate("(" + str(round(x[2], 2)) + ", " + str(round(y[2], 2)) + ")", xy=(x[2], y[2]), fontsize=10, xycoords='data')# 或者
####plt.annotate("({0},{1})".format(round(x[2], 2), round(y[2], 2)), xy=(x[2], y[2]), fontsize=10, xycoords='data')
# xycoords='data' 以data值为基准
# 设置字体大小为 10
# 12. 在图中某些位置显示文本：text
####plt.text(round(x[6], 2), round(y[6], 2), "good point", fontdict={'size': 10, 'color': 'red'})  # fontdict设置文本字体
# Add text to the axes.
# 13. 设置显示中文
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 14. 设置legend，【注意，'绘图测试’：一定要是可迭代格式，例如元组或者列表，要不然只会显示第一个字符，也就是legend会显示不全】
#plt.legend(['HUI'], loc=2, fontsize=10)
# plt.legend(['绘图测试'], loc='upper left', markerscale = 0.5, fontsize = 10) #这个也可
# markerscale：The relative size of legend markers compared with the originally drawn ones.
# 15. 保存图片 savefig
#plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
# dpi: The resolution in dots per inch，设置分辨率，用于改变清晰度
# If *True*, the axes patches will all be transparent
# 16. 显示图片 show
show()