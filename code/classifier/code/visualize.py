import numpy as np
import os
import matplotlib.pyplot as plt
 
''' 箱线图 '''
'''
#读取数据
res_LWAN = np.array([0.603174627,
0.559459459,
0.589189189,
0.581081081,
0.575675676,
0.589189189,
0.586486486,
0.605405405,
0.57027027,
0.578378378
])
res_prototype = np.array([0.6,
0.621621622,
0.602702703,
0.618918919,
0.605405405,
0.613513514,
0.616216216,
0.608108108,
0.613513514,
0.597297297
])
box_1, box_2 = res_LWAN, res_prototype
 
plt.figure(figsize=(4,3))#设置画布的尺寸
# plt.title('Re',fontsize=20)#标题，并设定字号大小
#boxprops：color箱体边框色，facecolor箱体填充色；
bplot = plt.boxplot([box_1, box_2,],patch_artist = True, labels=["LWAN", "Prototype"])
# colors = ["#d9ddef", "#f6d7b5"]
# facecolors = ["#4169e1", "#ffa500"]
facecolors = ["#4169e1", "#ffa500"]
for patch, facecolor in zip(bplot['boxes'], facecolors):
        patch.set_facecolor(facecolor)

plt.savefig(os.path.join("/media/data/3/lyp", "box_fig.png")) 
'''

''' 折线图，在colab上跑，目前的numpy会报错 '''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

point_colors = ["#d62728", "#ff7f0e", "#1f77b4"] # 红 黄 蓝
# point_colors = ["d62728", "ff7f0e", "1f77b4"] # 红 黄 蓝
face_colors = ["#f6d3d4", "#ffe5ce", "#d2e3f0"] # 红 黄 蓝
# face_colors = ["f6d3d4", "ffe5ce", "d2e3f0"] # 红 黄 蓝
x = [150,300,450,600,750]
# x_label=[32, 64, 128, 256, 1024]
# x_label=[0.0, 0.25, 0.5, 0.75, 1.0]
x_label=[5, 10, 25, 50, 76]

# y1=[0.610899873, 0.626086957, 0.607594937, 0.612945839, 0.631168831] # 最高
# y2=[0.597402597, 0.586666667, 0.604298357, 0.605729877, 0.609150327] # 最低
# y3=[0.608247423, 0.613636364, 0.605757196, 0.612945839, 0.623243934] # 中位数

# y1=[0.249628529, 0.631168831, 0.618181818, 0.597468354, 0.62972973] # 最高
# y2=[0.249443207, 0.609150327, 0.589580686, 0.580729167, 0.558459422] # 最低
# y3=[0.249443207, 0.623243934, 0.610303831, 0.581151832, 0.569491525] # 中位数

y1=[0.610303831, 0.616352201, 0.614775726, 0.620599739, 0.641460235] # 最高
y2=[0.586206897, 0.593710692, 0.597701149, 0.596273292, 0.617486339] # 最低
y3=[0.592207792, 0.602631579, 0.606382979, 0.617524339, 0.625310174] # 中位数
# plt.xlabel('value of prototype dimension')
# plt.xlabel('value of temperature')
plt.xlabel('number of weak learners')
plt.ylabel('micro F1 score')

#把x轴的刻度间隔设置为1，并存在变量里
plt.xticks(x, x_label)

plt.plot(x, y3, color=point_colors[2], linestyle='solid', marker='D', markersize=6)
plt.fill_between(x, y1, y2, #上限，下限
        facecolor=face_colors[2], #填充颜色
        edgecolor='white', #边界颜色
        alpha=1) #透明度
plt.show()

