''' 接口，可视化一组向量 '''
''' 可视化proto在训练过程中的变化结果 '''

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import matplotlib as mpl


# 待可视化的proto: np.array(epoch_num，proto_dim, num_classes), 
# 处理成：x=(batch_size[这里等于epoch_num], proto_dim), y=(1, num_classes)[one-hot]

input_dir = "/media/data/3/lyp/CAMS_result/CAMS_length_4096_seed_52_2022-12-24-21-30-21-466959120_proto" # e.g. CAMS_length_4096_seed_52_2022-12-22-21-04-46-636856542_proto
epoch_num = len(os.listdir(input_dir))

proto = []
Y = []
for ep in range(epoch_num):
    input_path = os.path.join(input_dir, "proto_epoch_{}.npy".format(str(ep)))  
    ori_proto = np.load(input_path)
    num_classes = ori_proto.shape[1]
    for cl in range(num_classes):
        proto.append(ori_proto[:, cl].reshape(1, -1))
        y = np.zeros((1, num_classes), dtype=int)
        y[0, cl] = 1
        Y.append(y)
batch_size = len(proto)
proto = np.stack(proto).reshape(batch_size, -1)
Y = np.stack(Y).reshape(batch_size, -1)
assert(proto.shape[0] == Y.shape[0])
print(proto.shape)
print(Y.shape)

# 设置颜色
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list \
        ('cmap', ['#8B0000', '#FF6A6A', '#00FF00', '#F4A460', '#00CDCD', '#0000FF'], 256)
        #('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256) #white, CadetBlue1, Green1, Yellow1, Red,DarkRed

cm = colormap()

# 原始数据 X 、Y 的可视化 
color = Y
color = [np.argmax(i) for i in color]
color = np.stack(color, axis=0) # one-hot -> np.array 不可

fig = plt.figure(figsize=(8, 12))		# 指定图像的宽和高

'''t-SNE'''
n_components = 2 # 2D可视化
# n_components = 3 # 3D可视化
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0) 
y = tsne.fit_transform(proto)
 
ax1 = fig.add_subplot(1, 1, 1) # 添加新的子图可用 (3, 1, 2) (3, 1, 3) (子图个数，行idx，列idx)
color_list = ['#8B0000', '#FF6A6A', '#00FF00', '#F4A460', '#00CDCD', '#0000FF']
marker_size = 20
for i in range(len(color)):# 将标签号和每个类的画法进行对应。
    if color[i] == 0:
        s0 = plt.scatter(y[i, 0], y[i, 1], c=color_list[0], cmap=cm, s=marker_size, marker='*')
    elif color[i] == 1:
        s1 = plt.scatter(y[i, 0], y[i, 1], c=color_list[1], cmap=cm, s=marker_size, marker='*')
    elif color[i] == 2:
        s2 = plt.scatter(y[i, 0], y[i, 1], c=color_list[2], cmap=cm, s=marker_size, marker='*')
    elif color[i] == 3:
        s3 = plt.scatter(y[i, 0], y[i, 1], c=color_list[3], cmap=cm, s=marker_size, marker='*')
    elif color[i] == 4:
        s4 = plt.scatter(y[i, 0], y[i, 1], c=color_list[4], cmap=cm, s=marker_size, marker='*')
    else:
        s5 = plt.scatter(y[i, 0], y[i, 1], c=color_list[5], cmap=cm, s=marker_size, marker='*')

# for i in range(len(proto)):
#     plt.scatter(proto_y[i, 0], proto_y[i, 1], c = color_list[i], cmap=cm, s=4, marker='*')

plt.legend((s0,s1,s2,s3,s4,s5),('0','1','2','3','4','5') ,loc = 'best')#添加图例
ax1.set_title('Scatter Plot of Raw Data with Lagend', fontsize=14)
# 显示图像
# plt.show()
plt.savefig(os.path.join(input_dir, "fig.png")) 
#这里是要保存的路径/home/img_save_folder/和保存文件名Picture0.png、Picture1.png...
plt.close()





'''
https://blog.csdn.net/hesongzefairy/article/details/113527780

matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)
 
x,y组成了散点的坐标；s为散点的面积；c为散点的颜色（默认为蓝色'b'）；marker为散点的标记；alpha为散点的透明度（0与1之间的数，0为完全透明，1为完全不透明）;linewidths为散点边缘的线宽；如果marker为None，则使用verts的值构建散点标记；edgecolors为散点边缘颜色。
'''