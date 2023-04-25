import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import io, draw, color, filters,segmentation

# 绘制图样示例函数
def img_show(image, nrows=1,ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    # plt.show()
    return fig, ax

img = io.imread(r'H:\图片\素材\Gril_7.jpg')
# plt.imshow(img)
# plt.show()

# 滤波去噪/ 灰度转换
img_gray = color.rgb2gray(img)
# img_show(img_gray)

###################################################
#################### 一、监督分割 ###################
###################################################

############### 1. 主动轮廓分割 ##############

# 人的头部周围画一个圆来初始化snake
def circle_points(resolution, center, radius):
    radians = np.linspace(0, 2*np.pi, resolution)
    col = center[1] + radius*np.cos(radians)
    row = center[0] + radius*np.sin(radians)
    return np.array([col, row]).T

# Exclude last point because a closed path should not have duplicate points

# points = circle_points(200, [300, 500], 200)[:-1]
# fig, ax = img_show(img)
# ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
# plt.show()

# 算法通过将闭合曲线拟合到脸部的边缘来从人的图像的其余部分分割人的面部
# 调整alpha和beta的参数。较高的alpha值会使snake收缩得更快，而beta会让snake变得更加平滑。

# snake = segmentation.active_contour(img_gray, points, alpha=0.06, beta=0.3)
# fig, ax = img_show(img)
# ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
# ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# plt.show()


###################################################
#################### 二、无监督分割 #################
###################################################

########## 1. SLIC(简单线性迭代聚类) ##########

img_slic = segmentation.slic(img, n_segments=50, sigma=5)
# img_show(color.label2rgb(img_slic, img, kind='avg'))
#显示分割后的边界线条
img_seg = segmentation.mark_boundaries(img, img_slic, color=(1, 1, 0), mode='outer', background_label=0)
img_show(img_seg)
plt.show()

