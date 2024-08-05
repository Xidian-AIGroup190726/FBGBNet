import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
# 读取图像
image = cv2.imread('/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1/train_image/000744.jpg',0)
#image = cv2.imread('/media/ExtDisk/yxt/HRSID/test_image/P0008_0_800_7200_8000.jpg',0)
img_array = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
plt.imshow(image,cmap = 'rainbow')
plt.show()
print(img_array)
# 将RGB图像转换为灰度图像
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 显示灰度图像
#cv2.imshow('Gray image', gray_image)

# 保存灰度图像
cv2.imwrite('/media/ExtDisk/gray_image.jpg', img_array)

# 关闭所有窗口
