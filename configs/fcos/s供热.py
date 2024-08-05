import json

import os, cv2

from PIL import Image

import numpy as np

import json

import os, cv2

import math

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def image_hist(image_path: str):
    # def image_hist(img):

    """

    图像直方图是反映一个图像像素分布的统计表，其横坐标代表了图像像素的种类，可以是灰度的，也可以是彩色的。

    纵坐标代表了每一种颜色值在图像中的像素总数或者占所有像素个数的百分比。

    图像是由像素构成，那么反映像素分布的直方图往往可以作为图像一个很重要的特征。

    直方图的显示方式是左暗又亮，左边用于描述图像的暗度，右边用于描述图像的亮度。

    :param image_path: 传入查找像素的图像文件

    :return: 无返回值

    """

    # 一维直方图（单通道直方图）

    img = cv2.imread(image_path, 0)

    # cv2.imshow('input', img)

    color = ('blue', 'green', 'red')

    # 使用plt内置函数直接绘制

    ##plt.hist(img.ravel(), 20, [0, 256])

    ##plt.show()

    # 一维像素直方图，也即是单通道直方图

    # for i, color in enumerate(color):

    global hist

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # print(hist)

    plt.plot(hist, color="r")

    # plt.plot(hist, color=color)

    # plt.xlim([0, 256])

    #plt.show()

    # return hist

    # plt.show()

    # cv2.waitKey(0)

    # cv2.destroyAllWindows()


def cal_res(img_path, image_name):
    tmp = []

    for i in range(256):
        tmp.append(0)

    val = 0

    k = 0

    res = 0

    # I = Image.open('/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1/train_image/000603.jpg' )

    I = Image.open(img_path)

    # I1 = Image.open(img_path)

    # I = I1.resize((400, 400))

    # I = I1.resize((400, 400))

    greyIm = I.convert('L')

    #greyIm.show()

    img = np.array(greyIm)

    x, y = img.shape

    dst = np.zeros([x, y])

    for i in range(x):

        for j in range(y):

            if img[i, j] > 150:

                dst[i, j] = 255

            else:

                dst[i, j] = 0

    # retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # cv2.imwrite('binary.jpg', dst)

    # cv2.imwrite('/content/sample_data/2/'+str(image_name).zfill(5), dst)

    m = 0

    for i in range(dst.shape[0]):

        for j in range(dst.shape[1]):

            # print(dst[i,j])

            if dst[i, j] == 255:
                m += 1

    for i in range(len(img)):

        for j in range(len(img[i])):
            val = img[i][j]

            tmp[val] = float(tmp[val] + 1)

            k = float(k + 1)

    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)

    tmp = np.array(tmp)

    for i in range(len(tmp)):

        if (tmp[i] == 0):

            res = res

        else:

            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))

    # print("res:",res)

    # if res-7<=0.1 and res>=6:

    # if res>=6:

    # print(img_path)

    # print("res:",res)

    ##image_hist(img_path)

    # cv2.imwrite('/content/sample_data/3/'+str(image_name).zfill(5), dst)

    # print("ratio:",m*100/(dst.shape[0]*dst.shape[1]),'%')

    # print("res",res)

    # hist = image_hist(img_path)

    # cv2.imwrite('/media/ExtDisk/yxt/zhifang/' + str(image_name).zfill(5), hist)

    return res, dst


def getFileList(dir, Filelist, ext=None):
    """

    获取文件夹及其子文件夹中文件列表

    输入 dir：文件夹根目录

    输入 ext: 扩展名

    返回： 文件路径列表

    """

    newDir = dir

    if os.path.isfile(dir):

        if ext is None:

            Filelist.append(dir)

        else:

            if ext in dir[-3:]:
                Filelist.append(dir)



    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)

            getFileList(newDir, Filelist, ext)

    return Filelist


def visualization_bbox1(num_image, json_path, img_path):
    with open(json_path) as annos:
        annotation_json = json.load(annos)

        image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名

        id = annotation_json['images'][num_image - 1]['id']  # 读取图片id

        image_path = os.path.join(img_path, str(image_name).zfill(5))  # 拼接图像路径

        res, dst = cal_res((str(image_path)), str(image_name))

        image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像

        img = cv2.imread(image_path, 0)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        # plt.hist(dst.ravel(), bins=30, alpha = 0.5)

        # plt.plot(hist, color="r")

        return hist, res


train_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1/train.json'
test_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/test.json'
val_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val.json'
test_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/test_image/'
val_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val_image/'
train_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1//train_image/'

#hrsid
train_json1 = '/media/ExtDisk/yxt/HRSID/annotations/train2017.json'
test_json1 = '/media/ExtDisk/yxt/HRSID/annotations/test2017.json'
train_path1 = '/media/ExtDisk/yxt/HRSID/train_image/'
test_path1 = '/media/ExtDisk/yxt/HRSID/test_image/'
d = 0

a = 0

b = 0

c = 0

e = 0

f = 0

g = 0

imglist = getFileList(train_path, [], 'jpg')

# cal_res()

res1 = []

for imgpath in imglist:

    d = d + 1

    hist, res = visualization_bbox1(d, train_json, train_path)

    res1.append(res)

    print(imgpath)

    print("res:", res)
    if res >= 2 and res < 3:

        plt.plot(hist, color="y", alpha=1, linestyle='--', linewidth=0.5, marker='o', markersize=1)

    elif res >= 3 and res < 4:

        plt.plot(hist, color="g", alpha=1, linestyle='-.', linewidth=0.5, marker='>', markersize=1)

    elif res >= 4 and res < 5:

            plt.plot(hist, color="r", alpha=1, linestyle=':', linewidth=0.5, marker='*', markersize=1)
    elif res >= 5 and res < 6:

        plt.plot(hist, color="b", alpha=1, linestyle=':', linewidth=0.5, marker='x', markersize=1)

    elif res >= 6 and res < 7 :

        plt.plot(hist, color="c", alpha=1, linestyle=':', linewidth=0.5, marker='D', markersize=1)

    elif res >= 7:

        plt.plot(hist, color="m", alpha=1, linestyle=':', linewidth=0.5, marker='1', markersize=1)

    print("d", d)

    if d==39:
        plt.plot(hist, color="y", alpha=1, linestyle='--', linewidth=1, marker='o', markersize=4, label='2≤G<3')
    elif d==794:
        plt.plot(hist, color="g", alpha=1, linestyle='-.', linewidth=1, marker='>', markersize=4, label='3≤G<4')
    elif d==809:
        plt.plot(hist, color="r", alpha=1, linestyle=':', linewidth=1, marker='*', markersize=4, label='4≤G<5')
    elif d == 810:
        plt.plot(hist, color="b", alpha=1, linestyle=':', linewidth=1, marker='x', markersize=4, label='5≤G<6')
    elif d == 811:
        plt.plot(hist, color="c", alpha=1, linestyle=':', linewidth=1, marker='D', markersize=4, label='6≤G<7')
    elif d == 812:

        plt.plot(hist, color="m", alpha=1, linestyle=':', linewidth=1, marker='1', markersize=4, label='G≥7')

    plt.legend()

print(max(res1), min(res1))

# plt.legend(['res<=2','res<=3','res<=4'])

plt.legend(loc=1, fontsize='10')

plt.xlabel('Pixel')

plt.ylabel('Number')

plt.savefig('/media/ExtDisk/yxt/IMG/pic.jpg', dpi=500)

#plt.show()