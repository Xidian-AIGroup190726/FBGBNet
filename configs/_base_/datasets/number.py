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
#ssdd
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

train_json2 = '/media/ExtDisk/yxt/sardataset/train.json'
test_json2 = '/media/ExtDisk/yxt/sardataset/test.json'
val_json2 = '/media/ExtDisk/yxt/sardataset/val.json'
path = '/media/ExtDisk/yxt/sardataset/images/'
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    #print(dir)
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


def embedding_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist


def sigmoid_bianhua(x, p1):
    p = (np.log((1 / 0.99) - 1) + 6) / (p1)
    x = p * (x)
    x = x - 6
    return 1 / (1 + np.exp(x))  # sigmoid函数

def cal_res(img_path):
        tmp = []
        for i in range(256):
            tmp.append(0)
        val = 0
        k = 0
        res = 0
        # I = Image.open('/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1/train_image/001080.jpg' )
        I = Image.open(img_path)
        # I = I.resize((400, 400))
        greyIm = I.convert('L')
        # greyIm.show()
        img = np.array(greyIm)
        x, y = img.shape
        dst = np.zeros([x, y])
        for i in range(x):
            for j in range(y):
                if img[i, j] > 80:
                    dst[i, j] = 255
                else:
                    dst[i, j] = 0
        # retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imwrite('binary.jpg', dst)
        # cv2.imwrite('/media/ExtDisk/yxt/er/'+str(image_name).zfill(5), dst)
        m = 0
        for i in range(dst.shape[0]):
            for j in range(dst.shape[1]):
                # print(dst[i,j])
                if dst[i, j] == 60:
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
        if res >= 4:
            print("res", res)
        return res

def visualization_bbox1(num_image, json_path, img_path,numbox,numsmall,numbig,numsmallcom,numcom):
    with open(json_path) as annos:
        annotation_json = json.load(annos)
        image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
        id = annotation_json['images'][num_image - 1]['id']  # 读取图片id
        image_path = os.path.join(img_path, str(image_name).zfill(5))# 拼接图像路径
        res1 = cal_res(str(image_path))
        if(res1>=4):
            numcom+=1
        print(str(image_path))
        image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像 # 统计一幅图片中bbox的数量
        k = 0
        img = Image.open(image_path)
        img_array = np.array(img)  # 把图像转成数组格式img = np.asarray(image)
        #print(img_array)
        shape = img_array.shape
        img1 = img
        #print(shape[0],shape[1])
        for i in range(len(annotation_json['annotations'][::])):
            if annotation_json['annotations'][i - 1]['image_id'] == id:
                numbox = numbox + 1
                point = []
                distance = []
                x, y, w, h = annotation_json['annotations'][i - 1]['bbox']  # 读取边框
                #print(w*h,(w*h)/(shape[0]*shape[1]))
                if (w*h)<=1024:
                    numsmall = numsmall+1
                if (w*h)>=9216:
                    numbig = numbig +1
                if ((res1 >= 4) and (w * h) <= 1024):
                    numsmallcom += 1
        return numbox,numsmall,numbig,numsmallcom,numcom
                #cv2.imwrite("/media/ExtDisk/yxt/HRSID/after_test_image70/" + str(image_name).zfill(5),img_array)
if __name__ == "__main__":
    d = 0
    org_img_folder = '/media/ExtDisk/yxt/sardataset/images/'
    numbox = 0
    numsmall = numbig = numsmallcom=numcom=0
    #org_img_folder = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val_image/'
    imglist = getFileList(org_img_folder, [], 'jpg')
    #print("aaa:",imglist)
    for imgpath in imglist:
        d = d + 1
        numbox,numsmall,numbig,numsmallcom,numcom= visualization_bbox1(d, test_json2, path,numbox,numsmall,numbig,numsmallcom,numcom)
        print(d,numbox,numsmall,numbig,numsmall/numbox,numbig/numbox,numsmallcom,numcom)
# 只有距离
#python mmdetection/tools/train.py mmdetection/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py#

