import json
import os, cv2
from PIL import Image
import numpy as np
import json
import os, cv2
from PIL import ImageDraw
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
#ssdd


train_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1/train.json'
test_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/test.json'
val_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val.json'
inshore_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test_inshore.json'
test_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/test_image/'
val_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val_image/'
train_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1//train_image/'
all_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/coco_style/images/test_inshore/'
#hrsid
train_json1 = '/media/ExtDisk/yxt/HRSID/annotations/train2017.json'
test_json1 = '/media/ExtDisk/yxt/HRSID/annotations/test2017.json'
train_path1 = '/media/ExtDisk/yxt/HRSID/train_image/'
test_path1 = '/media/ExtDisk/yxt/HRSID/test_image/'
inside_json1 = '/media/ExtDisk/yxt/HRSID/annotations/inshore.json'
all_path1 = '/media/ExtDisk/yxt/HRSID/all_image/'

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


def visualization_bbox1(num_image, json_path, img_path):
    with open(json_path) as annos:
        annotation_json = json.load(annos)

        image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
        id = annotation_json['images'][num_image - 1]['id']  # 读取图片id
        image_path = os.path.join(img_path, str(image_name).zfill(5))# 拼接图像路径
        print(str(image_path))
        image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
        #num_bbox = 0  # 统计一幅图片中bbox的数量
        k = 0
        #num_small=0
        num_big = 0
        img = Image.open(image_path)
        img_array = np.array(img)  # 把图像转成数组格式img = np.asarray(image)
        #print(img_array)
        shape = img_array.shape
        img1 = img
        a = ImageDraw.ImageDraw(img1)
        for i in range(len(annotation_json['annotations'][::])):
            if annotation_json['annotations'][i - 1]['image_id'] == id:
                point = []
                distance = []
                x, y, w, h = annotation_json['annotations'][i - 1]['bbox']  # 读取边框
                ImageDraw.Draw(img).rectangle([(x,y),(x+w,y+h)], outline='green', width=4)

                #print(w*h,(w*h)/(shape[0]*shape[1]))

        plt.imshow(img)
        plt.show()
        #img2.save("/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/after_test_image/" + str(image_name).zfill(5), 'JPG')
        cv2.imwrite("/media/ExtDisk/vis/" + str(image_name).zfill(5),img_array)
        #cv2.imwrite("/media/ExtDisk/yxt/HRSID/after_test_image70/" + str(image_name).zfill(5),img_array)
if __name__ == "__main__":
    #d = 1961
    d = 3130
    num_bbox = 0
    num_small= 0
    org_img_folder = '/media/ExtDisk/yxt/HRSID/test_image/'
    #org_img_folder = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/coco_style/image/test_inshore'
    #'/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val_image/'
    imglist = getFileList(org_img_folder, [], 'jpg')
    for imgpath in imglist:
        d = d + 1
        visualization_bbox1(d,train_json1,train_path1)
        print(d)
        #break
# 只有距离
#python mmdetection/tools/train.py mmdetection/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py#

