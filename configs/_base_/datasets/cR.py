import json
import os, cv2
from PIL import Image
import numpy as np
import json
import os, cv2
from PIL import ImageDraw
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


def visualization_bbox1(num_image, json_path, img_path):
    with open(json_path) as annos:
        annotation_json = json.load(annos)

        image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
        id = annotation_json['images'][num_image - 1]['id']  # 读取图片id
        image_path = os.path.join(img_path, str(image_name).zfill(5))# 拼接图像路径
        print(str(image_path))
        image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
        num_bbox = 0  # 统计一幅图片中bbox的数量
        k = 0
        num_small=0
        num_big = 0
        img = Image.open(image_path)
        img_array = np.array(img)  # 把图像转成数组格式img = np.asarray(image)
        #print(img_array)
        shape = img_array.shape
        img1 = img
        a = ImageDraw.ImageDraw(img1)
        #print(shape[0],shape[1])
        dscore = []
        cscore = []
        smian = []
        for i in range(0, shape[0] * shape[1]):
            dscore.append(0)
        for i in range(0, shape[0] * shape[1]):
            cscore.append(0)
        for i in range(len(annotation_json['annotations'][::])):
            if annotation_json['annotations'][i - 1]['image_id'] == id:
                num_bbox = num_bbox + 1
                point = []
                distance = []
                x, y, w, h = annotation_json['annotations'][i - 1]['bbox']  # 读取边框
                #print(w*h,(w*h)/(shape[0]*shape[1]))
                if (w*h)<=1024:
                    num_small = num_small+1
                if (w*h)>=4096:
                    num_big = num_big +1
                smian.append(w*h)
                x1 = x + w / 2
                y1 = y + h / 2  # (x1,y1) 为中心点 (x2,y2)为右下角点 (x,y)为右上角点
                x2 = x + w
                y2 = y + h
              # 用a来表示
                # 在边界框的两点（左上角、右下角）画矩形，无填充，边框红色，边框像素为5
                #a.rectangle(((x, y), (x2, y2)), fill=None, outline='green', width=4)
                #img.save("/media/ExtDisk/yxt/image.jpg")
                if x2 >= shape[1]:
                    x2 = shape[1] - 1
                if y2 >= shape[0]:
                    y2 = shape[0] - 1
                if x1 >= shape[1]:
                    x1 = shape[1] - 1
                if y1 >= shape[0]:
                    y1 = shape[0] - 1
                # img_array[int(x1), int(y1)]=(255,255,255)
                center = (int(x + w / 2), int(y + h / 2))

                # 计算每个像素点与中心点的距离
                for i in range(0, shape[1]):
                    for j in range(0, shape[0]):
                        # i是纵坐标，j是横坐标
                        dis = embedding_distance([y1, x1], [j, i])
                        # print(dis)
                        distance.append([dis, [j, i]])
                # 计算距离分数
                R = max(w / 2, h / 2)
                r1 = 1.5*R
                for i in range(0, len(distance)):
                    if dscore[i] < sigmoid_bianhua(distance[i][0], r1):
                        dscore[i] = sigmoid_bianhua(distance[i][0], r1)
                # x-Y
        smian.clear()
        if num_small/num_bbox!=0:
            a = 0.94 #4R （0，1.5R）=1 （1.5R，3R）d\(3R,4R)c
        else:
            a = 0.94

        print(a,num_bbox,num_small,num_small/num_bbox)
        #a = min(r1,1-r1)
        for i in range(0, len(distance)):
            if dscore[i] <= a:
                img_array[(distance[i][1])[0], (distance[i][1])[1]] = 0#(0, 0, 0)
        img2 = Image.fromarray(np.uint8(img_array))
        #img2.save("/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/after_test_image/" + str(image_name).zfill(5), 'JPG')
        cv2.imwrite("/media/ExtDisk/yxt/sardataset/after_images/" + str(image_name).zfill(5),img_array)
        #cv2.imwrite("/media/ExtDisk/yxt/HRSID/after_test_image70/" + str(image_name).zfill(5),img_array)
if __name__ == "__main__":
    d = 0
    org_img_folder = '/media/ExtDisk/yxt/sardataset/images/'
    #org_img_folder = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val_image/'
    imglist = getFileList(org_img_folder, [], 'jpg')
    #print("aaa:",imglist)
    for imgpath in imglist:
        d = d + 1
        visualization_bbox1(d, test_json2, path)
        print(d)
# 只有距离
#python mmdetection/tools/train.py mmdetection/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py#

