"""
1、标准VOC格式数据集转标准YOLO-darknet训练格式（YOLOv3、YOLOv4通用）
2、生成对应的names标签文件
"""

import os
from lxml import etree
from tqdm import tqdm
import shutil

# 转换后YOLO文件保存路径
save_file_root = "D:\\datasets\\ALL-YOLO"
if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)

# 转换前VOC文件路径
voc_root = "D:\\datasets\\ALL-VOC"
voc_version = 'VOC2007'
voc_images_path = os.path.join(voc_root,voc_version,"JPEGImages") #图片路径
voc_xml_path = os.path.join(voc_root,voc_version,"Annotations") #标注信息路径
train_txt_path = os.path.join(voc_root,voc_version,'ImageSets','Main','train.txt') #训练集图片名称路径
test_txt_path = os.path.join(voc_root,voc_version,'ImageSets','Main','test.txt') #测试集图片路径

# 验证文件是否存在
assert os.path.exists(voc_images_path),"voc_images_path not exist"
assert os.path.exists(voc_xml_path),"voc_xml_path not exist"
assert os.path.exists(train_txt_path),"train_txt_path not exist"
assert os.path.exists(test_txt_path),"test_txt_path not exist"

def xml_to_dict(xml):
    """
    把xml递归成嵌套字典，其中object按列表存放
    :param xml: xml标签文件
    :return: 字典:包含xml各种信息
    """
    if len(xml) == 0:
        return xml.text  # 遍历到底层 返回信息

    result={}
    for child in xml:
        child_child = xml_to_dict(child)  # 递归获取信息
        if child.tag != 'object':  # 本层非object 直接按字典存
            result[child.tag] = child_child  # 本层tag的内容是下层
        else:  # 本层是object 用列表存 不然多个object 会只保存最后一个object
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_child)
    return result


def trans_info(file_names, classes_dict, save_path, train_or_test="train"):
    """
    把 voc 格式转成 yolo 格式，注意 yolo 类别从 0 开始
    :param file_names: list: 文件名（不包括后缀）
    :param classes_dict: 类别信息
    :param save_path: 要保存的路径
    :param train_or_test: train or test
    :return: None
    """
    with open(os.path.join(save_file_root, train_or_test + '.txt'), 'w') as file:
        for file_name in tqdm(file_names, desc="转换{}文件中".format(train_or_test)):
            if file_name.strip() == '':
                continue
            xml_path = os.path.join(voc_xml_path, file_name+'.xml')
            # 读取对应file_name
            with open(xml_path) as f:
                xml_str = f.read()
            # read读出来str 再转回xml文件
            xml = etree.fromstring(xml_str)
            # 传入xml文件，返回转换后的字典
            annotations_info = xml_to_dict(xml)
            # 跳过没有目标的样本
            if "object" not in annotations_info:
                continue
            # 写train.txt文件
            file.write("data/obj/" + file_name + '.jpg' + '\n')
            # 读取信息
            img_height = float(annotations_info['size']['height'])
            img_width = float(annotations_info['size']['width'])
            if os.path.exists(os.path.join(save_path,train_or_test)) is False:
                os.makedirs(os.path.join(save_path,train_or_test))
            with open(os.path.join(save_path,train_or_test,file_name+'.txt'),'w') as f:
                for i,obj in enumerate(annotations_info['object']):
                    # 获取坐标信息
                    xmin = float(obj['bndbox']['xmin'])
                    xmax = float(obj['bndbox']['xmax'])
                    ymin = float(obj['bndbox']['ymin'])
                    ymax = float(obj['bndbox']['ymax'])
                    # 等会要相对化所以先转成浮点
                    name = obj['name']

                    # 标签重定义
                    if name == "van":
                        name = "truck"
                    elif name == "tricycle1" or name == "tricycle2":
                        name = "bike"
                    elif name == "pedestrian":
                        name = "person"
                    index = classes_dict[name] - 1  # yolo的编号是从0开始 voc是从1开始

                    # 转成 yolo （label xcenter ycenter width height）
                    x_center = xmin+(xmax-xmin)/2
                    y_center = ymin+(ymax-ymin)/2
                    w = xmax-xmin
                    h = ymax-ymin
                    # 保留6位小数
                    x_center = round(x_center/img_width,6)
                    y_center = round(y_center/img_height,6)
                    w = round(w/img_width,6)
                    h = round(h/img_height,6)

                    # 转换成字符串
                    info = [str(j) for j in [index, x_center, y_center, w, h]]

                    # 保存
                    if i == 0:
                        f.write(' '.join(info))  # 用空格隔开
                    else:
                        f.write('\n'+' '.join(info))

            # 把图片复制到对应文件夹
            if os.path.exists(os.path.join(save_path, 'Image')) is False:
                os.makedirs(os.path.join(save_path, 'Image'))
            shutil.copyfile(os.path.join(voc_images_path, file_name+'.jpg'), os.path.join(save_path, 'Image', file_name+'.jpg'))


def create_class_names(class_dict: dict):
    """
    生成 classes.names
    :param class_dict: 类别字典
    :return: None
    """
    keys = class_dict.keys()
    with open(os.path.join(save_file_root, 'classes.names'), "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")

def main():
    class_dict = {"person":1, "rider":2, "car":3, "bus":4, "truck":5, "bike":6}

    # 读取 train.txt
    with open(train_txt_path) as f:
        train_file_names = [file_name for file_name in f.read().split('\n')]
        print(train_file_names)
    trans_info(train_file_names,class_dict,save_file_root,"train")

    # 读取 test.txt
    with open(test_txt_path) as f:
        train_file_names = [file_name for file_name in f.read().split('\n')]
        print(train_file_names)
    trans_info(train_file_names, class_dict, save_file_root, "test")

    # todo 读取 json 文件, 生成 classes.names
    create_class_names(class_dict)

if __name__ == '__main__':
    main()
