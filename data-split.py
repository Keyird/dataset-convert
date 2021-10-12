"""
对标准VOC格式数据集进行划分，分为：trainval、train、val、test
生成的txt文件均存在于/ImageSets/Main/路径下
"""

import os
import random 
random.seed(0)

xmlfilepath = './Annotations'
saveBasePath = "./ImageSets/Main/"

trainval_percent = 0.9  # train+val所占比例
train_percent = 1  # train占train+val的比例

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num=len(total_xml)  
list=range(num)  
num_trainval = int(num*trainval_percent)
num_train = int(num_trainval*train_percent)
trainval= random.sample(list,num_trainval)
train=random.sample(trainval,num_train)
 
print("train and val size:",num_trainval)
print("train size:",num_train)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i in list:
    name = total_xml[i][:-4]+'\n'
    if i in trainval:  
        ftrainval.write(name)   # 写入trainval.txt
        if i in train:  
            ftrain.write(name)   # 写入train.txt
        else:  
            fval.write(name)   # 写入val.txt
    else:  
        ftest.write(name)  # 写入test.txt
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
