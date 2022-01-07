# -*- coding:utf-8 -*-
'''
DatalistGenerate: 模块功能，从分好类的图片集(一类一个文件夹)，生成文件路径与对应标签列表。
DataFile:   imagePath:  待生成图片集根目录（根目录下不要有其他文件和文件夹）
            extensions： 文件扩展名
            training_ratio:    数据集中用于训练的样本比例
            validation_ratio:  数据集中用于验证的样本比例
            除训练集和验证集外，剩余的样本为测试集
'''
import os
import numpy as np
import cv2
import shutil
import json

class DataFile:
    def __init__(self, imagePath, extensions, training_ratio, validation_ratio):
        self.imagePath = imagePath
        self.extensions = extensions

        # correct training ratio based on the range
        if training_ratio < 0:
            self.training_ratio = 0
        elif training_ratio > 1:
            self.training_ratio = 1
        else:
            self.training_ratio = training_ratio

        #  correct validation ratio based on the range
        if validation_ratio < 0:
            self.validation_ratio = 0
        elif validation_ratio > 1:
            self.validation_ratio = 1
        else:
            self.validation_ratio = validation_ratio

        # correct the training ratio and validation ratio based on the sum
        if training_ratio + validation_ratio > 0.9:
            self.training_ratio = 0.8
            self.validation_ratio = 0.1

        self.output = None
        self.lists = None
        self.categroy_num = None
        self.category2label = None
        self.label2category = None

        if not self.isExists():                                         # Create json file
            self.lists = createImageList(self.imagePath, self.extensions)
            self.record_count = len(self.lists)
            self.training_count = int(self.record_count * self.training_ratio)
            self.validation_count = int(self.record_count * self.validation_ratio)
            self.testing_count = self.record_count - self.training_count - self.validation_count
            self.getPathAndLabel()

        # read data from json file
        train_filelist, train_labellist, valid_filelist, valid_labellist, test_filelist, test_labellist = get_filelist_labellist(
            self.imagePath)

        self.training_count = len(train_filelist)
        self.validation_count = len(valid_filelist)
        self.testing_count = len(test_filelist)
        self.record_count = self.training_count + self.validation_count + self.testing_count

        self.train_filelist = np.array(train_filelist)
        self.train_labellist = np.array(train_labellist)
        self.valid_filelist = np.array(valid_filelist)
        self.valid_labellist = np.array(valid_labellist)
        self.test_filelist = np.array(test_filelist)
        self.test_labellist = np.array(test_labellist)
        self.label2category = get_label_dict(self.imagePath)
        self.category2label = {self.label2category[elem]: elem for elem in self.label2category}
        self.num_classes = len(self.label2category)

    def resize_data(self, resize_savepath, shape):
        self.resize_savepath = resize_savepath
        path_dict = {}
        for elem in self.label2category:
            folder_name = self.label2category[elem]
            if not os.path.exists(self.resize_savepath + "/" + folder_name):
                os.mkdir(self.resize_savepath + "/" + folder_name)
            path_dict[elem] = self.resize_savepath + "/" + folder_name

        new_output = []
        for i in range(len(self.train_filelist)):
            images = cv_imread(self.train_filelist[i])
            images = cv2.resize(images, (shape, shape))

            images_savepath = path_dict[self.train_labellist[i]]
            _, images_savename = os.path.split(self.train_filelist[i])
            cv2.imencode('.jpg', images)[1].tofile(images_savepath + "/" + images_savename)
            new_output.append([images_savepath + "/" + images_savename, self.train_labellist[i]])

        for i in range(len(self.valid_filelist)):
            images = cv_imread(self.valid_filelist[i])
            images = cv2.resize(images, (shape, shape))

            images_savepath = path_dict[self.valid_labellist[i]]
            _, images_savename = os.path.split(self.valid_filelist[i])
            cv2.imencode('.jpg', images)[1].tofile(images_savepath + "/" + images_savename)
            new_output.append([images_savepath + "/" + images_savename, self.valid_labellist[i]])

        for i in range(len(self.test_filelist)):
            images = cv_imread(self.test_filelist[i])
            images = cv2.resize(images, (shape, shape))
            images_savepath = path_dict[self.test_labellist[i]]
            _, images_savename = os.path.split(self.test_filelist[i])
            cv2.imencode('.jpg', images)[1].tofile(images_savepath + "/" + images_savename)
            new_output.append([images_savepath + "/" + images_savename, self.test_labellist[i]])

        self.imagePath = self.resize_savepath
        self.output = new_output
        self.saveFile()
    def isExists(self):
        return os.path.exists(self.imagePath + '/DataInfo.json')

    def getPathAndLabel(self):
        np.random.shuffle(self.lists)
        categorys, paths = zip(*self.lists)
        category = np.unique(categorys)
        self.categroy_num = len(category)
        self.category2label = dict(zip(category, range(self.categroy_num)))
        self.label2category = {l: k for k, l in self.category2label.items()}
        labels = [self.category2label[l] for l in categorys]
        output = zip(paths, labels)
        self.output = [list(elem) for elem in output]
        # self.saveFile()
        self.saveFile()
        return self.output

    def saveFile(self):
        DataInfo = {}

        # create data list
        train_data = [[elem[0], str(elem[1])] for elem in self.output[0: self.training_count]]
        vilid_data = [[elem[0], str(elem[1])] for elem in self.output[self.training_count: self.training_count + self.validation_count]]
        test_data = [[elem[0], str(elem[1])] for elem in self.output[self.training_count + self.validation_count: self.record_count]]

        # save the data list in dict
        DataInfo['train_data'] = train_data
        DataInfo['valid_data'] = vilid_data
        DataInfo['test_data'] = test_data
        DataInfo['label_dict'] = self.label2category

        # save the data info in dict
        DataInfo['training_count'] = self.training_count
        DataInfo['validation_count'] = self.validation_count
        DataInfo['testing_count'] = self.testing_count

        # save the info to json file
        with open(self.imagePath + '/DataInfo.json', 'w') as f:
            json.dump(DataInfo, f)

        return print("Json file saved!")


def createImageList(imagePath, extensions):
    paths = []
    categoryList = [c for c in sorted(os.listdir(imagePath)) if
                    c[0] != '.' and os.path.isdir(os.path.join(imagePath, c))]
    for category in categoryList:
        if category:
            walkPath = os.path.join(imagePath, category)
        else:
            walkPath = imagePath
            category = os.path.split(imagePath)[1]
        w = _walk(walkPath)
        while True:
            try:
                dirpath, dirnames, filenames = w.__next__()
            except StopIteration:
                break
            # Don't enter directories that begin with '.'
            for d in dirnames[:]:
                if d.startswith('.'):
                    dirnames.remove(d)
            dirnames.sort()
            # Ignore files that begin with '.'
            filenames = [f for f in filenames if not f.startswith('.')]
            # Only load images with the right extension
            filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in extensions]
            filenames.sort()
            for f in filenames:
                paths.append([category, os.path.join(dirpath, f).replace("\\", "/")])
    return paths


def _walk(top):
    names = os.listdir(top)
    dirs, nondirs = [], []
    for name in names:
        if os.path.isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    yield top, dirs, nondirs
    for name in dirs:
        path = os.path.join(top, name)
        for x in _walk(path):
            yield x


def get_random_image(filepath):
    walk_results = _walk(filepath)
    filelist = []
    for root, path, namelist in walk_results:
        root.replace("\\", "/")
        for elem in namelist:
            filelist.append(root + "/" + elem)
    np.random.shuffle(filelist)
    return filelist


def copy_file(filelist, newpath, count):
    for i in range(0, min(count, len(filelist))):
        filename = filelist[i]
        fp, fn = os.path.split(filename)
        shutil.copyfile(filename, newpath + "/" + fn)


def get_filelist_labellist(filepath):

    with open(filepath + "/" + "DataInfo.json", "r") as f:
        DataInfo = json.load(f)

        train_data = DataInfo['train_data']
        train_filelist = [elem[0] for elem in train_data]
        train_labellist = [elem[1] for elem in train_data]

        valid_data = DataInfo['valid_data']
        valid_filelist = [elem[0] for elem in valid_data]
        valid_labellist = [elem[1] for elem in valid_data]

        test_data = DataInfo['test_data']
        test_filelist = [elem[0] for elem in test_data]
        test_labellist = [elem[1] for elem in test_data]

    return train_filelist, train_labellist, valid_filelist, valid_labellist, test_filelist, test_labellist


def get_label_dict(filepath):
    with open(filepath + "/" + "DataInfo.json", "r") as f:
        DataInfo = json.load(f)
        label_dict = DataInfo['label_dict']
    return label_dict

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img
# # --------------------------Test Case---------------------------------
if __name__=="__main__":
    # d = DataFile(r'D:\DataSet\Congestion','.png', 0.8, 0.1)
    # doc = open(r'D:\CVProject\CBAM-keras-master\data\DataInfo.json','r',encoding='utf-8')
    # setting = json.load(doc)
    # print(setting)
    
    # print('样本路径文件生成成功！')

    d = DataFile(r'D:\CVProject\CBAM-keras-master\data','.png', 0.8, 0.1)
    # d.resize_data('D:/DataSet/WeatherClasifer_Chosen/Resized_Data', 224)
    print(len(d.train_filelist))
    print(len(d.train_labellist))
    print(len(d.valid_filelist))
    print(len(d.valid_labellist))
    print(len(d.test_filelist))
    print(len(d.test_labellist))
    print(d.train_filelist)
    print(d.train_labellist)
    print(d.record_count)
    print('样本路径文件生成成功！')