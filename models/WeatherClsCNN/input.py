from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import cv2
import os
from keras.utils import np_utils
import time

import models.WeatherClsCNN.DataListGenerate as DataListGenerate

class INPUT_DATA():
    def __init__(self, filepath = None, extension = ['.jpg', '.png'], training_ratio = 0.8, validation_ratio = 0.1, shape = (224, 224, 3), type = 'original'):
        self.filepath = filepath
        self.extension = extension
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.shape = shape
        self.type = type

        # init output
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None
        self.label_dict = None

    def get_all_resized_data(self):
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test, self.label_dict = input_original(
            self.filepath, self.extension, self.training_ratio, self.validation_ratio, self.shape)

    def get_all_original_data(self):
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test, self.label_dict = input_original(
            self.filepath, self.extension, self.training_ratio, self.validation_ratio, self.shape)

    def get_training_data(self):
        if self.X_train is None or self.Y_train is None:
            if self.type == 'resized':
                self.get_all_resized_data()
            else:
                self.get_all_original_data()
        return self.X_train, self.Y_train

    def get_validation_data(self):
        if self.X_valid is None or self.Y_valid is None:
            if self.type == 'resized':
                self.get_all_resized_data()
            else:
                self.get_all_original_data()
        return self.X_valid, self.Y_valid

    def get_testing_data(self):
        if self.X_test is None or self.Y_test is None:
            if self.type == 'resized':
                self.get_all_resized_data()
            else:
                self.get_all_original_data()
        return self.X_test, self.Y_test

    def get_label(self):
        if self.label_dict is None:
            if self.type == 'resized':
                self.get_all_resized_data()
            else:
                self.get_all_original_data()
        length = len(self.label_dict)
        labels = [""]*length
        for elem in self.label_dict:
            index_n = self.label_dict[elem].find("_")
            if index_n != -1:
                labels[int(elem)] = self.label_dict[elem][index_n+1:]   # 个人习惯，label格式为：0_sunny, 此处功能为提取sunny，可根据规则修改
            else:
                labels[int(elem)] = self.label_dict[elem]               # 无其他命名规则时，全部输出
        return labels

# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def input_original(filepath, extension, training_ratio, validation_ratio, shape):
    (width, height, channels) = shape
    DataList = DataListGenerate.DataFile(filepath, extension, training_ratio, validation_ratio)
    train_files = DataList.train_filelist
    train_labels = DataList.train_labellist
    valid_files = DataList.valid_filelist
    valid_labels = DataList.valid_labellist
    test_files = DataList.test_filelist
    test_labels = DataList.test_labellist
    num_classes = DataList.num_classes
    label_dict = DataList.label2category

    X_train = []
    Y_train = np_utils.to_categorical(train_labels[:len(train_labels)], num_classes)
    X_valid = []
    Y_valid = np_utils.to_categorical(valid_labels[:len(valid_labels)], num_classes)
    X_test = []
    Y_test = np_utils.to_categorical(test_labels[:len(test_labels)], num_classes)

    for elem in train_files:
        train_images = cv_imread(elem)
        train_images = cv2.resize(train_images, (width, height))
        train_images = train_images.reshape(-1, width, height, channels)
        X_train.append(train_images)
    for elem in valid_files:
        valid_images = cv_imread(elem)
        valid_images = cv2.resize(valid_images, (width, height))
        valid_images = valid_images.reshape(-1, width, height, channels)
        X_valid.append(valid_images)
    for elem in test_files:
        test_images = cv_imread(elem)
        test_images = cv2.resize(test_images, (width, height))
        test_images = test_images.reshape(-1, width, height, channels)
        X_test.append(test_images)
    X_train = np.concatenate(X_train, axis=0)
    X_valid = np.concatenate(X_valid, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, label_dict

def input_resized(filepath, extension, training_ratio, validation_ratio, shape):
    (width, height, channels) = shape
    DataList = DataListGenerate.DataFile(filepath, extension, 0.8, 0.2)
    train_files = DataList.train_filelist
    train_labels = DataList.train_labellist
    valid_files = DataList.valid_filelist
    valid_labels = DataList.valid_labellist
    test_files = DataList.test_filelist
    test_labels = DataList.test_labellist
    num_classes = DataList.num_classes
    label_dict = DataList.label2category

    X_train = []
    Y_train = np_utils.to_categorical(train_labels[:len(train_labels)], num_classes)
    X_valid = []
    Y_valid = np_utils.to_categorical(valid_labels[:len(valid_labels)], num_classes)
    X_test = []
    Y_test = np_utils.to_categorical(test_labels[:len(test_labels)], num_classes)

    for elem in train_files:
        images = cv_imread(elem)
        images = images.reshape(-1, width, height, channels)
        X_train.append(images)
    for elem in valid_files:
        images = cv_imread(elem)
        images = images.reshape(-1, width, height, channels)
        X_valid.append(images)
    for elem in test_files:
        images = cv_imread(elem)
        images = images.reshape(-1, width, height, channels)
        X_test.append(images)
    X_train = np.concatenate(X_train, axis=0)
    X_valid = np.concatenate(X_valid, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, label_dict

# ------------------------------------------Test case----------------------------------------
# X_train, Y_train, X_valid,  Y_valid, X_test,  Y_test, label_dict = input_original('D:/DataSet/WeatherClasifer_Chosen/Data', '.png', 0.8, 0.1, (224, 224, 3))
# X_train, Y_train, X_valid, Y_valid, X_test, Y_test, label_dict = input_resized('D:/DataSet/WeatherClasifer_Chosen/Resized_Data', '.png', 0.8, 0.1, (224, 224, 3))

# time_start = time.time()
# input_data = INPUT_DATA('D:/DataSet/WeatherClasifer_Chosen/Resized_Data', '.png', 0.8, 0.1, (224, 224, 3), 'resized')
# #
# X_train, Y_train = input_data.get_training_data()
# X_valid, Y_valid = input_data.get_validation_data()
# X_test,  Y_test = input_data.get_testing_data()
# label_dict = input_data.get_label()
# print(X_train.shape)
# print(Y_train.shape)
# print(X_valid.shape)
# print(Y_valid.shape)
# print(X_test.shape)
# print(Y_test.shape)
# print(len(label_dict))
# time_end = time.time()
#
# print(time_end-time_start)
