# _*_ coding: utf-8 _*_
# @Time : 2020/10/24 17:08
# @Author : Sun Zhu
# @Version：V 1.0
# @File : Feature.py
# @desc : Weather features extraction for weather classification.
# env: CV_TF2_3

import os
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import time
import random


import keras.backend as K

from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras.utils import np_utils
from keras.layers import Input, Dense, concatenate, TimeDistributed, Flatten, LSTM, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint


# import tensorflow.keras.backend as K
#
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.xception import preprocess_input
# # from tensorflow.keras.utils import np_utils
# from tensorflow.keras.layers import Input, Dense, concatenate, TimeDistributed, Flatten, LSTM, Dropout, BatchNormalization, Activation, MaxPooling2D, Conv2D
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint


import models.WeatherClsCNN.test_performance as tp
import pandas as pd

import models.WeatherClsCNN.densenet121
import models.WeatherClsCNN.resnet_101
import models.WeatherClsCNN.vgg19

# Configuration

# metadata_json = r'/home/shiyanshi/sz/Image2Weather/metadata.json'
metadata_json = r'D:\CVProject\CBAM-keras-master\Image2Weather\metadata.json'
# image_path = r'/home/shiyanshi/sz/Image2Weather/Image'
image_path = r'D:\CVProject\CBAM-keras-master\Image2Weather\Image'
# datainfo_json = r'/home/shiyanshi/sz/Image2Weather/datainfo.json'
datainfo_json = r'D:\CVProject\CBAM-keras-master\Image2Weather\DataInfo.json'

image_path_for_temp_pred_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
metadata_for_temp_pred_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"


def get_configure_parameters_for_temp_pred():
    """
    get some default parameters
    :return: a dict contains some default parameters
    """
    base_input_dict = {"filepath": r'D:\CVProject\CBAM-keras-master\Image2Weather\Image',
                        "savepath": r'D:\CVProject\CBAM-keras-master\temp_results',
                        "weightpath": r'D:\CVProject\CBAM-keras-master\weights',
                        "batch_size": 4,
                        "size": 224,
                        "channels": 3,
                        "epoches": 1,
                        "lr": 1e-3}
    return base_input_dict


def split_data(data_dict, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    split dataset
    :param data_dict: input data list
    :param train_ratio: training ratio
    :param valid_ratio: validating ratio
    :param test_ratio: testing ratio
    :return: list of data for training, validating and testing
    """
    total_count = len(data_dict)
    train_count = int(train_ratio * total_count)
    valid_count = int(valid_ratio * total_count)

    return data_dict[:train_count], data_dict[train_count:train_count + valid_count], data_dict[
                                                                                      train_count + valid_count:]


def gen_TimeValue_TValue(data_dict, batch_size=1):
    """
    Data generator:
        input data: Time_location(Value);
        label data: temperature(Value);
    :param data_dict: original data dict
    :param batch_size: batch size affecting output
    :return: [Time(Value)], [temp(Value)]
    """
    data_dict = np.squeeze(data_dict)
    TOTAL_COUNT = len(data_dict)
    i = 0
    while True:
        if i + batch_size > TOTAL_COUNT:
            tmp_data = np.concatenate((data_dict[i:], data_dict[:(i + batch_size) - TOTAL_COUNT]))
            i = i + batch_size - TOTAL_COUNT
        else:
            tmp_data = data_dict[i:i + batch_size]
            i += batch_size

        time_data = []

        output_temp = []

        for elem in tmp_data:

            # # time info           # 36D
            # input_date = np_utils.to_categorical(int(elem[8]) - 1, 12)  # 从0开始算
            # input_time = np_utils.to_categorical(int(elem[10]), 24)
            # time_data.append(np.append(input_date, input_time))

            # ==== NEW: 0524, time info 4D ====
            input_date = int(elem[8])
            input_time = int(elem[10])
            time_data.append([input_date, input_time, 37, 107])

            # location info       # 54D
            # input_lng = np_utils.to_categorical(float(elem['lng']) // 10 + 18, 36)
            # input_lat = np_utils.to_categorical(float(elem['lat']) // 10 + 9, 18)
            # loc_data.append(np.append(input_lng, input_lat))

            # # weather condition
            output_temp.append(float(elem[3]))
            # output_hum.append(float(elem['hum']))

        yield [np.array(time_data)], [np.array(output_temp)]


def gen_ImgTimeOnehot_TValue(data_dict, batch_size=1, image_root=None):
    """
    Data generator:
        input data: Image, Time(Onehot);
        label data: temperature(Value);
    :param data_dict: original data dict
    :param batch_size: batch size affecting output
    :param image_root: directory that contains image files
    :return: [Image, Time(Value)], [temp(Value)]
    """
    data_dict = np.squeeze(data_dict)
    TOTAL_COUNT = len(data_dict)
    i = 0
    while True:
        if i + batch_size > TOTAL_COUNT:
            tmp_data = np.concatenate((data_dict[i:], data_dict[:(i + batch_size) - TOTAL_COUNT]))
            i = i + batch_size - TOTAL_COUNT
        else:
            tmp_data = data_dict[i:i + batch_size]
            i += batch_size

        img_data = []
        time_data = []

        output_temp = []

        for elem in tmp_data:
            # img = image.load_img(os.path.join(image_root, elem[4]+'/'+ elem[2]), target_size=(224, 224))
            img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
            if img:
                input_image = image.img_to_array(img)
                input_image = preprocess_input(input_image)
                img_data.append(input_image)
            else:
                continue

            # # time info           # 36D
            # input_date = np_utils.to_categorical(int(elem[8]) - 1, 12)  # 从0开始算
            # input_time = np_utils.to_categorical(int(elem[10]), 24)
            # time_data.append(np.append(input_date, input_time))

            # ==== NEW: 0524, time info 4D ====
            input_date = int(elem[8])
            input_time = int(elem[10])
            time_data.append([input_date, input_time, 37, 107])

            # location info       # 54D
            # input_lng = np_utils.to_categorical(float(elem['lng']) // 10 + 18, 36)
            # input_lat = np_utils.to_categorical(float(elem['lat']) // 10 + 9, 18)
            # loc_data.append(np.append(input_lng, input_lat))

            # # weather condition
            output_temp.append(float(elem[3]))
            # output_hum.append(float(elem['hum']))

        yield [np.array(img_data), np.array(time_data)], [np.array(output_temp)]


def gen_ImgTimeOnehot_TOnehot(data_dict, batch_size=1, image_root=None, sigma=2.5):
    """
    Data generator:
        input data: Image, Time(Onehot);
        label data: temperature(LDE);
    :param data_dict: original data dict
    :param batch_size: batch size affecting output
    :param image_root: directory that contains image files
    :return: [Image, Time(Onehot)], [temp(LDE)]
    """
    data_dict = np.squeeze(data_dict)
    TOTAL_COUNT = len(data_dict)
    i = 0
    while True:
        if i + batch_size > TOTAL_COUNT:
            tmp_data = np.concatenate((data_dict[i:], data_dict[:(i + batch_size) - TOTAL_COUNT]))
            i = i + batch_size - TOTAL_COUNT
        else:
            tmp_data = data_dict[i:i + batch_size]
            i += batch_size

        img_data = []
        time_data = []

        output_temp = []

        for elem in tmp_data:
            # img = image.load_img(os.path.join(image_root, elem[4]+'/'+ elem[2]), target_size=(224, 224))
            img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
            if img:
                input_image = image.img_to_array(img)
                input_image = preprocess_input(input_image)
                img_data.append(input_image)
            else:
                continue

            # # time info           # 36D
            # input_date = np_utils.to_categorical(int(elem[8]) - 1, 12)  # 从0开始算
            # input_time = np_utils.to_categorical(int(elem[10]), 24)
            # time_data.append(np.append(input_date, input_time))

            # ==== NEW: time info 4D ====
            input_date = int(elem[8])
            input_time = int(elem[10])
            time_data.append([input_date, input_time, 37, 107])

            # location info       # 54D
            # input_lng = np_utils.to_categorical(float(elem['lng']) // 10 + 18, 36)
            # input_lat = np_utils.to_categorical(float(elem['lat']) // 10 + 9, 18)
            # loc_data.append(np.append(input_lng, input_lat))

            # # weather condition
            # output_temp.append(float(elem[3]))
            # output_hum.append(float(elem['hum']))
            output_temp.append(get_LDE_code(int(elem[3]), sigma=sigma))

        yield [np.array(img_data), np.array(time_data)], [np.array(output_temp)]


def gen_ImgTimeValue_TOnehot(data_dict, batch_size=1, image_root=None):
    """
    Data generator:
        input data: Image, Time(Value);
        label data: temperature(Onehot);
    :param data_dict: original data dict
    :param batch_size: batch size affecting output
    :param image_root: directory that contains image files
    :return: [Image, Time(Value)], [temp(Onehot)]
    """
    data_dict = np.squeeze(data_dict)
    TOTAL_COUNT = len(data_dict)
    i = 0
    while True:
        if i + batch_size > TOTAL_COUNT:
            tmp_data = np.concatenate((data_dict[i:], data_dict[:(i + batch_size) - TOTAL_COUNT]))
            i = i + batch_size - TOTAL_COUNT
        else:
            tmp_data = data_dict[i:i + batch_size]
            i += batch_size

        img_data = []
        time_data = []
        # loc_data = []

        # label_data = []
        output_temp = []
        # output_hum = []

        for elem in tmp_data:
            # # image data
            # img = image.load_img(os.path.join(image_root, elem[4]+'/'+ elem[2]), target_size=(224, 224))
            img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
            if img:
                input_image = image.img_to_array(img)
                input_image = preprocess_input(input_image)
                img_data.append(input_image)
            else:
                continue

            # # time info           # 36D
            # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
            input_date = int(elem[8])
            input_time = int(elem[10])
            time_data.append([input_date, input_time, 37, 107])

            # # # location info       # 54D
            # input_lng = np_utils.to_categorical(float(elem['lng']) // 10 + 18, 36)
            # input_lat = np_utils.to_categorical(float(elem['lat']) // 10 + 9, 18)
            # loc_data.append(np.append(input_lng, input_lat))

            # # weather condition
            # output_class = np_utils.to_categorical(elem['label'], 5)  # 5 conditions
            # label_data.append(output_class)
            # output_temp.append(float(elem[3]))
            # output_hum.append(float(elem['hum']))

            output_temp.append(get_Onehot_code(int(elem[3])))

        yield [np.array(img_data), np.array(time_data)], [np.array(output_temp)]


def gen_Img_TOnehot(data_dict, batch_size=1, image_root=None):
    """
    Data generator:
        input data: Image;
        label data: temperature(Onehot);
    :param data_dict: original data dict
    :param batch_size: batch size affecting output
    :param image_root: directory that contains image files
    :return: [Image], [temp(Onehot)]
    """
    data_dict = np.squeeze(data_dict)
    TOTAL_COUNT = len(data_dict)
    i = 0
    while True:
        tmp_data = []
        if i + batch_size > TOTAL_COUNT:
            tmp_data = np.concatenate((data_dict[i:], data_dict[:(i + batch_size) - TOTAL_COUNT]))
            i = i + batch_size - TOTAL_COUNT
        else:
            tmp_data = data_dict[i:i + batch_size]
            i += batch_size

        img_data = []
        time_data = []

        output_temp = []

        for elem in tmp_data:
            # img = image.load_img(os.path.join(image_root, elem[4]+'/'+ elem[2]), target_size=(224, 224))
            img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
            if img:
                input_image = image.img_to_array(img)
                input_image = preprocess_input(input_image)
                img_data.append(input_image)
            else:
                continue

            # # time info           # 36D
            # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
            input_date = np_utils.to_categorical(int(elem[8]) - 1, 12)  # 从0开始算
            input_time = np_utils.to_categorical(int(elem[10]), 24)
            time_data.append(np.append(input_date, input_time))
            output_temp.append(get_Onehot_code(int(elem[3])))           # -30~50


        yield np.array(img_data), np.array(output_temp)


def gen_Img_TLDE(data_dict, batch_size=1, image_root=None, sigma=3.5):
    """
    data generator:
        input data: Image;
        label data: temperature(LDE)
    :param data_dict: original data dict
    :param batch_size: batch size affecting output
    :param image_root: directory that contains image files
    :param sigma: distribution factors for LDE encoding
    :return: [Image], [temp(LDE)]
    """
    data_dict = np.squeeze(data_dict)
    TOTAL_COUNT = len(data_dict)
    i = 0
    while True:
        tmp_data = []
        if i + batch_size > TOTAL_COUNT:
            tmp_data = np.concatenate((data_dict[i:], data_dict[:(i + batch_size) - TOTAL_COUNT]))
            i = i + batch_size - TOTAL_COUNT
        else:
            tmp_data = data_dict[i:i + batch_size]
            i += batch_size

        img_data = []
        output_temp = []

        for elem in tmp_data:
            # img = image.load_img(os.path.join(image_root, elem[4]+'/'+ elem[2]), target_size=(224, 224))
            img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
            if img:
                input_image = image.img_to_array(img)
                input_image = preprocess_input(input_image)
                img_data.append(input_image)
            else:
                continue

            # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
            output_temp.append(get_LDE_code(int(elem[3]), sigma=sigma))


        yield np.array(img_data), np.array(output_temp)


def all_TimeValue_TValue(data_dict):
    """
    All data for training or testing:
        input data: Time(Value);
        label data: temperature(Value)
    :param data_dict: original data dict
    :return: [Time(Value)], [temp(Value)]
    """
    image_data = []
    time_data = []
    temp_data = []
    for elem in data_dict:

        # time_data.append(np.append(input_date, input_time))
        input_date = int(elem[8])
        input_time = int(elem[10])
        time_data.append([input_date, input_time, 37, 107])

        temp_data.append(float(elem[3]))
    return [np.array(time_data)], [np.array(temp_data)]

def all_ImgTimeOnehot_TValue(data_dict, image_root=None):
    """
    All data for training or testing:
        input data: Image, Time(Onehot);
        label data: temperature(Value)
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :return: [Image, Time(Onehot)], [temp(Value)]
    """
    image_data = []
    time_data = []
    temp_data = []
    for elem in data_dict:
        # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        # img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(224, 224))
        img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
        if img:
            input_image = image.img_to_array(img)
            input_image = preprocess_input(input_image)
        else:
            continue
        # input_date = np_utils.to_categorical(int(elem[8]) - 1, 12)  # 从0开始算
        # input_time = np_utils.to_categorical(int(elem[10]), 24)


        image_data.append(input_image)
        # time_data.append(np.append(input_date, input_time))
        input_date = int(elem[8])
        input_time = int(elem[10])
        time_data.append([input_date, input_time, 37, 107])

        temp_data.append(float(elem[3]))
    return [np.array(image_data), np.array(time_data)], [np.array(temp_data)]


def all_ImgTimeValue_TValue(data_dict, image_root=None):
    """
    All data for training or testing:
        input data: Image, Time(Value);
        label data: temperature(Value)
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :return: [Image, Time(Value)], [temp(Value)]
    """
    image_data = []
    time_data = []
    temp_data = []
    for elem in data_dict:
        # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        # img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(224, 224))
        img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
        if img:
            input_image = image.img_to_array(img)
            input_image = preprocess_input(input_image)
        else:
            continue

        input_date = int(elem[8])
        input_time = int(elem[10])

        image_data.append(input_image)
        time_data.append([input_date, input_time, 37, 107])
        temp_data.append(float(elem[3]))
    return [np.array(image_data), np.array(time_data)], [np.array(temp_data)]


def all_ImgTimeValue_TOnehot(data_dict, image_root=None):
    """
    All data for training or testing:
        input data: Image, Time(Value);
        label data: temperature(Value)
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :return: [Image, Time(Value)], [temp(Onehot)]
    """
    image_data = []
    time_data = []
    temp_data = []
    for elem in data_dict:
        # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        # img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(224, 224))
        img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
        if img:
            input_image = image.img_to_array(img)
            input_image = preprocess_input(input_image)
        else:
            continue

        input_date = int(elem[8])
        input_time = int(elem[10])

        image_data.append(input_image)
        time_data.append([input_date, input_time, 37, 107])

        temp_data.append(get_Onehot_code(int(elem[3])))
    return [np.array(image_data), np.array(time_data)], [np.array(temp_data)]



def all_Img_TValue(data_dict, image_root=None):
    """
    All data for training or testing:
        input data: Image;
        label data: temperature(Value)
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :return: [Image], [temp(Value)]
    """
    image_data = []
    temp_data = []
    for elem in data_dict:
        # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(224, 224))
        if img:
            input_image = image.img_to_array(img)
            input_image = preprocess_input(input_image)
        else:
            continue
        image_data.append(input_image)
        temp_data.append(float(elem[3]))
    return np.array(image_data), np.array(temp_data)


def all_Img_TOnehot(data_dict, image_root=None):
    """
    All data for training or testing:
        input data: Image;
        label data: temperature(Onehot)
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :return: [Image], [temp(Onehot)]
    """
    image_data = []
    temp_data = []
    for elem in data_dict:
        # # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        # img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(224, 224))
        img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
        if img:
            input_image = image.img_to_array(img)
            input_image = preprocess_input(input_image)
        else:
            continue
        image_data.append(input_image)
        temp_data.append(get_Onehot_code(int(elem[3])))
    return np.array(image_data), np.array(temp_data)


def get_single_img_TOnehot_by_path(img_path):

    """
    All data for training or testing:
        input data: Image;
        label data: temperature(Onehot)
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :return: [Image], [temp(Onehot)]
    """
    temp = get_info_by_imgpath(img_path)

    image_data = []
    temp_data = []

    img = image.load_img(img_path, target_size=(320, 240))
    if img:
        input_image = image.img_to_array(img)
        input_image = preprocess_input(input_image)
    else:
        pass
    image_data.append(input_image)
    temp_data.append(get_Onehot_code(temp))
    return np.array(image_data), np.array(temp_data)


def all_ImgSequence_TValueSequence(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image Sequence;
        label data: temperature Sequence
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :param sigma: distribution factor for LDE encoding
    :return: [Image(Sequence)], [temp(Sequence)]
    """
    image_data = []
    temp_data = []
    for elem in data_dict:
        # meta data: 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        image_data.append(load_image_for_step(image_root=image_root, batch_data=elem))
        sub_temp_data = []
        for i in range(len(elem)):
            sub_temp_data.append(int(elem[i][3]))
        temp_data.append(sub_temp_data)
    return np.array(image_data), np.array(temp_data)
    # return tf.data.Dataset.from_tensors(image_data), tf.data.Dataset.from_tensors(temp_data)


def all_ImgSequence_TValueSequence_v2(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image Sequence;
        label data: temperature Sequence
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :param sigma: distribution factor for LDE encoding
    :return: [Image(Sequence)], [st-data(Sequence)], [temp(Sequence)]
    """
    image_data = []
    st_data = []
    temp_data = []
    for elem in data_dict:
        # meta data: 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        image_data.append(load_image_for_step(image_root=image_root, batch_data=elem))

        # space-time data
        sub_st_data = []
        for i in range(len(elem)):
            input_date = int(elem[i][8])
            input_time = int(elem[i][10])
            sub_st_data.append([input_date, input_time, 37, 107])
        st_data.append(sub_st_data)
        # temp data
        sub_temp_data = []
        for i in range(len(elem)):
            sub_temp_data.append(int(elem[i][3]))
        temp_data.append(sub_temp_data)
    return np.array(image_data), np.array(st_data), np.array(temp_data)
    # return tf.data.Dataset.from_tensors(image_data), tf.data.Dataset.from_tensors(temp_data)


def all_ImgSequence_TOnehotSequence(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image Sequence;
        label data: temperature Sequence
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :param sigma: distribution factor for LDE encoding
    :return: [Image(Sequence)], [temp(Sequence)]
    """
    image_data = []
    temp_data = []
    for elem in data_dict:
        # meta data: 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        image_data.append(load_image_for_step(image_root=image_root, batch_data=elem))
        sub_temp_data = []
        for i in range(len(elem)):
            sub_temp_data.append(get_Onehot_code(int(elem[i][3])))
        temp_data.append(sub_temp_data)
    return np.array(image_data), np.array(temp_data)
    # return tf.data.Dataset.from_tensors(image_data), tf.data.Dataset.from_tensors(temp_data)


def all_ImgSequence_TOnehotSequence_v2(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image Sequence;
        label data: temperature Sequence
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :param sigma: distribution factor for LDE encoding
    :return: [Image(Sequence)], [st-data(Sequence)], [temp(Sequence)]
    """
    image_data = []
    st_data = []
    temp_data = []
    for elem in data_dict:
        # meta data: 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        image_data.append(load_image_for_step(image_root=image_root, batch_data=elem))

        # space-time data
        sub_st_data = []
        for i in range(len(elem)):
            input_date = int(elem[i][8])
            input_time = int(elem[i][10])
            sub_st_data.append([input_date, input_time, 37, 107])
        st_data.append(sub_st_data)

        # temp data
        sub_temp_data = []
        for i in range(len(elem)):
            sub_temp_data.append(get_Onehot_code(int(elem[i][3])))
        temp_data.append(sub_temp_data)
    return np.array(image_data), np.array(st_data), np.array(temp_data)
    # return tf.data.Dataset.from_tensors(image_data), tf.data.Dataset.from_tensors(temp_data)


def all_ImgSequence_TLDESequence(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image Sequence;
        label data: temperature Sequence
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :param sigma: distribution factor for LDE encoding
    :return: [Image(Sequence)], [temp(Sequence)]
    """
    image_data = []
    temp_data = []
    for elem in data_dict:
        # meta data: 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        image_data.append(load_image_for_step(image_root=image_root, batch_data=elem))
        sub_temp_data = []
        for i in range(len(elem)):
            sub_temp_data.append(get_LDE_code(int(elem[i][3]), sigma=sigma))
        temp_data.append(sub_temp_data)
    return np.array(image_data), np.array(temp_data)
    # return tf.data.Dataset.from_tensors(image_data), tf.data.Dataset.from_tensors(temp_data)


def all_ImgSequence_TLDESequence_v2(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image Sequence;
        label data: temperature Sequence
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :param sigma: distribution factor for LDE encoding
    :return: [Image(Sequence)], [st-data(Sequence)], [temp(Sequence)]
    """
    image_data = []
    st_data = []
    temp_data = []
    for elem in data_dict:
        # meta data: 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        image_data.append(load_image_for_step(image_root=image_root, batch_data=elem))

        # space-time data
        sub_st_data = []
        for i in range(len(elem)):
            input_date = int(elem[i][8])
            input_time = int(elem[i][10])
            sub_st_data.append([input_date, input_time, 37, 107])
        st_data.append(sub_st_data)

        # temp data
        sub_temp_data = []
        for i in range(len(elem)):
            sub_temp_data.append(get_LDE_code(int(elem[i][3]), sigma=sigma))
        temp_data.append(sub_temp_data)
    return np.array(image_data), np.array(st_data), np.array(temp_data)
    # return tf.data.Dataset.from_tensors(image_data), tf.data.Dataset.from_tensors(temp_data)


def all_ImgSequence_TLDESequence_v3(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image Sequence;
        label data: temperature Sequence
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :param sigma: distribution factor for LDE encoding
    :return: [Image(Sequence)], [st-data(Sequence)], [temp(Sequence)]
    """
    image_data = []
    st_data = []
    temp_data = []
    for elem in data_dict:
        # meta data: 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        image_data.append(load_image_for_step(image_root=image_root, batch_data=elem))

        # time data
        input_date = int(elem[-1][8])
        input_time = int(elem[-1][10])
        st_data.append([input_date, input_time, 37, 107])

        # temp data
        # sub_temp_data = []
        # for i in range(len(elem)):
        #     sub_temp_data.append(get_LDE_code(int(elem[i][3]), sigma=sigma))
        # temp_data.append(sub_temp_data)
        temp_data.append(get_LDE_code(int(elem[-1][3]), sigma=sigma))

    return np.array(image_data), np.array(st_data), np.array(temp_data)
    # return tf.data.Dataset.from_tensors(image_data), tf.data.Dataset.from_tensors(temp_data)



def all_ImgTimeOnehot_TLDE(data_dict, image_root=None, sigma=3.5):
    """
    All data for training or testing:
        input data: Image, Time(Onehot);
        label data: temperature(LDE)
    :param data_dict: original data dict
    :param image_root: directory that contains image files
    :return: [Image], [temp(LDE)]
    """
    image_data = []
    time_data = []
    temp_data = []
    for elem in data_dict:
        # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
        # img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(224, 224))
        img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
        if img:
            input_image = image.img_to_array(img)
            input_image = preprocess_input(input_image)
        else:
            continue
        input_date = np_utils.to_categorical(int(elem[8]) - 1, 12)  # 从0开始算
        input_time = np_utils.to_categorical(int(elem[10]), 24)

        image_data.append(input_image)
        time_data.append(np.append(input_date, input_time))
        temp_data.append(get_LDE_code(int(elem[3]), sigma=sigma))
    return [np.array(image_data), np.array(time_data)], [np.array(temp_data)]


def load_image_for_step(image_root, batch_data):
    """
    Load several images once for training or testing
    :param image_root: directory that contains image files
    :param batch_data: meta data
    :return:
    """
    output_list = []
    for elem in batch_data:
        img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(224, 224))
        # img = image.load_img(os.path.join(image_root, elem[4] + '/' + elem[2]), target_size=(320, 240))
        if img:
            input_image = image.img_to_array(img)
            input_image = preprocess_input(input_image)
            output_list.append(input_image)
        else:
            continue
    return output_list


def load_data_from_image2weather():
    '''
    load data from the dataset: image2weather;
    :return:
    '''
    pass


# loss weight; callback
class LossWeightsScheduler(Callback):
    '''
    调用
    obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'], model.loss_weights['pred_hum'], factor=1)
    '''
    def __init__(self,
                 classLossInitWeight,
                 tempLossInitWeight,
                 humLossInitWeight,
                 factor = 1.0):
        self.classLossInitWeight = classLossInitWeight
        self.tempLossInitWeight = tempLossInitWeight
        self.humLossInitWeight = humLossInitWeight
        self.factor = factor

        self.class_loss = 1.0
        self.temp_loss = 1.0
        self.hum_loss = 1.0

        self.logs = {}

        self.histroy = {}

    def calculate_loss_weights(self):
        if self.logs:
            self.classLossInitWeight = self.factor *  (1.0 / self.class_loss)
            self.tempLossInitWeight = self.factor * (1.0 / self.temp_loss)
            self.humLossInitWeight = self.factor * (1.0 / self.hum_loss)

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.logs = logs or {}

        self.class_loss = self.logs.get('val_pred_class_loss')  # 取验证损失
        self.temp_loss = self.logs.get('val_pred_temp_loss')
        self.hum_loss = self.logs.get('val_pred_hum_loss')

        self.calculate_loss_weights()
        print("logs output:", self.class_loss, self.temp_loss, self.hum_loss)
        print("Callback output:", self.classLossInitWeight, self.tempLossInitWeight, self.humLossInitWeight)
        print("epoch:", epoch)

    def on_train_end(self, logs={}):
        pass


def build_model_vgg(lr=0.001, info_dict=None, layer=5):
    """
    Build vgg model for temperature predicting
        input data : img
        label data : temp(Value)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: vgg model
    """

    input_img = Input(shape=(224, 224, 3), name='input_image')

    # conv features
    img_model = models.WeatherClsCNN.vgg19.vgg19_chosen_layer(
        img_rows=info_dict["size"],
        img_cols=info_dict["size"],
        channel=info_dict["channels"],
        # num_classes=len(info_dict["classname"]),
        weights_path=info_dict['weightpath'],
        info_dict={},
        trainable=True,
        chosen_layer=layer)
    # # ==== pool 2 =====
    # img_model.summary()
    # pool3 = Model(img_model.input, img_model.get_layer(name="block2_pool").output)
    # pool3.summary()
    # new_output = pool3(input_img)

    # # ==== pool 3 =====
    # pool3 = Model(img_model.input, img_model.get_layer(name="block3_pool").output)
    # pool3.summary()
    # new_output = pool3(input_img)

    # # ==== pool 4 =====
    # # [] batch_size=16, epochs=60, lr=0.0001 # batch_size # 影响还挺大
    # # [Scene 6,7] batch_size=8, epochs=60,lr=0.0001
    # pool3 = Model(img_model.input, img_model.get_layer(name="block4_pool").output)
    # pool3.summary()
    # new_output = pool3(input_img)

    # # ==== pool 5 =====
    # [scene0-4 best] batch_size=64, epochs=60, lr=0.000001
    # pool3 = Model(img_model.input, img_model.get_layer(name="block5_pool").output)


    img_model.summary()
    new_output = img_model(input_img)

    new_output = Flatten(name='flatten')(new_output)
    #vnew_output = Dense(1024, activation='relu', name='fc1')(new_output)
    # new_output = Dense(4096, activation='relu', name='fc1')(new_output)
    # new_output = Dense(4096, activation='relu', name='fc2')(new_output)
    new_output = Dense(1, name='pred_temp')(new_output)

    model = Model(inputs=input_img, outputs=new_output)
    # model.summary()

    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss={
                          'pred_temp': 'mean_squared_error',
                      },
                      metrics={
                          'pred_temp': root_mean_squared_error,
                      },
                  )

    return model


def build_model_MLP_TimeValue_TValue(lr=0.001):
    """
     Build simpleCNN+MLP model for temperature predicting
         input data : img, Time(Onehot)
         label data : temp(Value)
     :param lr:  learning rate
     :param info_dict: configuration
     :return: simpleCNN+MLP model
     """
    # Input data
    input_time = Input(shape=(4,), name='input_time')

    # Time feature
    x_time = Dense(64, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(64, activation='relu', name='time_dense_2')(x_time)
    x_time = Dense(64, activation='relu', name='time_dense_3')(x_time)
    x_time = Dense(64, activation='relu', name='time_dense_4')(x_time)
    # x_concat = Dropout(0.5)(x_time)

    # Output: Temperature(Value)
    # y_temp = Dense(1, name="pred_temp")(x_concat)
    y_temp = Dense(1, name="pred_temp")(x_time)

    model = Model(input_time, y_temp)

    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  loss={
                      'pred_temp': 'mean_squared_error',
                  },
                  metrics={
                      'pred_temp': r_squared_score,
                  },
                  )
    return model


def build_model_simpleCNN_Img_TValue(lr=0.001):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Value)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN model
    """
    # input_img = Input(shape=(224, 224, 3), name='input_image')
    input_img = Input(shape=(320, 240, 3), name='input_image')

    # # ======== FORMAL VERSION ========
    # x_img = BatchNormalization(name="BN_1")(input_img)
    # x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    # x_img = BatchNormalization(name="BN_2")(x_img)
    # x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    # x_img = BatchNormalization(name="BN_3")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    # x_img = BatchNormalization(name="BN_4")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    # # =================================


    # # # ======== FORMAL VERSION 5 layers ========
    # x_img = BatchNormalization(name="BN_1")(input_img)
    # x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    # x_img = BatchNormalization(name="BN_2")(x_img)
    # x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    # x_img = BatchNormalization(name="BN_3")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    # x_img = BatchNormalization(name="BN_4")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)
    # ## =================================


    # ========= NO BN LAYERS ==========
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(input_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)

    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)

    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)

    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)

    x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)
    # # ===============================

    # # ======= CONV+BN+ACTIVATEION =========
    # x_img = Conv2D(32, (3, 3), name="Conv_1")(input_img)
    # x_img = Activation("relu")(x_img)
    # x_img = BatchNormalization(name="BN_1")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    #
    # x_img = Conv2D(32, (3, 3), name="conv_2")(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = BatchNormalization(name="BN_2")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    #
    # x_img = Conv2D(64, (3, 3), name='Conv_3')(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = BatchNormalization(name="BN_3")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    #
    # x_img = Conv2D(64, (3, 3), name='Conv_4')(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = BatchNormalization(name="BN_4")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = Conv2D(128, (3, 3), name='Conv_5')(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)
    # # ======================================


    x_img = Flatten()(x_img)
    x_img = Dense(512, name='cnn_dense_1')(x_img)
    x_img = Dense(1, name='pred_temp')(x_img)

    model = Model(input_img, x_img)

    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss={
                          'pred_temp': 'mean_squared_error',
                      },
                      metrics={
                          'pred_temp': root_mean_squared_error,
                      },
                  )
    return model


def build_model_simpleCNN_Img_TOnehot(lr=0.001):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN model
    """
    # input_img = Input(shape=(224, 224, 3), name='input_image')
    input_img = Input(shape=(320, 240, 3), name='input_image')

    # # ======= FORMAL SimpleCNN ========
    # x_img = BatchNormalization(name="BN_1")(input_img)
    # x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    # x_img = BatchNormalization(name="BN_2")(x_img)
    # x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    # x_img = BatchNormalization(name="BN_3")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    # x_img = BatchNormalization(name="BN_4")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_4")(x_img)
    # # ==================================

    # # ======= FORMAL SimpleCNN: 5 Conv_layers ========
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)

    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)

    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)

    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_4")(x_img)

    x_img = BatchNormalization(name="BN_5")(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_5")(x_img)
    # # ==================================

    # # # # ==== no BN layers : 5 layers ====
    # x_img = BatchNormalization(name="BN_1")(input_img)
    # x_img = input_img
    # x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    # # x_img = BatchNormalization(name="BN_2")(x_img)
    # x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    # # x_img = BatchNormalization(name="BN_3")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    # # x_img = BatchNormalization(name="BN_4")(x_img)
    # x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_4")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_5")(x_img)
    # # ====================
    # # #


    # # # ======= CONV+BN+ACTIVATION : 5 layers ========
    # x_img = Conv2D(32, (3, 3), name="Conv_1")(input_img)
    # x_img = BatchNormalization(name="BN_1")(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    #
    # x_img = Conv2D(32, (3, 3), name="conv_2")(x_img)
    # x_img = BatchNormalization(name="BN_2")(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    #
    # x_img = Conv2D(64, (3, 3), name='Conv_3')(x_img)
    # x_img = BatchNormalization(name="BN_3")(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    #
    # x_img = Conv2D(64, (3, 3), name='Conv_4')(x_img)
    # x_img = BatchNormalization(name="BN_4")(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_4")(x_img)
    #
    # x_img = Conv2D(128, (3, 3), name='Conv_5')(x_img)
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Activation("relu")(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name="Pool_5")(x_img)
    # # # ==================================


    x_img = Flatten()(x_img)
    x_img = Dense(128, name='cnn_dense_1')(x_img)
    x_img = Dense(80, activation='softmax', name='pred_temp')(x_img)

    model = Model(input_img, x_img)
    # model.summary()

    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss={
                          # 'pred_temp': LDE_loss,
                          'pred_temp': 'kullback_leibler_divergence',
                      },
                      metrics={
                          'pred_temp': LDE_acc,
                      },
                  )
    return model


def build_model_simpleCNNMLP_ImgTimeOnehot_TValue(lr=0.001):
    """
    Build simpleCNN+MLP model for temperature predicting
        input data : img, Time(Onehot)
        label data : temp(Value)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+MLP model
    """
    # Input data
    # input_img = Input(shape=(224, 224, 3), name='input_image')
    input_img = Input(shape=(320, 240, 3), name='input_image')
    # input_time = Input(shape=(36,), name='input_time')
    input_time = Input(shape=(4,), name='input_time')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)

    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)

    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)

    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_4")(x_img)

    x_img = BatchNormalization(name="BN_5")(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_5")(x_img)

    x_img = Flatten()(x_img)

    # Time feature
    x_time = Dense(32, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(32, activation='relu', name='time_dense_2')(x_time)

    x_concat = concatenate([x_img, x_time])

    x_concat = Dense(512, activation='relu', name='concat_dense')(x_concat)
    x_concat = Dropout(0.5)(x_concat)

    # Output: Temperature(Value)
    y_temp = Dense(1, name="pred_temp")(x_concat)

    model = Model([input_img, input_time], y_temp)

    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss={
                          'pred_temp': 'mean_squared_error',
                      },
                      metrics={
                          # 'pred_temp': r_squared_score,
                          'pred_temp': root_mean_squared_error,
                      },
                  )
    return model


def build_model_simpleCNNMLP_ImgTimeOnehot_TOnehot(lr=0.001):
    """
    Build simpleCNN+MLP model for temperature predicting
        input data : img, Time(Onehot)
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+MLP model
    """
    # Input data
    # input_img = Input(shape=(224, 224, 3), name='input_image')
    input_img = Input(shape=(320, 240, 3), name='input_image')
    # input_time = Input(shape=(36,), name='input_time')
    input_time = Input(shape=(4,), name='input_time')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    x_img = BatchNormalization(name="BN_5")(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)
    x_img = Flatten()(x_img)

    # Time feature
    x_time = Dense(32, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(32, activation='relu', name='time_dense_2')(x_time)

    x_concat = concatenate([x_img, x_time])

    x_concat = Dense(512, activation='relu', name='concat_dense')(x_concat)

    # Output: Temperature(Onehot)
    y_temp = Dense(80, activation='softmax', name="pred_temp")(x_concat)

    model = Model([input_img, input_time], y_temp)

    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss={
                          'pred_temp': 'kullback_leibler_divergence',
                      },
                      metrics={
                          'pred_temp': LDE_acc,
                      },
                  )
    return model


def build_model_simpleCNNLSTM_TValue(lr=0.001, step=3):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+LSTM model
    """
    # Input data
    input_img = Input(shape=(224, 224, 3), name='input_image')
    # input_img = Input(shape=(320, 240, 3), name='input_image')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)

    x_img = Flatten()(x_img)
    x_img = Dense(1024, activation='relu', name='cnn_dense_1')(x_img)
    x_img = Dense(128, name='cnn_dense_2')(x_img)
    x_img = BatchNormalization(name="cnn_dense_bn")(x_img)
    cnn_model = Model(input_img, x_img)
    cnn_model.summary()

    # Sequence feature
    input_shape = (step, 224, 224, 3)
    # input_shape = (step, 320, 240, 3)
    model_cnnrnn = Sequential()
    model_cnnrnn.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model_cnnrnn.add(
        LSTM(units=100,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=True,
             ))

    model_cnnrnn.add(Dense(1, name='pred_temp'))

    model_cnnrnn.summary()

    model_cnnrnn.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss={
                          'pred_temp': 'mean_squared_error',
                      },
                      metrics={
                          'pred_temp': r_squared_score,
                      },
                  )
    return model_cnnrnn


def build_model_simpleCNNLSTM_TValue_v2(lr=0.001, step=3):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+LSTM model
    """
    # Input data
    input_img = Input(shape=(224, 224, 3), name='input_image')
    input_time = Input(shape=(4,), name='input_time')
    # input_img = Input(shape=(320, 240, 3), name='input_image')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)

    x_img = Flatten()(x_img)
    x_img = Dense(1024, activation='relu', name='cnn_dense_1')(x_img)
    x_img = Dense(512, name='cnn_dense_2')(x_img)
    x_img = BatchNormalization(name="cnn_dense_bn")(x_img)
    cnn_model = Model(input_img, x_img)
    cnn_model.summary()

    # space-time features
    # Time feature
    x_time = Dense(32, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(32, activation='relu', name='time_dense_2')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_1')(input_time)
    # x_time = Dense(64, activation='relu', name='time_dense_2')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_3')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_4')(x_time)
    st_model = Model(input_time, x_time)
    st_model.summary()


    # Image Sequence feature
    input_shape = (step, 224, 224, 3)
    # input_shape = (step, 320, 240, 3)
    model_cnnrnn = Sequential()
    model_cnnrnn.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model_cnnrnn.add(
        LSTM(units=100,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=True,
             ))

    model_cnnrnn.add(Dense(32, name='cnn_pred_temp'))

    # Space-time Sequence feature
    st_input_shape = (step, 4)
    # input_shape = (step, 320, 240, 3)
    model_st = Sequential()
    model_st.add(TimeDistributed(st_model, input_shape=st_input_shape))
    model_st.add(
        LSTM(units=32,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=True,
             ))
    model_st.add(Dense(32, name='mlp_pred_temp'))
    model_st.summary()

    step_input_img = Input(shape=(step, 224, 224, 3), name='step_input_image')
    step_input_time = Input(shape=(step, 4), name='step_input_time')

    cnn_x = model_cnnrnn(step_input_img)
    mlp_x = model_st(step_input_time)
    concat_x = concatenate([cnn_x, mlp_x])
    concat_x = Dense(32, activation='relu', name='concat_dense')(concat_x)
    temp_output = Dense(1, name='pred_temp')(concat_x)

    model_stc_lstm = Model([step_input_img, step_input_time], temp_output)


    model_stc_lstm.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss={
                          'pred_temp': 'mean_squared_error',
                      },
                      metrics={
                          'pred_temp': r_squared_score,
                      },
                  )
    return model_stc_lstm


def build_model_simpleCNNLSTM_TLDE(lr=0.001, step=3):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+LSTM model
    """
    # Input data
    input_img = Input(shape=(224, 224, 3), name='input_image')
    # input_img = Input(shape=(320, 240, 3), name='input_image')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)

    x_img = Flatten()(x_img)
    x_img = Dense(1024, activation='relu', name='cnn_dense_1')(x_img)
    x_img = Dense(128, name='cnn_dense_2')(x_img)
    x_img = BatchNormalization(name="cnn_dense_bn")(x_img)
    cnn_model = Model(input_img, x_img)
    cnn_model.summary()

    # Sequence feature
    input_shape = (step, 224, 224, 3)
    # input_shape = (step, 320, 240, 3)
    model_cnnrnn = Sequential()
    model_cnnrnn.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model_cnnrnn.add(
        LSTM(units=100,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=True,
             ))

    model_cnnrnn.add(Dense(80, activation='softmax', name='pred_temp'))

    model_cnnrnn.summary()

    model_cnnrnn.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                             loss={
                                'pred_temp': 'kullback_leibler_divergence',
                             },
                             metrics={
                                 'pred_temp': LDE_acc,
                             },
                         )
    return model_cnnrnn


def build_model_simpleCNNLSTM_TLDE_v4(lr=0.001, step=3):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+LSTM model
    """
    # Input data
    input_img = Input(shape=(224, 224, 3), name='input_image')
    # input_img = Input(shape=(320, 240, 3), name='input_image')
    input_time = Input(shape=(4, ), name='input_image')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)

    x_img = Flatten()(x_img)
    x_img = Dense(1024, activation='relu', name='cnn_dense_1')(x_img)
    x_img = Dense(128, name='cnn_dense_2')(x_img)
    x_img = BatchNormalization(name="cnn_dense_bn")(x_img)
    cnn_model = Model(input_img, x_img)
    cnn_model.summary()

    # Sequence feature
    input_shape = (step, 224, 224, 3)
    # input_shape = (step, 320, 240, 3)
    model_cnnrnn = Sequential()
    model_cnnrnn.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model_cnnrnn.add(
        LSTM(units=100,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=True,
             ))

    # model_cnnrnn.add(Dense(80, activation='softmax', name='pred_temp'))
    model_cnnrnn.summary()

    # space-time features
    # Time feature
    x_time = Dense(32, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(32, activation='relu', name='time_dense_2')(x_time)
    st_model = Model(input_time, x_time)
    st_model.summary()

    # Space-time Sequence feature
    st_input_shape = (step, 4)
    model_strnn = Sequential()
    model_strnn.add(TimeDistributed(st_model, input_shape=st_input_shape))
    # model_strnn.add(
    #     LSTM(units=32,
    #          activation='softsign',
    #          # activation='tanh',
    #          kernel_initializer='orthogonal',
    #          bias_initializer='ones',
    #          dropout=0.5,
    #          recurrent_dropout=0.5,
    #          recurrent_regularizer=l2(0.00001),
    #          kernel_regularizer=l2(0.00001),
    #          return_sequences=True,
    #          ))
    # model_strnn.add(Dense(16, name='mlp_pred_temp'))
    model_strnn.summary()


    step_input_img = Input(shape=(step, 224, 224, 3), name='step_input_image')
    step_input_time = Input(shape=(step, 4), name='step_input_time')

    cnn_x = model_cnnrnn(step_input_img)
    mlp_x = model_strnn(step_input_time)
    concat_x = concatenate([cnn_x, mlp_x])
    concat_x = Dense(128, activation='relu', name='concat_dense')(concat_x)
    temp_output = Dense(80, activation='softmax', name='pred_temp')(concat_x)

    model_stc_lstm = Model([step_input_img, step_input_time], temp_output)



    model_stc_lstm.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                             loss={
                                'pred_temp': 'kullback_leibler_divergence',
                             },
                             metrics={
                                 'pred_temp': LDE_acc,
                             },
                         )
    return model_stc_lstm

# NOT PASS
def build_model_simpleCNNLSTM_TLDE_v2(lr=0.001, step=3):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+LSTM model
    """
    # Input data
    input_img = Input(shape=(224, 224, 3), name='input_image')
    input_time = Input(shape=(4,), name='input_time')
    # input_img = Input(shape=(320, 240, 3), name='input_image')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)

    x_img = Flatten()(x_img)
    x_img = Dense(1024, activation='relu', name='cnn_dense_1')(x_img)
    x_img = Dense(128, name='cnn_dense_2')(x_img)
    x_img = BatchNormalization(name="cnn_dense_bn")(x_img)
    cnn_model = Model(input_img, x_img)
    cnn_model.summary()

    # space-time features
    # Time feature
    x_time = Dense(32, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(32, activation='relu', name='time_dense_2')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_1')(input_time)
    # x_time = Dense(64, activation='relu', name='time_dense_2')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_3')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_4')(x_time)
    st_model = Model(input_time, x_time)
    st_model.summary()

    # Sequence feature
    input_shape = (step, 224, 224, 3)
    # input_shape = (step, 320, 240, 3)
    model_cnnrnn = Sequential()
    model_cnnrnn.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model_cnnrnn.add(
        LSTM(units=100,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=True,
             ))

    # model_cnnrnn.add(Dense(80, activation='softmax', name='pred_temp'))
    # model_cnnrnn.add(Dense(100, activation='relu', name='cnn_pred_temp'))
    # model_cnnrnn.add(Dense(80, name='pred_temp'))

    # Space-time Sequence feature
    st_input_shape = (step, 4)
    # input_shape = (step, 320, 240, 3)
    model_st = Sequential()
    model_st.add(TimeDistributed(st_model, input_shape=st_input_shape))
    model_st.add(
        LSTM(units=32,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=True,
             ))
    # model_st.add(Dense(16, name='mlp_pred_temp'))
    model_st.summary()

    step_input_img = Input(shape=(step, 224, 224, 3), name='step_input_image')
    step_input_time = Input(shape=(step, 4), name='step_input_time')

    cnn_x = model_cnnrnn(step_input_img)
    mlp_x = model_st(step_input_time)
    concat_x = concatenate([cnn_x, mlp_x])
    concat_x = Dense(128, activation='relu', name='concat_dense')(concat_x)
    temp_output = Dense(80, activation='softmax', name='pred_temp')(concat_x)

    model_stc_lstm = Model([step_input_img, step_input_time], temp_output)


    model_stc_lstm.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                             loss={
                                'pred_temp': 'kullback_leibler_divergence',
                             },
                             metrics={
                                 'pred_temp': LDE_acc,
                             },
                         )
    return model_stc_lstm


def build_model_simpleCNNLSTM_TLDE_v3(lr=0.001, step=3):
    """
    Build simpleCNN model for temperature predicting
        input data : img
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+LSTM model
    """
    # Input data
    input_img = Input(shape=(224, 224, 3), name='input_image')
    input_time = Input(shape=(4,), name='input_time')
    # input_img = Input(shape=(320, 240, 3), name='input_image')

    # WeatherClsCNN feature
    x_img = BatchNormalization(name="BN_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_1")(x_img)
    x_img = BatchNormalization(name="BN_2")(x_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_2")(x_img)
    x_img = BatchNormalization(name="BN_3")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name="Pool_3")(x_img)
    x_img = BatchNormalization(name="BN_4")(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((3, 3), strides=2, name='Pool_4')(x_img)
    #
    # x_img = BatchNormalization(name="BN_5")(x_img)
    # x_img = Conv2D(128, (3, 3), activation='relu', name='Conv_5')(x_img)
    # x_img = MaxPooling2D((3, 3), strides=2, name='Pool_5')(x_img)

    x_img = Flatten()(x_img)
    x_img = Dense(1024, activation='relu', name='cnn_dense_1')(x_img)
    x_img = Dense(128, name='cnn_dense_2')(x_img)
    x_img = BatchNormalization(name="cnn_dense_bn")(x_img)
    cnn_model = Model(input_img, x_img)
    cnn_model.summary()

    # space-time features
    # Time feature
    x_time = Dense(32, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(32, activation='relu', name='time_dense_2')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_1')(input_time)
    # x_time = Dense(64, activation='relu', name='time_dense_2')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_3')(x_time)
    # x_time = Dense(64, activation='relu', name='time_dense_4')(x_time)
    st_model = Model(input_time, x_time)
    st_model.summary()

    # Sequence feature
    input_shape = (step, 224, 224, 3)
    # input_shape = (step, 320, 240, 3)
    model_cnnrnn = Sequential()
    model_cnnrnn.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model_cnnrnn.add(
        LSTM(units=100,
             activation='softsign',
             # activation='tanh',
             kernel_initializer='orthogonal',
             bias_initializer='ones',
             dropout=0.5,
             recurrent_dropout=0.5,
             recurrent_regularizer=l2(0.00001),
             kernel_regularizer=l2(0.00001),
             return_sequences=False,
             ))

    # Sequence st


    # model_cnnrnn.add(Dense(80, activation='softmax', name='pred_temp'))
    # model_cnnrnn.add(Dense(100, name='cnn_pred_temp'))

    step_input_img = Input(shape=(step, 224, 224, 3), name='step_input_image')
    # step_input_st = Input(shape=(step, 4, 1), name='step_input_st')

    # model_mlprnn = Sequential()
    # model_mlprnn.add(TimeDistributed(st_model, input_shape=step_input_st))

    cnn_x = model_cnnrnn(step_input_img)
    mlp_x = st_model(input_time)

    concat_x = concatenate([cnn_x, mlp_x])
    concat_x = Dense(128, activation='relu', name='concat_dense')(concat_x)
    temp_output = Dense(80, name='pred_temp')(concat_x)

    model_stc_lstm = Model([step_input_img, input_time], temp_output)
    model_stc_lstm.summary()


    model_stc_lstm.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                             loss={
                                'pred_temp': 'kullback_leibler_divergence',
                             },
                             metrics={
                                 'pred_temp': LDE_acc,
                             },
                         )
    return model_stc_lstm



def build_model_simpleCNNMLP_TValue(lr=0.001, step=3):
    """
    Build simpleCNN+MLP model for temperature predicting
        input data : img, Time(Value), Location(Value)
        label data : temp(Onehot/LDE)
    :param lr:  learning rate
    :param info_dict: configuration
    :return: simpleCNN+MLP model
    """
    # Input data
    # input_img = Input(shape=(224, 224, 3), name='input_image')
    input_img = Input(shape=(320, 240, 3), name='input_image')
    input_time = Input(shape=(4,), name='input_time')

    # WeatherClsCNN feature
    x_img = Conv2D(32, (3, 3), activation='relu', name="Conv_1")(input_img)
    x_img = Conv2D(32, (3, 3), activation='relu', name="conv_2")(x_img)
    x_img = MaxPooling2D((2, 2), name="Pool_1")(x_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_3')(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', name='Conv_4')(x_img)
    x_img = MaxPooling2D((2, 2), name='Pool_2')(x_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Flatten()(x_img)

    # Time feature
    x_time = Dense(32, activation='relu', name='time_dense_1')(input_time)
    x_time = Dense(32, activation='relu', name='time_dense_2')(x_time)

    x_concat = concatenate([x_img, x_time])

    x_concat = Dense(512, activation='relu', name='concat_dense')(x_concat)
    x_concat = Dropout(0.5)(x_concat)

    # Output: Temperature(Value)
    y_temp = Dense(1, name="pred_temp")(x_concat)

    model = Model([input_img, input_time], y_temp)

    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  loss={
                      'pred_temp': 'mean_squared_error',
                  },
                  metrics={
                      # 'pred_temp': root_mean_squared_error,
                      'pred_temp': r_squared_score,
                  },
                  )
    return model


def r_squared_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# ======== Prediction ========
def pred_by_MLP_Tvalue(cam_id=0):
    """

    Predict the temperature using simpleCNN model
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)
    # Load configure
    info_dict = get_configure_parameters_for_temp_pred()
    info_dict["epoches"] = 40
    info_dict["lr"] = 1e-3
    part_string = tp.get_part_string(info_dict=info_dict)

    t_start = time.clock()
    # Load Data
    # data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)
    data_list = obj.get_source_data_for_cnn()
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    train_gen = gen_TimeValue_TValue(train_data, batch_size=4)  # one hot with augementation
    valid_gen = gen_TimeValue_TValue(valid_data, batch_size=4)  # one hot
    test_gen = gen_TimeValue_TValue(test_data, batch_size=4)  # one hot

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # model = bulid_model_v1(loss_weights=[1.0, 0.01, 0.001], lr=0.001, info_dict=info_dict)
    model = build_model_MLP_TimeValue_TValue(lr=info_dict["lr"])   # 0.01是效果最好的学习率
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=5, verbose=1)
    # save the best model in training process
    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 # monitor='val_LDE_acc',     # 还真的可以...
                                 monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 # mode='max',
                                 # mode='min',
                                 period=1)
    # training model
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=500,
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")


    # model = load_model('/home/shiyanshi/sz/Image2Weather/Results/V1.0/model.h5', custom_objects={'Scale': Scale, 'root_mean_squared_error': root_mean_squared_error})
    # model.summary()
    model.load_weights(model_save_filename)

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # x, y = test_gen.__next__()
    #
    # x1, y1 = x.copy(), y.copy()
    #
    # evaluate = model.evaluate(x1, y1)
    loaded_test_data_x, loaded_test_data_y = all_TimeValue_TValue(test_data)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    predict = [np.squeeze(predict)]
    print(predict)
    print(loaded_test_data_y)

    data_dict = {
        'truth': loaded_test_data_y[0],
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict, tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    pass


def pred_by_vgg_TValue(cam_id=0, layer=5):
    """
    Predict the temperature using vgg model
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    t_start = time.clock()
    # Load Data
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    train_data, valid_data, test_data = split_data(data_list)


    print(len(train_data), len(valid_data), len(test_data))
    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    train_gen = gen_ImgTimeOnehot_TValue(train_data, image_root=obj.image_root, batch_size=8)  # one hot with augementation
    valid_gen = gen_ImgTimeOnehot_TValue(valid_data, image_root=obj.image_root, batch_size=8)  # one hot
    # test_gen = gen_ImgTimeOnehot_TValue(test_data, image_root=obj.image_root, batch_size=4)  # one hot
    # train_gen_x, train_gen_y = all_Img_TValue(train_data, image_root=obj.image_root)  # one hot with augementation
    # valid_gen_x, valid_gen_y = all_Img_TValue(valid_data, image_root=obj.image_root)  # one hot
    # Load configure
    info_dict = get_configure_parameters_for_temp_pred()
    info_dict["epoches"] = 200   # 60
    info_dict["lr"] = 1e-8    # 0.0001    # layer2: 1e-8,
    info_dict["batch_size"] = 32 # 32
    info_dict["steps_per_epoch"] = len(train_data)//info_dict["batch_size"]
    # info_dict["steps_per_epoch"] = 500
    part_string = tp.get_part_string(info_dict=info_dict)

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + "layer" + str(layer) + "_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # model = bulid_model_v1(loss_weights=[1.0, 0.01, 0.001], lr=0.001, info_dict=info_dict)
    model = build_model_vgg(lr=info_dict["lr"], info_dict=info_dict, layer=layer)
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=5, verbose=1)
    model_save_filename = info_dict["savepath"] + "/layer" + str(layer) + "_" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 # monitor='val_root_mean_squared_error',     # 还真的可以...
                                 monitor='val_root_mean_squared_error',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 # mode='max',
                                 mode='min',
                                 period=1)
    # training model
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=info_dict["steps_per_epoch"],
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + part_string + "_model.h5")
    model.load_weights(model_save_filename)

    # model = load_model('/home/shiyanshi/sz/Image2Weather/Results/V1.0/model.h5', custom_objects={'Scale': Scale, 'root_mean_squared_error': root_mean_squared_error})
    # model.summary()

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/layer" + str(layer) + "_" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    #
    # evaluate = model.evaluate(x1, y1)
    loaded_test_data_x, loaded_test_data_y = all_Img_TValue(test_data, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    predict = [np.squeeze(predict)]
    print(predict)
    print(loaded_test_data_y)

    data_dict = {
        'truth': loaded_test_data_y,
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/layer" + str(layer) + "_" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note="layer"+str(layer)+"_")
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict[0], tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNN_Tvalue(cam_id=0, epoches=40, lr=0.0001):
    """

    Predict the temperature using simpleCNN model
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)
    # Load configure
    info_dict = get_configure_parameters_for_temp_pred()
    info_dict["epoches"] = epoches
    info_dict["lr"] = lr
    part_string = tp.get_part_string(info_dict=info_dict)

    t_start = time.clock()
    # Load Data
    cam_id=1
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)
    cam_id=2
    data_list.extend(obj.get_source_data_for_cnn(selected_cam=cam_id))
    cam_id=3
    data_list.extend(obj.get_source_data_for_cnn(selected_cam=cam_id))
    cam_id=5
    data_list.extend(obj.get_source_data_for_cnn(selected_cam=cam_id))
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    random.shuffle(data_list)
    cam_id=4
    data_list_2 = obj.get_source_data_for_cnn(selected_cam=cam_id)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    train_gen = gen_ImgTimeOnehot_TValue(data_list, image_root=obj.image_root, batch_size=4)  # one hot with augementation
    valid_gen = gen_ImgTimeOnehot_TValue(valid_data, image_root=obj.image_root, batch_size=4)  # one hot
    test_gen = gen_ImgTimeOnehot_TValue(test_data, image_root=obj.image_root, batch_size=4)  # one hot

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # model = bulid_model_v1(loss_weights=[1.0, 0.01, 0.001], lr=0.001, info_dict=info_dict)
    model = build_model_simpleCNN_Img_TValue(lr=info_dict["lr"])
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=5, verbose=1)
    # save the best model in training process
    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 # monitor='val_LDE_acc',     # 还真的可以...
                                 monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 # mode='max',
                                 # mode='min',
                                 period=1)
    # training model
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=500,
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")


    # model = load_model('/home/shiyanshi/sz/Image2Weather/Results/V1.0/model.h5', custom_objects={'Scale': Scale, 'root_mean_squared_error': root_mean_squared_error})
    # model.summary()
    model.load_weights(model_save_filename)

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # x, y = test_gen.__next__()
    #
    # x1, y1 = x.copy(), y.copy()
    #
    # evaluate = model.evaluate(x1, y1)
    loaded_test_data_x, loaded_test_data_y = all_ImgTimeOnehot_TValue(data_list_2, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    predict = [np.squeeze(predict)]
    print(predict)
    print(loaded_test_data_y)

    data_dict = {
        'truth': loaded_test_data_y[0],
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict, tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    pass


def pred_by_simpleCNN_TOnehot(cam_id=0, epoches=40, batch_size=16, lr=1e-4):
    """

    Predict the temperature using simpleCNN model, the labeled temperature data are in Onehot code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    t_start = time.clock()
    # Load Data
    info_dict = get_configure_parameters_for_temp_pred()
    part_string = tp.get_part_string(info_dict=info_dict)
    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    random.shuffle(data_list)
    data_list_2 = obj.get_source_data_for_cnn(selected_cam=cam_id)
    train_data, valid_data, test_data = split_data(data_list)
    print(len(train_data), len(valid_data), len(test_data))

    # Load configure

    info_dict["epoches"] = epoches
    info_dict["batch_size"] = batch_size
    info_dict["steps_per_epoch"] = len(train_data)//info_dict["batch_size"]
    info_dict["lr"] = lr
    info_dict["notion"] = "输入:单幅图像，WeatherClsCNN:简单，输出:Onehot"

    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    # train_gen = gen_Img_TOnehot(train_data, image_root=obj.image_root, batch_size=info_dict["batch_size"])  # one hot with augementation
    train_gen = gen_Img_TOnehot(data_list, image_root=obj.image_root,
                                batch_size=info_dict["batch_size"])  # one hot with augementation
    valid_gen = gen_Img_TOnehot(valid_data, image_root=obj.image_root, batch_size=info_dict["batch_size"])  # one hot
    test_gen = gen_Img_TOnehot(test_data, image_root=obj.image_root, batch_size=info_dict["batch_size"])  # one hot

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # model = bulid_model_v1(loss_weights=[1.0, 0.01, 0.001], lr=0.001, info_dict=info_dict)
    model = build_model_simpleCNN_Img_TOnehot(lr=info_dict["lr"])
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=10, verbose=1)
    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)
    # training model
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=info_dict["steps_per_epoch"],
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")
    model.load_weights(model_save_filename)

    # model = load_model('/home/shiyanshi/sz/Image2Weather/Results/V1.0/model.h5', custom_objects={'Scale': Scale, 'root_mean_squared_error': root_mean_squared_error})
    # model.summary()

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # x, y = test_gen.__next__()
    #
    # x1, y1 = x.copy(), y.copy()
    #
    # evaluate = model.evaluate(x1, y1)
    # loaded_test_data_x, loaded_test_data_y = all_Img_TOnehot(test_data, image_root=obj.image_root)  # 非迭代方式
    loaded_test_data_x, loaded_test_data_y = all_Img_TOnehot(data_list_2, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    # predict = np.argmax(predict, axis=-1)-30
    predict = get_Onehot_value(predict)
    predict = [np.squeeze(predict)]
    print(predict)
    print(get_Onehot_value(loaded_test_data_y))

    truth = get_Onehot_value(loaded_test_data_y)

    data_dict = {
        'truth': truth,
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict[0]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict[0]), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=truth, Y_predict=predict[0], tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    pass


def pred_by_simpleCNN_TLDE(cam_id=0):
    """

    Predict the temperature using simpleCNN model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    # images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    # labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    images_root = r"D:\CVProject\Data\xiyan_temperature\images"
    labels_root = r"D:\CVProject\Data\xiyan_temperature\metadata"
    t_start = time.clock()
    # Load Data
    info_dict = get_configure_parameters_for_temp_pred()
    part_string = tp.get_part_string(info_dict=info_dict)
    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)
    print(len(train_data), len(valid_data), len(test_data))

    # Load configure

    info_dict["epoches"] = 100
    info_dict["batch_size"] = 16
    info_dict["steps_per_epoch"] = len(train_data)//info_dict["batch_size"]
    # info_dict["steps_per_epoch"] = 500
    # info_dict["steps_per_epoch"] = 50
    info_dict["lr"] = 0.001 # 0.001
    info_dict["sigma"] = 3
    info_dict["notion"] = "输入:单幅图像，WeatherClsCNN:简单，输出:Onehot"

    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    train_gen = gen_Img_TLDE(train_data, image_root=obj.image_root, batch_size=info_dict["batch_size"], sigma=info_dict["sigma"] )  # one hot with augementation
    valid_gen = gen_Img_TLDE(valid_data, image_root=obj.image_root, batch_size=info_dict["batch_size"], sigma=info_dict["sigma"] )  # one hot
    test_gen = gen_Img_TLDE(test_data, image_root=obj.image_root, batch_size=info_dict["batch_size"], sigma=info_dict["sigma"] )  # one hot

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # Onehot-model have same architecture with model based on LDE, thus we use Onehot version here.
    model = build_model_simpleCNN_Img_TOnehot(lr=info_dict["lr"])
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=10, verbose=1)

    # save the best model in training process
    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)
    # training model
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=info_dict["steps_per_epoch"],
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               # callbacks=[obj_callback, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")
    model.load_weights(model_save_filename)

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # x, y = test_gen.__next__()
    #
    # x1, y1 = x.copy(), y.copy()
    #
    # evaluate = model.evaluate(x1, y1)
    loaded_test_data_x, loaded_test_data_y = all_Img_TOnehot(test_data, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    # predict = np.argmax(predict, axis=-1)-30
    predict = get_Onehot_value(predict)
    predict = [np.squeeze(predict)]
    print(predict)
    print(get_Onehot_value(loaded_test_data_y))

    truth = get_Onehot_value(loaded_test_data_y)

    data_dict = {
        'truth': truth,
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict[0]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict[0]), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=truth, Y_predict=predict[0], tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    pass



def pred_by_simpleCNN_TLDE_xiyan(cam_id=0):
    """

    Predict the temperature using simpleCNN model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    # images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    # labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    images_root = r"D:\CVProject\Data\xiyan_temperature\images"
    labels_root = r"D:\CVProject\Data\xiyan_temperature\metadata"
    t_start = time.clock()
    # Load Data
    info_dict = get_configure_parameters_for_temp_pred()
    part_string = tp.get_part_string(info_dict=info_dict)
    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)
    print(len(train_data), len(valid_data), len(test_data))

    # Load configure

    info_dict["epoches"] = 100
    info_dict["batch_size"] = 16
    info_dict["steps_per_epoch"] = len(train_data)//info_dict["batch_size"]
    # info_dict["steps_per_epoch"] = 500
    # info_dict["steps_per_epoch"] = 50
    info_dict["lr"] = 0.001 # 0.001
    info_dict["sigma"] = 3
    info_dict["notion"] = "输入:单幅图像，WeatherClsCNN:简单，输出:Onehot"

    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    train_gen = gen_Img_TLDE(train_data, image_root=obj.image_root, batch_size=info_dict["batch_size"], sigma=info_dict["sigma"] )  # one hot with augementation
    valid_gen = gen_Img_TLDE(valid_data, image_root=obj.image_root, batch_size=info_dict["batch_size"], sigma=info_dict["sigma"] )  # one hot
    test_gen = gen_Img_TLDE(test_data, image_root=obj.image_root, batch_size=info_dict["batch_size"], sigma=info_dict["sigma"] )  # one hot

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # Onehot-model have same architecture with model based on LDE, thus we use Onehot version here.
    K.clear_session()
    model = build_model_simpleCNN_Img_TOnehot(lr=info_dict["lr"])
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    # reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=10, verbose=1)
    #
    # # save the best model in training process
    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    # checkpoint = ModelCheckpoint(model_save_filename,
    #                              monitor='val_LDE_acc',     # 还真的可以...
    #                              # monitor='val_loss',  # 还真的可以...
    #                              verbose=1,
    #                              save_best_only=True,
    #                              mode='max',
    #                              # mode='min',
    #                              period=1)
    # # training model
    # hist = model.fit_generator(train_gen,
    #                            steps_per_epoch=info_dict["steps_per_epoch"],
    #                            epochs=info_dict["epoches"],
    #                            validation_data=valid_gen,
    #                            validation_steps=1,
    #                            verbose=1,
    #                            # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
    #                            # callbacks=[obj_callback, reduce_lr_on_plateu]
    #                            callbacks=[checkpoint, reduce_lr_on_plateu],
    #                            )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")
    # K.clear_session()
    model.load_weights(model_save_filename)




    froot = r"D:\CVProject\Data\xiyan_temperature\images\158"
    fname = "di000000000000000158_20161106160036651_4401_.png"
    loaded_test_data_x,  loaded_test_data_y = get_single_img_TOnehot_by_path(img_path=os.path.join(froot, fname))
    # loaded_test_data_x, loaded_test_data_y = all_Img_TOnehot(test_data[:1], image_root=obj.image_root)  # 非迭代方式
    # print(len(loaded_test_data_x))        # 184


    # ==== feature_map ====
    import cv2

    # model_1 = Model(inputs=model.input, outputs=model.get_layer("Pool_1").output)
    # model_2 = Model(inputs=model.input, outputs=model.get_layer("Pool_2").output)
    # model_3 = Model(inputs=model.input, outputs=model.get_layer("Pool_3").output)
    # model_4 = Model(inputs=model.input, outputs=model.get_layer("Pool_4").output)
    # model_5 = Model(inputs=model.input, outputs=model.get_layer("Pool_5").output)

    # graph = tf.get_default_graph()

    filename = os.path.join(froot, fname)

    predict_final = model.predict(loaded_test_data_x)

    # img = cv2.imread(filename)
    #
    # img = cv2.resize(img, (240, 320))
    #
    # input_img = np.expand_dims(img, axis=0)
    input_img = loaded_test_data_x


    preds = model.predict(input_img)
    # print(preds)

    class_idx = np.argmax(preds[0])

    class_output = model.output[:, class_idx]
    # 获取最后一层
    last_conv_layer = model.get_layer("Pool_1")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.sum(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_img])

    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # relu激活。
    heatmap = heatmap / (np.max(heatmap) + 1e-10)

    img = cv2.imread(os.path.join(froot, fname))
    img = cv2.resize(img, (240, 320))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imshow('Grad-cam', superimposed_img)
    cv2.waitKey()

    return
    # Record the loss and accuracy
    # df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # x, y = test_gen.__next__()
    #

    # x1, y1 = x.copy(), y.copy()
    #
    # evaluate = model.evaluate(x1, y1)
    loaded_test_data_x, loaded_test_data_y = all_Img_TOnehot(test_data, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    # predict = np.argmax(predict, axis=-1)-30
    predict = get_Onehot_value(predict)
    predict = [np.squeeze(predict)]
    print(predict)
    print(get_Onehot_value(loaded_test_data_y))

    truth = get_Onehot_value(loaded_test_data_y)

    data_dict = {
        'truth': truth,
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict[0]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict[0]), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=truth, Y_predict=predict[0], tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    pass



def pred_by_simpleCNNMLP_TValue(cam_id=0,lr=0.0001):
    """
    收敛：2.47
    Predict the temperature using simpleCNN+MLP model
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"


    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    t_start = time.clock()
    # Load Data
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)

    random.shuffle(data_list)
    data_list_2 = obj.get_source_data_for_cnn(selected_cam=cam_id)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    train_data, valid_data, test_data = split_data(data_list)


    print(len(train_data), len(valid_data), len(test_data))
    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    # train_gen = gen_ImgTimeOnehot_TValue(train_data, image_root=obj.image_root, batch_size=4)  # one hot with augementation
    train_gen = gen_ImgTimeOnehot_TValue(data_list, image_root=obj.image_root,
                                         batch_size=4)  # one hot with augementation
    valid_gen = gen_ImgTimeOnehot_TValue(valid_data, image_root=obj.image_root, batch_size=4)  # one hot
    test_gen = gen_ImgTimeOnehot_TValue(test_data, image_root=obj.image_root, batch_size=4)  # one hot


    # Load configure
    info_dict = get_configure_parameters_for_temp_pred()
    info_dict["epoches"] = 60
    info_dict["lr"] = lr
    part_string = tp.get_part_string(info_dict=info_dict)

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # model = bulid_model_v1(loss_weights=[1.0, 0.01, 0.001], lr=0.001, info_dict=info_dict)
    model = build_model_simpleCNNMLP_ImgTimeOnehot_TValue(lr=info_dict["lr"])
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=10, verbose=1)

    # save the best model in training process
    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 # monitor='val_r_squared_score',     # 还真的可以...
                                 monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 # mode='max',
                                 mode='min',
                                 period=1)
    # training model
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=500,
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")
    model.load_weights(model_save_filename)

    # model = load_model('/home/shiyanshi/sz/Image2Weather/Results/V1.0/model.h5', custom_objects={'Scale': Scale, 'root_mean_squared_error': root_mean_squared_error})
    # model.summary()

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # x, y = test_gen.__next__()
    #
    # x1, y1 = x.copy(), y.copy()
    #
    # evaluate = model.evaluate(x1, y1)
    # loaded_test_data_x, loaded_test_data_y = all_ImgTimeOnehot_TValue(test_data, image_root=obj.image_root)  # 非迭代方式
    loaded_test_data_x, loaded_test_data_y = all_ImgTimeOnehot_TValue(data_list_2, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    predict = [np.squeeze(predict)]
    print(predict)
    print(loaded_test_data_y)

    data_dict = {
        'truth': loaded_test_data_y[0],
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict, tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNMLP_TLDE(cam_id=0, lr=0.0001):
    """
    收敛：err：2.14
    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)


    t_start = time.clock()
    # Load Data
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)
    random.shuffle(data_list)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    train_data, valid_data, test_data = split_data(data_list)
    print(len(train_data), len(valid_data), len(test_data))

    info_dict = get_configure_parameters_for_temp_pred()
    info_dict["epoches"] = 60
    info_dict["lr"] = lr
    info_dict["batch_size"] = 16
    info_dict["steps_per_epoch"] = len(train_data)//info_dict["batch_size"]
    # info_dict["steps_per_epoch"] = 500
    info_dict["sigma"] = 3

    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    # train_x, train_y = all_ImgTimeOnehot_TLDE(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    # valid_x, valid_y = all_ImgTimeOnehot_TLDE(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot
    # test_x, test_y = all_ImgTimeOnehot_TLDE(test_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot
    train_gen = gen_ImgTimeOnehot_TOnehot(train_data, image_root=obj.image_root, batch_size=4)  # one hot with augementation
    valid_gen = gen_ImgTimeOnehot_TOnehot(valid_data, image_root=obj.image_root, batch_size=4)  # one hot
    test_gen = gen_ImgTimeOnehot_TOnehot(test_data, image_root=obj.image_root, batch_size=4)  # one hot
    # Load configure

    part_string = tp.get_part_string(info_dict=info_dict)

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # model = bulid_model_v1(loss_weights=[1.0, 0.01, 0.001], lr=0.001, info_dict=info_dict)
    model = build_model_simpleCNNMLP_ImgTimeOnehot_TOnehot(lr=info_dict["lr"])
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=5, verbose=1)

    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)
    # training model
    # hist = model.fit(train_x, train_y,
    #                            steps_per_epoch=info_dict["steps_per_epoch"] ,
    #                            epochs=info_dict["epoches"],
    #                            validation_data=(valid_x, valid_y),
    #                            validation_steps=1,
    #                            verbose=1,
    #                            # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
    #                            # callbacks=[checkpoint, reduce_lr_on_plateu],
    #                            )
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=info_dict["steps_per_epoch"],    # 500
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")
    model.load_weights(model_save_filename)

    # model = load_model('/home/shiyanshi/sz/Image2Weather/Results/V1.0/model.h5', custom_objects={'Scale': Scale, 'root_mean_squared_error': root_mean_squared_error})
    # model.summary()

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    #
    # evaluate = model.evaluate(x1, y1)
    loaded_test_data_x, loaded_test_data_y = all_ImgTimeOnehot_TValue(test_data, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    predict = [get_LDE_value(np.squeeze(predict))]
    print(predict)
    print(loaded_test_data_y)

    data_dict = {
        'truth': loaded_test_data_y[0],
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict, tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNMLP_TimeValue_TOnehot(cam_id=0):
    """
    收敛：2.67
    Predict the temperature using simpleCNN+MLP model, the input time data are single-valued and 
        the labeled temperature data are in Onehot code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"


    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    t_start = time.clock()
    # Load Data
    data_list = obj.get_source_data_for_cnn(selected_cam=cam_id)


    data_list_2 = obj.get_source_data_for_cnn(selected_cam=cam_id)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    train_data, valid_data, test_data = split_data(data_list)
    random.shuffle(train_data)
    random.shuffle(valid_data)


    print(len(train_data), len(valid_data), len(test_data))

    info_dict = get_configure_parameters_for_temp_pred()
    info_dict["batch_size"] = 16
    # train_gen = data_generator_v3(train_data, batch_size=32)  # one hot with augementation
    train_gen = gen_ImgTimeValue_TOnehot(train_data, image_root=obj.image_root, batch_size=info_dict["batch_size"])  # one hot with augementation
    valid_gen = gen_ImgTimeValue_TOnehot(valid_data, image_root=obj.image_root, batch_size=info_dict["batch_size"])  # one hot
    test_gen = gen_ImgTimeValue_TOnehot(test_data, image_root=obj.image_root, batch_size=info_dict["batch_size"])  # one hot


    # Load configure
    info_dict["epoches"] = 60
    info_dict["lr"] = 0.001
    info_dict["steps_per_epoch"] = len(train_data)//info_dict["batch_size"]
    part_string = tp.get_part_string(info_dict=info_dict)

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    # model = bulid_model_v1(loss_weights=[1.0, 0.01, 0.001], lr=0.001, info_dict=info_dict)
    # model = build_model_simpleCNNMLP_TValue(lr=0.0001)
    model = build_model_simpleCNNMLP_ImgTimeOnehot_TOnehot(lr=info_dict["lr"])
    model.summary()
    # print("model_loss_weights", model.loss_weights)

    # set training strategy
    # obj_callback = LossWeightsScheduler(model.loss_weights['pred_class'], model.loss_weights['pred_temp'],
    #                                     model.loss_weights['pred_hum'], factor=1)

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001)
    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=10, verbose=1)

    # save the best model in training process
    model_save_filename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 # monitor='val_r_squared_score',     # 还真的可以...
                                 monitor='val_LDE_acc',  # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)
    # training model
    hist = model.fit_generator(train_gen,
                               steps_per_epoch=info_dict["steps_per_epoch"],
                               epochs=info_dict["epoches"],
                               validation_data=valid_gen,
                               validation_steps=1,
                               verbose=1,
                               # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                               callbacks=[checkpoint, reduce_lr_on_plateu],
                               )

    # save model
    # model.save(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_model.h5")
    model.load_weights(model_save_filename)

    # model = load_model('/home/shiyanshi/sz/Image2Weather/Results/V1.0/model.h5', custom_objects={'Scale': Scale, 'root_mean_squared_error': root_mean_squared_error})
    # model.summary()

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # x, y = test_gen.__next__()
    #
    # x1, y1 = x.copy(), y.copy()
    #
    # evaluate = model.evaluate(x1, y1)
    loaded_test_data_x, loaded_test_data_y = all_ImgTimeValue_TOnehot(test_data, image_root=obj.image_root)  # 非迭代方式
    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    predict = get_Onehot_value(predict)
    predict = [np.squeeze(predict)]
    print("predict:", predict)
    truth = get_Onehot_value(loaded_test_data_y)
    print("truth:", truth)



    data_dict = {
        'truth': truth[0],
        'predict': predict[0]
    }

    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/" + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # y_true = np.argmax(y1[0], axis=1)
    # y_pred = np.argmax(predict[0], axis=1)

    # Display
    # print("precise:", len(np.where(y_true == y_pred)[0]) / len(y_true))
    # print("temp_err:", np.sqrt(np.mean(np.square(y1[0] - np.squeeze(np.array(predict[0]))), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(truth - predict), axis=-1)), file=doc)
    # print("hum_err:", np.sqrt(np.mean(np.square(y1[2] - np.squeeze(np.array(predict[2]))), axis=-1)))

    # test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id)
    # test_performance.get_score_predict(y_true, y_pred)
    # test_performance.get_score_value(y1[1], predict[1], categray_actual=y_true, tag="temperature")
    test_performance.get_score_value(Y_actual=truth, Y_predict=predict, tag="temperature")
    # test_performance.get_score_value(y1[2], predict[2], categray_actual=y_true, tag="humidity")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNLSTM_TValue(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 100  # 100
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.0001  # 0.0001
    info_dict["sigma"] = 4
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data) // info_dict["batch_size"]

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    train_gen = all_ImgSequence_TValueSequence(train_data, image_root=obj.image_root,
                                             sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TValueSequence(valid_data, image_root=obj.image_root,
                                             sigma=info_dict["sigma"])  # one hot with augementation

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/step_{}_value_training_log_".format(step) + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TValue(lr=info_dict["lr"], step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/LSTM_value_HOUR_{0}_STEP_{1}_".format(chosen_hour, step) + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_r_squared_score',  # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=train_gen[0], y=train_gen[1],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=(valid_gen[0], valid_gen[1]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )



    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_value_HOUR_{0}_STEP_{1}_".format(chosen_hour,step) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_y = all_ImgSequence_TValueSequence(test_data, image_root=obj.image_root)  # 非迭代方式

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    predict = [np.squeeze(predict)]

    predict = np.squeeze(np.array([elem[:, -1] for elem in predict]))
    print("predict:", predict)
    print("predict shape：", np.array(predict).shape)
    loaded_test_data_y = [elem[-1] for elem in loaded_test_data_y]
    print("truth:", loaded_test_data_y)
    print("truth shape:", np.array(loaded_test_data_y).shape)

    data_dict = {
        'truth': np.array(loaded_test_data_y),
        'predict': predict
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_value_HOUR_{0}_STEP_{1}_".format(chosen_hour, step) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note="LSTM_Value_HOUR_{0}_STEP_{1}".format(chosen_hour, step))
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict, tag="temperature_value_")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNLSTM_TValue_v2(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 100  # 100
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.0001  # 0.0001
    info_dict["sigma"] = 4
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data) // info_dict["batch_size"]

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    train_gen = all_ImgSequence_TValueSequence_v2(train_data, image_root=obj.image_root,
                                             sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TValueSequence_v2(valid_data, image_root=obj.image_root,
                                             sigma=info_dict["sigma"])  # one hot with augementation


    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/step_{}_value_training_log_".format(step) + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TValue_v2(lr=info_dict["lr"], step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/LSTM_value_HOUR_{0}_STEP_{1}_".format(chosen_hour, step) + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_r_squared_score',  # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=[train_gen[0], train_gen[1]], y=train_gen[2],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=([valid_gen[0], valid_gen[1]], valid_gen[2]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )


    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)


    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_value_HOUR_{0}_STEP_{1}_".format(chosen_hour,step) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_t, loaded_test_data_y = all_ImgSequence_TValueSequence_v2(test_data, image_root=obj.image_root)  # 非迭代方式
    print(loaded_test_data_x)

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))


    loss, accuracy = model.evaluate([loaded_test_data_x, loaded_test_data_t], loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict([loaded_test_data_x, loaded_test_data_t])
    predict = [np.squeeze(predict)]


    predict = np.squeeze(np.array([elem[:, -1] for elem in predict]))
    print("predict:", predict)
    print("predict shape：", np.array(predict).shape)
    loaded_test_data_y = [elem[-1] for elem in loaded_test_data_y]
    print("truth:", loaded_test_data_y)
    print("truth shape:", np.array(loaded_test_data_y).shape)

    data_dict = {
        'truth': np.array(loaded_test_data_y),
        'predict': predict
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_value_HOUR_{0}_STEP_{1}_".format(chosen_hour, step) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - np.array(predict)), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note="LSTM_Value_HOUR_{0}_STEP_{1}".format(chosen_hour, step))
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict, tag="temperature_value_")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)
    data_list_2 = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])

    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 100  # 100/60
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.0001    # 0.0001
    info_dict["sigma"] = 3.5      # 3.5
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data)//info_dict["batch_size"]

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    train_gen = all_ImgSequence_TOnehotSequence_v2(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TOnehotSequence_v2(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/onehot_training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TLDE_v2(lr=info_dict["lr"] , step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/onehot_" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=[train_gen[0], train_gen[1]], y=train_gen[2],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=([valid_gen[0], valid_gen[1]], valid_gen[2]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )

    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_onehot_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_t, loaded_test_data_y = all_ImgSequence_TOnehotSequence_v2(data_list_2, image_root=obj.image_root)  # 非迭代方式

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate([loaded_test_data_x, loaded_test_data_t], loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict([loaded_test_data_x, loaded_test_data_t])
    #
    print(predict)
    print(loaded_test_data_y)

    predict = np.squeeze(predict)
    predict = get_Onehot_value(predict)
    loaded_test_data_y = get_Onehot_value(loaded_test_data_y)

    print("predict:", predict)
    print("truth:", loaded_test_data_y)
    data_dict = {
        'truth': loaded_test_data_y[:, -1],
        'predict': predict[:, -1]
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_onehot_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note = "LSTM_Onehot_HOUR_{}_".format(chosen_hour))
    test_performance.get_score_value(Y_actual=loaded_test_data_y[:, -1], Y_predict=predict[:, -1], tag="temperature_onehot_")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 60   # 100/60
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.0001    # 0.0001
    info_dict["sigma"] = 3.5      # 3.5
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data)//info_dict["batch_size"]

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    train_gen = all_ImgSequence_TOnehotSequence(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TOnehotSequence(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/onehot_training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TLDE(lr=info_dict["lr"] , step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/onehot_" + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=train_gen[0], y=train_gen[1],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=(valid_gen[0], valid_gen[1]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )

    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_onehot_HOUR_{}_STEP_{}".format(chosen_hour,step) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_y = all_ImgSequence_TOnehotSequence(test_data, image_root=obj.image_root)  # 非迭代方式

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    #
    print(predict)
    print(loaded_test_data_y)

    predict = np.squeeze(predict)
    predict = get_Onehot_value(predict)
    loaded_test_data_y = get_Onehot_value(loaded_test_data_y)

    print("predict:", predict)
    print("truth:", loaded_test_data_y)
    data_dict = {
        'truth': loaded_test_data_y[:, -1],
        'predict': predict[:, -1]
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_onehot_HOUR_{}_STEP_{}".format(chosen_hour, step) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note = "LSTM_Onehot_HOUR_{}_STEP_{}".format(chosen_hour, step))
    test_performance.get_score_value(Y_actual=loaded_test_data_y[:, -1], Y_predict=predict[:, -1], tag="temperature_onehot_")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    data_list_2 = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"]+1)
    random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 60    # 100/60
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.0001    # 0.0001
    info_dict["sigma"] = 3.5
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data)//info_dict["batch_size"]

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    # train_gen = all_ImgSequence_TLDESequence(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    # valid_gen = all_ImgSequence_TLDESequence(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    train_gen = all_ImgSequence_TLDESequence(data_list, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TLDESequence(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TLDE(lr=info_dict["lr"] , step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=train_gen[0], y=train_gen[1],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=(valid_gen[0], valid_gen[1]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )

    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_y = all_ImgSequence_TLDESequence(data_list_2, image_root=obj.image_root)  # 非迭代方式

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate(loaded_test_data_x, loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict(loaded_test_data_x)
    #
    print(predict)
    print(loaded_test_data_y)

    predict = np.squeeze(predict)
    predict = get_LDE_value(predict)
    loaded_test_data_y = get_LDE_value(loaded_test_data_y)

    print("predict:", predict)
    print("truth:", loaded_test_data_y)
    data_dict = {
        'truth': loaded_test_data_y[:, -1],
        'predict': predict[:, -1]
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note="LSTM_LDE_HOUR_{}_".format(chosen_hour))
    test_performance.get_score_value(Y_actual=loaded_test_data_y[:, -1], Y_predict=predict[:, -1], tag="temperature")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNLSTM_TLDE_v4(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    data_list_2 = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"]+1)
    # random.shuffle(data_list)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 100    # 100/60
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.0001    # 0.0001
    info_dict["sigma"] = 3.5
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data)//info_dict["batch_size"]

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    # train_gen = all_ImgSequence_TLDESequence(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    # valid_gen = all_ImgSequence_TLDESequence(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    train_gen = all_ImgSequence_TLDESequence_v2(data_list, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TLDESequence_v2(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TLDE_v4(lr=info_dict["lr"] , step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=[train_gen[0], train_gen[1]], y=train_gen[2],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=([valid_gen[0], valid_gen[1]], valid_gen[2]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )

    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_t, loaded_test_data_y = all_ImgSequence_TLDESequence_v2(data_list_2, image_root=obj.image_root)  # 非迭代方式

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate([loaded_test_data_x, loaded_test_data_t], loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict([loaded_test_data_x, loaded_test_data_t])
    #
    print(predict)
    print(loaded_test_data_y)

    predict = np.squeeze(predict)
    predict = get_LDE_value(predict)
    loaded_test_data_y = get_LDE_value(loaded_test_data_y)

    print("predict:", predict)
    print("truth:", loaded_test_data_y)
    data_dict = {
        'truth': loaded_test_data_y[:, -1],
        'predict': predict[:, -1]
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note="LSTM_LDE_HOUR_{}_".format(chosen_hour))
    test_performance.get_score_value(Y_actual=loaded_test_data_y[:, -1], Y_predict=predict[:, -1], tag="temperature")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))

# NOT PASS, 不收敛
def pred_by_simpleCNNLSTM_TLDE_v2(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    random.shuffle(data_list)
    data_list_2 = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    train_data, valid_data, test_data = split_data(data_list)


    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 100    # 100/60
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.0001    # 0.0001
    info_dict["sigma"] = 3.5
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data)//info_dict["batch_size"]
    # steps_per_epoch = 100

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    # train_gen = all_ImgSequence_TLDESequence(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    # valid_gen = all_ImgSequence_TLDESequence(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    train_gen = all_ImgSequence_TLDESequence_v2(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TLDESequence_v2(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TLDE_v2(lr=info_dict["lr"] , step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=[train_gen[0], train_gen[1]], y=train_gen[2],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=([valid_gen[0], valid_gen[1]], valid_gen[2]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )

    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_t, loaded_test_data_y = all_ImgSequence_TLDESequence_v2(test_data, image_root=obj.image_root)  # 非迭代方式

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate([loaded_test_data_x, loaded_test_data_t], loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict([loaded_test_data_x, loaded_test_data_t])
    #
    print(predict)
    print(loaded_test_data_y)

    predict = np.squeeze(predict)
    predict = get_LDE_value(predict)
    loaded_test_data_y = get_LDE_value(loaded_test_data_y)

    print("predict:", predict)
    print("truth:", loaded_test_data_y)
    data_dict = {
        'truth': loaded_test_data_y[:, -1],
        'predict': predict[:, -1]
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y[:, -1] - predict[:, -1]), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note="LSTM_LDE_HOUR_{}_".format(chosen_hour))
    test_performance.get_score_value(Y_actual=loaded_test_data_y[:, -1], Y_predict=predict[:, -1], tag="temperature")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))


def pred_by_simpleCNNLSTM_TLDE_v3(cam_id=0, step=0, chosen_hour=11):
    """

    Predict the temperature using simpleCNN+MLP model, the labeled temperature data are in LDE code
    :param cam_id: id of camera(scene)
    :return:
    """
    import models.TempPredCNN.TempDataGen as TempDataGen
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)

    # Load configuration
    info_dict = get_configure_parameters_for_temp_pred()

    info_dict["step"] = step
    info_dict["chosen_hour"] = chosen_hour

    t_start = time.clock()

    # source data
    data_list = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    random.shuffle(data_list)
    data_list_2 = obj.get_chosen_data_from_one_cam(obj.source_data[obj.cam_list[cam_id]], step=info_dict["step"],
                                                 chosen_hour=info_dict["chosen_hour"])
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    # configure for this
    info_dict["epoches"] = 100    # 100/60
    info_dict["batch_size"] = 8
    info_dict["lr"] = 0.01    # 0.0001
    info_dict["sigma"] = 3.5
    info_dict["notion"] = "multi-images, WeatherClsCNN+LSTM, temp: LDE, lr=0.001, epoches=100, steps_per_epoch=20"
    steps_per_epoch = len(train_data)//info_dict["batch_size"]
    # steps_per_epoch = 100

    print("steps_per_epoch:", steps_per_epoch)
    # steps_per_epoch = 50
    validation_steps = 1

    part_string = tp.get_part_string(info_dict=info_dict)

    # train_gen = all_ImgSequence_TLDESequence(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    # valid_gen = all_ImgSequence_TLDESequence(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    train_gen = all_ImgSequence_TLDESequence_v3(train_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation
    valid_gen = all_ImgSequence_TLDESequence_v3(valid_data, image_root=obj.image_root, sigma=info_dict["sigma"])  # one hot with augementation

    # set recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + str(cam_id) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    print("model parameters:")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    for elem in info_dict:
        print(elem, ":", info_dict[elem])

    # Load network model
    model = build_model_simpleCNNLSTM_TLDE_v3(lr=info_dict["lr"] , step=info_dict["step"])
    model.summary()

    # K.utils.plot_model(model, r'E:\Project\CV\WeatherPred\Results\DL\model.png', show_shapes=True)

    reduce_lr_on_plateu = ReduceLROnPlateau(monitor="val_loss", factor=0.5, mode='auto', patience=20, verbose=1)
    # training model

    model_save_filename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_model.h5"
    checkpoint = ModelCheckpoint(model_save_filename,
                                 monitor='val_LDE_acc',     # 还真的可以...
                                 # monitor='val_loss',  # 还真的可以...
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 # mode='min',
                                 period=1)

    hist = model.fit(x=[train_gen[0], train_gen[1]], y=train_gen[2],
                     steps_per_epoch=steps_per_epoch,
                     epochs=info_dict["epoches"],
                     validation_data=([valid_gen[0], valid_gen[1]], valid_gen[2]),
                     # validation_steps=validation_steps,
                     verbose=1,
                     # callbacks=[obj_callback, early_stopping, reduce_lr_on_plateu]
                     callbacks=[checkpoint, reduce_lr_on_plateu],
                     )

    # Record the loss and accuracy
    print(hist.history)

    model.load_weights(model_save_filename)
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # test
    loaded_test_data_x, loaded_test_data_t, loaded_test_data_y = all_ImgSequence_TLDESequence_v3(train_data, image_root=obj.image_root)  # 非迭代方式

    print("x1:", len(loaded_test_data_x), "y1:", len(loaded_test_data_y))
    loss, accuracy = model.evaluate([loaded_test_data_x, loaded_test_data_t], loaded_test_data_y)
    print("loss:", loss, "accuracy:", accuracy)
    print("loss:", loss, "accuracy:", accuracy, file=doc)
    predict = model.predict([loaded_test_data_x, loaded_test_data_t])
    #
    print(predict)
    print(loaded_test_data_y)

    predict = np.squeeze(predict)
    predict = get_LDE_value(predict)
    loaded_test_data_y = get_LDE_value(loaded_test_data_y)

    print("predict:", predict)
    print("truth:", loaded_test_data_y)
    data_dict = {
        'truth': loaded_test_data_y,
        'predict': predict
    }
    #
    pd_pdoc = pd.DataFrame.from_dict(data_dict)
    pd_pdoc.to_csv(info_dict["savepath"] + "/LSTM_LDE_HOUR_{}_".format(chosen_hour) + str(cam_id) + "_" + part_string + "_truth_predcit.csv")

    # Display
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - predict), axis=-1)))
    print("temp_err:", np.sqrt(np.mean(np.square(loaded_test_data_y - predict), axis=-1)), file=doc)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict, cam_id=cam_id, Note="LSTM_LDE_HOUR_{}_".format(chosen_hour))
    test_performance.get_score_value(Y_actual=loaded_test_data_y, Y_predict=predict, tag="temperature")

    t_end = time.clock()
    duration = t_end - t_start
    hours = duration // 3600
    sh_hours = duration % 3600
    mins = sh_hours // 60
    seconds = sh_hours % 60
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds))
    print("time of duration:{}h {}m {}s".format(hours, mins, seconds), file=doc)
    test_performance.record_sth("time of duration:{}h {}m {}s".format(hours, mins, seconds))



def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def LDE_loss(y_true, y_pred):
    """ LDE loss"""
    return K.sum(K.square(y_true - y_pred))


def LDE_acc(y_true, y_pred):
    """ LDE acc"""
    value_true = K.cast(get_LDE_value(y_true), tf.float32)
    value_pred = K.cast(get_LDE_value(y_pred), tf.float32)
    SS_res = K.sum(K.square(value_true - value_pred))
    SS_tot = K.sum(K.square(value_true - K.mean(value_true)))
    return (1.0 - SS_res / (SS_tot + K.epsilon()))


def get_LDE_code(t, minv=-30, maxv=50, sigma=3.5):
    """
    Translate a temperature value to a LDE code
    :param t: temperature value
    :param minv: minimum value of temperature range
    :param maxv: maximum value of temperature range
    :param sigma: distribution factor
    :return: temperature LDE code
    """
    value_range = maxv-minv
    t_idx = t-minv
    value_code = np.zeros(shape=(value_range,))
    for idx, _ in enumerate(value_code):
        value_code[idx] = get_gauss_value(t_idx-idx, sigma=sigma)
    return value_code


def get_LDE_value(c, minv=-30, maxv=50):
    """
    Translate a LDE code to a temperature value
    :param c: temperature LDE code
    :param minv: minimum value of temperature range
    :param maxv: maximum value of temperature range
    :return: temperature value
    """
    max_index = K.argmax(c, axis=-1)    # Note: axis in Keras and numpy are different!!!
    return max_index+minv


def get_Onehot_code(t, minv=-30, maxv=50):
    """
    Translate a temperature value to an Onehot code
    :param t: temperature value
    :param minv: minimum value of temperature range
    :param maxv: maximum value of temperature range
    :return: temperature Onehot code
    """
    value_range = maxv-minv
    t_idx = t-minv
    value_code = np.zeros(shape=(value_range,))
    value_code[t_idx] = 1
    return value_code


def get_Onehot_value(c, minv=-30, maxv=50):
    """
    Translate a Onehot code to a temperature value
    :param c: temperature LDE code
    :param minv: minimum value of temperature range
    :param maxv: maximum value of temperature range
    :return: temperature value
    """
    max_index = K.argmax(c, axis=-1)    # Note: axis in Keras and numpy are different!!!
    return max_index+minv


def get_gauss_value(val, sigma=3.5):
    """Get Gaussian value"""
    rst = (1/(sigma*np.sqrt(np.pi)))*np.exp(-1*val*val/(2*sigma*sigma))
    return rst if rst > 1e-6 else 0


def img_test():
    import models.TempPredCNN.TempDataGen as TempDataGen
    from tensorflow.python.keras.applications import imagenet_utils
    images_root = r"D:\CVProject\CBAM-keras-master\weather_data\dataset2"
    labels_root = r"D:\CVProject\CBAM-keras-master\weather_data\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)
    data_list = obj.get_source_data_for_cnn(selected_cam=0)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    print(train_data[1])
    img = image.load_img(os.path.join(images_root, train_data[1][4] + '/' + train_data[1][2]), target_size=(224, 224))
    if img:
        input_image = image.img_to_array(img)
        # input_image = preprocess_input(input_image)
        imagenet_utils.preprocess_input(input_image, data_format=None, mode='torch')
        print(input_image)

def get_confusion_matrix_from_csv(filename=None):
    root = r"D:\CVProject\CBAM-keras-master\temp_results\_NEW_2021\ZCURVE"
    filename = filename
    filepath = os.path.join(root, filename)
    import pandas as pd
    csv_data = pd.read_csv(filepath)
    truth_data = csv_data['truth']
    predict_data = csv_data['predict']
    # print(csv_data['truth'], csv_data['predict'])

    maxv = np.max(np.array(truth_data))
    minv = np.min(np.array(truth_data))

    classes = np.unique(np.array(truth_data[-650:]))

    int_predict_data = []
    for elem in predict_data:
        int_predict_data.append(round(elem))
        # int_predict_data.append(int(elem))
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true=truth_data[-650:], y_pred=int_predict_data[-650:])
    print(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    cm_norm = np.around(cm_norm, decimals=2)
    plt.figure()
    # plt.switch_backend('agg')
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(cm_norm, cmap=plt.cm.Blues)
    # plt.imshow(matrix, cmap=plt.cm.binary)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(classes))
    plt.xticks(indices, classes, fontsize=5)
    plt.yticks(indices, classes, fontsize=5)

    plt.colorbar()

    plt.xlabel('Prediction')
    plt.ylabel('GroundTruth')
    plt.title('ConfusionMatrix')
    plt.savefig(os.path.join(r"D:\CVProject\CBAM-keras-master\temp_results\_NEW_2021\ConfusionMatrix", filename.replace('.csv',".png")))
    # plt.show()

    return

# pass
def get_temp_test():
    import models.TempPredCNN.TempDataGen as TempDataGen
    from tensorflow.python.keras.applications import imagenet_utils
    images_root = r"D:\CVProject\Data\xiyan_temperature\images"
    labels_root = r"D:\CVProject\Data\xiyan_temperature\metadata"

    obj = TempDataGen.DataGenerator(images_root=images_root,
                                    label_root=labels_root)
    data_list = obj.get_source_data_for_cnn(selected_cam=0)
    # load_data = all_ImgTimeOnehot_TValue(data_list, image_root=image_root)
    train_data, valid_data, test_data = split_data(data_list)

    print(len(train_data), len(valid_data), len(test_data))
    print(train_data[1])
    img = image.load_img(os.path.join(images_root, train_data[1][4] + '/' + train_data[1][2]), target_size=(224, 224))
    if img:
        input_image = image.img_to_array(img)
        # input_image = preprocess_input(input_image)
        imagenet_utils.preprocess_input(input_image, data_format=None, mode='torch')
        print(input_image)

def get_info_by_imgpath(img_path):
    csv_root = r"D:\CVProject\Data\xiyan_temperature\metadata"
    froot, fname = os.path.split(img_path)
    folder = froot.split("\\")[-1]
    csv_path = os.path.join(csv_root, folder+".csv")
    import pandas as pd
    csv_data = pd.read_csv(csv_path)
    # df[df['a']>30]
    print(list(csv_data[csv_data["Filename"] == fname]["TempM"]))
    return -14
    temp = list(csv_data[csv_data["Filename"] == fname]["TempM"])[0]
    # return temp

def resize_images():
    import glob
    import cv2
    root = r"D:\CVProject\CBAM-keras-master\temp_results\_NEW_2021\SZ\resized\CLASS_ACT\04"
    folders = os.listdir(root)
    for folder in folders:
        filelist_in_folder = glob.glob(os.path.join(os.path.join(root, folder), "*.png"))
        for file in filelist_in_folder:
            img = cv2.imread(file)
            img = cv2.resize(img, (320, 240))
            cv2.imwrite(file, img)
    # filelist = glob.glob(os.path.join(root, "*.png"))
    # for elem in filelist:
    #     img = cv2.imread(elem)
    #     img = cv2.resize(img, (320, 240))
    #     cv2.imwrite(elem, img)
    # img = cv2.imread(filelist[0])
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    pass

    # resize_images()



if __name__ == "__main__":
    # # ===== MLP_TimeValue_TValue =====
    # pred_by_MLP_Tvalue(cam_id=0)    # input: time_location, label: value, model: MLP
    # pred_by_MLP_Tvalue(cam_id=1)
    # pred_by_MLP_Tvalue(cam_id=2)
    # pred_by_MLP_Tvalue(cam_id=3)
    # # pred_by_MLP_Tvalue(cam_id=4)
    # pred_by_MLP_Tvalue(cam_id=5)
    # pred_by_MLP_Tvalue(cam_id=6)
    # pred_by_MLP_Tvalue(cam_id=7)
    # pred_by_MLP_Tvalue(cam_id=8)

    # # ===== VGG =====

    # pred_by_vgg_TValue(cam_id=0, layer=2)            # input: image, label: value, model: vgg
    # pred_by_vgg_TValue(cam_id=3, layer=2)
    # pred_by_vgg_TValue(cam_id=5, layer=2)        #
    # pred_by_vgg_TValue(cam_id=6, layer=2)
    # pred_by_vgg_TValue(cam_id=7, layer=2)
    #
    # pred_by_vgg_TValue(cam_id=0, layer=3)            # input: image, label: value, model: vgg
    # pred_by_vgg_TValue(cam_id=3, layer=3)
    # pred_by_vgg_TValue(cam_id=5, layer=3)        #
    # pred_by_vgg_TValue(cam_id=6, layer=3)
    # pred_by_vgg_TValue(cam_id=7, layer=3)
    # #
    # pred_by_vgg_TValue(cam_id=0, layer=4)            # input: image, label: value, model: vgg
    # pred_by_vgg_TValue(cam_id=3, layer=4)
    # pred_by_vgg_TValue(cam_id=5, layer=4)        #
    # pred_by_vgg_TValue(cam_id=6, layer=4)
    # pred_by_vgg_TValue(cam_id=7, layer=4)
    #
    # pred_by_vgg_TValue(cam_id=0, layer=5)            # input: image, label: value, model: vgg
    # pred_by_vgg_TValue(cam_id=3, layer=5)
    # pred_by_vgg_TValue(cam_id=5, layer=5)        #
    # pred_by_vgg_TValue(cam_id=6, layer=5)
    # pred_by_vgg_TValue(cam_id=7, layer=5)
    #
    # # ==== simpleCNN  ====
    # pred_by_simpleCNN_Tvalue(cam_id=0, epoches=30, lr=1e-4)      # input: image, label: value, model: simpleCNN  # work, err：3.4
    # pred_by_simpleCNN_Tvalue(cam_id=3, epoches=40, lr=1e-4)
    # pred_by_simpleCNN_Tvalue(cam_id=5, epoches=40, lr=1e-3)
    # pred_by_simpleCNN_Tvalue(cam_id=6, epoches=40, lr=1e-3)
    # pred_by_simpleCNN_Tvalue(cam_id=7, epoches=40, lr=1e-3)
    # #
    # pred_by_simpleCNN_TOnehot(cam_id=0, epoches=60, lr=1e-4)
    # pred_by_simpleCNN_TOnehot(cam_id=0, epoches=200, lr=1e-3)
    # pred_by_simpleCNN_TOnehot(cam_id=0, epoches=40, lr=1e-4, batch_size=8)
    # pred_by_simpleCNN_TOnehot(cam_id=3, epoches=40, lr=1e-4)
    # pred_by_simpleCNN_TOnehot(cam_id=5, epoches=200, lr=1e-4)
    # pred_by_simpleCNN_TOnehot(cam_id=6, epoches=200, lr=1e-4)
    # pred_by_simpleCNN_TOnehot(cam_id=7, epoches=200, lr=1e-4)
    #
    # pred_by_simpleCNN_TLDE(cam_id=0)        # input: image, label: LDE, model: simpelCNN    # work, err：2.89




    # pred_by_simpleCNN_TLDE_xiyan(cam_id=0)



    # pred_by_simpleCNN_TLDE(cam_id=3)
    # pred_by_simpleCNN_TLDE(cam_id=5)
    # pred_by_simpleCNN_TLDE(cam_id=6)
    # pred_by_simpleCNN_TLDE(cam_id=7)
    # # ==== simpleCNN+MLP  ====
    # pred_by_simpleCNNMLP_TValue(cam_id=0)   # input: image; time(Onehot), label: value, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TValue(cam_id=3)   # input: image; time(Onehot), label: value, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TValue(cam_id=5)   # input: image; time(Onehot), label: value, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TValue(cam_id=6, lr=0.0001)   # input: image; time(Onehot), label: value, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TValue(cam_id=7, lr=0.0001)   # input: image; time(Onehot), label: value, model: simpleCNN+MLP
    #
    # pred_by_simpleCNNMLP_TLDE(cam_id=0)     # input: image; time(Onehot), label: LDE, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TLDE(cam_id=3)     # input: image; time(Onehot), label: LDE, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TLDE(cam_id=5, lr=0.001)     # input: image; time(Onehot), label: LDE, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TLDE(cam_id=6, lr=0.001)     # input: image; time(Onehot), label: LDE, model: simpleCNN+MLP
    # pred_by_simpleCNNMLP_TLDE(cam_id=7)     # input: image; time(Onehot), label: LDE, model: simpleCNN+MLP

    #

    # pred_by_simpleCNNMLP_TimeValue_TOnehot(cam_id=0)# input: image; time(value), label: Onehot, model: simpelCNN+MLP
    # pred_by_simpleCNNMLP_TimeValue_TOnehot(cam_id=3)
    # pred_by_simpleCNNMLP_TimeValue_TOnehot(cam_id=5)
    # pred_by_simpleCNNMLP_TimeValue_TOnehot(cam_id=6)
    # pred_by_simpleCNNMLP_TimeValue_TOnehot(cam_id=7)
    # # ==== simpleCNN+LSTM 收敛：1.9====
    step = 5
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=10) # input: image sequence, label: LDE sequence, model:
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=11)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=12)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=13)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=14)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=15)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=16)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=17)

    # pred_by_simpleCNNLSTM_TLDE(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TLDE(cam_id=3, step=step, chosen_hour=10) # input: image sequence, label: LDE sequence, model:
    # pred_by_simpleCNNLSTM_TLDE(cam_id=5, step=step, chosen_hour=10) # input: image sequence, label: LDE sequence, model:
    # pred_by_simpleCNNLSTM_TLDE(cam_id=6, step=step, chosen_hour=10) # input: image sequence, label: LDE sequence, model:
    # pred_by_simpleCNNLSTM_TLDE(cam_id=7, step=step, chosen_hour=10) # input: image sequence, label: LDE sequence, model:
    # # #                                                                 # simpleCNN+LSTM
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=8)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=11)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=12)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=13)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=14)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=15)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=16)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=17)
    # step = 2
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=3, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=5, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=6, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=7, step=step, chosen_hour=10)
    # step = 3
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=3, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=5, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=6, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=7, step=step, chosen_hour=10)
    # step = 4
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=3, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=5, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=6, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=7, step=step, chosen_hour=10)
    # step = 5
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=3, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=5, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=6, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=7, step=step, chosen_hour=10)
    # step = 6
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=3, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=5, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=6, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=7, step=step, chosen_hour=10)
    # step = 7
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=3, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=5, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=6, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot(cam_id=7, step=step, chosen_hour=10)
    #
    # step = 3
    # pred_by_simpleCNNLSTM_TValue(cam_id=0, step=2, chosen_hour=10)
    # PASS

    # def pred_by_step(hour=8):
    #     pred_by_simpleCNNLSTM_TValue(cam_id=0, step=2, chosen_hour=hour)
    #     pred_by_simpleCNNLSTM_TValue(cam_id=0, step=3, chosen_hour=hour)
    #     pred_by_simpleCNNLSTM_TValue(cam_id=0, step=4, chosen_hour=hour)
    #     pred_by_simpleCNNLSTM_TValue(cam_id=0, step=5, chosen_hour=hour)
    #     pred_by_simpleCNNLSTM_TValue(cam_id=0, step=6, chosen_hour=hour)
    #     pred_by_simpleCNNLSTM_TValue(cam_id=0, step=7, chosen_hour=hour)
    #     pred_by_simpleCNNLSTM_TValue(cam_id=0, step=8, chosen_hour=hour)


    # pred_by_step(hour=8)
    # pred_by_step(hour=9)
    # pred_by_step(hour=11)
    # pred_by_step(hour=12)
    # pred_by_step(hour=13)
    # pred_by_step(hour=14)
    # pred_by_step(hour=15)
    # pred_by_step(hour=16)
    # pred_by_step(hour=17)
    # pred_by_simpleCNNLSTM_TValue(cam_id=0, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TValue(cam_id=3, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TValue(cam_id=5, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TValue(cam_id=6, step=step, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TValue(cam_id=7, step=step, chosen_hour=10)
    # # main_v10(cam_id=3, step=step)




    # ====== TEST =====
    # get_confusion_matrix_from_csv("VTENET_Value_0_4_40_0.001_truth_predcit.csv")
    # get_temp_test()

    # pred_by_simpleCNNLSTM_TValue_v2(cam_id=0, step=4, chosen_hour=9)   # PASS 2021-8-8 17:57:13
    # pred_by_simpleCNNLSTM_TValue_v2(cam_id=3, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TValue_v2(cam_id=5, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TValue_v2(cam_id=6, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TValue_v2(cam_id=7, step=4, chosen_hour=9)
    # #
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=9)   # PASS 2021-8-8 17:57:02
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=3, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=5, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=6, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=7, step=4, chosen_hour=9)

    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=8)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=10)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=11)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=12)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=13)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=14)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=15)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=16)
    # pred_by_simpleCNNLSTM_TOnehot_v2(cam_id=0, step=4, chosen_hour=17)


    # #
    # pred_by_simpleCNNLSTM_TLDE_v2(cam_id=0, step=4, chosen_hour=9)        # PASS 2021-8-8 17:56:38
    # pred_by_simpleCNNLSTM_TLDE_v2(cam_id=3, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TLDE_v2(cam_id=5, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TLDE_v2(cam_id=6, step=4, chosen_hour=9)
    # pred_by_simpleCNNLSTM_TLDE_v2(cam_id=7, step=4, chosen_hour=9)

    # pred_by_simpleCNNLSTM_TLDE_v3(cam_id=0, step=3, chosen_hour=10)         # NOT PASS
    # pred_by_simpleCNNLSTM_TLDE_v4(cam_id=0, step=step, chosen_hour=10)      # PASS 2021-08-08

    pass