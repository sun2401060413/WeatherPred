# _*_ coding: utf-8 _*_
# @Time : 2020/10/24 17:08
# @Author : Sun Zhu
# @Version：V 1.0
# @File : Feature.py
# @desc : Weather features extraction for weather classification.

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class DataGenerator(object):
    """数据生成器，根据图像路径和label路径生成训练测试数据"""
    def __init__(self,
                 images_root=None,  # 图像根目录
                 label_root=None    # label相关文件根目录
                 ):
        self.image_root = images_root
        self.label_root = label_root
        self.metadata_filepath = os.listdir(self.label_root)
        # print(self.metadata_filepath)
        # print(self.get_cam_id(filename=self.metadata_filepath[0]))
        self.source_data = self.get_source_data()

    @staticmethod
    def get_cam_id(filename=None):
        """从csv文件名中提取cam id 信息，初始化时即生成"""
        return os.path.splitext(filename)[0]

    def get_source_data(self):
        """根据csv文件抽取所有原始数据，初始化时即生成
            数据组织：data_dict: key:cam_id,value:cam_data
                    key:date_id, key:data
        """
        self.source_data = {}
        self.cam_list = []
        for elem in self.metadata_filepath:
            self.source_data[self.get_cam_id(elem)] = self.get_cam_data(elem)
            self.cam_list.append(os.path.splitext(elem)[0])
        return self.source_data

    def get_cam_data(self, csv_file):
        """从csv文件中读取数据，并组织成一个字典。
        字典包括key：date_id: 数据中可用的date_id, ["20100101", "20100102",...]
                data: 数据字典。key为date_id, 将每日的数据放置于一个list内
                例如："20100101":[[0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour],...]
        """
        filepath = os.path.join(self.label_root, csv_file)
        csv_data = pd.read_csv(filepath)
        cam_id = os.path.splitext(csv_file)[0]

        data_dict = {"date_id": [], "data": {}}

        # 信息抽取
        for index, row in csv_data.iterrows():
            filename = row["Filename"]
            date_id = os.path.splitext(filename)[0].split("_")[0]
            temp = row["TempM"]
            year = row["Year"]
            month = row["Month"]
            day = row["Day"]
            hour = row["Hour"]
            dtime = datetime(year=year, month=month, day=day, hour=hour)
            daycount = get_day_count(dtime)
            mincount = get_mins_count(dtime)
            # 0-daycount, 1-mincount, 2-filename, 3-temp, 4-cam_id, 5-date_id, 6-dtime, 7-year, 8-month, 9-day, 10-hour
            info_list = [daycount, mincount, filename, temp, cam_id, date_id, dtime, year, month, day, hour]
            # 时间原因，先不考虑数据缺失的问题。
            if date_id not in data_dict["date_id"]:
                data_dict["date_id"].append(date_id)
            if not data_dict["data"].__contains__(date_id):
                data_dict["data"][date_id] = []
            data_dict["data"][date_id].append(info_list)

        # print(data_dict)
        # png_savepath = r"E:\Project\CV\WeatherPred\temp"
        # vis_dateset(dtime_collection, os.path.join(png_savepath, cam_id+"_new.png"))
        return data_dict

    def get_chosen_data(self, chosen_cam=[], step=3, chosen_hour=12):
        """从选定的cam_id中导出数据，chosen_cam为选定cam_id的列表
            可选参数:chosen_cam: 可选摄像机cam_id列表
            step:步长
            chosen_hour: 选定时间
        """
        output_list = []
        if len(chosen_cam) == 0:
            for elem in self.source_data:
                output_list.extend(self.get_chosen_data_from_one_cam(self.source_data[elem], step=step, chosen_hour=chosen_hour))
        else:
            for elem in self.source_data:
                if elem in chosen_cam:
                    output_list.extend(self.get_chosen_data_from_one_cam(self.source_data[elem], step=step, chosen_hour=chosen_hour))
        return output_list

    def get_chosen_data_from_one_cam(self, data_dict=None, step=3, chosen_hour=12):
        """从选定cam_id获取数据，TEST:PASS
            可选参数：data_dict: 某摄像机数据：key：date_id, data
        """
        output_list = []
        for elem in data_dict["data"]:
            available_date_id = generate_available_date_id(elem, step=step)
            # print(elem, available_date_id)
            record = [self.find_data_in_chosen_time(data_dict["data"][elem], chosen_hour)]
            flag = True
            for subelem in available_date_id:
                if not data_dict["data"].__contains__(subelem):
                    flag = False
                    break
                else:
                    data_chosen = self.find_data_in_chosen_time(data_dict["data"][subelem], chosen_hour)
                    if data_chosen is not None:
                        record.append(data_chosen)
                    else:
                        flag = False
                        break
            if flag is True and None not in record:
                output_list.append(record)
        return output_list

    def find_data_in_chosen_time(self, datalist=None, chosen_hour=None):
        """查看给定的list中，各元素中的hour要素是否存在与给定相一致的"""
        output = None
        for elem in datalist:
            if elem[-1] == chosen_hour:
                output = elem
                break
        return output

    def get_source_data_for_cnn(self, selected_cam=None):
        output = []
        if selected_cam is None:
        # ====== 全部场景 ======
            for elem in self.source_data:
                for subelem in self.source_data[elem]["data"]:
                    output.extend(self.source_data[elem]["data"][subelem])
        else:
        # ======================
            for subelem in self.source_data[self.cam_list[selected_cam]]["data"]:
                output.extend(self.source_data[self.cam_list[selected_cam]]["data"][subelem])
                pass
        return output

# =========== UTILS FUNCTION ===========


def generate_available_date_id(date_id, step=3):
    """以data_id为基准，计算步长范围内所需数据来源的date_id
        例如：date_id为'20100131'时，取步长5的date_id,则可得到['20100201', '20100202', '20100203', '20100204']
        可正向取，即取当前点未来的date_id, 亦可反向取，取当前点历史的date_id
    """
    current_data = datetime.strptime(date_id, "%Y%m%d")
    output_list = []
    if step > 0:
        for i in range(1, step):
            # print((current_data+timedelta(days=i)).strftime("%Y%m%d"))
            output_list.append((current_data+timedelta(days=i)).strftime("%Y%m%d"))
    else:
        for i in range(step, -1):
            # print((current_data+timedelta(days=i)).strftime("%Y%m%d"))
            output_list.append((current_data+timedelta(days=i)).strftime("%Y%m%d"))
    return output_list


def get_day_count(targetDay):
    """从日期数据计算属于该年第几天"""
    dayCount = targetDay - datetime(targetDay.year-1, 12, 31)
    return dayCount.days


def get_mins_count(targetDay):
    """从时间数据获取分钟"""
    return targetDay.hour*60+targetDay.minute


def vis_dateset(data, filename):
    plt.figure()
    X, Y = [], []
    plt.title(filename)
    for elem in data:
        X.append(elem[0])
        Y.append(elem[1])
    plt.scatter(X, Y)
    plt.grid()
    # plt.show()
    plt.savefig(filename, dpi=300)

# ======= TEST FUNCTIONS ========


def DataGeneratorTest():
    """数据生成器测试数据"""
    obj = DataGenerator(images_root=r"D:\CVProject\CBAM-keras-master\weather_data\dataset2",
                        label_root=r"D:\CVProject\CBAM-keras-master\weather_data\metadata")
    chosen_data = obj.get_chosen_data(chosen_cam=[obj.cam_list[0], obj.cam_list[1]], step=0)
    print(chosen_data)



if __name__=="__main__":
    # # ==== TEST: DataGeneratorTest =====
    DataGeneratorTest()

    pass