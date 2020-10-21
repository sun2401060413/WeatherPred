# -*- coding:utf-8 -*-
# WeatherClassify: Feature: Version1.0 by SunZhu, 2018.05.16 in Chang'an University
'''Weather features extraction'''

import os
from enum import Enum
import math
import numpy as np
import pylab as pl
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import cv2
import scipy
# from scipy.misc import imread  1.4.0 版本中被弃用
import imageio
import maxflow
import SkyDetection
import datetime
from scipy import io
import tqdm

def version():
    print("WeatherClassify: Feature: Version1.0 by SunZhu, 2018.05.16 in Chang'an University")
    return

class DisplayType(Enum):
    "图像显示类型"
    TYPE_ALL = 0            # 显示所有图像
    TYPE_SRC = 1            # 显示原图像
    TYPE_GRAY = 2           # 显示灰度图
    TYPE_DARK = 3           # 显示暗通道图像
    TYPE_TEXTURE = 4        # 显示纹理图像

class WeatherFeature:
    "Weather feature base class"

    def __init__(self):
        # self.SrcImage = image
        # self.GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.SkyDetector = SkyDetection.SkyDetection()
    def set_image(self, filename):
        self.filename = filename
        self.SrcImage = cv_imread(filename)
        self.GrayImage = cv2.cvtColor(self.SrcImage, cv2.COLOR_BGR2GRAY)
        try:
            self.MaskImage = self.SkyDetector.get_sky_region(self.SrcImage)
        except:
            self.MaskImage = np.zeros(self.SrcImage[:2], np.uint8)
        self.MaskImage_sum = np.sum(np.reshape(self.MaskImage, (self.MaskImage.size, )))
    def getFeatures(self):
        # Global feature
        self.time_feature = f_time(self.filename)
        self.dark_channel, self.haze_feature = f_haze(self.SrcImage)          # Haze feature
        self.contrast_feature = f_contrast(self.SrcImage)                     # Contrast feature

        # Local feature
        if self.MaskImage_sum == 0:
            self.texture_feature = np.zeros(32, np.float)                    # Texture feature
            self.lbp_feature = np.zeros(64, np.float)                        # LBP feature
            self.color_feature = np.zeros(192, np.float)                      # color feature
            self.intensity_feature = np.zeros(64, np.float)                  # color feature
            self.cloud_feature = np.zeros(192, np.float)                      # cloud feature
        else:
            self.texture_image, self.texture_feature = f_texture_in_mask(self.SrcImage, self.MaskImage)   # Texture feature
            self.lbp_image, self.lbp_feature = f_LBP_in_mask(self.SrcImage, self.MaskImage)               # LBP feature
            self.color_feature = f_color_in_mask(self.SrcImage, self.MaskImage)                             # color feature
            self.intensity_feature = f_color_in_mask(self.GrayImage, self.MaskImage)                          # color feature
            self.cloud_feature = f_cloud_in_mask(self.SrcImage, self.MaskImage)                             # cloud feature

        # 总体特征，共801 D
        self.feature = np.concatenate([self.time_feature,            # Time                      2 D
                                      self.color_feature,           # RGB color histogram       64*3 = 192 D
                                      self.texture_feature,         # Gabor wavelet             32 D
                                      self.intensity_feature,       # Intensity histogram       64 D
                                      self.cloud_feature,           # Cloud features            64*3 = 192 D
                                      self.lbp_feature,             # Local binary pattern      64 D
                                      self.haze_feature,            # Haze features             84 D
                                      self.contrast_feature],       # Contrast features         171 D
                                      axis=0)
    def imshow(self, *args):
        if args[0] == DisplayType.TYPE_ALL:
            args = [DisplayType.TYPE_SRC,
                    DisplayType.TYPE_GRAY,
                    DisplayType.TYPE_DARK,
                    DisplayType.TYPE_TEXTURE]

        for value in args:
            if value == DisplayType.TYPE_SRC:
                cv2.namedWindow("SrcImage", cv2.WINDOW_NORMAL)
                cv2.imshow("SrcImage", self.SrcImage)

            if value == DisplayType.TYPE_GRAY:
                cv2.namedWindow("GrayImage", cv2.WINDOW_NORMAL)
                cv2.imshow("GrayImage", self.GrayImage)

            if value == DisplayType.TYPE_DARK:
                cv2.namedWindow("DarkChannel", cv2.WINDOW_NORMAL)
                cv2.imshow("DarkChannel", self.dark_channel)

            if value == DisplayType.TYPE_TEXTURE:
                pl.figure(1)
                for temp in range(len(self.texture_image)):
                    pl.subplot(4, 4, temp+1)
                    pl.imshow(self.texture_image[temp], cmap='gray')
                pl.show()
        cv2.waitKey()
## 特征提取函数
def f_haze(image):   # Haze feature         # 全局提取
    '''Haze特征提取功能
    :param image: 待分析图像
    :return: dark_image:DarkChannel图像，f_vector:提取的特征向量
    '''
    # 1.将图像拉伸为512×512，规格见Ref[1];
    resize_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_CUBIC)
    # 2.计算其暗通道图像;
    dark_image = J_dark(resize_image, 4)    # The size of local patch is 8×8 in Ref[1]
    # 3.划分为不同规格统计区域，计算中位数(划分方案见Ref[1]: 2×2，4×4，8×8)
    f_vector = get_median_dark(dark_image, 2, 4, 8) # 2,4,8 denote the partation size
    return dark_image, f_vector

def f_contrast(image):  # Contrast feature  # 全局提取
    '''
    :param image: 待分析图像
    :return: f_vector: 提取的特征向量
    '''
    # 1. 将图像转为LCH颜色空间;
    Lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lch_image_c = Lab2LCH_C(Lab_image)
    # 2. LCH色彩空间的饱和度维排序
    pixels_count = lch_image_c.shape[0]*lch_image_c.shape[1]
    # reshape_lch_image_c = np.reshape(lch_image_c, (1, pixels_count))
    # sort_lch_image_c = np.sort(reshape_lch_image_c[0])
    reshape_lch_image_c = lch_image_c.flatten()
    sort_lch_image_c = np.sort(reshape_lch_image_c)

    # 3. 求取%5：5%：95% 百分位的饱和度
    precentages = [x for x in np.arange(0.05, 1.00, 0.05)]
    v_precentage = []
    for v in precentages:
        v_precentage.append(sort_lch_image_c[int(v*pixels_count)])

    # 4. 不同百分位饱和度比值ratio
    f_vector = []
    for i in range(0, len(v_precentage)):          # 对于任意i>j,输出，因此共171维
        for j in range(0, i):
            if v_precentage[j]==0:
                f_vector.append(v_precentage[i] / 0.0001)
            else:
                f_vector.append(v_precentage[i]/v_precentage[j])
    return f_vector

def f_texture(image):
    ''' 提取纹理特征，目前仅实现使用Gabor滤波器实现的纹理特征提取
    :param image: 待分析图像
    :return: f_vector: 纹理特征
    '''
    # 1.建立滤波器组
    filters = BuiltGaborFilter(4, [7, 9, 11, 13])     # 特征参数见Ref[3]Camera as Weather Sensor: Estimating Weather Information from Single Images
    # 2.应用滤波器组获取特征图像
    res_images = getGabor(image, filters)
    # 3.抽取纹理特征
    f_vector = []
    for res_image in res_images:
        data = np.array(res_image).flatten()
        f_vector.append(np.mean(data))                  # 均值
        f_vector.append(np.var(data))                   # 方差
    return res_images, f_vector

def f_texture_in_mask(image, mask):
    '''
    :param image:
    :param mask:
    :return:
    '''
    # 1.建立滤波器组
    filters = BuiltGaborFilter(4, [7, 9, 11, 13])     # 特征参数见Ref[3]Camera as Weather Sensor: Estimating Weather Information from Single Images
    # 2.应用滤波器组获取特征图像
    res_images = getGabor(image, filters)
    # 3.抽取纹理特征
    x_pixels, y_pixels = np.where(mask > 0)
    f_vector = []
    for res_image in res_images:
        data = [res_image[x_pixels[i],y_pixels[i]] for i in range(len(x_pixels))]
        f_vector.append(np.mean(data))                  # 均值
        f_vector.append(np.var(data))                   # 方差
    return res_images, f_vector

def f_LBP(image):
    ''' LBP 特征提取
    :param image: 待分析图像
    :return:  f_vector:LBP(局部二值模式)
    '''
    # 1. 计算LBP特征图像
    LBP_image = getLBP(image, 3, 8)
    # 2. 统计LBP特征直方图
    res = cv2.calcHist([image], [0], None, [64], [0, 256])
    f_vector = np.array(res).flatten()
    f_vector = f_vector/(LBP_image.shape[0]*LBP_image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(f_vector)), f_vector)
    # pl.show()
    return LBP_image, f_vector

def f_LBP_in_mask(image, mask):
    ''' LBP 特征提取
    :param image: 待分析图像
    :return:  f_vector:LBP(局部二值模式)
    '''
    # 1. 计算LBP特征图像
    LBP_image = getLBP(image, 3, 8)
    # 2. 统计LBP特征直方图
    res = cv2.calcHist([image], [0], mask, [64], [0, 256])
    f_vector = np.array(res).flatten()
    f_vector = f_vector/(LBP_image.shape[0]*LBP_image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(f_vector)), f_vector)
    # pl.show()
    return LBP_image, f_vector

def f_color(image):
    ''' 颜色特征
    :param image: 待分析图像
    :return: f_vector:颜色特征
    '''
    # 1.统计色彩直方图
    if len(image.shape) > 2:        # 多通道时，输出特征为色彩特征
        b_hist = cv2.calcHist([image], [0], None, [64], [0, 256])         # 64个bins是Ref[3]中的选择
        g_hist = cv2.calcHist([image], [1], None, [64], [0, 256])         # 同上
        r_hist = cv2.calcHist([image], [2], None, [64], [0, 256])         # 同上
        f_vector = np.array([b_hist, g_hist, r_hist]).flatten()
        f_vector = f_vector/(image.shape[0]*image.shape[1])
    else:                           # 单通道时，输出为亮度特征
        # f_vector = np.array(cv2.calcHist(image, [0], None, [64], [0.0, 255.0])).flatten()
        f_vector = np.array(cv2.calcHist([image], [0], None, [64], [0, 256])).flatten()
        f_vector = f_vector / (image.shape[0] * image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(b_hist)), b_hist, 'b')
    # pl.plot(range(len(g_hist)), g_hist, 'g')
    # pl.plot(range(len(r_hist)), r_hist, 'r')
    # pl.show()
    return  f_vector

def f_color_in_mask(image, mask):
    '''
    :param image:   待处理图像
    :param mask:    sky region
    :return:        天空区域内的颜色特征
    '''
    # 1.统计色彩直方图
    if len(image.shape) > 2:        # 多通道时，输出特征为色彩特征
        b_hist = cv2.calcHist([image], [0], mask, [64], [0, 256])         # 64个bins是Ref[3]中的选择
        g_hist = cv2.calcHist([image], [1], mask, [64], [0, 256])         # 同上
        r_hist = cv2.calcHist([image], [2], mask, [64], [0, 256])         # 同上
        f_vector = np.array([b_hist, g_hist, r_hist]).flatten()
        f_vector = f_vector/(image.shape[0]*image.shape[1])
    else:                           # 单通道时，输出为亮度特征
        # f_vector = np.array(cv2.calcHist(image, [0], None, [64], [0.0, 255.0])).flatten()
        f_vector = np.array(cv2.calcHist([image], [0], None, [64], [0, 256])).flatten()
        f_vector = f_vector / (image.shape[0] * image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(b_hist)), b_hist, 'b')
    # pl.plot(range(len(g_hist)), g_hist, 'g')
    # pl.plot(range(len(r_hist)), r_hist, 'r')
    # pl.show()
    return  f_vector

def f_SIFT(image):
    '''
    :param image:
    :return:
    '''
    # sift = cv2.SIFT()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kps, features = sift.detectAndCompute(gray, None)
    # print(kps)
    # print(features)
    # print(image.shape)
    return features

def f_cloud_in_mask(image, mask):
    ''' 颜色特征
    :param image: 待分析图像
    :return: f_vector:颜色特征
    '''
    # 1.统计色彩直方图
    ro_image = getRO(image)
    sa_image = getSA(image)
    ed_image = getEGD(image)

    ro_feature = get_hist(ro_image, mask, [-1, 1])
    sa_feature = get_hist(sa_image, mask, [0, 1])
    ed_feature = get_hist(ed_image, mask, [0, 208])
    f_vector = np.concatenate((ro_feature, sa_feature, ed_feature), axis=0)             # list 拼接
    return  f_vector

def f_time(filepath):
    '''
    :return:
    '''
    time_str = get_time_str(filepath)
    current_time = datetime.datetime.strptime(time_str[0][0:14], "%Y%m%d%H%M%S")
    d_number = current_time.timetuple().tm_yday
    s_number = current_time.timetuple().tm_hour*3600 + current_time.timetuple().tm_min*60 + current_time.timetuple().tm_sec
    if is_leap(current_time.timetuple().tm_year):
        d_number = d_number / 366
    else:
        d_number = d_number / 365
    s_number = s_number / (24*3600)
    return [d_number, s_number]

## 功能函数
def is_leap(year):
    '''
    判断是否为闰年
    :param year:
    :return:
    '''
    if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
        return True
    return False

def get_hist(image, mask, value_range):
    '''
    计算特征图像的直方图
    :param image:
    :param mask:
    :param value_range:
    :return:
    '''
    # flatten_image = image.flatten()
    flatten_image = []
    x, y = np.where(mask>0)
    for i in range(len(x)):
        flatten_image.append(image[x[i], y[i]])
    f_vector, bin_edges = np.histogram(flatten_image, bins=64, range=value_range)
    f_vector = f_vector/len(flatten_image)
    # f_vector = f_vector/(image.shape[0]*image.shape[1])
    # f_vector = np.array(cv2.calcHist([image], [0], mask, [64], value_range)).flatten()
    return f_vector

def J_dark(image, r):
    '''区域通道最小值滤波
    :param image: 待分析图像(3通道)
    :param r: 分析区域半径
    :return: DarkChannel图像
    '''
    if r <= 0:
        return np.min(image, 2)     #R, G, B 通道最小值
    h, w = image.shape[:2]
    temp_image = image
    res = np.minimum(temp_image, temp_image[[0] + [x for x in range(h-1)], :])  #[0, 0 ,1,..., h-1]
    res = np.minimum(res, temp_image[[x for x in range(1, h)] + [h-1], :])      #[0, 1,...,h-1, h-1]
    temp_image = res
    res = np.minimum(temp_image, temp_image[:, [0] + [x for x in range(w-1)]])  #[0, 0, 1,..., w-1]
    res = np.minimum(res, temp_image[:, [x for x in range(1, w)] + [w-1]])      #[0, 1,..., w-1, w-1]
    return J_dark(res, r-1)  # 递归调用

def get_median_dark(image, *args):
    '''
    获取不同图像划分方案下，区域中DarkChannel中位数
    :param image: 待分析图像(暗通道图像)
    :param args: 图像划分方案集(2:2×2，4:4×4，8:8×8)
    :return: res:特征向量
    '''
    res = []               # 特征向量
    rows, cols = image.shape
    for arg in args:
        r = int(rows/arg)
        c = int(cols/arg)
        for i in range(arg):
            for j in range(arg):
                sorted_image = sorted(image[r*i:r*(i+1), c*j:c*(j+1)].reshape(1, r*c)[0, :])
                res.append(sorted_image[int(r*c/2)])
    return res

def Lab2LCH_C(image):
    ''' Lab 色彩空间 转 LCH色彩空间(为减少计算量，仅计算C通道，即饱和度通道)
    :param image: 待转换Lab色彩空间的图像 (3维)
    :return: cvt_image: 转换完成的LCH色彩空间图像 （1维）
    色彩空间转换公式参考Ref[2]，其中：L = L; C = sqrt(a^2 + b^2); H = atan(b/a)
    e.g. Lab = [52, -14, -32] → LCH = [52 ， 34.93 ， 246.37°] 参考工具地址：http://www.colortell.com/labto （色彩管理网）
    '''
    # 1.L = l
    # 2.C = sqrt(a^2 + b^2)
    cvt_image = np.sqrt(np.array(image[:, :, 1]) ** 2 + np.array(image[:, :, 2]) ** 2)
    # 3.H = atan(b/a)
    return cvt_image

def BuiltGaborFilter(direction_count = 4,kernel_size = [7, 9, 11, 13, 15, 17]):
    ''' 生成Gabor滤波器
    :param direction_count: 滤波器方向个数（默认为4个）
    :param kernel_size: 滤波核尺度list（默认为[7, 9, 11, 13, 15, 17] 6个尺度）
    :return: 返回指定参数下的滤波器集合（默认参数下会生成4个方向，6个尺度共24个滤波器核）
    '''
    filters = []
    lamda = np.pi/2.0                                               # 波长
    for theta in np.arange(0, np.pi, np.pi/direction_count):        # Gabor 滤波器方向，默认为0°,45°,90°,135°
        for k in range(len(kernel_size)):
            kernel = cv2.getGaborKernel((kernel_size[k], kernel_size[k]),   # size of the Gabor kernel
                                        1.0,                                # standard deviation of the Gaussian function
                                        theta,                              # orientation of the normal to the parallel stripes of the Gabor function
                                        lamda,                              # wavelength of the sinusoidal factor in the above equation
                                        0.5,                                # spatial aspect ratio.
                                        0,                                  # phase offset.
                                        cv2.CV_32F)                         # the type
            kernel /= 1.5*kernel.sum()
            filters.append(kernel)
    return filters

def GaborProcess(image, filters):
    accum = np.zeros_like(image)
    for kernel in filters:
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        np.maximum(accum, filtered_image, accum)
    return accum

def getGabor(image, filters):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = []                    # 滤波结果
    for i in range(len(filters)):
        res.append(np.asarray(GaborProcess(gray_image, filters[i])))
    # pl.figure(1)
    # for temp in range(len(filters)):
    #     pl.subplot(4, 4, temp+1)
    #     pl.imshow(filters[temp], cmap='gray')
    # pl.figure(2)
    # for temp in range(len(res)):
    #     pl.subplot(4, 4, temp+1)
    #     pl.imshow(res[temp], cmap='gray')
    # pl.show()
    return  res

def getLBP(image, radius, n_points):
    ''' 获取LBP特征值图像
    :param image: 待分析图像
    :param radius: 计算半径
    :param n_points: 进制位数(8，16，32)
    :return: res:编码结果图像
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = local_binary_pattern(gray_image, radius*n_points, radius)
    # pl.imshow(res, cmap='gray')
    # pl.show()
    return res

def getRO(image):
    ''' cloud feature：normalized blue/red ratio (see ref.[4])
        defination ：ro(s) = (b - r)/(b + r), Clear sky appears blue with high ro and cloud appears white or gray
                 with low ro
    :param image:
    :return:
    '''
    # 注意：imdecode读取的是rgb，不是bgr(cv2.imread读入的时bgr)
    row, col, depth = image.shape
    ro_image = np.zeros((row, col), np.float)
    (B, G, R) = cv2.split(image)
    for i in range(row):
        for j in range(col):
            if (int(B[i, j]) + int(R[i, j])) == 0:
                ro_image[i, j] = (int(B[i, j]) - int(R[i, j])) / 0.001
            else:
                ro_image[i, j] = (int(B[i, j]) - int(R[i, j]))/(int(B[i, j]) + int(R[i, j]))
    return ro_image

def getSA(image):
    ''' cloud feature：saturation (see ref.[4])
        defination ：sa(s) = 1 - 3*min(r,g,b)/(r+g+b)
    :param image:
    :return:
    '''
    # 注意：imdecode读取的是rgb，不是bgr(cv2.imread读入的时bgr)
    row, col, depth = image.shape
    sa_image = np.zeros((row, col), np.float)
    (B, G, R) = cv2.split(image)
    for i in range(row):
        for j in range(col):
            if (int(R[i, j]) + int(G[i, j]) + int(B[i, j])) == 0:
                sa_image[i, j] = 1 - 3 * min(int(R[i, j]), int(G[i, j]), int(B[i, j])) / 0.001
            else:
                sa_image[i, j] = 1 - 3 * min(int(R[i, j]), int(G[i, j]), int(B[i, j])) / (int(R[i, j]) + int(G[i, j]) + int(B[i, j]))
    return sa_image

def getEGD(image):
    ''' cloud feature：EGD (see ref.[4])
        defination ：ed(s) = sqrt(r^2+g^2+b^2-((r+g+b)^2)/3)
    :param image:
    :return:
    '''
    # 注意：imdecode读取的是rgb，不是bgr(cv2.imread读入的时bgr)
    row, col, depth = image.shape
    ed_image = np.zeros((row, col), np.float)
    (B, G, R) = cv2.split(image)
    for i in range(row):
        for j in range(col):
            ed_image[i, j] = math.sqrt(int(R[i, j])**2 + int(G[i, j])**2 + int(B[i, j])**2 - (int((R[i, j]) + int(G[i, j]) + int(B[i, j]))**2)/3)
    return ed_image

def graph_cut_test():
    # img = imread("C:/Users/15092/Desktop/a2.png")
    # img = imageio.imread("E:/DataSet/WeatherClasifer_Chosen/0_sunny/34020000000270000055_20161123143749846_2501_小雨_多云.png")
    img = imageio.imread(
        "34020000000270000055_20161123135350025_2501_.png")
    # img = cv2.imread("E:/DataSet/WeatherClasifer_Chosen/0_sunny/34020000000270000055_20161123135350025_2501_.png")
    # Create the graph.
    g = maxflow.Graph[int]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    # nodeids = g.add_grid_nodes(img.shape)
    nodeids = g.add_grid_nodes(img.shape)
    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, 50)
    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    g.add_grid_tedges(nodeids, img, 255 - img)
    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))
    # Show the result.
    plt.imshow(img2)
    plt.show()

def get_time_str(filepath):
    path, name = os.path.split(filepath)
    split_str = name.split("_")
    time_str = [elem for elem in split_str if len(elem) == 17 and elem[0:2] == "20"]
    return time_str

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def get_filename_label(txtfilename, type='txt'):
    import json
    datalist = []
    with open(txtfilename, 'r') as doc:
        if type == 'txt':
            lines = doc.readlines()
            for line in lines:
                line_str = line.strip("\n").split(" ")
                datalist.append(line_str)
        elif type == 'json':
            datalist = json.load(doc)['train_data']
    return datalist

if __name__ == '__main__':
    # # 功能测试
    solver = WeatherFeature()
    # datalist = get_filename_label("D:/DataSet/WeatherClasifer_Chosen/Data/train.txt")
    datalist = get_filename_label(r"E:\DataSet\WeatherClasifer_Chosen\ver0.0\DataInfo.json", type='json')
    training_data = []
    pbar = tqdm.tqdm(total=len(datalist))
    for elem in datalist:
        pbar.update(1)
        solver.set_image(elem[0])
        solver.getFeatures()
        training_data.append(np.concatenate([solver.feature, [int(elem[1])]], axis=0))
    pbar.close()
    # # print(training_data)
    # # print(len(training_data[0]))
    #
    # io.savemat("D:/DataSet/WeatherClasifer_Chosen/training_data.mat", {'array': training_data})
    # print("特征提取完成！")
    # training_data = io.loadmat("D:/DataSet/WeatherClasifer_Chosen/training_data.mat")
    # print(training_data["array"])
    # print(len(training_data["array"][0]))

    # filepath = "E:/DataSet/SkyDetection/MaskImage/processed/34020000000270000055_20161028133111581_2501_小雨_多云.png"
    # solver.set_image(filepath)
    # solver.getFeatures()
    #
    # print(len(solver.time_feature))
    # print(len(solver.color_feature))
    # print(len(solver.texture_feature))
    # print(len(solver.intensity_feature))
    # print(len(solver.cloud_feature))
    # print(len(solver.lbp_feature))
    # print(len(solver.haze_feature))
    # print(len(solver.contrast_feature))
    # print(len(solver.feature))

    # Reference
    # [1] Lu C, Lin D, Jia J, et al. Two-Class Weather Classification[C]//
    #     IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 2014:3718-3725.
    # [2] LAB与LCH颜色空间及其应用：https://wenku.baidu.com/view/42f8f03ab5daa58da0116c175f0e7cd1842518a0.html
    # [3] Chu W T, Zheng X Y, Ding D S. Camera as weather sensor: Estimating weather information from single images[J].
    #     Journal of Visual Communication & Image Representation, 2017, 46:233-249.
    # [4] Q. Li, W. Lu, J. Yang and J. Z. Wang (2012). "Thin Cloud Detection of All-Sky Images Using Markov Random Fields"
    #     IEEE Geoscience and Remote Sensing Letters 9:417-421