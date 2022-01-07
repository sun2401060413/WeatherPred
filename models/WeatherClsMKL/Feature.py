# _*_ coding: utf-8 _*_
# @Time : 2018/5/16 17:08
# @Author : Sun Zhu
# @Version：V 1.0
# @File : Feature.py
# @desc : Weather features extraction for weather classification.
# @env: CV_TF_1

import sys

# PCANet path
sys.path.append(r"sys.path.append(D:\CVProject\PCANet")

import os
from enum import Enum
import math
import numpy as np
import pylab as pl
from skimage.feature import local_binary_pattern

import cv2
import models.WeatherClsMKL.SkyDet as SkyDetection
import datetime
from scipy import io

import tqdm


def version():
    return print("WeatherPred-Feature: Version1.0 Created by SunZhu at 2018.05.16 in Chang'an University")

class DisplayType(Enum):
    """ Mode for displaying the result images """
    TYPE_ALL = 0            # Display all results
    TYPE_SRC = 1            # Display the source images
    TYPE_GRAY = 2           # Display the gray images
    TYPE_DARK = 3           # Display the Dark channels images
    TYPE_TEXTURE = 4        # Display the textures images
    TYPE_SYKMASK = 5        # Display the non-sky region


class WeatherFeature:
    """ Base class for Weather features extraction

    This class creates feature extractor for weather recognition. The features include
    time, color, texture, cloud, lbp, haze, shadow, snow , pca.

    # Arguments
        feature_list : feature type for extreation. e.g. ['time', 'color', 'texture', 'cloud', 'lbp', 'haze', 'contrast', 'pca'
        save_mode: if save_mode is not None, the class will save the chosen features into some npy files
        feature_files_saved_path: file saved path (feature data);

    # Returns

    """


    def __init__(self,
                 feature_list=['time', 'color', 'texture', 'cloud', 'lbp', 'haze', 'contrast', 'pca'],
                 save_mode=None,
                 feature_files_saved_path=None
                 ):
        self.feature_list = feature_list
        self.SkyDetector = SkyDetection.SkyDetection()

        if "pca" in self.feature_list:              # read the saved pca features.
            # pca features are obtained from the PCANet project (The results are saved in a npy file).
            feature_dict_filepath = r"D:\CVProject\CBAM-keras-master\PCANet\data\data_new.npy"
            self.pca_feature_dict = get_pca_feature_dict(feature_dict_filepath)

        if save_mode is not None:
            # file saved path
            self.feature_file_root = feature_files_saved_path

            # === Each feature ===
            # time
            self.time_feature_dict = {}
            self.time_feature_file = os.path.join(self.feature_file_root, "time_feature.npy")
            # color
            self.color_feature_dict = {}
            self.color_feature_file = os.path.join(self.feature_file_root, "color_feature.npy")
            # texture
            self.texture_feature_dict = {}
            self.texture_feature_file = os.path.join(self.feature_file_root, "texture_feature.npy")
            # lbp
            self.lbp_feature_dict = {}
            self.lbp_feature_file = os.path.join(self.feature_file_root, "lbp_feature.npy")
            # cloud
            self.cloud_feature_dict = {}
            self.cloud_feature_file = os.path.join(self.feature_file_root, "cloud_feature.npy")
            # haze
            self.haze_feature_dict = {}
            self.haze_feature_file = os.path.join(self.feature_file_root, "haze_feature.npy")
            # contrast
            self.contrast_feature_dict = {}
            self.contrast_feature_file = os.path.join(self.feature_file_root, "contrast_feature.npy")
            # shadow
            self.shadow_feature_dict = {}
            self.shadow_feature_file = os.path.join(self.feature_file_root, "shadow_feature.npy")
            # snow
            self.snow_feature_dict = {}
            self.snow_feature_file = os.path.join(self.feature_file_root, "snow_feature.npy")
            # pac
            # # self.pca_feature_dict = {}
            pass

    def set_image(self, filename):
        # Get the gray image and mask of skyregion.
        self.filename = filename
        self.SrcImage = cv_imread(filename)
        self.GrayImage = cv2.cvtColor(self.SrcImage, cv2.COLOR_BGR2GRAY)
        try:
            self.MaskImage = self.SkyDetector.get_sky_region(self.SrcImage)
        except:
            self.MaskImage = np.zeros(self.SrcImage[:2], np.uint8)
        self.MaskImage_sum = np.sum(np.reshape(self.MaskImage, (self.MaskImage.size, )))

    def getFeatures(self):

        self.feature = []
        # time
        if 'time' in self.feature_list:                     # 2
            # Global features
            self.time_feature = f_time(self.filename)
            self.feature.extend(self.time_feature)
            print(len(self.time_feature), end=" ")

        # color
        if 'color' in self.feature_list:                    # 90
            if self.MaskImage_sum == 0:
                self.color_feature = np.zeros(90, np.float)  # color feature
            else:
                self.color_feature = f_color_in_mask_v2(self.SrcImage, self.MaskImage)  # color feature
            print(len(self.color_feature), end=" ")
            self.feature.extend(self.color_feature)

        # texture
        if 'texture' in self.feature_list:                  # 32
            if self.MaskImage_sum == 0:
                self.texture_feature = np.zeros(32, np.float)  # Texture feature
            else:
                self.texture_image, self.texture_feature = f_texture_in_mask(
                                                                    self.SrcImage,
                                                                    self.MaskImage)  # Texture feature
            self.feature.extend(self.texture_feature)

        # lbp
        if 'lbp' in self.feature_list:                      # 32
            if self.MaskImage_sum == 0:
                self.lbp_feature = np.zeros(32, np.float)                        # LBP feature
            else:
                self.lbp_image, self.lbp_feature = f_LBP_in_mask(self.SrcImage, self.MaskImage)  # LBP feature
            self.feature.extend(self.lbp_feature)


        # cloud
        if 'cloud' in self.feature_list:                    # 192
            if self.MaskImage_sum == 0:
                self.cloud_feature = np.zeros(192, np.float)  # cloud feature
            else:
                self.cloud_feature = f_cloud_in_mask(self.SrcImage, self.MaskImage)  # cloud feature
            self.feature.extend(self.cloud_feature)


        # haze
        if 'haze' in self.feature_list:                     # 84
            self.dark_channel, self.haze_feature = f_haze(self.SrcImage)  # Haze feature
            self.feature.extend(self.haze_feature)

        # contrast
        if 'contrast' in self.feature_list:                 # 171
            self.contrast_feature = f_contrast(self.SrcImage)  # Contrast feature
            self.feature.extend(self.contrast_feature)


        # shadow
        if 'shadow' in self.feature_list:                   # 3*4*4=48
            split_list = self.filename.split("/")
            root = r"D:\CVProject\CBAM-keras-master\prediction\prediction"
            self.shadow_path = os.path.join(root, split_list[-2], split_list[-1])
            self.shadow_img = cv_imread(self.shadow_path, mode="None")
            shadow_count, shadow_sum, edge_count = f_shadow(self.shadow_img)
            self.shadow_feature = shadow_count+shadow_sum+edge_count
            self.feature.extend(self.shadow_feature)

        # snow
        if 'snow' in self.feature_list:                     # 2*8*8=32

            snow_count, edge_count = f_snow(self.SrcImage, self.MaskImage)
            self.snow_feature = snow_count+edge_count
            self.feature.extend(self.snow_feature)

        # pca
        if 'pca' in self.feature_list:                      # 864
            self.pca_feature = f_PCA(self.filename, self.pca_feature_dict)
            self.feature.extend(self.pca_feature)

    def getFeaturesToSave(self):
        """
            save the features to the files
        """
        self.feature = []
        # TIME
        if 'time' in self.feature_list:                     # 2
            # Global features
            self.time_feature = f_time(self.filename)
            self.feature.extend(self.time_feature)
            self.time_feature_dict[self.filename] = self.time_feature
        # COLOR
        if 'color' in self.feature_list:                    # 90
            if self.MaskImage_sum == 0:
                self.color_feature = np.zeros(90, np.float)  # color feature
            else:
                self.color_feature = f_color_in_mask_v2(self.SrcImage, self.MaskImage)  # color feature
            self.feature.extend(self.color_feature)
            self.color_feature_dict[self.filename] = self.color_feature
        # TEXTURE
        if 'texture' in self.feature_list:                  # 32
            if self.MaskImage_sum == 0:
                self.texture_feature = np.zeros(32, np.float)  # Texture feature
            else:
                self.texture_image, self.texture_feature = f_texture_in_mask(self.SrcImage,
                                                                             self.MaskImage)  # Texture feature
            self.feature.extend(self.texture_feature)
            self.texture_feature_dict[self.filename] = self.texture_feature
        # LBP
        if 'lbp' in self.feature_list:                      # 64
            if self.MaskImage_sum == 0:
                self.lbp_feature = np.zeros(32, np.float)                        # LBP feature
            else:
                self.lbp_image, self.lbp_feature = f_LBP_in_mask(self.SrcImage, self.MaskImage)  # LBP feature
            self.feature.extend(self.lbp_feature)
            self.lbp_feature_dict[self.filename] = self.lbp_feature
        # CLOUD
        if 'cloud' in self.feature_list:                    # 192
            if self.MaskImage_sum == 0:
                self.cloud_feature = np.zeros(192, np.float)  # cloud feature
            else:
                self.cloud_feature = f_cloud_in_mask(self.SrcImage, self.MaskImage)  # cloud feature
            self.feature.extend(self.cloud_feature)
            self.cloud_feature_dict[self.filename] = self.cloud_feature
        # HAZE
        if 'haze' in self.feature_list:                     # 84
            self.dark_channel, self.haze_feature = f_haze(self.SrcImage)  # Haze feature
            self.feature.extend(self.haze_feature)
            self.haze_feature_dict[self.filename] = self.haze_feature
        # CONTRAST
        if 'contrast' in self.feature_list:                 # 171
            self.contrast_feature = f_contrast(self.SrcImage)  # Contrast feature
            self.feature.extend(self.contrast_feature)
            self.contrast_feature_dict[self.filename] = self.contrast_feature
        # SHADOW
        if 'shadow' in self.feature_list:                   # 3*4*4=48
            split_list = self.filename.split("/")
            root = r"D:\CVProject\CBAM-keras-master\prediction\prediction"
            self.shadow_path = os.path.join(root, split_list[-2], split_list[-1])
            self.shadow_img = cv_imread(self.shadow_path, mode="None")
            shadow_count, shadow_sum, edge_count = f_shadow(self.shadow_img)
            self.shadow_feature = shadow_count+shadow_sum+edge_count
            self.feature.extend(self.shadow_feature)
            self.shadow_feature_dict[self.filename] = self.shadow_feature
        # SNOW
        if 'snow' in self.feature_list:                     # 2*4*4=32
            snow_count, edge_count = f_snow(self.SrcImage, self.MaskImage)
            self.snow_feature = snow_count+edge_count
            self.feature.extend(self.snow_feature)
            self.snow_feature_dict[self.filename] = self.snow_feature

        # PCA
        if 'pca' in self.feature_list:                      # 864
            self.pca_feature = f_PCA(self.filename, self.pca_feature_dict)
            self.feature.extend(self.pca_feature)


    def getFeaturesFromFile(self):
        '''
        从文件中获取特征
        '''
        # Init all feature_dicts
        result_dict = {}

        time_feature_dict = None
        color_feature_dict = None
        texture_feature_dict = None
        lbp_feature_dict = None
        cloud_feature_dict = None
        haze_feature_dict = None
        contrast_feature_dict = None
        shadow_feature_dict = None
        snow_feature_dict = None
        # pac_feature_dict = None           # 提取模型使用另外的模型，需要训练

        if 'time' in self.feature_list:
            time_feature_dict = np.load(self.time_feature_file, allow_pickle=True).item()
            result_dict['time'] = time_feature_dict
        if 'color' in self.feature_list:
            color_feature_dict = np.load(self.color_feature_file, allow_pickle=True).item()
            result_dict['color'] = color_feature_dict
        if 'texture' in self.feature_list:
            texture_feature_dict = np.load(self.texture_feature_file, allow_pickle=True).item()
            result_dict['texture'] = texture_feature_dict
        if 'lbp' in self.feature_list:
            lbp_feature_dict = np.load(self.lbp_feature_file, allow_pickle=True).item()
            result_dict['lbp'] = lbp_feature_dict
        if 'cloud' in self.feature_list:
            cloud_feature_dict = np.load(self.cloud_feature_file, allow_pickle=True).item()
            result_dict['cloud'] = cloud_feature_dict
        if 'haze' in self.feature_list:
            haze_feature_dict = np.load(self.haze_feature_file, allow_pickle=True).item()
            result_dict['haze'] = haze_feature_dict
        if 'contrast' in self.feature_list:
            contrast_feature_dict = np.load(self.contrast_feature_file, allow_pickle=True).item()
            result_dict['contrast'] = contrast_feature_dict
        if 'shadow' in self.feature_list:
            shadow_feature_dict = np.load(self.shadow_feature_file, allow_pickle=True).item()
            result_dict['shadow'] = shadow_feature_dict
        if 'snow' in self.feature_list:
            snow_feature_dict = np.load(self.snow_feature_file, allow_pickle=True).item()
            result_dict['snow'] = snow_feature_dict
        if 'pca' in self.feature_list:
            # result_dict['pca'] = self.pca_feature_dict
            pca_feature_dict = {elem: np.squeeze(self.pca_feature_dict[elem][0], axis=0) for elem in self.pca_feature_dict}
            result_dict['pca'] = pca_feature_dict

        return result_dict


    def saveFeaturesToFile(self):
        if 'time' in self.feature_list:
            np.save(self.time_feature_file, self.time_feature_dict)
        if 'color' in self.feature_list:
            np.save(self.color_feature_file, self.color_feature_dict)
        if 'texture' in self.feature_list:
            np.save(self.texture_feature_file, self.texture_feature_dict)
        if 'lbp' in self.feature_list:
            np.save(self.lbp_feature_file, self.lbp_feature_dict)
        if 'cloud' in self.feature_list:
            np.save(self.cloud_feature_file, self.cloud_feature_dict)
        if 'haze' in self.feature_list:
            np.save(self.haze_feature_file, self.haze_feature_dict)
        if 'contrast' in self.feature_list:
            np.save(self.contrast_feature_file, self.contrast_feature_dict)
        if 'shadow' in self.feature_list:
            np.save(self.shadow_feature_file, self.shadow_feature_dict)
        if 'snow' in self.feature_list:
            np.save(self.snow_feature_file, self.snow_feature_dict)
        pass


    def combinNewFeatures(self, datalist):
        chosen_feature_dict = self.getFeaturesFromFile()
        # ==== feature info ====
        # e.g.
        # types : 10
        # type : ['time', 'color',....]
        # size : {'time': 2, 'color': 90, ...}
        # pos : {'time': [0, 2], 'color': [2, 92],...}
        # ======================
        result_features_dict = {}
        # feature number
        result_features_dict['types'] = len(self.feature_list)
        # feature list
        result_features_dict['type'] = self.feature_list
        # each feature size
        result_features_dict['size'] = {}
        # each feature pos
        result_features_dict['pos'] = {}
        result_features = []
        for elem in datalist:
            # elem : img filepath
            item_feature = []
            # cls : features type
            for cls in self.feature_list:
                if chosen_feature_dict.__contains__(cls):
                    if chosen_feature_dict[cls].__contains__(elem[0]):
                        item_feature.extend(chosen_feature_dict[cls][elem[0]])
                        if not result_features_dict['size'].__contains__(cls):
                            result_features_dict['size'][cls] = len(chosen_feature_dict[cls][elem[0]])
            if len(item_feature) > 0:
                item_feature.extend([int(elem[1])])
            result_features.append(item_feature)

        cur_pos = 0
        for elem in self.feature_list:
            result_features_dict['pos'][elem] = [cur_pos, cur_pos+result_features_dict['size'][elem]]
            cur_pos = cur_pos+result_features_dict['size'][elem]
        return result_features_dict, result_features


    def imshow(self, *args):

        if args[0] == DisplayType.TYPE_ALL:
            args = [DisplayType.TYPE_SRC,
                    DisplayType.TYPE_GRAY,
                    DisplayType.TYPE_DARK,
                    DisplayType.TYPE_TEXTURE,
                    DisplayType.TYPE_SYKMASK]

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

            if value == DisplayType.TYPE_SYKMASK:
                cv2.namedWindow("Sky Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Sky Mask", self.MaskImage*255)
        cv2.waitKey()


## ========== Functions for extracting features ==========


def f_haze(image):   # Haze feature         # Global features
    """ Haze feature
    :param image:
    :return: dark_image:DarkChannel image，f_vector:feature vector
    """

    # 1.resize the image into 512×512，see Ref[1];
    resize_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    # 2.get the dark channel image;
    dark_image = J_dark(resize_image, 4)    # The size of local patch is 8×8 in Ref[1]
    # 3.divide the image into some patches with different size, and get the median intensity
    # of each patch. (See Ref[1] for specific size of patches: 2×2，4×4，8×8 )
    f_vector = get_median_dark(dark_image, 2, 4, 8)     # 2,4,8 denote the partition size
    return dark_image, f_vector


def f_contrast(image):  # Contrast feature  # Global feature
    """
        :param image: input image
        :return: f_vector: output feature vector
    """
    # 1. convert the image from RGB color space to LCH;
    Lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lch_image_c = Lab2LCH_C(Lab_image)
    # 2. sort the saturation channel;
    pixels_count = lch_image_c.shape[0]*lch_image_c.shape[1]
    # reshape_lch_image_c = np.reshape(lch_image_c, (1, pixels_count))
    # sort_lch_image_c = np.sort(reshape_lch_image_c[0])
    reshape_lch_image_c = lch_image_c.flatten()
    sort_lch_image_c = np.sort(reshape_lch_image_c)

    # 3. get the 1/20,2/20,...,19/20 quantiles.(19 values in total)
    percentages = [x for x in np.arange(0.05, 1.00, 0.05)]
    v_percentage = []
    for v in percentages:
        v_percentage.append(sort_lch_image_c[int(v*pixels_count)])

    # 4. get the ratio of quantiles.
    f_vector = []
    for i in range(0, len(v_percentage)):  # get the ratio for any i>j, thus the final vector has 171 dim in total
        for j in range(0, i):
            if v_percentage[j] == 0:
                f_vector.append(v_percentage[i] / 0.0001)
            else:
                f_vector.append(v_percentage[i]/v_percentage[j])
    return f_vector


def f_texture(image):   # texture feature # global feature
    """
        Extract the texture feature by using a bank of Gabor filter.
        :param image: input image
        :return: f_vector: output feature vector
    """

    # 1.build a Gabor filter banks;
    filters = BuiltGaborFilter(4, [7, 9, 11, 13])     # see details in Ref[3]
    # 2.get the response of filters;
    res_images = getGabor(image, filters)
    # 3.build the feature vector;
    f_vector = []
    for res_image in res_images:
        data = np.array(res_image).flatten()
        f_vector.append(np.mean(data))                  # mean
        f_vector.append(np.var(data))                   # var
    return res_images, f_vector


def f_texture_in_mask(image, mask):  # texture feature # local feature
    """
        Extract the texture feature of sky region by using a bank of Gabor filter.
        :param image:input image
        :param mask: mask image of sky region
        :return:output feature vector
    """

    # 1.build a Gabor filter banks;
    filters = BuiltGaborFilter(4, [7, 9, 11, 13])     # see details in Ref[3]
    # 2.get the response of filters;
    res_images = getGabor(image, filters)
    # 3.build the feature vector;
    x_pixels, y_pixels = np.where(mask > 0)
    f_vector = []
    for res_image in res_images:
        data = [res_image[x_pixels[i], y_pixels[i]] for i in range(len(x_pixels))]
        f_vector.append(np.mean(data))                  # mean
        f_vector.append(np.var(data))                   # var
    return res_images, f_vector


def f_LBP(image):   # texture feature # global feature
    """
        Get the LBP features
        :param image: input image
        :return:  f_vector:LBP(local binary patten)
    """

    # 1. get LBP features
    LBP_image = getLBP(image, 3, 8)
    # 2. get the histogram of LBP
    res = cv2.calcHist([image], [0], None, [32], [0, 256])
    f_vector = np.array(res).flatten()
    f_vector = f_vector/(LBP_image.shape[0]*LBP_image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(f_vector)), f_vector)
    # pl.show()
    return LBP_image, f_vector


def f_LBP_in_mask(image, mask):     # texture feature # local feature
    """
        Get the LBP features of sky region
        :param image: input image
        :return:  f_vector:LBP
    """

    # 1. get LBP features
    LBP_image = getLBP(image, 3, 8)
    # 2. get the histogram of LBP
    res = cv2.calcHist([image], [0], mask, [32], [0, 256])
    f_vector = np.array(res).flatten()
    f_vector = f_vector/(LBP_image.shape[0]*LBP_image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(f_vector)), f_vector)
    # pl.show()
    return LBP_image, f_vector


def f_color(image):     # color feature # global feature
    """
        color feature: RGB histogram with 64 bins
        :param image: input image
        :return: f_vector: feature vector
    """
    # 1.get histogram
    if len(image.shape) > 2:        # Multi-channels
        b_hist = cv2.calcHist([image], [0], None, [64], [0, 256])         # 64 bins, see params in Ref[3]
        g_hist = cv2.calcHist([image], [1], None, [64], [0, 256])         # same as above
        r_hist = cv2.calcHist([image], [2], None, [64], [0, 256])         # same as above
        f_vector = np.array([b_hist, g_hist, r_hist]).flatten()
        f_vector = f_vector/(image.shape[0]*image.shape[1])
    else:                           # Single channel
        # f_vector = np.array(cv2.calcHist(image, [0], None, [64], [0.0, 255.0])).flatten()
        f_vector = np.array(cv2.calcHist([image], [0], None, [64], [0, 256])).flatten()
        f_vector = f_vector / (image.shape[0] * image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(b_hist)), b_hist, 'b')
    # pl.plot(range(len(g_hist)), g_hist, 'g')
    # pl.plot(range(len(r_hist)), r_hist, 'r')
    # pl.show()
    return f_vector

def f_color_in_mask(image, mask):
    """
        [This version of implementation is deprecated]
        color feature in sky region: RGB histogram with 64 bins
        :param image:   input image
        :param mask:    mask image of sky region
        :return:        feature vector
    """
    import warnings
    warnings.warn("This version of color feature extraction is deprecated", DeprecationWarning)
    # 1. get histogram
    if len(image.shape) > 2:        # Multi-channels
        b_hist = cv2.calcHist([image], [0], mask, [64], [0, 256])         # 64 bins, see params in Ref[3]
        g_hist = cv2.calcHist([image], [1], mask, [64], [0, 256])         # same as above
        r_hist = cv2.calcHist([image], [2], mask, [64], [0, 256])         # same as above
        f_vector = np.array([b_hist, g_hist, r_hist]).flatten()
        f_vector = f_vector/(image.shape[0]*image.shape[1])
    else:                           # Single channel
        # f_vector = np.array(cv2.calcHist(image, [0], None, [64], [0.0, 255.0])).flatten()
        f_vector = np.array(cv2.calcHist([image], [0], None, [64], [0, 256])).flatten()
        f_vector = f_vector / (image.shape[0] * image.shape[1])
    # pl.figure(1)
    # pl.plot(range(len(b_hist)), b_hist, 'b')
    # pl.plot(range(len(g_hist)), g_hist, 'g')
    # pl.plot(range(len(r_hist)), r_hist, 'r')
    # pl.show()
    return f_vector


def f_color_in_mask_v2(image, mask):
    """
        color feature in sky region: RGB histogram with 64 bins
        :param image:   input image
        :param mask:    mask of sky region
        :return:        color feature vector
    """

    # 1.1 covert the image from the RGB color space to LAB
    # image_lab = RGB2Lab(image)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # 1.2 covert the image from the RGB color space to HSI
    # image_hsi = RGB2HSI(image)
    image_hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    mask_count = np.count_nonzero(mask > 0)

    bins_rgb, bins_lab, bins_hsi = 10, 10, 10
    # 2. get the color histogram

    if len(image.shape) > 2:        # Multi-channels
        b_hist = cv2.calcHist([image], [0], mask, [bins_rgb], [0, 256])         # 64 bins, see params in Ref[3]
        g_hist = cv2.calcHist([image], [1], mask, [bins_rgb], [0, 256])         # same as above
        r_hist = cv2.calcHist([image], [2], mask, [bins_rgb], [0, 256])         # same as above

        # L:[0,100], A,B:[-127,127] # change the value range to [0,255] for better viewing
        lab_l_hist = cv2.calcHist([image_lab], [0], mask, [bins_lab], [0, 255])
        lab_a_hist = cv2.calcHist([image_lab], [1], mask, [bins_lab], [0, 255])
        lab_b_hist = cv2.calcHist([image_lab], [2], mask, [bins_lab], [0, 255])

        hsi_h_hist = cv2.calcHist([image_hsi], [0], mask, [bins_hsi], [0, 255])
        hsi_s_hist = cv2.calcHist([image_hsi], [1], mask, [bins_hsi], [0, 255])
        hsi_i_hist = cv2.calcHist([image_hsi], [2], mask, [bins_hsi], [0, 255])

        f_vector = np.array([b_hist, g_hist, r_hist, lab_l_hist, lab_a_hist, lab_b_hist, hsi_h_hist, hsi_s_hist, hsi_i_hist]).flatten()
        # f_vector = f_vector/(image.shape[0]*image.shape[1])
        f_vector = f_vector / mask_count


    else:           # Single channel
        # f_vector = np.array(cv2.calcHist(image, [0], None, [64], [0.0, 255.0])).flatten()
        f_vector = np.array(cv2.calcHist([image], [0], None, [10], [0, 256])).flatten()
        f_vector = f_vector / mask_count
    # pl.figure(1)
    # pl.plot(range(len(b_hist)), b_hist, 'b')
    # pl.plot(range(len(g_hist)), g_hist, 'g')
    # pl.plot(range(len(r_hist)), r_hist, 'r')
    # pl.show()
    return f_vector


def f_SIFT(image):
    """
        :param image:
        :return:
    """

    # sift = cv2.SIFT()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kps, features = sift.detectAndCompute(gray, None)
    # print(kps)
    # print(features)
    # print(image.shape)
    return features


def f_cloud_in_mask(image, mask):
    """
        get cloud features from sky region
        :param image: input image
        :return: f_vector: feature vector
    """

    # 1. get RO, SA, EGD features (see details in Ref[3])
    ro_image = getRO(image)
    sa_image = getSA(image)
    ed_image = getEGD(image)

    # ro_feature = get_hist(ro_image, mask, [-1, 1])
    ro_feature = get_hist(ro_image, mask, [-0.3, 0.3])
    sa_feature = get_hist(sa_image, mask, [0, 1])
    # ed_feature = get_hist(ed_image, mask, [0, 208])
    ed_feature = get_hist(ed_image, mask, [0, 50])
    f_vector = np.concatenate((ro_feature, sa_feature, ed_feature), axis=0)             # list 拼接
    return f_vector


def f_time(filepath):
    """
        :param: filepath, path of image file
        :return: time feature vector
    """

    time_str = get_time_str(filepath)
    current_time = datetime.datetime.strptime(time_str[0][0:14], "%Y%m%d%H%M%S")
    d_number = current_time.timetuple().tm_yday
    s_number = current_time.timetuple().tm_hour*3600 + current_time.timetuple().tm_min*60 + \
               current_time.timetuple().tm_sec
    if is_leap(current_time.timetuple().tm_year):
        d_number = d_number / 366
    else:
        d_number = d_number / 365
    s_number = s_number / (24*3600)
    return [d_number, s_number]


def f_shadow_bak(image):
    """
        get shadow feature via MTMT-Net(see Ref[])
        :param: image
    """

    gray_im = image
    edge_im = cv2.Canny(gray_im, 50, 200, 255)
    shadow_count = np.count_nonzero(gray_im)
    shadow_sum = np.sum(image)
    edge_count = np.count_nonzero(edge_im)
    if shadow_count > 0:
        shadow_sum = shadow_sum/shadow_count/np.max(image)/3
    else:
        shadow_sum = 0
    shadow_count = shadow_count / (image.shape[0] * image.shape[1])
    return shadow_count, shadow_sum, edge_count


def f_shadow(image):
    """
        阴影特征
    """

    gray_im = image
    edge_im = cv2.Canny(gray_im, 50, 200, 255)

    devide_block_count = 4      # devide_block_count setting

    shadow_count = []
    shadow_sum = []
    edge_count = []

    rows, cols = image.shape
    r = int(rows/devide_block_count)    # row block index
    c = int(cols/devide_block_count)    # column block index
    for i in range(devide_block_count):
        for j in range(devide_block_count):
            # sorted_image = sorted(image[r*i:r*(i+1), c*j:c*(j+1)].reshape(1, r*c)[0, :])
            shadow_block_count = np.count_nonzero(gray_im[r*i:r*(i+1), c*j:c*(j+1)])
            shadow_count.append(shadow_block_count)
            shadow_block_sum = np.sum(gray_im[r*i:r*(i+1), c*j:c*(j+1)])
            shadow_sum.append(shadow_block_sum)
            edge_block_count = np.count_nonzero(edge_im[r*i:r*(i+1), c*j:c*(j+1)])
            edge_count.append(edge_block_count)
    return shadow_count, shadow_sum, edge_count


def f_snow(image, mask):
    """
        snow feature;
    """

    Reverse_MaskImage = np.ones(mask.shape) - mask
    rMask_count = np.count_nonzero(Reverse_MaskImage)
    mask_im = image.copy()
    mask_im[:, :, 0] = image[:, :, 0]*Reverse_MaskImage
    mask_im[:, :, 1] = image[:, :, 1]*Reverse_MaskImage
    mask_im[:, :, 2] = image[:, :, 2]*Reverse_MaskImage

    avg_r = np.sum(mask_im[:, :, 0])/rMask_count
    avg_g = np.sum(mask_im[:, :, 1])/rMask_count
    avg_b = np.sum(mask_im[:, :, 2])/rMask_count

    # avg_r = np.percentile(mask_im[np.where(mask_im[:, :, 0] > 0)], 50)
    # avg_g = np.percentile(mask_im[np.where(mask_im[:, :, 1] > 0)], 50)
    # avg_b = np.percentile(mask_im[np.where(mask_im[:, :, 2] > 0)], 50)

    mask_v = mask_im[np.where(mask_im>0)]

    threshold = (np.max(mask_im) - avg_r)*0.7

    snow_mask = Reverse_MaskImage
    snow_mask = snow_mask * ((mask_im[:, :, 0] - avg_r) > threshold)
    snow_mask = snow_mask * ((mask_im[:, :, 1] - avg_g) > threshold)
    snow_mask = snow_mask * ((mask_im[:, :, 2] - avg_b) > threshold)
    # #
    # cv2.imshow("image", image)
    # cv2.imshow("snow_mask", snow_mask)
    # cv2.waitKey()

    snow_mask_uint8 = snow_mask.astype(np.uint8)*255
    edge = cv2.Canny(snow_mask_uint8, 50, 200, 255)

    devide_block_count = 4      # devide_block_count setting

    snow_count = []
    edge_count = []


    rows, cols = snow_mask.shape
    r = int(rows/devide_block_count)    # row block index
    c = int(cols/devide_block_count)    # column block index
    for i in range(devide_block_count):
        for j in range(devide_block_count):
            snow_block_count = np.count_nonzero(snow_mask[r*i:r*(i+1), c*j:c*(j+1)])
            snow_count.append(snow_block_count)
            edge_block_count = np.count_nonzero(edge[r*i:r*(i+1), c*j:c*(j+1)])
            edge_count.append(edge_block_count)
    return snow_count, edge_count


def f_PCA(filepath, feature_dict={}):
    """
        features extracted by PCANet(Read features from saved npy file)
    """

    if feature_dict.__contains__(filepath):
        return np.squeeze(feature_dict[filepath][0], axis=0)
    return None

## Utility function

def is_leap(year):
    """
        判断是否为闰年
        :param year:
        :return:
    """

    if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
        return True
    return False


def get_hist(image, mask, value_range):
    """
        计算特征图像的直方图
        :param image:
        :param mask:
        :param value_range:
        :return:
    """

    # flatten_image = image.flatten()
    flatten_image = []
    x, y = np.where(mask > 0)
    for i in range(len(x)):
        flatten_image.append(image[x[i], y[i]])
    f_vector, bin_edges = np.histogram(flatten_image, bins=64, range=value_range)
    f_vector = f_vector/len(flatten_image)
    # f_vector = f_vector/(image.shape[0]*image.shape[1])
    # f_vector = np.array(cv2.calcHist([image], [0], mask, [64], value_range)).flatten()
    return f_vector


def J_dark(image, r):
    """
        区域通道最小值滤波
        :param image: 待分析图像(3通道)
        :param r: 分析区域半径
        :return: DarkChannel图像
    """

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
    """
        获取不同图像划分方案下，区域中DarkChannel中位数
        :param image: 待分析图像(暗通道图像)
        :param args: 图像划分方案集(2:2×2，4:4×4，8:8×8)
        :return: res:特征向量
    """

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
    """
        Lab 色彩空间 转 LCH色彩空间(为减少计算量，仅计算C通道，即饱和度通道)
        :param image: 待转换Lab色彩空间的图像 (3维)
        :return: cvt_image: 转换完成的LCH色彩空间图像 （1维）
        色彩空间转换公式参考Ref[2]，其中：L = L; C = sqrt(a^2 + b^2); H = atan(b/a)
        e.g. Lab = [52, -14, -32] → LCH = [52 ， 34.93 ， 246.37°] 参考工具地址：http://www.colortell.com/labto （色彩管理网）
    """

    # 1.L = l
    # 2.C = sqrt(a^2 + b^2)
    cvt_image = np.sqrt(np.array(image[:, :, 1]) ** 2 + np.array(image[:, :, 2]) ** 2)
    # 3.H = atan(b/a)
    return cvt_image


def Lab2LCH_C(image):
    """
        Lab 色彩空间 转 LCH色彩空间(为减少计算量，仅计算C通道，即饱和度通道)
        :param image: 待转换Lab色彩空间的图像 (3维)
        :return: cvt_image: 转换完成的LCH色彩空间图像 （1维）
        色彩空间转换公式参考Ref[2]，其中：L = L; C = sqrt(a^2 + b^2); H = atan(b/a)
        e.g. Lab = [52, -14, -32] → LCH = [52 ， 34.93 ， 246.37°] 参考工具地址：http://www.colortell.com/labto （色彩管理网）
    """

    # 1.L = l
    # 2.C = sqrt(a^2 + b^2)
    cvt_image = np.sqrt(np.array(image[:, :, 1]) ** 2 + np.array(image[:, :, 2]) ** 2)
    # 3.H = atan(b/a)
    return cvt_image


def __rgb2xyz__(pixel):
    """
        BGR色彩空间转XYZ
        :param pixel:
        :return:
    """
    # b, g, r = pixel[:, :, 0], pixel[:, :, 1], pixel[:, :, 2]
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])

    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    """
        XYZ空间转Lab空间
        :param xyz: 像素xyz空间下的值
        :return: 返回Lab空间下的值
    """
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab_pixel(pixel):
    """
        RGB空间转Lab空间，单像素
        :param pixel: RGB空间像素值，格式：[G,B,R]
        :return: 返回Lab空间下的值
    """
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab


def RGB2Lab(img):
    """
        RGB空间转Lab空间，整幅图像
        :param img:
        :return:lab图像
        lab图像值域：
            L:[0,100]
            A,B:[-127,127]
    """
    w = img.shape[0]
    h = img.shape[1]
    lab = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            Lab = RGB2Lab_pixel(img[i, j])
            lab[i, j] = (Lab[0], Lab[1], Lab[2])
    return lab


def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
# endregion

def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb


def RGB2HSI(rgb_img):
    """
        这是将RGB彩色图像转化为HSI图像的函数
        :param rgm_img: RGB彩色图像
        :return: HSI图像
        值域：
            # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    """
    # 保存原始图像的行列数
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    # 对原始图像进行复制
    hsi_img = rgb_img.copy()
    # 对图像进行通道拆分
    B, G, R = cv2.split(rgb_img)
    # 把通道归一化到[0,1]
    [B, G, R] = [ i/ 255.0 for i in ([B, G, R])]
    H = np.zeros((row, col))    # 定义H通道
    I = (R + G + B) / 3.0       # 计算I通道
    S = np.zeros((row, col))      # 定义S通道
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   # 计算夹角
        h = np.zeros(col)               # 定义临时数组
        # den>0且G>=B的元素h赋值为thetha
        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
        # den>0且G<=B的元素h赋值为thetha
        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
        # den<0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h/(2*np.pi)      # 弧度化后赋值给H通道
    #计算S通道
    for i in range(row):
        min = []
        # 找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j], G[i][j], R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        # 计算S通道
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        # I为0的值直接赋值0
        S[i][R[i]+B[i]+G[i] == 0] = 0
    # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_img[:, :, 0] = H*255
    hsi_img[:, :, 1] = S*255
    hsi_img[:, :, 2] = I*255
    return hsi_img


def HSI2RGB(hsi_img):
    """
        这是将HSI图像转化为RGB图像的函数
        :param hsi_img: HSI彩色图像
        :return: RGB图像
    """
    # 保存原始图像的行列数
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    #对原始图像进行复制
    rgb_img = hsi_img.copy()
    #对图像进行通道拆分
    H,S,I = cv2.split(hsi_img)
    #把通道归一化到[0,1]
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
    R,G,B = H,S,I
    for i in range(row):
        h = H[i]*2*np.pi
        #H大于等于0小于120度时
        a1 = h >=0
        a2 = h < 2*np.pi/3
        a = a1 & a2         #第一种情况的花式索引
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i]*(1+S[i]*np.cos(h)/tmp)
        g = 3*I[i]-r-b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        #H大于等于120度小于240度
        a1 = h >= 2*np.pi/3
        a2 = h < 4*np.pi/3
        a = a1 & a2         #第二种情况的花式索引
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        #H大于等于240度小于360度
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2             #第三种情况的花式索引
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1-S[i])
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255
    return rgb_img


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
    return res


def getLBP(image, radius, n_points):
    """
        获取LBP特征值图像
        :param image: 待分析图像
        :param radius: 计算半径
        :param n_points: 进制位数(8，16，32)
        :return: res:编码结果图像
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = local_binary_pattern(gray_image, radius*n_points, radius)
    # pl.imshow(res, cmap='gray')
    # pl.show()
    return res


def getRO(image):
    """
        cloud feature：normalized blue/red ratio (see ref.[4])
            defination ：ro(s) = (b - r)/(b + r), Clear sky appears blue with high ro and cloud appears white or gray
                     with low ro
        :param image:
        :return:
    """

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
    """
        cloud feature：saturation (see ref.[4])
            defination ：sa(s) = 1 - 3*min(r,g,b)/(r+g+b)
        :param image:
        :returns:
    """

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
    """
    cloud feature:EGD (see ref.[4])
        defination ：ed(s) = sqrt(r^2+g^2+b^2-((r+g+b)^2)/3)
    :param image:
    :return:
    """
    # 注意：imdecode读取的是rgb，不是bgr(cv2.imread读入的时bgr)
    row, col, depth = image.shape
    ed_image = np.zeros((row, col), np.float)
    (B, G, R) = cv2.split(image)
    for i in range(row):
        for j in range(col):
            ed_image[i, j] = math.sqrt(int(R[i, j])**2 + int(G[i, j])**2 + int(B[i, j])**2 - (int((R[i, j]) + int(G[i, j]) + int(B[i, j]))**2)/3)
    return ed_image


def get_time_str(filepath):
    """
    :param filepath:
    :return:
    """
    path, name = os.path.split(filepath)
    split_str = name.split("_")
    time_str = [elem for elem in split_str if len(elem) == 17 and elem[0:2] == "20"]
    return time_str


def cv_imread(filePath, mode="RGB"):
    """
    To read the image in a file path containing chinese characters
    :param filePath:file path containing chinese characters.
    :param mode:color space. e.g. "RGB", "BGR"
    :return:
    """
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    # "cv_img" is an image in the BGR color space, if you want to output the image in the RGB color space, then convert the image.
    if mode is "BGR":
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
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


def get_filename_label_all(txtfilename, type='txt', data_split = ['train', 'valid', 'test']):
    import json
    datalist = []
    with open(txtfilename, 'r') as doc:
        if type == 'txt':
            lines = doc.readlines()
            for line in lines:
                line_str = line.strip("\n").split(" ")
                datalist.append(line_str)
        elif type == 'json':
            json_data = json.load(doc)
            if 'train' in data_split:
                datalist.extend(json_data['train_data'])
            if 'valid' in data_split:
                datalist.extend(json_data['valid_data'])
            if 'test' in data_split:
                datalist.extend(json_data['test_data'])

    return datalist


def get_selected_features(feature_list=None, datalist=None, savename=None):
    solver = WeatherFeature(feature_list=feature_list)
    training_data = []
    pbar = tqdm.tqdm(total=len(datalist))
    for elem in datalist:
        pbar.update(1)
        solver.set_image(elem[0])
        solver.getFeatures()
        training_data.append(np.concatenate([solver.feature, [int(elem[1])]], axis=0))
    pbar.close()
    print(training_data[0])
    print(training_data[0].shape)


    io.savemat(savename, {'array': training_data})
    print("特征提取完成！")
    training_data = io.loadmat(savename)
    print(training_data["array"])
    print(len(training_data["array"][0]))


def get_pca_feature_dict(filepath):
    return np.load(filepath, allow_pickle=True).item()

# ================ TEST FUNCTIONS =================

def extract_features_directly():
    """ Extract features and save them in the mat files directly """
    # chosen feature types
    feature_list = ['time', 'color', 'texture', 'lbp', 'cloud', 'haze', 'contrast', 'shadow', 'snow', 'pca']

    # solver
    solver = WeatherFeature(feature_list=feature_list)

    # input data
    datalist = get_filename_label_all(r"D:\CVProject\CBAM-keras-master\data\DataInfo.json", type='json')

    # init output data
    training_data = []

    # bar
    pbar = tqdm.tqdm(total=len(datalist))

    for i, elem in enumerate(datalist[:20]):
        pbar.update(1)
        solver.set_image(elem[0])
        solver.getFeatures()

        # solver.imshow(DisplayType.TYPE_ALL)
        # solver.getFeaturesToSave()

        training_data.append(np.concatenate([solver.feature, [int(elem[1])]], axis=0))

    pbar.close()

    # print(training_data[0])
    # print(training_data[0].shape)
    # print(training_data[1])
    # print(training_data[1].shape)

    # === Feature save and load ===
    # io.savemat(r"/results/features_with_pca_mat.mat", {'array': training_data})
    print("Extraction accomplised！")
    # training_data = io.loadmat(r"/results/features_with_pca_mat.mat")


def extract_features_to_npy_files():
    """ Extract features and save them in the npy files """
    # Types of features
    # include PCA
    # feature_list = ['time', 'color', 'texture', 'lbp', 'cloud', 'haze', 'contrast', 'shadow', 'snow', 'pca']  # item 5
    # not include PCA
    # feature_list = ['time', 'color', 'intensity', 'texture', 'lbp', 'cloud', 'haze', 'contrast', 'shadow', 'snow']  # item 5
    feature_list = ['shadow', 'snow']


    # # ===== 分特征保存为文件（方便训练、测试、进一步处理、以及组合分类）=====
    solver = WeatherFeature(feature_list=feature_list,
                            save_mode='file',
                            # feature_files_saved_path=r"D:\CVProject\CBAM-keras-master\handcraft\each_feature"
                            feature_files_saved_path=r"../../results"
                            )

    datalist = get_filename_label_all(r"D:\CVProject\CBAM-keras-master\data\DataInfo.json", type='json')

    training_data = []
    pbar = tqdm.tqdm(total=len(datalist))
    for i, elem in enumerate(datalist[:20]):
        pbar.update(1)
        solver.set_image(elem[0])
        solver.getFeaturesToSave()
        # print(i, len(solver.feature))
        training_data.append(np.concatenate([solver.feature, [int(elem[1])]], axis=0))
    pbar.close()

    # # ===== 保存特征为文件 =====
    # # == 生成之前需要调用getFeaturesToSave()以获得特征字典 ==
    solver.saveFeaturesToFile()
    print("Extraction accomplised！")

def extract_features_from_npy_files():
    """ 
        Load features from a saved file and combine them to bulid a new feature in npy file
    """
    # ====== Extract different types of features, and save them in different files ======
    # include PCA
    # feature_list = ['time', 'color', 'texture', 'lbp', 'cloud', 'haze', 'contrast', 'shadow', 'snow', 'pca']  # item 5
    # feature_list = ['lbp']
    feature_list = ['shadow', 'snow']

    # # ===== Save features in file, for better =====
    solver = WeatherFeature(feature_list=feature_list,
                            save_mode='file',
                            # feature_files_saved_path=r"D:\CVProject\CBAM-keras-master\handcraft\each_feature"
                            feature_files_saved_path=r"../../results"
                            )
    # # === Dataset info ===
    # datalist4train = get_filename_label_all(r"D:\CVProject\CBAM-keras-master\data\DataInfo.json", type='json',
    #                                         data_split=['train', 'valid'])
    # datalist4test = get_filename_label_all(r"D:\CVProject\CBAM-keras-master\data\DataInfo.json", type='json',
    #                                        data_split=['test'])
    datalist = get_filename_label_all(r"D:\CVProject\CBAM-keras-master\data\DataInfo.json", type='json')
    # # === Combine features ===
    disp_features_dict, disp_data = solver.combinNewFeatures(datalist)
    # print(disp_data)
    # result_feature_dict displaying
    for elem in disp_features_dict:
        print(elem, ":", disp_features_dict[elem])
    # # elem type displaying
    # for elem in disp_data[0]:
    #     print(elem, ":", type(elem))
    print(len(disp_data), len(disp_data[0]))

    # rcd = 0
    # for i, elem in enumerate(disp_data):
    #     if len(elem) - rcd > 0:
    #         print(i, len(elem))
    #     rcd = len(elem)

    # === Feature save and load ===
    # io.savemat(r"/results/features_with_pca_file_npy.mat", {'array': disp_data})
    # training_data = io.loadmat(r"/results/features_with_pca_file_npy.mat")

    print("Extraction accomplised！")


if __name__ == '__main__':

    # # === Extract and save features in a mat file ===
    # extract_features_directly()
    # # === Extract and save features in few seperated npy files ===
    extract_features_to_npy_files()
    # # === Combin features with seperated npy files ===
    # extract_features_from_npy_files()



    # ===== Reference =====
    # [1] Lu C, Lin D, Jia J, et al. Two-Class Weather Classification[C]//
    #     IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 2014:3718-3725.
    # [2] LAB与LCH颜色空间及其应用：https://wenku.baidu.com/view/42f8f03ab5daa58da0116c175f0e7cd1842518a0.html
    # [3] Chu W T, Zheng X Y, Ding D S. Camera as weather sensor: Estimating weather information from single images[J].
    #     Journal of Visual Communication & Image Representation, 2017, 46:233-249.
    # [4] Q. Li, W. Lu, J. Yang and J. Z. Wang (2012). "Thin Cloud Detection of All-Sky Images Using Markov Random
    #     Fields" IEEE Geoscience and Remote Sensing Letters 9:417-421
    # [5] CHAN T-H, JIA K, GAO S, et al. PCANet: A simple deep learning baseline for image classification? [J].
    #     IEEE transactions on image processing, 2015, 24(12): 5017-32.
    # [6] Chen Z, Zhu L, Wan L, et al. A Multi-task Mean Teacher for Semi-supervised Shadow Detection[C]. IEEE
    #     Conference on Computer Vision and Pattern Recognition, CVPR, 2020: 5611-5620.
    # =========================