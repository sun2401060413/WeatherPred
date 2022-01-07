# -*- coding:utf-8 -*-
# WeatherClassify: SkyDetection: Version1.0 by SunZhu, 2018.09.11 in Chang'an University
'''Sky Region extraction'''
import os
import cv2
import numpy as np
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.externals import joblib
# from sklearn.datasets import load_iris
# from sklearn import cross_validation,metrics

# filepath = "E:/DataSet/SkyDetection/MaskImage"
# savepath = "E:/DataSet/SkyDetection/MaskImage/bw_MaskImage"

# Srcfilepath = "E:/DataSet/SkyDetection/MaskImage/processed"
Srcfilepath = r"D:\CVProject\CBAM-keras-master\weather_data\SkyDetection\MaskImage\processed"
# Bwfilepath = "E:/DataSet/SkyDetection/MaskImage/bw_MaskImage"
Bwfilepath = r"D:\CVProject\CBAM-keras-master\weather_data\SkyDetection\MaskImage\bw_MaskImage"
# savepath = "E:/DataSet/SkyDetection/MaskImage/patches"
savepath = r"D:\CVProject\CBAM-keras-master\weather_data\SkyDetection\MaskImage\patches"

filelist = os.listdir(savepath)
patch_size = 15
training_ratio = 0.8

# main function    
# clf = joblib.load("E:/DataSet/SkyDetection/MaskImage/RandomForestModel.m") #调用
clf = joblib.load(r"D:\CVProject\CBAM-keras-master\weather_data\SkyDetection\MaskImage\RandomForestModel.m") #调用
def version():
    print("WeatherClassify: SkyDetection: Version1.0 by SunZhu, 2018.09.11 in Chang'an University")

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
def get_bw_image():
    for elem in filelist:
        (filename, extension) = os.path.splitext(elem)
        if extension == ".jpg":
            path_str = filepath+'/'+elem
            img=cv_imread(path_str)
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,bw_img = cv2.threshold(gray_img,128,255,cv2.THRESH_BINARY)
            # cv2.imwrite(savepath+'/'+elem,bw_img)
            cv2.imencode('.jpg', bw_img)[1].tofile(savepath+'/'+elem)
            print(elem)
def get_patch_image(): 
    for elem in filelist:
        (filename,extension) = os.path.splitext(elem)
        if extension == ".png":
            src_path_str = Srcfilepath+'/'+elem
            bw_path_str = Bwfilepath+'/'+filename+'.jpg'
            src_img=cv_imread(src_path_str)
            bw_img = cv_imread(bw_path_str)
            # cv2.imshow("src_img",src_img)
            # cv2.imshow("bw_img",bw_img)
            # cv2.waitKey()
            [rows,cols,channels] = src_img.shape
            print(rows,cols,channels)
            patch_count_row = int(rows/15)
            patch_count_col = int(cols/15)
            
            for i in range(patch_count_row):
                for j in range(patch_count_col):
                    src_crop_img = src_img[15*i:15*(i+1),15*j:15*(j+1)]
                    bw_crop_img = bw_img[15*i:15*(i+1),15*j:15*(j+1)]
                    bw_sum = np.sum(np.reshape(bw_crop_img,(bw_crop_img.size,)))/255
                    if bw_sum>255/3:
                        save_filename = savepath+"/"+filename+"_"+str(15*i+8)+"_"+str(15*j+8)+"_1"+".jpg"
                    else:
                        save_filename = savepath+"/"+filename+"_"+str(15*i+8)+"_"+str(15*j+8)+"_0"+".jpg"
                    print(save_filename)
                    cv2.imencode('.jpg', src_crop_img)[1].tofile(save_filename)
def get_image_and_label(filename):
    # 34020000000270000055_20161123074425174_2501_小雨_多云_98_98_0.jpg
    (patch_image_name,patch_image_extension) = os.path.splitext(filename)
    splited_patch_image_name = patch_image_name.split("_")
    label = splited_patch_image_name[-1]
    y = splited_patch_image_name[-2]
    x = splited_patch_image_name[-3]
    tmp_str = ""
    for elem in splited_patch_image_name[0:-4]:
        tmp_str = tmp_str + elem + "_"
    imagename = tmp_str+splited_patch_image_name[-4]
    return imagename, x, y, label
def get_image_feature(filename):
    # imagename,x,y,label = get_image_and_label(filename)
    img = cv_imread(savepath+"/"+filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kps, features = sift.detectAndCompute(gray_img, None)

    # sift features
    if np.any(features):
        sift_feature = features[0]
    else:
        sift_feature = np.zeros(128,)
        
    mean_R = np.mean(img[:, :, 0])
    mean_G = np.mean(img[:, :, 1])
    mean_B = np.mean(img[:, :, 2])
    
    color_feature = [mean_R, mean_G, mean_B]
    
    image_feature = np.append(sift_feature, color_feature)
    # image_feature = np.append(image_feature,[x,y])
    # print(image_feature)
    # cv2.imshow("test",img)
    # cv2.waitKey()
    return image_feature
def get_image_feature_v2(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kps,features = sift.detectAndCompute(gray_img, None)

    # sift features
    if np.any(features):
        sift_feature = features[0]
    else:
        sift_feature = np.zeros(128,)
        
    mean_R = np.mean(img[:, :, 0])
    mean_G = np.mean(img[:, :, 1])
    mean_B = np.mean(img[:, :, 2])
    
    color_feature = [mean_R, mean_G, mean_B]
    
    image_feature = np.append(sift_feature, color_feature)

    return image_feature
def get_feature_and_label(training_samples,testing_samples):
    training_feature_and_label = []
    testing_feature_and_label = []
    for elem in training_samples:
        imagename, x, y, label = get_image_and_label(elem)
        image_feature = get_image_feature(elem)
        output_feature = np.append(image_feature, [x, y])
        training_feature_and_label.append([output_feature, label])
    for elem in testing_samples:
        imagename, x, y, label = get_image_and_label(elem)
        image_feature = get_image_feature(elem)
        output_feature = np.append(image_feature, [x, y])
        testing_feature_and_label.append([output_feature, label])
    return training_feature_and_label, testing_feature_and_label
def get_data():
    np.random.shuffle(filelist)
    training_count = int(len(filelist)*training_ratio)
    training_samples = filelist[0:training_count]
    testing_samples = filelist[training_count:len(filelist)]

    training_feature_and_label,testing_feature_and_label= get_feature_and_label(training_samples,testing_samples)
    print(training_feature_and_label)
    print(testing_feature_and_label)

    np.save("E:/DataSet/SkyDetection/MaskImage/training_feature_and_label.npy", training_feature_and_label)
    np.save("E:/DataSet/SkyDetection/MaskImage/testing_feature_and_label.npy", testing_feature_and_label)


def get_proformance_score(input_list):
    '''
    分类评估指标
    混淆矩阵：
    -------------------------------------------------
    |   true_label  |   Positive    |   Negative    |
    |pred_labe -----|-------------------------------|
    |   Postive     |       TP      |       FP      |
    |   Negtive     |       FN      |       TN      |
    -------------------------------------------------
    
    TP（true positive）：表示样本的真实类别为正，最后预测得到的结果也为正；
    FP（false positive）：表示样本的真实类别为负，最后预测得到的结果却为正；
    FN（false negative）：表示样本的真实类别为正，最后预测得到的结果却为负；
    TN（true negative）：表示样本的真实类别为负，最后预测得到的结果也为负.
    
    Accuracy = (TP+TN)/(TP+FP+TN+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    
    Accuracy：   表示预测结果的精确度，预测正确的样本数除以总样本数。
    Precision：  准确率，表示预测结果中，预测为正样本的样本中，正确预测为正样本的概率；
    Recall：     召回率，表示在原始样本的正样本中，最后被正确预测为正样本的概率；
    Specificity：常常称作特异性，它研究的样本集是原始样本中的负样本，表示的是在这些负样本中最后被正确预测为负样本的概率。
    
    另外：
    F1-score：   表示的是precision和recall的调和平均评估指标。
    MCC：        Matthews correlation coefficient
    
    '''
    TP, FP, FN, TN = 0, 0, 0, 0
    for elem in input_list:
        y_actual = elem[0]
        y_pred = elem[1]
        if y_actual == 1 and y_pred == 1:
            TP+=1
        if y_actual == 1 and y_pred == 0:
            FN+=1
        if y_actual == 0 and y_pred == 1:
            FP+=1
        if y_actual == 0 and y_pred == 0:
            TN+=1
    Accuracy = (TP+TN)/(TP+FP+TN+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    F1_score = (2*Recall*Precision)/(Recall+Precision)
    MCC = (TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return Accuracy, Precision, Recall, Specificity, F1_score, MCC


def main_func_v1():
    print("Reading training data and testing data ...")
    training_feature_and_label = np.load("E:/DataSet/SkyDetection/MaskImage/training_feature_and_label.npy")
    testing_feature_and_label = np.load("E:/DataSet/SkyDetection/MaskImage/testing_feature_and_label.npy")

    training_data = [list(elem[0]) for elem in training_feature_and_label]
    training_label = training_feature_and_label[:,1]

    testing_data = [list(elem[0]) for elem in testing_feature_and_label]
    testing_label = testing_feature_and_label[:,1]

    print("Data reading is accomplished!")
    
    print("Building random forest classifier ...")
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(training_data,training_label)


    # Saved model to disk
    print("Saving model ...")
    joblib.dump(rf, "E:/DataSet/SkyDetection/MaskImage/RandomForestModel.m") #存储
    clf = joblib.load("E:/DataSet/SkyDetection/MaskImage/RandomForestModel.m") #调用
    print("model is saved!")
    prediction = clf.predict(testing_data)
    prediction = [int(round(elem)) for elem in prediction]
    label = [int(elem) for elem in testing_label]

    result_matrix = zip(prediction,label)
    # for i in range(len(prediction)):
        # print("prediction",[i],":",prediction[i],"label:",testing_label[i])
    # print(list(result_matrix))

    Accuracy,Precision,Recall,Specificity,F1_score,MCC = get_proformance_score(result_matrix)
    print("training data size:", len(training_label))
    print("testing data size:", len(testing_label))
    print("Accuracy:", Accuracy)
    print("Precision:", Precision)
    print("Recall:", Recall)
    print("Specificity:", Specificity)
    print("F1_score:", F1_score)
    print("MCC:", MCC)
def extend_img(img):
    '''
    SIFT特征提取需要最小15*15的图像块，为了避免部分像素无法被完整裁剪为图像块，这里对图像进行了边界扩展。
    高和宽扩展为原图像尺寸的最小整数倍。
    '''
    [row,col,depth] = img.shape
    # print(row,col,depth)

    ceil_row_count = math.ceil(row/15)
    ceil_col_count = math.ceil(col/15)
    floor_row_count = math.floor(row/15)
    floor_col_count = math.floor(col/15)

    extend_img = np.zeros((ceil_row_count*15,ceil_col_count*15,3), np.uint8)
    extend_img[0:row, 0:col, :] = img
    if ceil_row_count != floor_row_count:
        for i in range(row, ceil_row_count*15):
            extend_img[i, 0:col, :] = img[row-1, :, :]
    if ceil_col_count != floor_col_count:
        for i in range(col, ceil_col_count*15):
            extend_img[:, i, :] = extend_img[:, col-1, :]

    # cv2.imshow("img",img)
    # cv2.imshow("extend_img",extend_img)
    # cv2.waitKey(1)
    return extend_img
def get_prediction(classifier,img):
    extendimg = extend_img(img)
    row,col,depth = extendimg.shape
    row_count = int(row/15)
    col_count = int(col/15)
    image_feature = []
    for i in range(row_count):
        for j in range(col_count):
            tmp_img = extendimg[15*i:15*(i+1),15*j:15*(j+1)]
            # cv2.imshow("tmp_img",tmp_img)
            # cv2.waitKey(1)
            feature_pt_1 = get_image_feature_v2(tmp_img)
            feature_pt_2 = [15*i+8,15*j+8]
            feature = np.append(feature_pt_1,feature_pt_2)
            image_feature.append(list(feature))
    prediction = classifier.predict(image_feature)
    result_img = np.zeros((extendimg.shape[0],extendimg.shape[1]),np.uint8)
    for i in range(len(prediction)):
        x,y = image_feature[i][-2:]
        result_img[int(x-8):int(x+7),int(y-8):int(y+7)] = np.ones((15,15),np.uint8)*prediction[i]*255
        # result_img[x-8:x+7,y-8:y+7,:] = np.ones((15,15,3),np.uint8)*prediction[i]*255
    output_img = result_img[0:img.shape[0],0:img.shape[1]]
    # cv2.imshow("prediction",output_img)
    # cv2.imshow("source image",img)
    # cv2.waitKey()
    return output_img
def get_sky_region(classifier,img):
    prediction_image = get_prediction(classifier, img)
    # cv2.imshow("prediction_image",prediction_image)
    # cv2.waitKey()
    ret, bw_prediction_image = cv2.threshold(prediction_image, 250, 255, cv2.THRESH_BINARY)
    
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # ------------------------------RECT初始化方法---------------------------------------
    # r = np.where(bw_prediction_image == 255)[0]
    # c = np.where(bw_prediction_image == 255)[1]
    # min_r = np.min(r)
    # max_r = np.max(r)
    # min_c = np.min(c)
    # max_c = np.max(c)
    # rect = (min_c,min_r,max_c-min_c,max_r-min_r)    #划定区域
    
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    
    # cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    # -------------------------------Mask初始化方法-------------------------------------—-
    try:
        mask[(prediction_image-128)>0] = 3
        mask[(prediction_image-128)<=0] = 2
        mask[prediction_image == 0] = 0     # 0-cv2.GC_BGD
        mask[prediction_image == 255] = 1   # 1-cv2.GC_FGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        # -------------------------------联合方法--------------------------------------
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        tmp_mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        # cv2.imshow("tmp_mask2",tmp_mask2)
        mask3 = np.zeros(img.shape[:2], np.uint8)
        _, contours, hierarchy = cv2.findContours(tmp_mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_pixel_count = []
        for i in range(len(contours)):
            contour_pixel_count.append(cv2.contourArea(contours[i]))        # 计算轮廓的面积
        # print(contour_pixel_count)


        # print(max_contour_count)
        if len(contour_pixel_count) == 0:
            mask3 = mask2
        else:
            # print(contour_pixel_count)
            max_contour_count = np.max(contour_pixel_count)  # 如果天空不存在呢...
            for i in range(len(contours)):
                if cv2.contourArea(contours[i])/max_contour_count > 0.2:
                    x, y, w, h = cv2.boundingRect(contours[i])
                    rect = (x, y, w, h)
                    # print(rect)
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.grabCut(img, mask3, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask3 = np.where((mask3 == 2) | (mask3 == 0), 0, 1).astype('uint8')
        # # # ------------------------------------------------------------------------------------
        # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        # slected_img = img*mask3[:,:,np.newaxis]

        # cv2.imshow("source",img)
        # cv2.imshow("selected_img",slected_img)
        # cv2.waitKey()
    except:
        mask3 = np.zeros(img.shape[:2], np.uint8)
    return mask3


class SkyDetection():
    def __init__(self):
        # self.classifer = joblib.load("E:/DataSet/SkyDetection/MaskImage/RandomForestModel.m") #调用
        self.classifer = joblib.load(r"D:\CVProject\CBAM-keras-master\weather_data\SkyDetection\MaskImage\RandomForestModel.m")  # 调用
    def set_classifer(self,classifer_path):
        self.classifer = joblib.load(classifer_path)
    def get_sky_region(self,img):
        region_mask = get_sky_region(self.classifer, img)
        return region_mask



if __name__=="__main__":
    img = cv_imread(r"D:\CVProject\CBAM-keras-master\data\0_sunny\34020000000270000055_20161123135350025_2501_.png")
    SkyDetector = SkyDetection()
    sky_mask = SkyDetector.get_sky_region(img)
    slected_img = img*sky_mask[:, :, np.newaxis]
    cv2.imshow("source", img)
    cv2.imshow("selected_img", slected_img)
    cv2.waitKey()
# # prediction = clf.predict(testing_data)
# Srcfilelist = os.listdir(Srcfilepath)
# SkyDetector = SkyDetection()
# for elem in Srcfilelist:
#     img = cv_imread(Srcfilepath+"/"+elem)
#     sky_mask = SkyDetector.get_sky_region(img)
#     slected_img = img*sky_mask[:,:,np.newaxis]
#     cv2.imshow("source",img)
#     cv2.imshow("selected_img",slected_img)
#     cv2.waitKey()




# # iris=load_iris()  
# # #print iris#iris的４个属性是：萼片宽度　萼片长度　花瓣宽度　花瓣长度　标签是花的种类：setosa versicolour virginica  
# # print(iris)