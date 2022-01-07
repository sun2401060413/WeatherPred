from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,  precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from models.WeatherClsCNN.utils import xstr

class performance_score():
    def __init__(self, score_type = "binary_class", info_dict=None, cam_id=None, Note=''):
        '''
        :param score_type:
                binary_class  # 二分类
                multi_class  # 多分类
                binary_label  # 二标签
                multi_label  # 多标签
        '''
        self.score_type = score_type
        self.info_dict = info_dict
        self.cam_id = cam_id
        self.Note = Note
        self.initialization()

    def __del__(self):
        self.doc.close()

    def initialization(self):
        part_string = get_part_string(self.info_dict)
        if self.cam_id is not None:
            self.recorder_name = self.info_dict["savepath"] + "/record_" + self.Note + str(self.cam_id) + "_" + part_string + ".txt"
        else:
            self.recorder_name = self.info_dict["savepath"] + "/record_" + self.Note + part_string + ".txt"
        #   self.confusion_matrix_name = self.info_dict["savepath"] + "/confusion_matrix_" + part_string + ".png"
        self.doc = open(self.recorder_name, 'w')

    def get_score(self, Y_actual, Y_predict, info_dict = {}):
        part_string = get_part_string(info_dict)
        if self.cam_id is not None:
            part_string = str(self.cam_id) + "_" + part_string
        if info_dict.__contains__("attention_module"):
            recorder_name = info_dict["savepath"] + "/record_" + self.Note + xstr(info_dict["attention_module"]) + "_" + part_string + ".txt"
            confusion_matrix_name = info_dict["savepath"] + "/confusion_matrix_" + self.Note + xstr(info_dict["attention_module"]) + "_" + part_string + ".png"
            confusion_matrix_name_count = info_dict["savepath"] + "/confusion_matrix_" + self.Note + xstr(info_dict["attention_module"]) + "_" + part_string + "_count.png"
        else:
            recorder_name = info_dict["savepath"] + "/record_" + self.Note + part_string + ".txt"
            confusion_matrix_name = info_dict["savepath"] + "/confusion_matrix_" + self.Note + part_string + ".png"
            confusion_matrix_name_count = info_dict["savepath"] + "/confusion_matrix_" + self.Note + part_string + "_count.png"
        doc = open(recorder_name, 'w')
        # 二分类
        if self.score_type == "binary_class":
            # 计算准确率
            ps = precision_score(Y_actual, Y_predict, average='weighted')
            # 参数average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]
            # macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
            # weighted:对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。
            # micro：Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。
            # samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，
            #          通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
            print("precision_score:", ps)

            # 计算召回率
            rc = recall_score(Y_actual, Y_predict, average='weighted')
            print("recall_score:", rc)

            # 计算f1_score
            rc = f1_score(Y_actual, Y_predict, average='weighted')
            print("f1_score:", rc)

            # 计算ROC
            roc = roc_curve(Y_actual, Y_predict)
            print("roc_curve:", roc)

            # 计算roc_auc_score
            roc_auc = roc_auc_score(Y_actual, Y_predict)
            print("roc_auc:", roc_auc)

            # 计算分类报告
            clr = classification_report(Y_actual, Y_predict)
            print("classificaiton_report:\n", clr)

            # 计算混淆矩阵
            cfm = confusion_matrix(Y_actual, Y_predict)
            print("confusion_matrix:\n", cfm)

            plot_confusion_matrix_v2(info_dict["classname"], cfm, confusion_matrix_name)
            plot_confusion_matrix_v2(info_dict["classname"], cfm, confusion_matrix_name_count, type='count')
        # 多分类
        if self.score_type == "multi_class":
            doc = open(recorder_name, 'w')
            # 计算准确率
            print("Y_actual.shape",Y_actual.shape)
            print("Y_predict.shape",Y_predict.shape)
            ps = precision_score(Y_actual, Y_predict, average='weighted')
            print("precision_score:", ps, file=doc)

            # 计算召回率
            rc = recall_score(Y_actual, Y_predict, average='weighted')
            print("recall_score:", rc, file=doc)

            # 计算f1_score
            rc = f1_score(Y_actual, Y_predict, average='weighted')
            print("f1_score:", rc, file=doc)

            # 计算分类报告
            clr = classification_report(Y_actual, Y_predict)
            print("classificaiton_report:\n", clr, file=doc)

            # 计算混淆矩阵
            cfm = confusion_matrix(Y_actual, Y_predict)
            print("confusion_matrix:\n", cfm, file=doc)

            # plot_confusion_matrix_v2(info_dict["classname"], cfm, confusion_matrix_name)
            # plot_confusion_matrix_v2(info_dict["classname"], cfm, confusion_matrix_name_count, type='count')
        # 二标签
        if self.score_type == "binary_label":
            # 计算准确率
            ps = precision_score(Y_actual, Y_predict, average='micro')
            # 参数average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]
            # micro：Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。
            # samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，
            #          通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
            print("precision_score:", ps)

            # 计算召回率
            rc = recall_score(Y_actual, Y_predict, average='weighted')
            print("recall_score:", rc)
        # 多标签
        if self.score_type == "multi_label":
            # 计算准确率
            ps = precision_score(Y_actual, Y_predict, average='weighted')
            # 参数average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]
            # micro：Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。
            # samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，
            #          通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
            print("precision_score:", ps)

            # 计算召回率
            rc = recall_score(Y_actual, Y_predict, average='weighted')
            print("recall_score:", rc)
        doc.close()

    def get_score_value(self, Y_actual, Y_predict, categray_actual=None, tag=""):
        MSE = mean_squared_error(Y_actual, Y_predict)
        print(tag + "_MSE:\n", MSE, file=self.doc)
        MAE = mean_absolute_error(Y_actual, Y_predict)
        print(tag + "_MAE:\n", MAE, file=self.doc)
        R2 = r2_score(Y_actual, Y_predict)
        print(tag + "_R2:\n", R2, file=self.doc)

        # seg_Y = {}
        # for i in range(len(Y_actual)):
        #     if not seg_Y.__contains__(categray_actual[i]):
        #         seg_Y[categray_actual[i]] = []
        #     seg_Y[categray_actual[i]].append([Y_actual[i], Y_predict[i]])
        #
        # print("seg_Y:", seg_Y)

        # length = len(seg_Y)
        print(tag + "segment_MSE:\t", file=self.doc)
        # for i, elem in enumerate(seg_Y):
        #     input_1 = [x[0] for x in seg_Y[elem]]
        #     input_2 = [x[1][0] for x in seg_Y[elem]]
        #     nMSE = mean_squared_error(input_1, input_2)
        #     print("\t" + str(nMSE), file=self.doc)

        print(tag + "segment_MAE:\t", file=self.doc)
        # for i, elem in enumerate(seg_Y):
        #     input_1 = [x[0] for x in seg_Y[elem]]
        #     input_2 = [x[1][0] for x in seg_Y[elem]]
        #     nMAE = mean_absolute_error(input_1, input_2)
        #     print("\t" + str(nMAE), file=self.doc)

        print(tag + "segment_R2:\t", file=self.doc)
        # for i, elem in enumerate(seg_Y):
        #     input_1 = [x[0] for x in seg_Y[elem]]
        #     input_2 = [x[1][0] for x in seg_Y[elem]]
        #     nR2 = r2_score(input_1, input_2)
        #     print("\t" + str(nR2), file=self.doc)

    def record_sth(self, str):
        print(str, file=self.doc)


def plot_confusion_matrix(classes, matrix, savename):
    """classes: a list of class names"""
     # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    # save
    plt.savefig(savename)


def plot_confusion_matrix_v2(classes, matrix, savename, fontsize=12, type='ratio'):
    """classes: a list of class names"""
    # Normalize by row
    if type == 'ratio':
        matrix = matrix.astype(np.float)
        linesum = matrix.sum(1)
        linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
        matrix /= linesum

    maxv = np.max(matrix)
    thresh = 0.7*maxv

    plt.figure()
    # plt.switch_backend('agg')
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(matrix, cmap=plt.cm.Blues)
    # plt.imshow(matrix, cmap=plt.cm.binary)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(matrix))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # plt.xticks(indices, [0, 1, 2])
    # plt.yticks(indices, [0, 1, 2])
    plt.xticks(indices, classes, fontsize=fontsize)
    plt.yticks(indices, classes, fontsize=fontsize)

    plt.colorbar()

    plt.xlabel('Prediction')
    plt.ylabel('GroundTruth')
    plt.title('ConfusionMatrix')

    # # plt.rcParams两行是用于解决标签不能显示汉字的问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # str('%.2f' % (matrix[i, i] * 100))
    # 显示数据
    if type =='ratio':
        for first_index in range(len(matrix)):  # 第几行
            for second_index in range(len(matrix[first_index])):  # 第几列
                if matrix[first_index][second_index] > thresh:
                    plt.text(second_index, first_index, str('%.2f' % (matrix[first_index][second_index] * 100)), fontsize=fontsize,
                             color='white', ha='center', va='center')
                else:
                    plt.text(second_index, first_index, str('%.2f' % (matrix[first_index][second_index] * 100)), fontsize=fontsize,
                             ha='center', va='center')
    else:
        for first_index in range(len(matrix)):  # 第几行
            for second_index in range(len(matrix[first_index])):  # 第几列
                if matrix[first_index][second_index] > thresh:
                    plt.text(second_index, first_index, matrix[first_index][second_index], fontsize=fontsize, color='white', ha='center', va='center')
                else:
                    plt.text(second_index, first_index, matrix[first_index][second_index], fontsize=fontsize, ha='center', va='center')
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    # plt.show()
    plt.savefig(savename)

def get_part_string(info_dict={}):
    part_string = str(info_dict["batch_size"]) + "_" + str(info_dict["epoches"])
    if "lr" in info_dict.keys():
        part_string = part_string + "_" + str(info_dict["lr"])
    if "op" in info_dict.keys():
        part_string = part_string + "_" + info_dict["op"]
    if "loss" in info_dict.keys():
        part_string = part_string + "_" + info_dict["loss"]
    return part_string