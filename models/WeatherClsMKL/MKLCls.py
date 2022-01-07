# _*_ coding: utf-8 _*_
# @Time : 2020/10/24 17:08
# @Author : Sun Zhu
# @Version：V 1.0
# @File : Feature.py
# @desc : Weather features extraction for weather classification.
# @env: MKL
# requirment:
#     cvxopt      1.2.0
#     MKLpy       0.6
#     pytorch     1.6.0
#     scikit      0.23.2
#     scipy       1.5.3
#     numpy       1.19.1

import numpy
from scipy import io

# MKLpy packages
from MKLpy.metrics import pairwise
from MKLpy.generators import Multiview_generator
from sklearn.metrics import accuracy_score

def MultiView_learning():
    """MultiView learning"""
    print('loading dataset...', end='')

    training_data = io.loadmat(r"D:\CVProject\CBAM-keras-master\handcraft\features_with_pca_file_0202.mat")
    length = len(training_data['array'][0])
    X, Y = training_data['array'][:, 0:length - 2], training_data['array'][:, -1]
    print('done')

    # preprocess data
    print('preprocessing data...', end='')
    from MKLpy.preprocessing import normalization, rescale_01
    X = rescale_01(X)  # feature scaling in [0,1]
    X = normalization(X)  # ||X_i||_2^2 = 1

    # train/test split
    from sklearn.model_selection import train_test_split
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.1, random_state=42, shuffle=True)

    print(numpy.array(Xtr).shape)
    print(numpy.array(Ytr).shape)

    print('done')
    print('Training on {0} samples, Testing on {1} samples'.format(len(Xtr), len(Xte)))

    print('computing RBF Kernels...', end='')


    from MKLpy.metrics import pairwise
    from MKLpy.generators import Multiview_generator

    X1_tr = numpy.array(Xtr[:, :2])             # time
    X2_tr = numpy.array(Xtr[:, 2:92])          # color
    X3_tr = numpy.array(Xtr[:, 92:124])        # Gabor
    X4_tr = numpy.array(Xtr[:, 124:156])       # lbp
    X5_tr = numpy.array(Xtr[:, 156:348])       # cloud
    X6_tr = numpy.array(Xtr[:, 348:432])       # haze
    X7_tr = numpy.array(Xtr[:, 432:603])       # contrast
    X8_tr = numpy.array(Xtr[:, 603:606])       # shadow
    X9_tr = numpy.array(Xtr[:, 606:608])       # snow
    X10_tr = numpy.array(Xtr[:, 608:])          # pca

    X1_te = numpy.array(Xte[:, :2])             # time
    X2_te = numpy.array(Xte[:, 2:92])          # color
    X3_te = numpy.array(Xte[:, 92:124])        # Gabor
    X4_te = numpy.array(Xte[:, 124:156])       # lbp
    X5_te = numpy.array(Xte[:, 156:348])       # cloud
    X6_te = numpy.array(Xte[:, 348:432])       # haze
    X7_te = numpy.array(Xte[:, 432:603])       # contrast
    X8_te = numpy.array(Xte[:, 603:606])       # shadow
    X9_te = numpy.array(Xte[:, 606:608])       # snow
    X10_te = numpy.array(Xte[:, 608:])       # pca

    KLtr = Multiview_generator([X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.rbf_kernel)
    KLte = Multiview_generator([X1_te, X2_te, X3_te, X4_te, X5_te, X6_te, X7_te, X8_te, X9_te, X10_te], [X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.rbf_kernel)

    print('done')

    from MKLpy.algorithms import AverageMKL, EasyMKL
    print('training EasyMKL with one-vs-all multiclass strategy...', end='')
    from sklearn.svm import SVC
    base_learner = SVC(C=8)
    clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr, Ytr)

    print('the combination weights are:')
    for sol in clf.solution:
        print('(%d vs all): ' % sol, clf.solution[sol].weights)

    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
    y_pred = clf.predict(KLte)  # predictions
    y_score = clf.decision_function(KLte)  # rank
    accuracy = accuracy_score(Yte, y_pred)
    print('Accuracy score: %.4f' % (accuracy))
    recall = recall_score(Yte, y_pred, average='macro')
    print('Recall score: %.4f' %(recall))
    cm = confusion_matrix(Yte, y_pred)
    print('Confusion matrix', cm)

    print('training EasyMKL with one-vs-one multiclass strategy...', end='')
    clf = EasyMKL(lam=0.1, multiclass_strategy='ovo', learner=base_learner).fit(KLtr, Ytr)
    print('done')
    print('the combination weights are:')
    for sol in clf.solution:
        print('(%d vs %d): ' % (sol[0], sol[1]), clf.solution[sol].weights)

    y_pred = clf.predict(KLte)  # predictions
    y_score = clf.decision_function(KLte)  # rank
    accuracy = accuracy_score(Yte, y_pred)
    print('Accuracy score: %.4f' % (accuracy))
    recall = recall_score(Yte, y_pred, average='macro')
    print('Recall score: %.4f' %(recall))
    cm = confusion_matrix(Yte, y_pred)
    print('Confusion matrix', cm)


def Learning_curve_using_weather_data():
    '''
         Cross validation using weather data: PASS: 2021.02.05
    '''
    # load data
    print('loading dataset...', end='')
    # from sklearn.datasets import load_breast_cancer as load
    # ds = load()
    # X, Y = ds.data, ds.target

    # # Files
    training_data = io.loadmat(r"D:\CVProject\CBAM-keras-master\handcraft\features_with_pca.mat")
    # training_data = io.loadmat(r"D:\CVProject\CBAM-keras-master\handcraft\features_with_pca_file.mat")
    # training_data = io.loadmat(r"D:\CVProject\CBAM-keras-master\handcraft\features_with_pca_file_0202.mat")
    results_data = open(r"D:\CVProject\CBAM-keras-master\handcraft\results\learning_curve_results_0202_01.txt", "w")

    # length = len(training_data['array'][0])
    length = len(training_data['array'][0])

    # X, Y = training_data['array'][:, 0:length - 1], training_data['array'][:, -1]

    X, Y = training_data['array'][:, 0:length - 1], training_data['array'][:, -1]


    print('done')

    # preprocess data
    print('preprocessing data...', end='')
    from MKLpy.preprocessing import normalization, rescale_01
    X = rescale_01(X)  # feature scaling in [0,1]
    X = normalization(X)  # ||X_i||_2^2 = 1
    print('done')


    from MKLpy.algorithms import EasyMKL, KOMD  # KOMD is not a WeatherClsMKL algorithm but a simple kernel machine like the SVM
    from MKLpy.model_selection import cross_val_score
    from sklearn.svm import SVC
    import numpy as np
    # base_learner = SVC(C=10000)  # "hard"-margin svm
    print("Build a base learner")
    base_learner = SVC(C=20)  # "hard"-margin svm

    # # # === parameters selection ===
    # best_results = {}
    # # for lam in [0, 0.01, 0.1, 0.2, 0.9, 1]:  # possible lambda values for the EasyMKL algorithm
    # for lam in [0]:  # possible lambda values for the EasyMKL algorithm
    #     # MKLpy.model_selection.cross_val_score performs the cross validation automatically, it may returns
    #     # accuracy, auc, or F1 scores
    #     # evaluation on the test set
    #     print("Model training with lam {}".format(lam))
    #     clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr, Ytr)
    #     scores = cross_val_score(KLtr, Ytr, clf, n_folds=5, scoring='accuracy')
    #     acc = np.mean(scores)
    #     if not best_results or best_results['score'] < acc:
    #         best_results = {'lam': lam, 'score': acc}

    print("Build EasyMKL classifier")
    # clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr, Ytr)
    # scores = cross_val_score(KLtr, Ytr, clf, n_folds=5, scoring='accuracy')
    # acc = np.mean(scores)
    # print("acc:", acc)

    # ====== Learning curve =======
    #
    # X1_tr = numpy.array(Xtr[:, :2])             # time
    # X2_tr = numpy.array(Xtr[:, 2:92])          # color
    # X3_tr = numpy.array(Xtr[:, 92:124])        # Gabor
    # X4_tr = numpy.array(Xtr[:, 124:156])       # lbp
    # X5_tr = numpy.array(Xtr[:, 156:348])       # cloud
    # X6_tr = numpy.array(Xtr[:, 348:432])       # haze
    # X7_tr = numpy.array(Xtr[:, 432:603])       # contrast
    # X8_tr = numpy.array(Xtr[:, 603:651])       # shadow
    # X9_tr = numpy.array(Xtr[:, 606:683])       # snow
    # X10_tr = numpy.array(Xtr[:, 683:])          # pca
    #
    # X1_te = numpy.array(Xte[:, :2])             # time
    # X2_te = numpy.array(Xte[:, 2:92])          # color
    # X3_te = numpy.array(Xte[:, 92:124])        # Gabor
    # X4_te = numpy.array(Xte[:, 124:156])       # lbp
    # X5_te = numpy.array(Xte[:, 156:348])       # cloud
    # X6_te = numpy.array(Xte[:, 348:432])       # haze
    # X7_te = numpy.array(Xte[:, 432:603])       # contrast
    # X8_te = numpy.array(Xte[:, 603:651])       # shadow
    # X9_te = numpy.array(Xte[:, 606:683])       # snow
    # X10_te = numpy.array(Xte[:, 683:])       # pca
    # #
    # # # # all features
    # KLtr = Multiview_generator([X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.rbf_kernel)
    # KLte = Multiview_generator([X1_te, X2_te, X3_te, X4_te, X5_te, X6_te, X7_te, X8_te, X9_te, X10_te], [X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.rbf_kernel)
    #
    # KYtr = Ytr[:]
    # KYte = Yte[:]

    # for elem in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for elem in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # for elem in [1]:
        learn_count = int(elem*X.shape[0])
        KLtr, KYtr, KLte, KYte = bulid_kernel_transform(X[:learn_count], Y[:learn_count])

        train_count, test_count = len(KYtr), len(KYte)

        clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr, KYtr)
        # scores = cross_val_score(KLtr, Ytr, clf, n_folds=5, scoring='accuracy')
        # acc = np.mean(scores)
        y_train_pred = clf.predict(KLtr)
        y_test_pred = clf.predict(KLte)

        train_set_accuracy = accuracy_score(KYtr, y_train_pred)
        tests_et_accuracy = accuracy_score(KYte, y_test_pred)

        # display the results
        print("Test on {0} train samples and {1} test samples,".format(train_count, test_count), end="")
        print('accuracy on the train set: %.3f and accuracy on the test set : %.3f' % (
        train_set_accuracy, tests_et_accuracy))

        # save the results in txt
        print("Test on {0} train samples and {1} test samples,".format(train_count, test_count), end="", file=results_data)
        print('accuracy on the train set: %.3f and accuracy on the test set : %.3f' % (
        train_set_accuracy, tests_et_accuracy), file=results_data)

    # from sklearn.metrics import accuracy_score
    print('done')
    # ==============================

    pass
    # # # ===== evaluate the model =====
    # # # Chose the model with high performance
    #
    # # Transform
    # X1_tr = numpy.array(Xtr[:, :2])             # time
    # X2_tr = numpy.array(Xtr[:, 2:92])          # color
    # X3_tr = numpy.array(Xtr[:, 92:124])        # Gabor
    # X4_tr = numpy.array(Xtr[:, 124:156])       # lbp
    # X5_tr = numpy.array(Xtr[:, 156:348])       # cloud
    # X6_tr = numpy.array(Xtr[:, 348:432])       # haze
    # X7_tr = numpy.array(Xtr[:, 432:603])       # contrast
    # X8_tr = numpy.array(Xtr[:, 603:606])       # shadow
    # X9_tr = numpy.array(Xtr[:, 606:608])       # snow
    # X10_tr = numpy.array(Xtr[:, 608:])          # pca
    #
    # X1_te = numpy.array(Xte[:, :2])             # time
    # X2_te = numpy.array(Xte[:, 2:92])          # color
    # X3_te = numpy.array(Xte[:, 92:124])        # Gabor
    # X4_te = numpy.array(Xte[:, 124:156])       # lbp
    # X5_te = numpy.array(Xte[:, 156:348])       # cloud
    # X6_te = numpy.array(Xte[:, 348:432])       # haze
    # X7_te = numpy.array(Xte[:, 432:603])       # contrast
    # X8_te = numpy.array(Xte[:, 603:606])       # shadow
    # X9_te = numpy.array(Xte[:, 606:608])       # snow
    # X10_te = numpy.array(Xte[:, 608:])       # pca
    #
    # # # all features
    # KLtr = Multiview_generator([X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.homogeneous_polynomial_kernel)
    # KLte = Multiview_generator([X1_te, X2_te, X3_te, X4_te, X5_te, X6_te, X7_te, X8_te, X9_te, X10_te], [X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.homogeneous_polynomial_kernel)
    #
    # KYtr = Ytr[:]
    # KYte = Yte[:]
    #
    # clf = EasyMKL(learner=base_learner, lam=0.1).fit(KLtr, KYtr)
    # y_train_pred = clf.predict(KLtr)
    # y_test_pred = clf.predict(KLte)
    #
    # train_set_accuracy = accuracy_score(KYtr, y_train_pred)
    # tests_et_accuracy = accuracy_score(KYte, y_test_pred)
    #
    # # print('accuracy on the test set: %.3f, with lambda=%.2f' % (accuracy, best_results['lam']))
    # print('accuracy on the train set: %.3f, and accuracy on the test set : %.3f' % (train_set_accuracy, tests_et_accuracy))
    # # ======================
    pass


def bulid_kernel_transform(X, Y, test_size=.1):
    '''
        Transform Orignal data into new data;
    '''

    from sklearn.model_selection import train_test_split
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.1, random_state=42, shuffle=True)

    # print(numpy.array(Xtr).shape)
    # print(numpy.array(Ytr).shape)

    # # # ======== all features (old version) =============

    # # #
    X1_tr = numpy.array(Xtr[:, :2])             # time
    X2_tr = numpy.array(Xtr[:, 2:92])          # color
    X3_tr = numpy.array(Xtr[:, 92:124])        # Gabor
    X4_tr = numpy.array(Xtr[:, 124:156])       # lbp
    X5_tr = numpy.array(Xtr[:, 156:348])       # cloud
    X6_tr = numpy.array(Xtr[:, 348:432])       # haze
    X7_tr = numpy.array(Xtr[:, 432:603])       # contrast
    X8_tr = numpy.array(Xtr[:, 603:651])       # shadow
    X9_tr = numpy.array(Xtr[:, 651:683])       # snow
    X10_tr = numpy.array(Xtr[:, 683:])          # pca

    X1_te = numpy.array(Xte[:, :2])             # time
    X2_te = numpy.array(Xte[:, 2:92])          # color
    X3_te = numpy.array(Xte[:, 92:124])        # Gabor
    X4_te = numpy.array(Xte[:, 124:156])       # lbp
    X5_te = numpy.array(Xte[:, 156:348])       # cloud
    X6_te = numpy.array(Xte[:, 348:432])       # haze
    X7_te = numpy.array(Xte[:, 432:603])       # contrast
    X8_te = numpy.array(Xte[:, 603:651])       # shadow
    X9_te = numpy.array(Xte[:, 651:683])       # snow
    X10_te = numpy.array(Xte[:, 683:])          # pca
    # sky+global features
    # KLtr = Multiview_generator([X1_tr, X2_tr, X3_tr], kernel=pairwise.rbf_kernel)
    # KLte = Multiview_generator([X1_te, X2_te, X3_te], [X1_tr, X2_tr, X3_tr], kernel=pairwise.rbf_kernel)
    # #

    KLtr = Multiview_generator([X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.rbf_kernel)
    KLte = Multiview_generator([X1_te, X2_te, X3_te, X4_te, X5_te, X6_te, X7_te, X8_te, X9_te, X10_te], [X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, X8_tr, X9_tr, X10_tr], kernel=pairwise.rbf_kernel)

    KYtr = Ytr
    KYte = Yte

    return KLtr, KYtr, KLte, KYte



if __name__=="__main__":

    # === Available ==
    MultiView_learning()       # 能跑出正确结果，不要改他(2021-0203)。

    # # === learning curve using weather data ===
    # Learning_curve_using_weather_data() # PASS
