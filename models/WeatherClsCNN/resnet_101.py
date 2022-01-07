# -*- coding: utf-8 -*-
# @Time : 2020/10/15 17:08
# @Author : Sun Zhu
# @Version：V 1.0
# @File : resnet101.py
# @desc :

from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

import pandas as pd
from models.WeatherClsCNN.custom_layers.scale_layer import Scale
import numpy as np
import models.WeatherClsCNN.input as input
import models.WeatherClsCNN.flags as flags
import models.WeatherClsCNN.test_performance as tp
import sys
import models.WeatherClsCNN.loss as loss
import absl.app as app
import os
import models.WeatherClsCNN.optimizers_setting as optimizers_setting
import time

import models.WeatherClsCNN.attention_module as am
from models.WeatherClsCNN.utils import xstr

sys.setrecursionlimit(3000)

# Basic architecture
def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    # x = merge([x, input_tensor], mode='sum', name='res' + str(stage) + block)
    x = Add(name='res' + str(stage) + block)([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    # x = merge([x, shortcut], mode='sum', name='res' + str(stage) + block)
    # x = add([x, shortcut])
    x = Add(name='res' + str(stage) + block)([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


# Architecture
def resnet101_model_with_attention_module(img_rows, img_cols, color_type=1, num_classes=None, weights_path=None, info_dict={}, attention_module=None):
    """
    Resnet 101 Model for Keras

    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    # if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == 'channels_last':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=0)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=1)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=2)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=3)

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    # if K.image_dim_ordering() == 'th':
    #   # Use pre-trained weights for Theano backend
    #   weights_path = 'imagenet_models/resnet101_weights_th.h5'
    # else:
    #   # Use pre-trained weights for Tensorflow backend
    #   weights_path = 'D:/Project/keras/keras_weights/resnet101_weights_tf.h5'
    weights_path = weights_path + "/resnet101_weights_tf.h5"

    model.load_weights(weights_path, by_name=True)

    # for layer in model.layers:
        # layer.trainable = False
    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers_setting.get_optimizer(info_dict=info_dict)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_resnet101_model_with_attention_module(img_rows, img_cols, color_type=1, num_classes=None, weights_path=None, info_dict={}, attention_module=None):
    """load frozen modle"""
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    # if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == 'channels_last':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=0)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=1)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=2)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module, num=3)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc8')(x)

    model = Model(img_input, x)

    model.load_weights(weights_path, by_name=True)

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers_setting.get_optimizer(info_dict=info_dict)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



# Process
def train(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, info_dict={}):
    print("start training ...")
    # Middle string in the filename of recorder
    part_string = tp.get_part_string(info_dict=info_dict)

    # Training process recorder
    if not os.path.exists(info_dict["savepath"]):
        os.mkdir(info_dict["savepath"])
    recorder_name = info_dict["savepath"] + "/training_log_" + xstr(info_dict["attention_module"]) + "_" + part_string + ".txt"
    doc = open(recorder_name, "w")
    for elem in info_dict:
        print(elem, ':', info_dict[elem], file=doc)
    print('Train on %d samples, validate on %d samples, test on %d samples'%(len(X_train), len(X_valid), len(X_test)), file=doc)

    # Load our model
    # model with attention module
    model = resnet101_model_with_attention_module(img_rows=info_dict["size"],
                            img_cols=info_dict["size"],
                            color_type=info_dict["channels"],
                            num_classes=len(info_dict["classname"]),
                            weights_path=info_dict['weightpath'],
                            info_dict=info_dict,
                            attention_module=info_dict["attention_module"]   # ‘cbam_block’/’se_block‘/None/'eca_net'
                            )
    model.summary()

    # Start Fine-tuning
    time_start = time.time()
    hist = model.fit(X_train, Y_train,
              batch_size=info_dict["batch_size"],
              epochs=info_dict['epoches'],
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )
    time_end = time.time()
    print('Totally time cost: %d mins, %d sec' % (int((time_end - time_start) // 60), int((time_end - time_start) % 60)),
        file=doc)

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=info_dict["batch_size"], verbose=1)

    # Cross-entropy loss score
    score = model.evaluate(X_valid, Y_valid, verbose=0)
    print('Test score:', score[0])
    print('Test score:', score[0], file=doc)            # Record the info on a txtfile
    print('Test accuracy:', score[1])
    print('Test accuracy:', score[1], file=doc)         # Record the info on a txtfile

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + "Resnet_101_" + xstr(info_dict["attention_module"]) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # Plot and save the loss_accuracy figure
    fig_savename = info_dict["savepath"] + "/" + "Resnet_101_" + xstr(info_dict["attention_module"] )+ "_" + part_string + ".png"
    loss.training_vis(hist, fig_savename)

    # Save the model
    model_savename = info_dict["savepath"] + "/" + "Resnet_101_" + xstr(info_dict["attention_module"]) + "_" + part_string + "_model.h5"
    # model.save(model_savename) # 同时保存模型的方式存在问题
    model.save_weights(model_savename)

    predictions_valid = model.predict(X_test, batch_size=info_dict["batch_size"], verbose=1)
    Y_predict = np.argmax(predictions_valid, axis=1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引

    Y_actual = np.argmax(Y_test, axis=1)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance.get_score(Y_actual, Y_predict, info_dict)


def pred(X_valid, Y_valid, info_dict={}):
    model = load_resnet101_model_with_attention_module(
                                        img_rows=info_dict["size"],
                                        img_cols=info_dict["size"],
                                        color_type=info_dict["channels"],
                                        num_classes=len(info_dict["classname"]),
                                        weights_path=info_dict['weightpath'],
                                        info_dict=info_dict,
                                        attention_module=info_dict["attention_module"])
    print("model loaded")

    predictions_valid = model.predict(X_valid, batch_size=info_dict["batch_size"], verbose=1)
    Y_predict = np.argmax(predictions_valid, axis=1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引

    Y_actual = np.argmax(Y_valid, axis=1)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance.get_score(Y_actual, Y_predict, info_dict)


def vis_prediction(mode='All', info_dict=None):
    """
    :param mode: Display mode: 'All'-显示所有; 'Err'-只显示错误;
    :param info_dict:
    :return:
    """
    model = load_resnet101_model_with_attention_module(
        img_rows=info_dict["size"],
        img_cols=info_dict["size"],
        color_type=info_dict["channels"],
        num_classes=len(info_dict["classname"]),
        weights_path=info_dict['weightpath'],
        info_dict=info_dict,
        attention_module=info_dict["attention_module"])

    model.summary()

    from models.WeatherClsCNN.DataListGenerate import DataFile
    from models.WeatherClsCNN.DataListGenerate import cv_imread
    import cv2
    import numpy as np

    obj_datafiles = DataFile(info_dict["filepath"],
                             ['.jpg', '.png'],
                             0.8,
                             0.1)
    label_dict = obj_datafiles.label2category

    for idx, elem in enumerate(obj_datafiles.test_filelist):
        src_img = cv_imread(elem)
        img = cv2.resize(src_img, (224, 224))
        expend_img = np.expand_dims(img, axis=0)
        preds = model.predict(expend_img)
        pred_class = np.argmax(preds[0])
        truth_class = obj_datafiles.test_labellist[idx]
        if mode is "Err":
            if str(pred_class) != str(truth_class):
                print(idx, label_dict[str(pred_class)], label_dict[str(truth_class)])
        else:
            print(idx, label_dict[str(pred_class)], label_dict[str(truth_class)])


def vis_activation(filename=None, info_dict=None):

    model = load_resnet101_model_with_attention_module(
        img_rows=info_dict["size"],
        img_cols=info_dict["size"],
        color_type=info_dict["channels"],
        num_classes=len(info_dict["classname"]),
        weights_path=info_dict['weightpath'],
        info_dict=info_dict,
        attention_module=info_dict["attention_module"])

    model.summary()

    from models.WeatherClsCNN.DataListGenerate import cv_imread
    import cv2
    import numpy as np
    #
    img = cv_imread(filename)

    img = cv2.resize(img, (224, 224))

    input_img = np.expand_dims(img, axis=0)

    preds = model.predict(input_img)
    print(preds)

    class_idx = np.argmax(preds[0])

    class_output = model.output[:, class_idx]
    # 获取最后一层
    last_conv_layer = model.get_layer("res5c_branch2c")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.sum(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_img])

    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # relu激活。
    heatmap = heatmap/(np.max(heatmap) + 1e-10)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imshow('Grad-cam', superimposed_img)
    cv2.waitKey()


def main():

    # --------------- Model configuration --------------
    input_dict= {}
    # input_dict["mode"] = "train"
    # input_dict["mode"] = "test"
    input_dict["mode"] = "vis_act"
    # input_dict["mode"] = "vis_pred"
    input_dict["filepath"] = r'D:\CVProject\CBAM-keras-master\data'
    input_dict["savepath"] = r'D:\CVProject\CBAM-keras-master\results'
    input_dict["weightpath"] = 'D:\CVProject\CBAM-keras-master\weights'
    input_dict["batch_size"] = 16
    input_dict["extension"] = [".jpg", ".png"]
    input_dict["training_ratio"] = 0.8
    input_dict["validation_ratio"] = 0.1
    input_dict["size"] = 224
    input_dict["channels"] = 3
    input_dict["input_mode"] = "resized"
    input_dict["epoches"] = 1
    input_dict["classname"] = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
    input_dict["attention_module"] = 'cbam_block'  # ‘cbam_block’/’se_block‘/'eca_net'/None(default)
    # --------------------------------------------------

    print("model parameters:")
    for elem in input_dict:
        print(elem, ":", input_dict[elem])
    input_data = input.INPUT_DATA(input_dict["filepath"],
                                  input_dict["extension"],
                                  input_dict["training_ratio"],
                                  input_dict["validation_ratio"],
                                  (input_dict["size"], input_dict["size"], input_dict["channels"]),
                                  input_dict["input_mode"])
    X_train, Y_train = input_data.get_training_data()
    X_valid, Y_valid = input_data.get_validation_data()
    X_test, Y_test = input_data.get_testing_data()
    input_dict["classname"] = input_data.get_label()
    if input_dict['mode'] == 'train':
        train(X_train[:50], Y_train[:50], X_valid[:50], Y_valid[:50], X_test[:50], Y_test[:50], info_dict=input_dict)
    if input_dict['mode'] == 'test':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\1104_01\Resnet_101_cbam_block_16_40_model.h5"
        pred(X_test, Y_test, info_dict=input_dict)
    if input_dict['mode'] == 'vis_act':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\1104_01\Resnet_101_cbam_block_16_40_model.h5"
        selected_image = r"D:\CVProject\CBAM-keras-master\results\image_error\src\1.jpg"
        vis_activation(filename=selected_image, info_dict=input_dict)
    if input_dict['mode'] == 'vis_pred':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\1104_01\Resnet_101_cbam_block_16_40_model.h5"
        vis_prediction(info_dict=input_dict)


if __name__ == '__main__':
    main()
