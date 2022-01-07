# -*- coding: utf-8 -*-

import tensorflow as tf

from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D, Concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

from sklearn.metrics import log_loss

import cv2
import models.WeatherClsCNN.DataListGenerate
from models.WeatherClsCNN.custom_layers.scale_layer import Scale
import models.WeatherClsCNN.input as input
import pandas as pd
import numpy as np
import models.WeatherClsCNN.test_performance as tp
import models.WeatherClsCNN.loss as loss

import os
import models.WeatherClsCNN.flags as flags
import models.WeatherClsCNN.optimizers_setting as optimizers_setting
import time
import models.WeatherClsCNN.attention_module as am

from models.WeatherClsCNN.utils import xstr

# Basic block architecture
def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4, attention_module=None):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    # attention_module
    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module)

    return x

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True, attention_module=None):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        # concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))
        concat_feat = Concatenate(axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch) )([concat_feat, x])

        if grow_nb_filters:
            nb_filter += growth_rate
    # attention_module
    if attention_module is not None:
        x = am.attach_attention_module(x, attention_module)

    return concat_feat, nb_filter


# Architecture
def densenet121_model_with_attension(img_rows,
                    img_cols,
                    color_type=1,
                    nb_dense_block=4,
                    growth_rate=32,
                    nb_filter=64,
                    reduction=0.5,
                    dropout_rate=0.0,
                    weight_decay=1e-4,
                    num_classes=None,
                    weights_path=None,
                    attention_module=None,
                    info_dict={}):
    '''
    DenseNet 121 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    # if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == 'channels_last':
      # concat_axis = 3
      concat_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      # concat_axis = 1
      concat_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 24, 16] # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    # if K.image_dim_ordering() == 'th':
    #   # Use pre-trained weights for Theano backend
    #   weights_path = 'imagenet_models/densenet121_weights_th.h5'
    # else:
    #   # Use pre-trained weights for Tensorflow backend
    #   weights_path = 'imagenet_models/densenet121_weights_tf.h5'
    weights_path = weights_path + '/densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers_setting.get_optimizer(info_dict=info_dict)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_densenet121_model_with_attention(img_rows,
                    img_cols,
                    color_type=1,
                    nb_dense_block=4,
                    growth_rate=32,
                    nb_filter=64,
                    reduction=0.5,
                    dropout_rate=0.0,
                    weight_decay=1e-4,
                    num_classes=None,
                    weights_path=None,
                    attention_module=None,
                    info_dict={}):
     eps = 1.1e-5

     # compute compression factor
     compression = 1.0 - reduction

     # Handle Dimension Ordering for different backends
     global concat_axis
     # if K.image_dim_ordering() == 'tf':
     if K.image_data_format() == 'channels_last':
       # concat_axis = 3
       concat_axis = 3
       img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
     else:
       # concat_axis = 1
       concat_axis = 1
       img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

     # From architecture for ImageNet (Table 1 in the paper)
     nb_filter = 64
     nb_layers = [6, 12, 24, 16] # For DenseNet-121

     # Initial convolution
     x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
     x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
     x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
     x = Scale(axis=concat_axis, name='conv1_scale')(x)
     x = Activation('relu', name='relu1')(x)
     x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
     x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

     # Add dense blocks
     for block_idx in range(nb_dense_block - 1):
         stage = block_idx+2
         x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

         # Add transition_block
         x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
         nb_filter = int(nb_filter * compression)

     final_stage = stage + 1
     x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

     x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
     x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
     x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

     x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
     x_fc = Dense(1000, name='fc6')(x_fc)
     x_fc = Activation('softmax', name='prob')(x_fc)

     x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
     x_newfc = Dense(num_classes, name='fc6')(x_newfc)
     x_newfc = Activation('softmax', name='prob')(x_newfc)

     model = Model(img_input, x_newfc)

     model.load_weights(weights_path, by_name=True)

     # Truncate and replace softmax layer for transfer learning
     # Cannot use model.layers.pop() since model is not of Sequential() type
     # The method below works since pre-trained weights are stored in layers but not in the model

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
    print('Train on %d samples, validate on %d samples, test on %d samples' % (len(X_train), len(X_valid), len(X_test)),
          file=doc)

    # Load our model
    model = densenet121_model_with_attension(img_rows=info_dict["size"],
                              img_cols=info_dict["size"],
                              color_type=info_dict["channels"],
                              num_classes=len(info_dict["classname"]),
                              weights_path=info_dict['weightpath'],
                            attention_module=info_dict["attention_module"],
                              info_dict=info_dict)

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
    print('Test score:', score[0], file=doc)  # Record the info on a txtfile
    print('Test accuracy:', score[1])
    print('Test accuracy:', score[1], file=doc)  # Record the info on a txtfile

    # Record the loss and accuracy
    df = pd.DataFrame.from_dict(hist.history)
    csv_savename = info_dict["savepath"] + "/" + "Densenet_121" + xstr(info_dict["attention_module"]) + "_" + part_string + "_loss.csv"
    df.to_csv(csv_savename, encoding='utf-8', index=False)

    # Plot and save the loss_accuracy figure
    fig_savename = info_dict["savepath"] + "/" + "Densenet_121" + xstr(info_dict["attention_module"]) + "_" + part_string + ".png"
    loss.training_vis(hist, fig_savename)

    # Save the model
    model_savename = info_dict["savepath"] + "/" + "Densenet_121" + xstr(info_dict["attention_module"]) + "_" + part_string + "_model.h5"
    # model.save(model_savename) # 同时保存模型的方式存在问题
    model.save_weights(model_savename)

    predictions_valid = model.predict(X_test, batch_size=info_dict["batch_size"], verbose=1)
    Y_predict = np.argmax(predictions_valid, axis=1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引

    Y_actual = np.argmax(Y_test, axis=1)

    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance.get_score(Y_actual, Y_predict, info_dict)


def pred(X_valid, Y_valid, info_dict={}):
    model = load_densenet121_model_with_attention(  img_rows=info_dict["size"],
                                                    img_cols=info_dict["size"],
                                                    color_type=info_dict["channels"],
                                                    num_classes=len(info_dict["classname"]),
                                                    weights_path=info_dict['weightpath'],
                                                    attention_module=info_dict["attention_module"],
                                                    info_dict=info_dict)
    model.summary()
    print("model loaded")

    predictions_valid = model.predict(X_valid, batch_size=info_dict["batch_size"], verbose=1)
    Y_predict = np.argmax(predictions_valid, axis=1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引

    Y_actual = np.argmax(Y_valid, axis=1)

    info_dict["batch_size"] = 0
    info_dict["epoches"] = 0
    test_performance = tp.performance_score(score_type="multi_class", info_dict=info_dict)
    test_performance.get_score(Y_actual, Y_predict, info_dict)

    return Y_predict


def vis_prediction(mode='All', info_dict=None):
    """
    :param mode: Display mode: 'All'-显示所有; 'Err'-只显示错误;
    :param info_dict:
    :return:
    """
    model = load_densenet121_model_with_attention(
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

    model = load_densenet121_model_with_attention(
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
    last_conv_layer = model.get_layer("conv5_16_x2")

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
    input_dict = {}
    # input_dict["mode"] = "train"
    # input_dict["mode"] = "test"
    input_dict["mode"] = "vis_act"
    # input_dict["mode"] = "vis_pred"
    input_dict["filepath"] = r'D:\CVProject\CBAM-keras-master\data'
    input_dict["savepath"] = r'D:\CVProject\CBAM-keras-master\results'
    input_dict["weightpath"] = r'D:\CVProject\CBAM-keras-master\weights'
    input_dict["batch_size"] = 1
    input_dict["extension"] = [".jpg", ".png"]
    input_dict["training_ratio"] = 0.8
    input_dict["validation_ratio"] = 0.1
    input_dict["size"] = 224
    input_dict["channels"] = 3
    input_dict["input_mode"] = "resized"
    input_dict["epoches"] = 1
    input_dict["attention_module"] = "cbam_block"   # ‘cbam_block’/’se_block‘/'eca_net'/None(default)
    input_dict["classname"] = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
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
    if input_dict['mode'] == 'train':
        train(X_train[:50], Y_train[:50], X_valid[:50], Y_valid[:50], X_test[:50], Y_test[:50], info_dict=input_dict)
    if input_dict['mode'] == 'test':
        input_dict['weightpath'] = r"D:\CVProject\CBAM-keras-master\results\1111_01\Densenet_121_16_41_model.h5"
        pred(X_test[0:100], Y_test[0:100], info_dict=input_dict)
    if input_dict['mode'] == 'vis_act':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\1111_01\Densenet_121_16_41_model.h5"
        selected_image = r"D:\CVProject\CBAM-keras-master\results\image_error\src\1.jpg"
        vis_activation(filename=selected_image, info_dict=input_dict)
    if input_dict['mode'] == 'vis_pred':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\1111_01\Densenet_121_16_41_model.h5"
        vis_prediction(info_dict=input_dict)

if __name__ == '__main__':
    main()
