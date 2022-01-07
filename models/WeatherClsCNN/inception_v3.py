# -*- coding: utf-8 -*-
# @Time : 2020/10/15 17:08
# @Author : Sun Zhu
# @Version：V 1.0
# @File : resnet101.py
# @desc :

from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K

from keras_applications.imagenet_utils import _obtain_input_shape

import os
import time
import numpy as np
import pandas as pd

import models.WeatherClsCNN.input as input
import models.WeatherClsCNN.loss as loss
import models.WeatherClsCNN.test_performance as tp
import models.WeatherClsCNN.optimizers_setting as optimizers_setting
from models.WeatherClsCNN.attention_module import attach_attention_module
from models.WeatherClsCNN.utils import xstr

# Basic architecture
def _conv2d_bn(x,
               filters,
               num_row,
               num_col,
               padding='same',
               strides=(1, 1),
               name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


# Architecture
def InceptionV3_model_with_attention_module(include_top=True,
                    weights=None,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    attention_module=None,
                    info_dict=None):
    """Instantiates the Squeeze and Excite Inception v3 architecture.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = _conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = _conv2d_bn(x, 32, 3, 3, padding='valid')
    x = _conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv2d_bn(x, 80, 1, 1, padding='valid')
    x = _conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=0)

    # mixed 1: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=1)

    # mixed 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=2)

    # mixed 3: 17 x 17 x 768
    branch3x3 = _conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=3)

    # mixed 4: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 128, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=4)

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 160, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

        # attention_module
        if attention_module is not None:
            x = attach_attention_module(x, attention_module, num=5+i)

    # mixed 7: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 192, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=7)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = _conv2d_bn(x, 192, 1, 1)
    branch3x3 = _conv2d_bn(branch3x3, 320, 3, 3,
                           strides=(2, 2), padding='valid')

    branch7x7x3 = _conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = _conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=8)

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 320, 1, 1)

        branch3x3 = _conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = _conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = _conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = _conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = _conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = _conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

        # attention_module
        if attention_module is not None:
            x = attach_attention_module(x, attention_module, num=9+i)

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    opt = optimizers_setting.get_optimizer(info_dict=info_dict)

    model = Model(inputs, x, name='inception_v3')
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def Load_InceptionV3_model_with_attention_module(include_top=True,
                    weights=None,
                    weights_path=None,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    attention_module=None,
                    info_dict=None):
    """Instantiates the Squeeze and Excite Inception v3 architecture.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = _conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = _conv2d_bn(x, 32, 3, 3, padding='valid')
    x = _conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv2d_bn(x, 80, 1, 1, padding='valid')
    x = _conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=0)

    # mixed 1: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=1)

    # mixed 2: 35 x 35 x 256
    branch1x1 = _conv2d_bn(x, 64, 1, 1)

    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=2)

    # mixed 3: 17 x 17 x 768
    branch3x3 = _conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=3)

    # mixed 4: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 128, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=4)

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 160, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

        # attention_module
        if attention_module is not None:
            x = attach_attention_module(x, attention_module, num=5+i)

    # mixed 7: 17 x 17 x 768
    branch1x1 = _conv2d_bn(x, 192, 1, 1)

    branch7x7 = _conv2d_bn(x, 192, 1, 1)
    branch7x7 = _conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = _conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=7)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = _conv2d_bn(x, 192, 1, 1)
    branch3x3 = _conv2d_bn(branch3x3, 320, 3, 3,
                           strides=(2, 2), padding='valid')

    branch7x7x3 = _conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = _conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, num=8)

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = _conv2d_bn(x, 320, 1, 1)

        branch3x3 = _conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = _conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = _conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = _conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = _conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = _conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

        # attention_module
        if attention_module is not None:
            x = attach_attention_module(x, attention_module, num=9+i)

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.


    model = Model(inputs, x, name='inception_v3')

    model.load_weights(weights_path, by_name=True)

    opt = optimizers_setting.get_optimizer(info_dict=info_dict)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


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
    model = InceptionV3_model_with_attention_module(
                    input_shape=(info_dict['size'], info_dict['size'], info_dict['channels']),
                    classes=len(info_dict["classname"]),
                    attention_module=info_dict["attention_module"], # ‘cbam_block’/’se_block‘/None/'eca_net'
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
    model = Load_InceptionV3_model_with_attention_module(
        input_shape=(info_dict['size'], info_dict['size'], info_dict['channels']),
        weights_path=info_dict['weightpath'],
        classes=len(info_dict["classname"]),
        attention_module=info_dict["attention_module"],  # ‘cbam_block’/’se_block‘/None/'eca_net'
        info_dict=info_dict)
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
    model = Load_InceptionV3_model_with_attention_module(
        input_shape=(info_dict['size'], info_dict['size'], info_dict['channels']),
        weights_path=info_dict['weightpath'],
        classes=len(info_dict["classname"]),
        attention_module=info_dict["attention_module"],  # ‘cbam_block’/’se_block‘/None/'eca_net'
        info_dict=info_dict)

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
        img = cv2.resize(src_img, (299, 299))
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

    model = Load_InceptionV3_model_with_attention_module(
        input_shape=(info_dict['size'], info_dict['size'], info_dict['channels']),
        weights_path=info_dict['weightpath'],
        classes=len(info_dict["classname"]),
        attention_module=info_dict["attention_module"],  # ‘cbam_block’/’se_block‘/None/'eca_net'
        info_dict=info_dict)

    model.summary()

    from models.WeatherClsCNN.DataListGenerate import cv_imread
    import cv2
    import numpy as np
    #
    img = cv_imread(filename)

    img = cv2.resize(img, (299, 299))

    input_img = np.expand_dims(img, axis=0)

    preds = model.predict(input_img)
    print(preds)

    class_idx = np.argmax(preds[0])

    class_output = model.output[:, class_idx]
    # 获取最后一层
    last_conv_layer = model.get_layer("conv2d_105")

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
    input_dict["size"] = 299
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
        train(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, info_dict=input_dict)
    if input_dict['mode'] == 'test':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\Resnet_101_cbam_block_16_1_model.h5"
        pred(X_test, Y_test, info_dict=input_dict)
    if input_dict['mode'] == 'vis_act':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\Resnet_101_cbam_block_16_1_model.h5"
        selected_image = r"D:\CVProject\CBAM-keras-master\results\image_error\src\1.jpg"
        vis_activation(filename=selected_image, info_dict=input_dict)
    if input_dict['mode'] == 'vis_pred':
        input_dict["weightpath"] = r"D:\CVProject\CBAM-keras-master\results\Resnet_101_cbam_block_16_1_model.h5"
        vis_prediction(info_dict=input_dict)


if __name__ == '__main__':
    main()