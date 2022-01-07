# _*_ coding: utf-8 _*_
# @Time : 2021/6/7 11:56 
# @Author : Sun Zhu
# @Version：V 0.1
# @File : vgg19_sz.py
# @desc :
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""VGG19 model for Keras.

Reference:
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
      https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export


WEIGHTS_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/'
                'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg19/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

layers = VersionAwareLayers()

def vgg19_chosen_layer(img_rows, img_cols, channel=3, num_classes=None, weights_path=None, trainable=False, info_dict={}, chosen_layer=5):
    weights = weights_path + '/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = VGG19(include_top=False,
                       weights=weights,
                       input_shape=(img_rows, img_cols, channel),
                       classes=num_classes,
                       chosen_layer=chosen_layer
                       )
    if trainable is False:
        for layer in base_model.layers:
            layer.trainable = False
        # Classification block

    return base_model


@keras_export('keras.applications.vgg19.VGG19', 'keras.applications.VGG19')
def VGG19(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    chosen_layer=0
):
  if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')
  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  # Block 1
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
          img_input)
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  if chosen_layer in [2, 3 ,4, 5]:
      x = layers.Conv2D(
          128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
      x = layers.Conv2D(
          128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
      x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  if chosen_layer in [3, 4 ,5]:
      x = layers.Conv2D(
          256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
      x = layers.Conv2D(
          256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
      x = layers.Conv2D(
          256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
      x = layers.Conv2D(
          256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
      x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  if chosen_layer in [4, 5]:
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
      x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  if chosen_layer in [5]:
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
      x = layers.Conv2D(
          512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
      x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  x = layers.GlobalAveragePooling2D()(x)
  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name='vgg19')

  # Load weights.
  model.load_weights(weights, by_name=True)

  return model


@keras_export('keras.applications.vgg19.preprocess_input')
def preprocess_input(x, data_format=None):
  return imagenet_utils.preprocess_input(
      x, data_format=data_format, mode='caffe')


@keras_export('keras.applications.vgg19.decode_predictions')
def decode_predictions(preds, top=5):
  return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode='',
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
