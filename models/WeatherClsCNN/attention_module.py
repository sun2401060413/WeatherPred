from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Conv1D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
import math
import tensorflow as tf

def attach_attention_module(net, attention_module, num=0):
	if attention_module == 'se_block': 		# SE_block
		net = se_block(net)
	elif attention_module == 'cbam_block': 	# CBAM_block
		net = cbam_block(net)
	elif attention_module == 'eca_net': 	# eca_net
		net = eca_layer(net, num)
	elif attention_module == 'mixed_block':
		net = mixed_block(net, num=num)
	else:
		raise Exception("'{}' is not supported attention module!".format(attention_module))
	return net

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._keras_shape[1:] == (1, 1, channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1, 1, channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1, 1, channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)		# 置换维度

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	# cbam_feature = channel_attention(cbam_feature, ratio)
	return cbam_feature

def mixed_block(mix_feature, ratio=8, num=0):
	mix_feature = spatial_attention(mix_feature)
	mix_feature = eca_layer(mix_feature, num=num)
	return mix_feature

def channel_attention(input_feature, ratio=8):
	
	# channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# channel = input_feature._keras_shape[channel_axis]
	#
	# shared_layer_one = Dense(channel//ratio,
	# 						 activation='relu',
	# 						 kernel_initializer='he_normal',
	# 						 use_bias=True,
	# 						 bias_initializer='zeros')
	# shared_layer_two = Dense(channel,
	# 						 kernel_initializer='he_normal',
	# 						 use_bias=True,
	# 						 bias_initializer='zeros')
	#
	# avg_pool = GlobalAveragePooling2D()(input_feature)
	# avg_pool = Reshape((1,1,channel))(avg_pool)
	# assert avg_pool._keras_shape[1:] == (1,1,channel)
	# avg_pool = shared_layer_one(avg_pool)
	# assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	# avg_pool = shared_layer_two(avg_pool)
	# assert avg_pool._keras_shape[1:] == (1,1,channel)
	#
	# max_pool = GlobalMaxPooling2D()(input_feature)
	# max_pool = Reshape((1,1,channel))(max_pool)
	# assert max_pool._keras_shape[1:] == (1,1,channel)
	# max_pool = shared_layer_one(max_pool)
	# assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	# max_pool = shared_layer_two(max_pool)
	# assert max_pool._keras_shape[1:] == (1,1,channel)
	#
	# cbam_feature = Add()([avg_pool,max_pool])
	# cbam_feature = Activation('sigmoid')(cbam_feature)
	#
	# if K.image_data_format() == "channels_first":
	# 	cbam_feature = Permute((3, 1, 2))(cbam_feature)
	#
	# return multiply([input_feature, cbam_feature])

	# get channel
	channel_axis = 1 if K.image_data_format() == "channels_first" else 3
	channel = int(input_feature.shape[channel_axis])
	maxpool_channel = GlobalMaxPooling2D()(input_feature)
	maxpool_channel = Reshape((1, 1, channel))(maxpool_channel)
	avgpool_channel = GlobalAveragePooling2D()(input_feature)
	avgpool_channel = Reshape((1, 1, channel))(avgpool_channel)
	Dense_One = Dense(units=int(channel/ratio), activation='relu', kernel_initializer='he_normal',
						 use_bias=True, bias_initializer='zeros')
	Dense_Two = Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
						 bias_initializer='zeros')
	# max path
	mlp_1_max = Dense_One(maxpool_channel)
	mlp_2_max = Dense_Two(mlp_1_max)
	mlp_2_max = Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
	# avg path
	mlp_1_avg = Dense_One(avgpool_channel)
	mlp_2_avg = Dense_Two(mlp_1_avg)
	mlp_2_avg = Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
	channel_attention_feature = Add()([mlp_2_max, mlp_2_avg])
	channel_attention_feature = Activation('sigmoid')(channel_attention_feature)
	return multiply([channel_attention_feature, input_feature])

def spatial_attention(input_feature):
	# kernel_size = 7
	#
	# if K.image_data_format() == "channels_first":
	# 	channel = input_feature._keras_shape[1]
	# 	cbam_feature = Permute((2,3,1))(input_feature)
	# else:
	# 	channel = input_feature._keras_shape[-1]
	# 	cbam_feature = input_feature
	#
	# avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	# assert avg_pool._keras_shape[-1] == 1
	# max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	# assert max_pool._keras_shape[-1] == 1
	# concat = Concatenate(axis=3)([avg_pool, max_pool])
	# assert concat._keras_shape[-1] == 2
	# cbam_feature = Conv2D(filters = 1,
	# 				kernel_size=kernel_size,
	# 				strides=1,
	# 				padding='same',
	# 				activation='sigmoid',
	# 				kernel_initializer='he_normal',
	# 				use_bias=False)(concat)
	# assert cbam_feature._keras_shape[-1] == 1
	#
	# if K.image_data_format() == "channels_first":
	# 	cbam_feature = Permute((3, 1, 2))(cbam_feature)
	#
	# return multiply([input_feature, cbam_feature])

	kernel_size = 3
	maxpool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
	avgpool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
	max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
	spatial_feature = Conv2D(filters=1, kernel_size=kernel_size, padding="same", activation='sigmoid',
					 kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

	return multiply([input_feature, spatial_feature])


class ECALayer():
    """
      ECA layer
    """
    def __init__(self, input_tensor, gamma=2, b=1):
        """
        :param input_tensor: input_tensor.shape=[batchsize,channels,h,w]
        :param gamma:
        :param b:
        """
        self.in_tensor = input_tensor
        self.gamma = gamma
        self.b = b
        self.channels = K.in_shape(self.in_tensor)[1]

    def forward(self, input):
        t = int(abs((math.log(self.channels, 2)+self.b)/self.gamma))
        k = t if t % 2 else t+1
        out = GlobalAveragePooling2D(data_format='channels_first')(input)
        out = Reshape((-1, self.channels, 1))(out)
        out = Conv1D(1, kernel_size=k, padding='same')(out)
        out = Activation('sigmoid')(out)
        out = tf.expand_dims(out, -1) #shape=[batchsize,channels,h,w]
        scale = multiply([self.in_tensor, out])
        return scale

def eca_layer(inputs_tensor=None, num=None, gamma=2, b=1, **kwargs):
    """
    ECA-NET
    :param inputs_tensor: input_tensor.shape=[batchsize,h,w,channels]
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channels = inputs_tensor._keras_shape[channel_axis]
    t = int(abs((math.log(channels, 2)+b)/gamma))
    k = t if t % 2 else t+1

    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels, 1))(x_global_avg_pool)
    x = Conv1D(1, kernel_size=k, padding="same", name="eca_conv1_" + str(num))(x)
    x = Activation('sigmoid', name='eca_conv1_relu_' + str(num))(x)  #shape=[batch,chnnels,1]
    # x = K.expand_dims(x, -1) 	#shape=[batch,chnnels,1,1]
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = Permute((2, 3, 1))(x)
    output = multiply([inputs_tensor, x])
    return output