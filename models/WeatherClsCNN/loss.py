from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
import keras
from sklearn.metrics import log_loss
import os
from keras import backend as K
import tensorflow as tf
K.set_image_dim_ordering='tf'

# if import matplotlib directly, it will throw error when call "figure" function!
# solution: Append "plt.switch_backend('agg')" after importing
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
from keras.utils import np_utils
from keras.callbacks import TensorBoard,Callback



def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# K.set_session(get_session(0.6))  # using 60% of total GPU Memory




class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'y', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
history = LossHistory()



def training_vis(hist, savename):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    # acc = hist.history['acc']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.semilogy(loss,label='train_loss',linewidth=1.0)
    ax1.semilogy(val_loss,label='val_loss', linewidth=1.0)
    ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax1.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.25')
    ax1.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax1.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.25')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc', linewidth=1.0)
    ax2.plot(val_acc, label='val_acc', linewidth=1.0)
    ax2.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax2.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.25')
    ax2.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax2.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.25')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    fig.savefig(savename, format='png', transparent=True, dpi=300, pad_inches=0)
    # plt.show()

def training_vis_v1(hist, savename):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.semilogy(loss,label='train_loss',linewidth=1.0)
    ax1.semilogy(val_loss,label='val_loss', linewidth=1.0)
    ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax1.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.25')
    ax1.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax1.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.25')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc', linewidth=1.0)
    ax2.plot(val_acc, label='val_acc', linewidth=1.0)
    ax2.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax2.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.25')
    ax2.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax2.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.25')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    fig.savefig(savename, format='png', transparent=True, dpi=300, pad_inches=0)
    # plt.show()

