  
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
#from keras_contrib.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
#from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf
#vgg = keras.applications.VGG16(
#    include_top=False,
#    weights="imagenet",
#    input_tensor=None,
#    input_shape=(1024, 1024, 3))





def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    pool_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(pool_3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    pool_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(pool_4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.8)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.8)(x)
    #classifying layer
    #x = Conv2D(4096, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    #x = BilinearUpSampling2D(size=(32, 32))(x)
    #### Block 7
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='fc7', kernel_regularizer=l2(weight_decay))(x)
    
    ### Block 8
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(512, (2, 2), strides=2, padding='same', name='fc8_Conv2DTranspose_1', kernel_regularizer=l2(weight_decay))(x)
    x = Add(name='fc8_conc')([pool_4, x])
    x = Conv2DTranspose(256, (2, 2), strides=2, padding='same', name='fc8_Conv2DTranspose_2', kernel_regularizer=l2(weight_decay))(x)
    #x = Conv2D(256, (1, 1), activation='relu', padding='same', name='fc8', kernel_regularizer=l2(weight_decay))(x)
    x = Add()([pool_3, x])
    x = Conv2DTranspose(256, (16, 16), strides=4, padding='same', name='fc8_Conv2DTranspose_3', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2DTranspose(3, (2, 2), strides=2, padding='same', name='fc8_Conv2DTranspose_4', kernel_regularizer=l2(weight_decay))(x)
    print(x)
    model = Model(img_input, x)

    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5.h5'))
    #weights_path = r'C:\Users\zeref\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #model.load_weights(weights_path, by_name=True)
    return model

print(FCN_Vgg16_32s((256, 256, 3)).summary())