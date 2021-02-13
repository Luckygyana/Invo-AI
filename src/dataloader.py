
import os
import cv2
import numpy as np
from PIL import Image
import keras
import random
import pandas as pd
import matplotlib.pyplot as plt

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_dir, csv_path=None, batch_size=32, shuffle=True):
        'Initialization'
        #self.dim = dim
        self.batch_size = batch_size
        self.img_dir = image_dir
        self.img = os.listdir(self.img_dir)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.convert_to_dict(csv_path)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img) / self.batch_size))

    def convert_to_dict(self, path):
        self.data = {}
        with open('mask.csv', 'r') as f:
            next(f)
            for line in f:
                filename, xmin, ymin, xmax, ymax = line.strip().split(',')
                if filename + '.bmp' not in self.data:
                    self.data[filename+'.bmp'] = [{'xmin' : int(xmin),
                                    'ymin' : int(ymin),
                                    'xmax' : int(xmax),
                                    'ymax' : int(ymax)}]
                else:
                    self.data[filename+'.bmp'].append({'xmin' : int(xmin),
                                    'ymin' : int(ymin),
                                    'xmax' : int(xmax),
                                    'ymax' : int(ymax)})


        


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.img[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        images = []
        output = []
        for filename in list_IDs_temp:
            img = cv2.imread(os.path.join(self.img_dir, filename))
            if img is not None:
                if filename in self.data:
                    print("shape of img : {}".format(img.shape))
                    h, w = img.shape[:2]
                    x_ratio = 1024 / h
                    y_ratio = 1024 / w
                    img = cv2.resize(img, (1024, 1024))
                    images.append(img)
                    out = img.copy()
                    _temp = self.data[filename]
                    for temp in _temp:
                        out[int(temp['xmin'] * x_ratio):int(temp['xmax'] * x_ratio), int(temp['ymin'] * y_ratio):int(temp['ymax'] * y_ratio), :] = 0
                    output.append(out)
                    p = np.concatenate((img, out), axis=1)
                    plt.imshow(p)
                    
                    plt.show()
                    # plt.imshow(img)
                    # plt.show()
        print(len(images))
        print(len(output))
        images = np.array(images)
        output = np.array(output)
        print("shape of images {} ".format(images.shape))
        print("shape of output {} ".format(output.shape))
        return images, output
