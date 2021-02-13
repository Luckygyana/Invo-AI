import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.dataloader import DataGenerator

import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

path = r'E:\source\sem_8\invoice\invoice\Images'
gen = DataGenerator(path, 'mask.csv')
while(True):
    gen.__getitem__(5)
# img = cv2.imread(path)
# img = cv2.resize(img, (1024, 1024))

# model = keras.models.load_model('E:\source\sem_8\invoice\invoice\weights-0.00.hdf5')
# img = np.reshape(img, (1, 1024, 1024, 3))
# result = model.predict(img)
# plt.imshow(result[0])
# plt.show()