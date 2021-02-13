import os
import cv2
import numpy as np
from cfg import config as CONF
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2


def binary_(image):
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(image) > 127:
            image = cv2.bitwise_not(image)
    return image


def deblur(img, filename, index, jobId, filepath):
    int_dir = os.path.join(CONF.out_dir, str(jobId))
    int_dir = os.path.join(int_dir, filename.split('.')[0])
    print(int_dir)
    if not os.path.exists(int_dir):
        os.makedirs(int_dir, exist_ok=True)
    _dict = dict()
    _dict['original'] = filepath
    print(cv2.Laplacian(img, cv2.CV_64F).var())
    cl = binary_(image=img)
    output = os.path.join(int_dir, 'open.jpg')
    _dict['open'] = output
    cv2.imwrite(output, cl)
    boundary = cv2.Canny(img, 50, 50)
    cl = cv2.addWeighted(cl, 1.5, boundary, 1.5, 0)
    output = os.path.join(int_dir, 'weight_open_1.5_boundary_1.5.jpg')
    _dict['open + edge detection'] = output
    cv2.imwrite(output, cl)
    cl = cv2.addWeighted(cl, 1.5, img, 0.5, 0)
    output = os.path.join(int_dir, 'final.jpg')
    _dict['clahe'] = output
    cv2.imwrite(output, cl)
    li = {"images" : _dict}
    return cl, li



