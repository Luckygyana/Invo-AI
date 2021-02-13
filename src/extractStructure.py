print('into src')
import os
import re
import sys
import cv2
import copy
import glob
import time
import random
import shutil
import joblib
import numpy as np
print('started')
from src.parallel_calc import parallelize
print('started')
from multiprocessing import Pool
from cfg import config as CONF
import matplotlib.pyplot as plt
from src.bbox_manager import BoundingBox
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from scipy import ndimage as ndi
from src.deblurr import deblur
from joblib import wrap_non_picklable_objects
from pdf2image import convert_from_path, convert_from_bytes
from skimage import io, morphology, img_as_bool, segmentation
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
print('started')
# import keras_ocr


4
font = cv2.FONT_HERSHEY_SIMPLEX 
thickness = 2
color = (0, 0, 255)
fontscale = 1


############ Extract basename ################################################

def extract_basename(path):
  """Extracts basename of a given path. Should Work with any OS Path on any OS"""
  basename = re.search(r'[^\\/]+(?=[\\/]?$)', path)
  if basename:
    return basename.group(0)

class ExtractStructure(object):

    """
    Class to manage the bounding box classes 


    extract_images : For each pdf image is extracted and stored in intermediate location
    self.images : List of file location of extracted images

    binary_images : Manipulates images to extract Contour
    self.mapping_l2r : dict() : Stores list of bounding boxes as values for each image as key  l2r : Left-to-right
    self.mapping_t2b : dict() : Stores list of bounding boxes as values for each image as key  l2r : top-to-bottom

    """

    def __init__(self, path, recognizer, jobId):
        super().__init__()
        self.filePath = path
        self.filename = re.split(r'\\|/', path)[-1]
        # print(self.filename)
        self.images = []
        self.recognition_kwargs = CONF.alphabet
        self.recognizer = recognizer
        self.jobId = jobId
        self.result = {}

    def check_blurr(self, image):
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        if(variance > 100):
            return True            # Not Blurr
        else:
            return False              # Blurr

    def extract_images(self, path='./temp', dpi=300):
        files = glob.glob(os.path.join(path, '*'))
        for file in files:
            try:
                os.remove(file)
            except:
                shutil.rmtree(file)
        if not os.path.exists(os.path.join(path, '{}'.format(dpi))):
            os.mkdir(os.path.join(path, '{}'.format(dpi)))
        pages = convert_from_path(self.filePath, dpi, fmt='jpeg')
        i = 0
        if not os.path.exists(os.path.join(os.getcwd(), '{}'.format(dpi))):
            os.mkdir(os.path.join(os.getcwd(), '{}'.format(dpi)))
        for page in pages:
            file_path = os.path.join(path, '{}/{}-{}.jpg'.format(dpi, self.filename.split('.')[0], i))
            page.save(file_path, 'jpeg')
            i += 1
        self.images = glob.glob(os.path.join(path, str(dpi)+ '/*'))

    def binary_images(self, img, first=True):
        """
        create binary image from original image to detect contours
        """
        t = time.time()
        # img = cv2.GaussianBlur(img,(5,5),0)
        thresh,img_bin = cv2.threshold(img,1,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
        img_bin = 255-img_bin
        # cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\cv_inverted.png',img_bin)
        # plotting = plt.imshow(img_bin,cmap='gray')
        kernel_len = np.array(img).shape[1]//100
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
        # cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\vertical.jpg',vertical_lines)
        # plotting = plt.imshow(image_1,cmap='gray')
        image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
        # cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\horizontal.jpg',horizontal_lines)
        # plotting = plt.imshow(image_2,cmap='gray')

        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        img_vh = cv2.erode(~img_vh, kernel, iterations=2)
        thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\img_vh.jpg', img_vh)
        bitxor = cv2.bitwise_xor(img,img_vh)
        bitnot = cv2.bitwise_not(bitxor)
        plotting = plt.imshow(bitxor,cmap='gray')
        # plt.show()
        
        if True:
            # image = img_vh.copy()
            # out = ndi.distance_transform_edt(~image)
            # out = out < 0.05 * out.max()
            # out = morphology.skeletonize(out)
            # out = morphology.binary_dilation(out, morphology.selem.disk(1))
            # out = segmentation.clear_border(out)
            # out = out | image
            img_vh = 255 - img_vh
            kernel = np.ones((1,15),np.uint8) 
            dilation = cv2.dilate(img_vh,kernel,iterations = 1)
            dilation = cv2.erode(dilation,kernel,iterations = 1)
            kernel = np.ones((15,1),np.uint8) 
            dilation = cv2.dilate(dilation,kernel,iterations = 1)
            dilation = cv2.erode(dilation,kernel,iterations = 1)
            dilation = 255 - dilation
            cv2.imwrite('dilation.jpg', dilation)
            return dilation
        # print(time.time()-t)
        return img_vh
        
        

    def bboxes_mapper(self, sorting = 'rows'):
        """
        Create dictionary mapping each image with its contours
        """
        if sorting == 'rows':
            self.mapping_l2r = dict()
        else:
            self.mapping_t2b = dict()
        for i in range(len(self.images)):
            org_img = cv2.imread(self.images[i])
            self.shape = org_img.shape
            img = cv2.imread(self.images[i], 0)
            file = self.images[i]
            _, li = deblur(img, self.filename, i, self.jobId, self.images[i])
            self.result[str(i)] = li
            # if(self.check_blurr(img)):
            #     img = self.unblurr(img)
            out = self.binary_images(img, True)
            contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            boundingBoxes = [cv2.boundingRect(c) for c in contours]
            if sorting=='rows':
                Boxes = BoundingBox(boundingBoxes)
                x, y, w, h = Boxes.bbox[1]
                img = cv2.rectangle(img, (x,y),(x+w,y+h), (10), 3)
                out = self.binary_images(img, False)
                cv2.imwrite('repeat.jpg', out)
                contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                boundingBoxes = [cv2.boundingRect(c) for c in contours]
                Boxes = BoundingBox(boundingBoxes)
                Boxes.bbox = Boxes.bbox[2:]
                Boxes.bbox = [box for box in Boxes.bbox if (box[2] > 15 and box[3] > 15)]
                Boxes.sort_increasing(mode='l2r')
                self.mapping_l2r[file] = Boxes
                # self.mapping_l2r[file].sort_row_wise()

            if sorting=='cols':
                Boxes = BoundingBox(boundingBoxes, mode='t2b')
                x, y, w, h = Boxes.bbox[1]
                img = cv2.rectangle(img, (x,y),(x+w,y+h), (10), 3)
                out = self.binary_imgaes(img, False)
                contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                boundingBoxes = [cv2.boundingRect(c) for c in contours]
                Boxes = BoundingBox(boundingBoxes)
                Boxes.bbox = Boxes.bbox[2:]
                Boxes.bbox = [box for box in Boxes.bbox if (box[2] > 15 and box[3] > 15)]
                Boxes.sort_increasing(mode='t2b')
                self.mapping_t2b[file] = Boxes
                # self.mapping_l2r[file].sort_row_wise()

    def row_keyword_mapper(self, psm=11, oem=3, sorting='rows'):

        ### Maps text in each bounding boxes for file names
        if sorting == 'rows':
            self.key_mapping_l2r = dict()
            
                
            for key in self.mapping_l2r.keys():
                self.img = cv2.imread(key)
                print(key)
                height, width = self.img.shape[:2]
                video = None
                if CONF.write_video:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fname = extract_basename(key)
                    video = cv2.VideoWriter('tess_{}.avi'.format(fname.split('.')[0]), fourcc, float(1), (width, height))
                # print(height, width)
                # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                box = self.mapping_l2r[key]  
                key_word = []  
                ###### Parallelization can be used
                box.sort_row_wise()
                img = self.img.copy()
                key_word = self.func(box,video=video)

                ####################################
#                 for row in box.rows:
#                     words = []
#                     for box in row:
#                         x, y, w, h = box
#                         # img2 = self.img[y-3:y+h+3, x-3:x+w+3, :]
#                         img2 = self.img[y:y+h, x:x+w, :]
#                         text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\ "  --psm {} --oem {}'.format(psm, oem))
#                         # print(text)
#                         text = text['text']
#                         # if text == '':
#                         #     plt.imshow(img2, cmap='gray')
#                         #     plt.show()
#                         #     li = [img2]
#                         #     text = self.recognizer.recognize(li)
#                         #     # text = self.recognizer.recognize_from_boxes([self.img], [[[x, y, w, h]]])
#                         #     print(text)
#                         img = cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),2)
                        
#                         if CONF.write_video:
#                             text = text.split('\n')
#                             text = [t.strip('\n') for t in text]
#                             text = [t for t in text if t != '']
#                             _text = copy.deepcopy(text)
#                             text = ' '.join(text)
#                             for t in _text:
#                                 temp_img = img.copy()
#                                 image = cv2.putText(temp_img, t, (x, y), font, fontscale, color, thickness, cv2.LINE_AA)
#                                 video.write(image)
#                         # plt.imshow(image)
#                         # plt.show()
#                         words.append(text)
#                     key_word.append(words)
#                     ################################
                self.key_mapping_l2r[key] = key_word
                cv2.imwrite('out.jpg', img)
        else:
            self.key_mapping_t2b = dict()
            for key in self.mapping_t2b.keys():
                self.img = cv2.imread(key)
                box = self.mapping_t2b[key]  
                key_word = []  
                ###### Parallelization can be used
                box.sort_col_wise()
                for row in box.rows:
                    words = []
                    for box in row:
                        x, y, w, h = box
                        img2 = self.img[y-3:y+h+3, x-3:x+w+3, :]
                        text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config=CONF.TESS_CONFIG)
#                         text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\\n "  --psm {} --oem {}'.format(psm, oem))

                        text = text['text'].split('\n')
                        text = [t.strip('\n') for t in text]
                        text = [t for t in text if t != '']
                        words.append(text)
                    key_word.append(words)
                self.key_mapping_t2b[key] = key_word



    def detect_header(self):
        ### Detecting the header row;
        for key in self.mapping_l2r.keys():
            self.mapping_l2r[key].row_string = self.key_mapping_l2r[key]
            # self.mapping_l2r[key].clean_rows()
            # print(self.mapping_l2r[key].row_string)
            # self.mapping_l2r[key].row_sim_index_new()
            self.mapping_l2r[key].row_sim_index()


    def extra_reader(self):
        
        for key in self.mapping_l2r.keys():
            self.img = cv2.imread(key)
            img = self.img.copy()
            for box in self.mapping_l2r[key].bbox:
                x, y, w, h = box
                img[y:y+h, x:x+w, :] = 0
            plt.imshow(img)
            # print("Showing blacked")
            text = pytesseract.image_to_string(img, output_type=pytesseract.Output.DICT, lang='eng', config=CONF.TESS_CONFIG)
            # text = text['text'].split('\n')
            # text = [t.strip('\n') for t in text]
            # text = [t for t in text if t != '']
            # sys.exit()
            self.mapping_l2r[key].outside = [text['text'].replace('\n', ' ')]
        # plt.show()




    def func(self, box, save_img=False, video=None):
        img = self.img.copy()
        rows = range(len(box.rows) - 1)
        # key_word = Parallel(n_jobs=2)(parallelize(r, box, img) for r in rows)
        # print(type(box))
        # sys.exit()
        if os.name == 'nt' or True:
            key_word = [[]] * len(box.rows) 
            # plt.show()
            for r in range(len(box.rows)):
                words = [""] * len(box.rows[r]) 
                row = box.rows[r]
                for i in range(len(row)):
                    x, y, w, h = row[i]
                    img2 = self.img[y-3:y+h+3, x-3:x+w+3, :]
                    # img2 = self.img[y:y+h, x:x+w, :]
                    text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\ "  --psm {} --oem {}'.format(CONF.psm, CONF.oem))
                    # print(text)
                    text = text['text']
                    text = text.replace('\x0c', '')
                    # if text == '':
                    #     plt.imshow(img2, cmap='gray')
                    #     plt.show()
                    #     li = [img2]
                    #     text = self.recognizer.recognize(li)
                    #     # text = self.recognizer.recognize_from_boxes([self.img], [[[x, y, w, h]]])
                    #     print(text)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),2)
                    if CONF.write_video:
                            text = text.split('\n')
                            text = [t.strip('\n') for t in text]
                            text = [t for t in text if t != '']
                            _text = copy.deepcopy(text)
                            text = ' '.join(text)
                            for t in _text:
                                temp_img = img.copy()
                                image = cv2.putText(temp_img, t, (x, y), font, fontscale, color, thickness, cv2.LINE_AA)
                                video.write(image)
                    # plt.imshow(image)
                    # plt.show()
                    words[i] = text
                key_word[r] = words
                # print(words)
        else:
            # print("Should not be here")
            key_word = parallelize(rows, box, img)
        plt.imshow(img)
        plt.show()
        cv2.imwrite(f'out1{str(random.randint(100, 100000))}.jpg', img)
        return key_word






            



            



            



            
