print('inro srch dvkhs')
import sys
import copy
import pytesseract
from cfg import config as CONF
print('started')
from joblib import delayed, Parallel
from multiprocessing import Pool
from src.bbox_manager import BoundingBox
print('started')
from joblib import wrap_non_picklable_objects
print('started')

class Saver(object):
    def __init__(self):
        super().__init__()
        self.bbox = None
        self.image = None
        
    def init(self, bbox, image):
        self.bbox = bbox
        self.image = image

saved = Saver()


def extractor(r, rows, img):
    # print(r)
    # print((box.rows))
    words = [""] * len(rows[r]) 
    row = rows[r]
    for i in range(len(row)):
        x, y, w, h = row[i]
        img2 = img[y:y+h, x:x+w, :]
        text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\ "  --psm {} --oem {}'.format(CONF.psm, CONF.oem))
        text = text['text']
        words[i] = text
    # key_word[r] = words
    # print(words)
    return words

row = None
img = None
def extractor2(r):
    print(r)
    # print((box.rows))
    words = [""] * len(rows[r]) 
    row = rows[r]
    for i in range(len(row)):
        x, y, w, h = row[i]
        img2 = img[y:y+h, x:x+w, :]
        text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\ "  --psm {} --oem {}'.format(CONF.psm, CONF.oem))
        text = text['text']
        words[i] = text
    # key_word[r] = words
    # print(words)
    return words

def parallelize(li, bbox, image):
    global row
    global img
    li = list(li)
    rows = bbox.rows
    img = image
    inp = [(r, rows, img) for r in li]
    # key_words = Parallel(n_jobs=-1, verbose=100)(__func(r) for r in li)
    with Pool(1) as p:
        key_word = p.starmap(extractor, inp)
    return key_word


def func2(r, rows, img):
    # print(r)
    # print((box.rows))
    words = [""] * len(rows[r]) 
    row = rows[r]
    for i in range(len(row)):
        x, y, w, h = row[i]
        # img2 = self.img[y-3:y+h+3, x-3:x+w+3, :]
        img2 = img[y:y+h, x:x+w, :]
        text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\ "  --psm {} --oem {}'.format(CONF.psm, CONF.oem))
        # print(text)
        text = text['text']
        # if text == '':
        #     plt.imshow(img2, cmap='gray')
        #     plt.show()
        #     li = [img2]
        #     text = self.recognizer.recognize(li)
        #     # text = self.recognizer.recognize_from_boxes([self.img], [[[x, y, w, h]]])
        #     print(text)
        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),2)
        # plt.imshow(image)
        # plt.show()
        words[i] = text
    # key_word[r] = words
    # print(words)
    return words
