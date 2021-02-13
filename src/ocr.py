import os
import sys
import cv2
import time
import copy
import imutils
import numpy as np
import matplotlib.pyplot as plt
#from .ShapeDetector import ShapeDetector
#import matplotlib.pyplot as plt
try:
    from PIL import Image, ImageChops
except ImportError:
    import Image
import pytesseract
from cv2 import VideoWriter, VideoWriter_fourcc

pytesseract.pytesseract.tesseract_cmd = r'E:\Applications\Tesseract-OCR\tesseract.exe'

#print(pytesseract.image_to_string(Image.open('example_01.png')))


inp_path = r'E:\source\sem_8\invoice\invoice\Sample Invoices\Sample Invoice_1.pdf'
out_path = r'E:\source\sem_8\invoice\invoice\Sample Invoices\Sample Invoice_1.jpg'


from pdf2image import convert_from_path, convert_from_bytes



def crop(img, path):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 249, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23,23))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    cv2.imwrite(path, dst)






def pdf_to_image(path_folder, dpi=200):
    files = os.listdir(path_folder)
    for file in files:
        inp_path = os.path.join(path_folder, file)
        print(inp_path)
        pages = convert_from_path(inp_path, dpi, fmt='jpeg')
        i = 0
        if not os.path.exists(os.path.join(os.getcwd(), 'imgs/{}'.format(dpi))):
            os.mkdir(os.path.join(os.getcwd(), 'imgs/{}'.format(dpi)))
        for page in pages:
            page.save('./imgs/{}/{}-{}.jpg'.format(dpi, file.split('.')[0], i), 'jpeg')
            #crop(page, './imgs/{}_{}.jpg'.format(file.split('.')[0], i))
            #line_detect('./imgs/{}_{}.jpg'.format(file.split('.')[0], i))
            i += 1
#pdf_to_image(r'E:\source\sem_8\invoice\invoice\Sample Invoices')


##d = pytesseract.image_to_data(img, output_type=Output.DICT, config="-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz")
def check(path=r'./imgs/300/Sample11-0.jpg', dpi=300):
    for p in range(11, 12):
        for o in range(3, 4):
            img = cv2.imread('./imgs/200/Sample21-0.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img1 = copy.deepcopy(img)
            #temp = sys.stdout
            #f = open('out.txt', 'w')
            #sys.stdout = f
            d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "  --psm {} --oem {}'.format(p, o))
            n_boxes = len(d['level'])
            print(d)
            #f.close()
            #sys.stdout = temp
            print('Read image')
            try:
                width = img.shape[1]
                height = img.shape[0]
                fps = 1
                font = cv2.FONT_HERSHEY_SIMPLEX 
                thickness = 2
                color = (0, 0, 255)
                fontscale = 1
                org = (width - 1000, height - 100)
                org1 = (width - 1000, height - 200)
                #fourcc = VideoWriter_fourcc(*'XVID')
                #video = VideoWriter('tess_with_sp_psm_{}_oem_{}.avi'.format(p, o), fourcc, float(fps), (width, height))

                for i in range(n_boxes):
                    #if(d['width'][i] <= 10 or d['height'][i] <= 10):
                    #    continue
                    if(d['text'][i] ==''):
                        continue
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    img1 = img1[y - 5 : y + h + 5, x - 5: x+ w + 5]
                    print(pytesseract.image_to_string(img1, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "  --psm {} --oem {}'.format(p, o)))
                    cv2.imwrite('temp/{}.jpg'.format(i), img1)
                    img1 = copy.deepcopy(img)
                    text = 'w_{}_h_{}'.format(d['width'][i], d['height'][i])
                    image = cv2.putText(img1, text, org, font, fontscale, color, thickness, cv2.LINE_AA) 
                    image = cv2.putText(img1, d['text'][i], org1, font, fontscale, color, thickness, cv2.LINE_AA) 
                    #video.write(img1)
                cv2.imwrite('./{}_tess_with_sp_psm_{}_oem_{}.jpg'.format(dpi, p, o), img)
                print('image saved')
            except Exception as e:
                print('fsdfa')
                print("{} {}".format(p, o))
                print(e)

#print(pytesseract.image_to_string(Image.open('./imgs/Sample21-0.jpg'), lang='eng', config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))  
#sys.stdout = temp

check()

