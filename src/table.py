import cv2
import sys
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'E:\Applications\Tesseract-OCR\tesseract.exe'

# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    x2 = np.add(x1, x2)
    y2 = np.add(y1, y2)
    # print(x1.shape, x2.shape)
    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def solve(org_image=None, img=None, first = True, psm=11):

    #thresholding the image to a binary image
    img = cv2.GaussianBlur(img,(5,5),0)
    # if not first:
    #     img = np.where(img < 190, img, 255)
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
    thresh,img_bin = cv2.threshold(img,1,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    #inverting the image 
    img_bin = 255-img_bin
    cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\cv_inverted.png',img_bin)
    #Plotting the image to see the output
    plotting = plt.imshow(img_bin,cmap='gray')
    # plt.show()

    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\vertical.jpg',vertical_lines)
    #Plot the generated image
    plotting = plt.imshow(image_1,cmap='gray')
    # plt.show()


    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\horizontal.jpg',horizontal_lines)
    #Plot the generated image
    plotting = plt.imshow(image_2,cmap='gray')
    # plt.show()


    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite(r'E:\source\sem_8\invoice\invoice\imgs\inverted\img_vh.jpg', img_vh)
    # bitxor = cv2.bitwise_xor(img,img_vh)
    # bitnot = cv2.bitwise_not(bitxor)
    # #Plotting the generated image
    # plotting = plt.imshow(bitnot,cmap='gray')
    # #plt.show()
    # image = img_vh.copy()
    # out = ndi.distance_transform_edt(~image)
    # out = out < 0.1 * out.max()
    # out = morphology.skeletonize(out)
    # out = morphology.binary_dilation(out, morphology.selem.disk(1))
    # out = segmentation.clear_border(out)
    # out = out | image
    if first:
        # out = np.where(out < 10, 0, 255)
        out = 255 - img_vh
        out = out.astype(np.uint8)
        kernel = np.ones((1,30),np.uint8) 
        dilation = cv2.dilate(out,kernel,iterations = 2)
        dilation = cv2.erode(dilation,kernel,iterations = 1)
        kernel = np.ones((30,1),np.uint8) 
        dilation = cv2.dilate(dilation,kernel,iterations = 2)
        dilation = cv2.erode(dilation,kernel,iterations = 1)
        kernel = [[1, 1, 1],
                [1, 10, 1],
                [1, 1, 1]]
        kernel = np.array(kernel)
        kernel = kernel.astype(np.uint8)
        # dilation = ndi.convolve(dilation, kernel, mode='constant',)
        plt.imshow(dilation, cmap='gray')
        plt.show()

    # if not first:
    # sys.exit()
    if first:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    else:
        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    print(hierarchy)



    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    def sort_contours_new(cnts):
        bboxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, bboxes),
        key=lambda b:b[1][2] * b[1][3], reverse=True))
        return (cnts, boundingBoxes)

    # Sort all the contours by top to bottom.
    # contours, boundingBoxes = sort_contours_new(contours)
    # if first:
    #     _contours = contours[1:]
    # else:
    #     contours = contours[1:]
    #     contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # print(boundingBoxes)
    def sort_bbox(boundingBoxes):
        return sorted(boundingBoxes, key=lambda x : x[2]*x[3], reverse=True)
    boundingBoxes = sort_bbox(boundingBoxes)

    # sys.exit(0)
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    widths = [boundingBoxes[i][2] for i in range(len(boundingBoxes))]
    #Get mean of heights
    mean = np.mean(heights)
    # print(np.mean(heights), np.mean(widths))

    box = []
    if first:
        boundingBoxes = boundingBoxes[1:]
    else:
        boundingBoxes = boundingBoxes[2:]
    for c in boundingBoxes:

        # x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = c
        if first:
            # print('plotting....')
            plt.imshow(img, cmap='gray')
            #plt.show()
            # print(x, y, w, h)
            img = cv2.rectangle(img, (x,y),(x+w,y+h), (10), 3)
            # print('After rect at {} {} {} {}'.format(x, y, w, h))
            solve(org_image, img, False, psm)
            return
        if (h<15 or w<15):
            continue
        # print(x, y, w, h)
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0),2)

        box.append([x,y,w,h])
    # plotting = plt.imshow(image,cmap='gray')
    # plt.show()
    # sys.exit(0)

    # print(box)
    row=[]
    column=[]
    j=0
    #Sorting the boxes to their respective row and column
    #for i in range(len(box)):
    #    if(i==0):
    #        column.append(box[i])
    #        previous=box[i]
    #    else:
    #        if(box[i][1]<=previous[1]+mean/2):
    #            column.append(box[i])
    #            previous=box[i]
    #            if(i==len(box)-1):
    #                row.append(column)
    #        else:
    #            row.append(column)
    #            column=[]
    #            previous = box[i]
    #            column.append(box[i])
    #print(column)
    #print(row)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    height, width = img.shape[:2]
    video = cv2.VideoWriter('tess_{}.avi'.format(psm), fourcc, float(1), (width, height))

    # print("{}, {}".format(height, width))
    img_white = org_image.copy()
    #org1 = (width - 1000, height - 200)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    thickness = 2
    color = (0, 0, 255)
    fontscale = 1
    probs = np.full((len(box)), 1)
    # def func(key1, key2):
    #     if(abs(key1[0] - key2[0]) < 5):
    #         if(key1[1] < key2[1]):
    #             return -1
    #         else:
    #             return 1
    #     elif(key1[0] < key2[0]):
    #         return -1
    #     else:
    #         return 1
    def func1(key1, key2):
        # print(key1, key2)
        if(abs(key1[1] - key2[1]) < 5):
            if(key1[0] < key2[0]):
                # print('a')
                return -1
            else:
                # print('b')
                return 1
        elif(key1[1] < key2[1]):
            # print('c')
            return -1
        else:
            # print('d')
            return 1
    box = sorted(box, key=functools.cmp_to_key(func1))
    # sys.exit(0)
    # box, probs = non_max_suppression_fast(box, probs, overlap_thresh=0.9)
    for b in box:
        x, y, w, h = b
        cv2.rectangle(img_white,(x,y),(x+w,y+h),(0,255,0),2)
        img1 = img_white.copy()
# <<<<<<< version-1.0
        img2 = org_image[ y:y+h, x:x+w, :]
# =======
#         img2 = org_image[ y-3:y+h+3, x-3:x+w+3, :]
# >>>>>>> master
        #plt.imshow(img2)
        #plt.show()
        text = pytesseract.image_to_string(img2, output_type=pytesseract.Output.DICT, lang='eng', config='-c  tessedit_char_whitelist="$%@.,&():ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/\\\n "  --psm {} --oem {}'.format(psm, 3))
        # print(text)
        if(text['text'] == ''):
            cv2.imwrite("temp/{}_{}.jpg".format(x, y), img2)
        text = text['text'].split('\n')
        text = [t.strip('\n') for t in text]
        text = [t for t in text if t != '']
        for t in text:
            image1 = img1.copy()
            image = cv2.putText(image1, t, (x, y), font, fontscale, color, thickness, cv2.LINE_AA)
            video.write(image)

    plt.imshow(img_white)
    cv2.imwrite('table_{}.jpg'.format(psm), img_white)
    # plt.show()

#countcol = 0
#for i in range(len(row)):
#    countcol = len(row[i])
#    if countcol > countcol:
#        countcol = countcol


#center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
#center=np.array(center)
#center.sort()

#finalboxes = []
#for i in range(len(row)):
#    lis=[]
#    for k in range(countcol):
#        lis.append([])
#    for j in range(len(row[i])):
#        diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
#        minimum = min(diff)
#        indexing = list(diff).index(minimum)
#        lis[indexing].append(row[i][j])
#    finalboxes.append(lis)

#outer=[]
#for i in range(len(finalboxes)):
#    for j in range(len(finalboxes[i])):
#        inner=''
#        if(len(finalboxes[i][j])==0):
#            outer.append(' ')
#        else:
#            for k in range(len(finalboxes[i][j])):
#                y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
#                finalimg = bitnot[x:x+h, y:y+w]
#                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
#                border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
#                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#                dilation = cv2.dilate(resizing, kernel,iterations=1)
#                erosion = cv2.erode(dilation, kernel,iterations=1)

                
#                out = pytesseract.image_to_string(erosion)
#                if(len(out)==0):
#                    out = pytesseract.image_to_string(erosion, config='--psm 3')
#                inner = inner +" "+ out
#            outer.append(inner)

#arr = np.array(outer)
#dataframe = pd.DataFrame(arr.reshape(len(row),countcol))
#print(dataframe)
#data = dataframe.style.set_properties(align="left")
##Converting it in a excel-file
#data.to_excel('output.xlsx')

file=r'E:\source\sem_8\invoice\invoice\imgs\300\Sample11-0.jpg'

org_image = cv2.imread(file)
img = cv2.imread(file,0)
# for i in range(13):
#     try:
#         print(f'psm={i}')
#         solve(org_image, img, True, i)
#     except Exception as e:
#         print(e)
#         continue
solve(org_image, img, True, 11)
from src.ocr import pdf_to_image
# 
# pdf_to_image(r'E:\source\sem_8\invoice\invoice\Sample Invoices', dpi=500)

