import re
import sys
import copy
import nltk
import functools
import numpy as np
from gensim.models import Word2Vec
from autocorrect import Speller
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering




spell = Speller()

class BoundingBox:

    def __init__(self, bbox, mode='l2r'):
        self.bbox = bbox
        self.bbox = sorted(self.bbox, key=lambda x : x[2]*x[3], reverse=True)
        self.header = ['S. No.', 'S. N.', 'S. No.', 'Sr. No.', 'Product ID', 'Item Code ', 'Cust Mat Code', 'Material Code',    ###List of header attribute
                        'Material Description', 'Part No', 'ISBN-13', 'Product Code', 'SKU', 'SKU\tSales Order/No/Pos/SeqHSN',  ## make nw additions
                        'HSN Code', 'HSN/SAC', 'HSN of Goods/ Services', 'Title', 'Item Description', 'Description of Goods',
                        'Desc of Goods/Services', 'Title & Author', 'Quantity', 'Quantity', 'QTY\tNo of packages',
                        'SUPPL Qty', 'Unit Price', 'Mrp per Unit', 'Basic Rate', 'Unit Price\t',
                         'Excise Duty', 'Freight']
                        # 'Discount Percentage', 'Disc.', 'Disc. (INR)', 'Cash Discount (CD / amount * 100)SGST Percentage',
                        # 'SGST/ UTGST', 'Tax %', 'Tax Rate', 'CGST Percentage', 'CGST Tax %', 'IGST Percentage', 'IGST Tax %',
                        # 'Cess Percentage',
                        #  'TCS Percentage', 'Total tax/Total amount * 100', 'Grand Total - Basic Total', 'Total Amount',
                        # 'Net Payable','Ord.' 'Total (R/O)', 'Grand Total\t', 'APP%', 'Line Total', 'Total quantity pcs'
                        # ]
        self.remove_list =['id', 'basic', 'total', 'tax', '']
        # self.split_header()
        # print(self.header)
        # sys.exit()
        self.header = [self.clean(head) for head in self.header]
        self.row_string = None
        self.similarity_index = None
        # self.sort_row_wise()
    def split_header(self):   ### optimization needed.

        """ Add optimization to header to increase accuracy 
        """
        temp = []
        for head in self.header:
            for _ in re.split('\\\\|/|\\ |\\\t|\\(|\\)', head):
                temp.append(_)

        temp = sorted(list(set(temp)))
        temp = [t for t in temp if t.lower() not in self.remove_list and len(t) > 1]
        self.header = temp


    def clean(self, word):
        word = re.sub('[&()-/\.!@#$%*0-9\\t\\n]', '', word)
        return word.lower()

    def sort_increasing(self, mode='l2r'):
        """
        Sort cells as per increasing order (based on left, top)
        """
        if mode=='l2r':
            def func1(key1, key2):
                if(abs(key1[1] - key2[1]) < 10):   ### parameter to optimize (variable allowed)
                    return -1 if(key1[0] < key2[0]) else 1
                elif(key1[1] < key2[1]):
                    return -1
                else:
                    return 1
        else:
            def func1(key1, key2):
                if(abs(key1[0] - key2[0]) < 10):
                    return -1 if(key1[1] < key2[1]) else 1
                elif(key1[0] < key2[0]):
                    return -1
                else:
                    return 1
        bbox_inc = copy.deepcopy(self.bbox)
        if mode == 'l2r':
            self.bbox_l2r = sorted(bbox_inc,key=functools.cmp_to_key(func1))
        else:
            self.bbox_t2b = sorted(bbox_inc,key=functools.cmp_to_key(func1))

    def sort_row_wise(self):
        """
        Arrange bboxes in row wise order.
        """
        rows = []
        i = 0
        temp = []
        for box in self.bbox_l2r:
            if(len(temp) == 0):
                temp.append(box)
            else:
                _box = temp[-1] 
                if(abs(box[1]-_box[1]) < 10):    ### parameter to be optimized (difference in value allowed in row in top i.e box[1]).
                    temp.append(box)
                else:
                    _temp = copy.deepcopy(temp)
                    rows.append(_temp)
                    temp = [box]

        self.rows = rows

    def sort_col_wise(self):
        rows = []
        i = 0
        temp = []
        for box in self.bbox_l2r:
            if(len(temp) == 0):
                temp.append(box)
            else:
                _box = temp[-1]
                if(abs(box[1]-_box[1]) < 5):
                    temp.append(box)
                else:
                    _temp = copy.deepcopy(temp)
                    rows.append(_temp)
                    temp = [box]

        self.col = rows

    def isSubSequence(self,str1,str2): 
        m = len(str1) 
        n = len(str2) 
        
        j = 0    # Index of str1 
        i = 0    # Index of str2 
        while j<m and i<n: 
            if str1[j] == str2[i]:     
                j = j+1    
            i = i + 1
            
        return j==m 
    def calculate_similarity(self, row):  ## can be optimized.
        similarity_count = 0
        for sentence in row:
            # print(sentence)clean
            sentence = spell(sentence)
            sentence = self.clean(sentence)
            for word in sentence.split(' '):
                if word in self.header:
                    similarity_count+=1
            # else:
            #     for head in self.header:
            #         if int(self.isSubSequence(head, sentence)) or int(self.isSubSequence(sentence,head)):
            #             similarity_count += 0.5

        return similarity_count

    def row_sim_index(self):  
        """ Create an index determining its chances to be table header.
        """
        self.similarity_index = []
        for row in self.row_string:
            self.similarity_index.append(self.calculate_similarity(row))

        self.similarity_index = np.array(self.similarity_index)
        print(self.similarity_index)

    def clean_rows(self):
        self.row_string = [[self.clean(r) for r in row] for row in self.row_string] 

    def row_sim_index_new(self): ### variant of self.row_sim_index (can create a new version altogether)
        sentences = copy.deepcopy(self.row_string)
        # sentences.append(self.header)
        # print('sentences')
        s = []
        for rows in sentences:
            temp = []
            for words in rows:
                for word in words.split(' '):
                    temp.append(self.clean(word))
            s.append(temp)
        sentences = s

        print(sentences)
        m = Word2Vec(sentences, size=50, min_count=1,sg=1)

        def vectorizer(sent, m):
            vec = []
            numw = 0
            for w in sent:
                try:
                    if numw == 0:
                        vec = m[w]
                    else:
                        vec = np.add(vec, m[w])
                except:
                    pass
                numw += 1
            return np.asarray(vec) / numw

        l = []
        for i in sentences:
            l.append(vectorizer(i, m))

        X = np.array(l)
        # print(X)
        clf = KMeans(n_clusters=3, max_iter=100, init='k-means++', n_init=1)
        labels = clf.fit_predict(X)
        print(labels)

            



