print('into bx')
import re
import sys
import copy
import nltk
import functools
import numpy as np
print('started')
from src.base import Bbox
print('started')
from gensim.models import Word2Vec
from autocorrect import Speller
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
print('started')
from cfg import config as CONF
print('started')
from src import similarity
print('started')



spell = Speller()

class BoundingBox(Bbox):

    def __init__(self, bbox, mode='l2r'):
        super(BoundingBox, self).__init__()
        self.bbox = bbox
        if self.bbox:
            self.bbox = sorted(self.bbox, key=lambda x : x[2]*x[3], reverse=True)
        
        # self.sort_row_wise()
    


    

    def sort_increasing(self, mode='l2r'):
        """
        Sort cells as per increasing order (based on left, top)
        """
        if mode=='l2r':
            def func1(key1, key2):
                if(abs(key1[1] - key2[1]) < CONF.ROW_TOLERANCE):   ### parameter to optimize (variable allowed)
                    return -1 if(key1[0] < key2[0]) else 1
                elif(key1[1] < key2[1]):
                    return -1
                else:
                    return 1
        else:
            def func1(key1, key2):
                if(abs(key1[0] - key2[0]) < CONF.COL_TOLERANCE):

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
                if(abs(box[1]-_box[1]) < CONF.ROW_TOLERANCE):    ### parameter to be optimized (difference in value allowed in row in top i.e box[1]).
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
        for box in self.bbox_t2b:
            if(len(temp) == 0):
                temp.append(box)
            else:
                _box = temp[-1]
                if(abs(box[1]-_box[1]) < CONF.COL_TOLERANCE):

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


    def calculate_similarity(self, row, func="edit_distance",close_matches_thres = None):  ## can be optimized.
        similarity_count = 0
        for sentence in row:
            sentence = spell(sentence)
            sentence = self.clean(sentence)
            if func == 'simple':
                for word in sentence.split(' '):
                    if word in self.header:
                        similarity_count+=1
            if func == 'sentence_encoder':
                for s in sentence.split(' '):
                    for t in self.header:
                        similarity_count += similarity.sentence_encoder(s,t)
            if func == 'levenshtein':
                for s in sentence.split(' '):
                    for t in self.header:
                        similarity_count += similarity.levenshtein_ratio_and_distance(s,t)
            if func == 'close_matches':
                for s in sentence.split(' '):
                    for t in self.header:
                        similarity_count += similarity.close_matches(s,t,close_matches_thres)
            if func == 'sequencematcher':
                for s in sentence.split(' '):
                    for t in self.header:
                        similarity_count += similarity.sequencematcher(s,t)
            if func == 'edit_distance':
                for s in sentence.split(' '):
                    for t in self.header:
                        similarity_count += similarity.edit_distance(s,t)
            if func == 'jaccard_distance':
                for s in sentence.split(' '):
                    for t in self.header:
                        similarity_count += similarity.jaccard_distance(s,t)

        return similarity_count


    def calculate_similarity_weights(self, row, func="edit_distance",close_matches_thres = None):
        similarity_count_n = 0
        similarity_count_m = 0
        n = 0
        m = 0
        for sentence in row:
            sentence = spell(sentence)
            sentence = self.clean(sentence)
            if func == 'simple':
                for word in sentence.split(' '):
                    if word in self.header:
                        if word in self.weights_header:
                            similarity_count_n +=1
                            n += 1
                        else :
                            similarity_count_m +=1
                            m +=1
            if func == 'sentence_encoder':
                for s in sentence.split(' '):
                    for t in self.header:
                        if t in self.weights_header:
                            similarity_count_n += similarity.sentence_encoder(s,t)
                            n += 1
                        else :
                            similarity_count_m += similarity.sentence_encoder(s,t)
                            m +=1
            if func == 'levenshtein':
                for s in sentence.split(' '):
                    for t in self.header:
                        if t in self.weights_header:
                            similarity_count_n += similarity.levenshtein_ratio_and_distance(s,t)
                            n += 1
                        else :
                            similarity_count_m += similarity.levenshtein_ratio_and_distance(s,t)
                            m +=1
            if func == 'close_matches':
                for s in sentence.split(' '):
                    for t in self.header:
                        if t in self.weights_header:
                            similarity_count_n += similarity.close_matches(s,t)
                            n += 1
                        else :
                            similarity_count_m += similarity.close_matches(s,t)
                            m +=1
            if func == 'sequencematcher':
                for s in sentence.split(' '):
                    for t in self.header:
                        if t in self.weights_header:
                            similarity_count_n += similarity.sequencematcher(s,t)
                            n += 1
                        else :
                            similarity_count_m += similarity.sequencematcher(s,t)
                            m +=1
            if func == 'edit_distance':
                for s in sentence.split(' '):
                    val = 999 
                    if s == "":
                        continue
                    m += 1
                    n += 1
                    for t in self.header:
                        val = min(val, similarity.edit_distance(s, t))
                    similarity_count_m += val

                    val = 999
                    for t in self.weights_header:
                        val = min(val, similarity.edit_distance(s, t))
                    similarity_count_n += val
                    # for t in self.header:
                    #     val = 999
                    #     for t in self.wei
                    #     if t in self.weights_header:
                    #         similarity_count_n += similarity.edit_distance(s,t)
                    #         n += 1
                    #     else :
                    #         similarity_count_m += similarity.edit_distance(s,t)
                    #         m +=1
            if func == 'jaccard_distance':
                for s in sentence.split(' '):
                    for t in self.header:
                        if t in self.weights_header:
                            similarity_count_n += similarity.jaccard_distance(s,t)
                            n += 1
                        else :
                            similarity_count_m += similarity.jaccard_distance(s,t)
                            m +=1
        # print("m : {}, n : {}".format(m, n))
        # print("sim_m : {}, sim_n : {}".format(similarity_count_m, similarity_count_n))
        try:
            similarity_count = similarity_count_n/(n) * (1) + similarity_count_m/(m) * (0.05)
        except:
            return 10
        # print(similarity_count)
        return similarity_count

    
    # def calculate_similarity(self, row):  ## can be optimized.
    #     similarity_count = 0
    #     for sentence in row:
    #         # print(sentence)clean
    #         sentence = spell(sentence)
    #         sentence = self.clean(sentence)
    #         # for word in sentence.split(' '):
    #             # print(word)
    #         if sentence in self.header:
    #             similarity_count+=1
    #         # else:
    #         #     for head in self.header:
    #         #         if int(self.isSubSequence(head, sentence)) or int(self.isSubSequence(sentence,head)):
    #         #             similarity_count += 0.5

    #     return similarity_count

    def row_sim_index(self):  
        """ Create an index determining its chances to be table header.
        """
        self.similarity_index = []
        for row in self.row_string:
            # self.similarity_index.append(self.calculate_similarity_weights(row))
            val = self.calculate_similarity_weights(row)
            print(row)
            print(val)
            _row = [r for r in row if r != '']
            if (len(_row) <= 4):
                self.similarity_index.append(10)
            else:
                self.similarity_index.append(val)

        print(self.similarity_index)


        self.similarity_index = np.array(self.similarity_index)
        # print(self.similarity_index)

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

        # print(sentences)
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
        # print(labels)


    def search_attributes(self):
        for key in self.search_details.keys():
            if key in self.details_status:
                continue
            for det in self.search_details[key]:
                detail = self.clean_details(det)

                # print(detail)
                if key in self.details_status:
                    continue
                for row in self.row_string:
                    if key in self.details_status:
                        continue
                    for word in row:
                        # print(word)
                        # word = word.replace('\n', ' ')
                        _word = self.clean_details(word)
                        # print(_word)
                        if key in self.details_status:
                            continue
                        if detail in _word:
                            start = _word.find(detail)
                            end = -1
                            temp = _word[start:]
                            for val in self.details_list:
                                val = self.clean_details(val)
                                if _word != val:
                                    # print(temp, val)
                                    if temp.find(val) != -1:
                                        # print(temp)
                                        # print(val)
                                        end = temp.find(val)
                                        if(end > 10):
                                            temp = temp[:end]
                                        # print(end)

                            self.details[key] = temp
                            self.details_status.append(key)
                            # print(f"{detail}\n{_word}\n{word}")
                            # if ':' in _word:

                            #     self.details[key] = word
                            #     if self.details[key] != '':
                            #         self.details_status.append(key)

            # modeterms ofpayment
            # modeterms ofpayment



