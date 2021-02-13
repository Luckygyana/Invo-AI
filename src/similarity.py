import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
import difflib 
from numpy import dot
from numpy.linalg import norm
import nltk

def sentence_encoder(s,t):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed"
    embed = hub.Module(module_url)
    input = tf.placeholder(tf.string, shape=(None))
    message_encoder = embed(input)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        message_embeddings_ = session.run(message_encoder, feed_dict={input: [s,t]})
        cos_sim = dot(message_embeddings_[0], message_embeddings_[1])/(norm(message_embeddings_[0])*norm(message_embeddings_[1]))
    return cos_sim

def levenshtein_ratio_and_distance(s, t):
        rows = len(s)+1
        cols = len(t)+1
        distance = np.zeros((rows,cols),dtype = int)
        for i in range(1, rows):
            for k in range(1,cols):
                distance[i][0] = i
                distance[0][k] = k
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row-1] == t[col-1]:
                    cost = 0 
                else:
                    if ratio_calc == True:
                        cost = 2
                    else:
                        cost = 1
                distance[row][col] = min(distance[row-1][col] + 1,
                                    distance[row][col-1] + 1,          
                                    distance[row-1][col-1] + cost)   
        
            ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
            return ratio


def close_matches(s,lst,threshold):
    similarity = 0
    out = difflib.get_close_matches(s,lst,cutoff=threshold)
    if out != None:
        similarity +=1 
    return similarity

def sequencematcher(s,t):
    return difflib.SequenceMatcher(None,s,t).ratio()

def edit_distance(s,t):
    return nltk.edit_distance(s, t)

def jaccard_distance(s,t):
    return nltk.jaccard_distance(s, t)
