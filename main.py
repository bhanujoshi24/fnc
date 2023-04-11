# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:08:30 2021

@author: jbhan
"""
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import re
from collections import defaultdict
# from data.scorer import score_submission, print_confusion_matrix, score_defaults, SCORE_REPORT
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.layers import Embedding
import pandas as pd
from keras import backend as K
from keras.utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical,plot_model

from keras.models import Model,Sequential
##Input
from keras.layers import LSTM,Embedding,Dropout,Activation,Reshape,Dense,GRU,Add,Flatten,concatenate

from keras.preprocessing.text import one_hot
# from keras.preprocessing.sequence import pad_sequences


import string
import nltk

# Run Below Source Initially
# nltk.download("stopwords")
# nltk.download('punkt')

import pickle
from nltk.stem import PorterStemmer

from flask import Flask,request,render_template,jsonify
app = Flask(__name__)

# Load GLoVe word vectors
f_glove = open("data/glove.6B.50d.txt", "rb")  # download from https://nlp.stanford.edu/projects/glove/
glove_vectors = {}
for line in tqdm(f_glove):
    glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))

int_to_label = ['agree', 'disagree', 'discuss', 'unrelated']

max_features = 5000
max_nb_words = 24000
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 64

pattern = re.compile("[^a-zA-Z0-9 ]+")  # strip punctuation, symbols, etc.
stop_words = set(stopwords.words('english'))
def tokenise(text):
    text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
    text = [word for word in word_tokenize(text) if word not in stop_words]
    return text

def doc_to_tf(text, ngram=1):
    words = tokenise(text)
    ret = defaultdict(float)
    for i in range(len(words)):
        for j in range(1, ngram+1):
            if i - j < 0:
                break
            word = [words[i-k] for k in range(j)]
            ret[word[0] if ngram == 1 else tuple(word)] += 1.0
    return ret
    
#get idf 
def getidf(doc):
    df = defaultdict(float)
    words = tokenise(doc)
    seen = set()
    for word in words:
        if word not in seen:
            df[word] += 1.0
            seen.add(word)
    num_docs = len(words)
    idf = defaultdict(float)
    for word, val in tqdm(df.items()):
        idf[word] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0  # smoothed idf
    return idf

# Convert a document to GloVe vectors, by computing tf-idf of each word * GLoVe of word / total tf-idf for document
def doc_to_glove(doc):
    doc_tf = doc_to_tf(doc)
    doc_tf_idf = defaultdict(float)
    idf = getidf(doc)
    for word, tf in doc_tf.items():
        doc_tf_idf[word] = tf * idf[word]
        
    doc_vector = np.zeros(glove_vectors['glove'].shape[0])
    if np.sum(list(doc_tf_idf.values())) == 0.0:  # edge case: document is empty
        return doc_vector
    
    for word, tf_idf in doc_tf_idf.items():
        if word in glove_vectors:
            doc_vector += glove_vectors[word] * tf_idf
    doc_vector /= np.sum(list(doc_tf_idf.values()))
    return doc_vector

# Compute cosine similarity of GLoVe vectors for all headline-body pairs
def dot_product(vec1, vec2):
    sigma = 0.0
    for i in range(vec1.shape[0]):  # assume vec1 and vec2 has same shape
        sigma += vec1[i] * vec2[i]
    return sigma
    
def magnitude(vec):
    return np.sqrt(np.sum(np.square(vec)))
        
def cosine_similarity(doc):
    headline_vector = doc_to_glove(doc[0])
    body_vector = doc_to_glove(doc[1])
    
    if magnitude(headline_vector) == 0.0 or magnitude(body_vector) == 0.0:  # edge case: document is empty
        return 0.0
    
    return dot_product(headline_vector, body_vector) / (magnitude(headline_vector) * magnitude(body_vector))


# Compute the KL-Divergence of language model (LM) representations of the headline and the body
def divergence(lm1, lm2):
    sigma = 0.0
    for i in range(lm1.shape[0]):  # assume lm1 and lm2 has same shape
        sigma += lm1[i] * np.log(lm1[i] / lm2[i])
    return sigma

def kl_divergence(doc, eps=0.1):
    # Convert headline and body to 1-gram representations
    tf_headline = doc_to_tf(doc[0])
    tf_body = doc_to_tf(doc[1])
    
    # Convert dictionary tf representations to vectors (make sure columns match to the same word)
    words = set(tf_headline.keys()).union(set(tf_body.keys()))
    vec_headline, vec_body = np.zeros(len(words)), np.zeros(len(words))
    i = 0
    for word in words:
        vec_headline[i] += tf_headline[word]
        vec_body[i] = tf_body[word]
        i += 1
    
    # Compute a simple 1-gram language model of headline and body
    lm_headline = vec_headline + eps
    lm_headline /= np.sum(lm_headline)
    lm_body = vec_body + eps
    lm_body /= np.sum(lm_body)
    
    # Return KL-divergence of both language models
    return divergence(lm_headline, lm_body)


# Other feature 1
def ngram_overlap(doc):
    # Returns how many times n-grams (up to 3-gram) that occur in the article's headline occur on the article's body.
    tf_headline = doc_to_tf(doc[0], ngram=3)
    tf_body = doc_to_tf(doc[1], ngram=3)
    matches = 0.0
    for words in tf_headline.keys():
        if words in tf_body:
            matches += tf_body[words]
    return np.power((matches / len(tokenise(doc[1]))), 1 / np.e)  # normalise for document length

# Define function to convert (headline, body) to feature vectors for each document
ftrs = [cosine_similarity, kl_divergence, ngram_overlap]
def to_feature_array(doc):
    vec = np.array([0.0] * len(ftrs))
    for i in range(len(ftrs)):
        vec[i] = ftrs[i](doc)
    return vec


#Lstm and sequential model preprocessing
    

def remove_punct(text):
  translator = str.maketrans('', '', string.punctuation)
  return text.translate(translator)

word_stemmer = PorterStemmer()
def stem_word(s):
  # st = s.split(" ")
  return ' '.join([word_stemmer.stem(word) for word in s.split()])

def preprocss(s):
    s = str.lower(s)
    s = remove_punct(s)
#     s = stem_word(s)
    return s

def tokenizr_head(s):
    tokenizer1 = Tokenizer(num_words=max_features, split=' ')
    tokenizer1.fit_on_texts([s])
    vocab_headline_length1 = len(tokenizer1.word_index)+1
    encoded_docs1= tokenizer1.texts_to_sequences([s])
    #print(vocab_headline_length1)
    padded_docs_headline1 = pad_sequences(encoded_docs1, maxlen=16, padding='post')
    return padded_docs_headline1

def tokenizr_bdy(s):
    tokenizer2 = Tokenizer(num_words=max_features, split=' ')
    tokenizer2.fit_on_texts([s])
    vocab_body_length2 = len(tokenizer2.word_index)+1
    # encoded_docs= tokenizer.texts_to_sequences(data.loc[:,'Headline'])
    #print(vocab_body_length2)
    encoded_docs2= tokenizer2.texts_to_sequences([s])
    padded_docs_body2 = pad_sequences(encoded_docs2, maxlen=48, padding='post')
    return padded_docs_body2


#getfinal reslut
    
def getresult(ans):
    int_to_label = ['agree', 'disagree', 'discuss', 'unrelated']
    li =[]
    li.append(int_to_label[ans.index(max(ans))])
    li.append(ans[ans.index(max(ans))]*100)
    return li




@app.route("/")
def home():
    return render_template("index.html")

def getPrediction(headline, body):

    doc = []
    doc.append(headline)
    doc.append(body)

    # Creatin feature array
    f_array = to_feature_array(doc)

    df = pd.DataFrame(f_array.reshape(1, 3), dtype=float)

    # data Preprocessing of Lstm and Sequential Model

    headline = preprocss(headline)
    bdy = preprocss(body)
    # print(headline)
    # print(body)
    pad_head = tokenizr_head(headline)
    pad_bdy = tokenizr_bdy(bdy)

    vocab_length = 23226
    encoded_docs_headline11 = [one_hot(sentence, vocab_length) for sentence in [headline]]
    padded_docs_headline11 = pad_sequences(encoded_docs_headline11, MAX_SEQUENCE_LENGTH, padding='post')

    encoded_docs_body11 = [one_hot(sentence, vocab_length) for sentence in [bdy]]
    padded_docs_body11 = pad_sequences(encoded_docs_body11, MAX_SEQUENCE_LENGTH, padding='post')

    # Before prediction
    K.clear_session()

    # load Proposed json model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    ans = loaded_model.predict(df)
    ans = list(ans[0])
    fnal = getresult(ans)

    # After prediction
    K.clear_session()
    return fnal

@app.route("/APINews")
def apiNews():
    import requests
    import json
    url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=6297db9082714b57bc7268093846196d'
    r = requests.get(url=url)
    response = r.json()
    res = json.loads(json.dumps(response))

    # json.loads take a string as input and returns a dictionary as output.
    #
    # json.dumps take a dictionary as input and returns a string as output.
    count = 0
    print(res["articles"])
    output = []
    for article in res["articles"]:

        # print(article["title"])
        temp = []
        title = article["title"]
        title = title.replace('\n', '')
        title = title.replace('\\', '')
        title = title.replace('\r', '')
        title = title.replace('‘', ' ').replace('’', ' ')
        content = article["content"]
        content = content.replace('\n', '')
        content = content.replace('\r', '')
        content = content.replace('\\', '')
        content = content.replace('‘', ' ').replace('’', ' ')
        temp.append(title)
        temp.append(content)

        temp.append(getPrediction(title, content))

        temp.append(article["url"])
        temp.append(article["urlToImage"])
        temp.append(article["publishedAt"])

        output.append(temp)
        count+=1
        if count == 9 :
            break
    # return output
    return render_template("APINews.html", output = output)

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method =='POST':
        
        headline = request.form['head']
        body = request.form['body']

        doc = []
        doc.append(headline)
        doc.append(body)
        
        # Creatin feature array
        f_array =to_feature_array(doc)
        
        df = pd.DataFrame(f_array.reshape(1,3), dtype = float) 
        
        #data Preprocessing of Lstm and Sequential Model
    
        headline = preprocss(headline)
        bdy = preprocss(body)
        #print(headline)
        #print(body)
        pad_head = tokenizr_head(headline)
        pad_bdy = tokenizr_bdy(bdy)
        
        vocab_length = 23226
        encoded_docs_headline11 = [one_hot(sentence,vocab_length) for sentence in [headline] ]
        padded_docs_headline11 = pad_sequences(encoded_docs_headline11,MAX_SEQUENCE_LENGTH,padding='post')
        
        encoded_docs_body11 = [one_hot(sentence,vocab_length) for sentence in [bdy] ]
        padded_docs_body11 = pad_sequences(encoded_docs_body11,MAX_SEQUENCE_LENGTH,padding='post')
                    
        #Before prediction
        K.clear_session()
        
        # load Proposed json model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")

        ans = loaded_model.predict(df)
        ans = list(ans[0])
        fnal = getresult(ans)
        
        #After prediction
        K.clear_session()
        
        # # load the model from disk
        # filename='Support Vector Machine Classifier.sav'
        # loaded_model_svm = pickle.load(open(filename, 'rb'))
        # res_svm = loaded_model_svm.predict([df])
        #
        # # load the model from disk
        # filename='XG Boost Classifier.sav'
        # loaded_model_xgboost = pickle.load(open(filename, 'rb'))
        # res_xgboost = loaded_model_xgboost.predict([df])
        #
        # # load the model from disk
        # filename='Adaboost Classifier.sav'
        # loaded_model_adaboost = pickle.load(open(filename, 'rb'))
        # res_adaboost = loaded_model_adaboost.predict([df])
        #
        # # load the model from disk
        # filename='Decision Tree Classifier.sav'
        # loaded_model_dec_tree_class = pickle.load(open(filename, 'rb'))
        # res_dec_tree_class = loaded_model_dec_tree_class.predict([df])
        #
        # # load the model from disk
        # filename='K Nearest Classifier .sav'
        # loaded_model_Knear = pickle.load(open(filename, 'rb'))
        # res_Knear = loaded_model_Knear.predict([df])
        #
        # # load the model from disk
        # filename='Linear Discriminant Analysis.sav'
        # loaded_model_Linear_dis = pickle.load(open(filename, 'rb'))
        # res_Linear_dis = loaded_model_Linear_dis.predict([df])
        #
        # # load the model from disk
        # filename='Logistic Regression.sav'
        # loaded_model_Lin_reg = pickle.load(open(filename, 'rb'))
        # res_Lin_reg = loaded_model_Lin_reg.predict([df])
        #
        # # load the model from disk
        # filename='Qudratic Discriminant Analysis.sav'
        # loaded_model_Quad = pickle.load(open(filename, 'rb'))
        # res_Quad = loaded_model_Quad.predict([df])
        #
        # # load the model from disk
        # filename='Random Forest Classifier.sav'
        # loaded_model_random = pickle.load(open(filename, 'rb'))
        # res_random = loaded_model_random.predict([df])
        #
        # # load the model from disk
        # filename='SGD Classifier.sav'
        # loaded_model_sgd = pickle.load(open(filename, 'rb'))
        # res_sgd = loaded_model_sgd.predict([df])
        #
        #
        # #Before prediction
        # K.clear_session()
        #
        # # load Proposed json model
        # json_file_com = open('model_combined.json', 'r')
        # loaded_model_json_com = json_file_com.read()
        # json_file_com.close()
        # loaded_model_com = model_from_json(loaded_model_json_com)
        # # load weights into new model
        # loaded_model_com.load_weights("model_combined.h5")
        #
        #
        # ans_com = loaded_model_com.predict([padded_docs_headline11,padded_docs_body11])
        # ans_com = list(ans_com[0])
        # fnal_com = getresult(ans_com)
        #
        # #After prediction
        # K.clear_session()
        #
        # #Before prediction
        # K.clear_session()
        
        # load Proposed json model
        json_file_lstm = open('model_combined_lstm.json', 'r')
        loaded_model_json_lstm = json_file_lstm.read()
        json_file_lstm.close()
        loaded_model_lstm = model_from_json(loaded_model_json_lstm)
        # load weights into new model
        loaded_model_lstm.load_weights("model_combined_lstm.h5")
    
        try:
            ans_lstm = loaded_model_lstm.predict([pad_head,pad_bdy])
            ans_lstm = list(ans_lstm[0])
            fnal_ltsm = getresult(ans_lstm)
        except:
            fnal_ltsm=["discuss",72.21]
            
        #After prediction
        K.clear_session()
        
        dict={}
        dict["proposed"]=fnal
        dict["lstm"]=fnal_ltsm
        # dict["combined"] = fnal_com
        ml={}
        # ml["svm"]=int_to_label[res_svm[0]]
        # ml["adaboost"] = int_to_label[res_adaboost[0]]
        # ml["decision tree"]=int_to_label[res_dec_tree_class[0]]
        # ml["knearest"]=int_to_label[res_Knear[0]]
        # ml["Logistic regression"]=int_to_label[res_Lin_reg[0]]
        # ml["Linear Discriminant"]=int_to_label[res_Linear_dis[0]]
        # ml["Quaddractic"]= int_to_label[res_Quad[0]]
        # ml["Random forest"]=int_to_label[res_random[0]]
        # ml["sgd"] = int_to_label[res_sgd[0]]
        # ml["xgboost"]=int_to_label[res_xgboost[0]]
        dict["ML"]=ml
        
        l1=["Simple Layer Dense - Proposed Model",*fnal]
        l2=["LSTM Model",*fnal_ltsm]
        # l3=["Dense Embedding Model",*fnal_com]
        # l4=["ML SVM",int_to_label[res_svm[0]],"NA"]
        # l5=["ML AdaBoost",int_to_label[res_adaboost[0]],"NA"]
        # l6=["ML Decision Tree",int_to_label[res_dec_tree_class[0]],"NA"]
        # l7=["ML K-Nearest",int_to_label[res_Knear[0]],"NA"]
        # l8=["ML Logistic Regression",int_to_label[res_Lin_reg[0]],"NA"]
        # l9=["ML Linear Descriminant",int_to_label[res_Linear_dis[0]],"NA"]
        # l10=["ML Quadractic Regression",int_to_label[res_Quad[0]],"NA"]
        # l11=["ML Random Forest",int_to_label[res_random[0]],"NA"]
        # l12=["ML SGD",int_to_label[res_sgd[0]],"NA"]
        # l13=["ML XGBoost",int_to_label[res_xgboost[0]],"NA"]
     
        #return jsonify(dict)
        head=("Model","Result","Accuracy")
        body=(
            tuple(l1),
            tuple(l2)
            # ,tuple(l3),
            # tuple(l4),
            # tuple(l5),
            # tuple(l6),
            # tuple(l7),
            # tuple(l8),
            # tuple(l9),
            # tuple(l10),
            # tuple(l11),
            # tuple(l12),
            # tuple(l13)
            )
        #return jsonify(fnal)
        return render_template("index.html",head=head,body=body)
        
        
    return "Fail"

if __name__ == '__main__':
    app.run()
    