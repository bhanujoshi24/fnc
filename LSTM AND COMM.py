# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:32:28 2021

@author: jbhan
"""
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical,plot_model

from keras.models import Input,Model,Sequential
from keras.layers import LSTM,Embedding,Dropout,Activation,Reshape,Dense,GRU,Add,Flatten,concatenate

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

import string
import nltk 
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_english = set(stopwords.words('english'))

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
    print(vocab_headline_length1)
    padded_docs_headline1 = pad_sequences(encoded_docs1, maxlen=16, padding='post')
    return padded_docs_headline1

def tokenizr_bdy(s):
    tokenizer2 = Tokenizer(num_words=max_features, split=' ')
    tokenizer2.fit_on_texts([s])
    vocab_body_length2 = len(tokenizer2.word_index)+1
    # encoded_docs= tokenizer.texts_to_sequences(data.loc[:,'Headline'])
    print(vocab_body_length2)
    encoded_docs2= tokenizer2.texts_to_sequences([s])
    padded_docs_body2 = pad_sequences(encoded_docs2, maxlen=48, padding='post')
    return padded_docs_body2


def getresult(ans):
    int_to_label = ['agree', 'disagree', 'discuss', 'unrelated']
    li =[]
    li.append(int_to_label[ans.index(max(ans))])
    li.append(ans[ans.index(max(ans))]*100)
    return li
  
    
max_features = 5000
max_nb_words = 24000
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 64

headline = input("Enter headline : ")
bdy = input("Enter Body : ")
headline = preprocss(headline)
bdy = preprocss(bdy)
print(headline)
print(body)
pad_head = tokenizr_head(headline)
pad_bdy = tokenizr_bdy(bdy)


# load Proposed json model
json_file_com = open('model_combined.json', 'r')
loaded_model_json_com = json_file_com.read()
json_file_com.close()
loaded_model_com = model_from_json(loaded_model_json_com)
# load weights into new model
loaded_model_com.load_weights("model_combined.h5")

# load Proposed json model
json_file_lstm = open('model_combined_lstm.json', 'r')
loaded_model_json_lstm = json_file_lstm.read()
json_file_lstm.close()
loaded_model_lstm = model_from_json(loaded_model_json_lstm)
# load weights into new model
loaded_model_lstm.load_weights("model_combined_lstm.h5")


try:
    ans_lstm = loaded_model_lstm.predict([pad_head,pad_bdy])
    print(ans_lstm[0])
    ans_lstm = list(ans_lstm[0])
    fnal_lstm = getresult(ans_lstm)
except:
    print("Exception has occured")
print("Proposed Output :",*fnal_lstm)
vocab_length = 23226
encoded_docs_headline11 = [one_hot(sentence,vocab_length) for sentence in [headline] ]
padded_docs_headline11 = pad_sequences(encoded_docs_headline11,MAX_SEQUENCE_LENGTH,padding='post')

encoded_docs_body11 = [one_hot(sentence,vocab_length) for sentence in [bdy] ]
padded_docs_body11 = pad_sequences(encoded_docs_body11,MAX_SEQUENCE_LENGTH,padding='post')


print(loaded_model_com.predict([padded_docs_headline11,padded_docs_body11]))