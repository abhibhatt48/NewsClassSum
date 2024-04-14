# imports
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import requests
import json
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx


app = Flask(__name__)

# global
title = []
author = []
date = []
sdescription = []
ldescription = []
readurl = []
imgurl = []
classification = []
unique_classification = []
embeded_data = None


# glove
f = open("./model/glove.6B.50d.txt", encoding='utf8')
embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    emb = np.array(values[1:], dtype='float')
    embedding_index[word] = emb


# models
fake_model = load_model('./model/fake_news.h5')
classification_model = joblib.load("./model/mnb.pkl")
tfidf = joblib.load("./model/tfidf.pkl")
le = joblib.load("./model/le.pkl")

# init
sw = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# functions extra
def get_embedding_output(X):
    maxLen = 20
    embedding_output = np.zeros((len(X), maxLen, 50))
    for ix in range(X.shape[0]):
        my_example = X[ix].split()
        for ij in range(len(my_example)):
            if (embedding_index.get(my_example[ij].lower()) is not None) and (ij < maxLen):
                embedding_output[ix][ij] = embedding_index[my_example[ij].lower()]

    return embedding_output


def clean_text(data):
    data = data.lower()
    data = re.sub("[^a-z]+", " ", data)
    data = data.split()
    data = [lemmatizer.lemmatize(word) for word in data if word not in sw]
    data = " ".join(data)
    return data


# functions main
def fakeornot(title, author, date, ldescription, readurl, imgurl):
    t = []
    a = []
    d = []
    # sd = []
    ld = []
    read = []
    img = []
    embeded_data = get_embedding_output(np.array(title))
    k = np.argmax(fake_model.predict(embeded_data), axis=1)

    for i in range(len(title)):
        if k[i] == 1:
            t.append(title[i])
            a.append(author[i])
            d.append(date[i])
            # sd.append(sdescription[i])
            ld.append(ldescription[i])
            read.append(readurl[i])
            img.append(imgurl[i])

    title[:] = t
    author[:] = a
    date[:] = d
    # sdescription[:] = sd
    ldescription[:] = ld
    readurl[:] = read
    imgurl[:] = img


def news_classifier(title):
    title_clean = pd.DataFrame([x for x in title])
    title_clean = title_clean[0].apply(clean_text)
    title_clean = tfidf.transform(title_clean)
    title_clean = title_clean.toarray()

    y_pred = classification_model.predict(title_clean)

# conv to category
    category = dict(zip(le.classes_, le.transform(le.classes_)))
    key_list = list(category.keys())
    val_list = list(category.values())
    y = [key_list[val_list.index(i)] for i in y_pred]

    classification[:] = y
    unique_classification[:] = np.unique(np.array(y))

# app
if __name__ == "__main__":
    app.run()
