# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 01:31:50 2018

@author: pathouli
"""

import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import sys
import tweepy  
from textwrap import TextWrapper
from tweepy.utils import import_simplejson
import argparse
json = import_simplejson()


class classifier_workshop():
    def fetch_directories(self, the_path):
        the_dirs = os.listdir(the_path) #get directory names
        if '.git' in the_dirs:
            the_dirs.remove('.git') #remove the hidden git IF you have it    
    
        return the_dirs

    def gen_corpus(self, theText):
        
        #set dictionaries
        stopWords = set(stopwords.words('english'))
        # theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
        
        #pre-processing
        lines_temp = [text.strip() for text in theText]
        lines = " ".join(lines_temp)
        theText = lines.split()
        tokens = [token.lower() for token in theText] #ensure everything is lower case
        tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
        tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
        tokens = [word for word in tokens if word not in stopWords] #rid of stop words
        #tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
        tokens = " ".join(tokens) #need to pass string seperated by spaces       
    
        return tokens
    
    def body_matrix(self, the_directories, the_path):
        raw_data = pd.DataFrame()
        for word in the_directories:
            tmp = os.listdir(the_path + word)
            for in_file in tmp:
                try:
                    f = open(the_path + word + "/" + in_file, "r")
                    lines = f.readlines()
                    stemmed_words = self.gen_corpus(lines)
                    raw_data = raw_data.append({'body':stemmed_words, 'labels':word}, ignore_index=True)
                except:
                    pass  
        return raw_data
    
    def model_train(self, model, the_matrix):
        ############
        ##Train##
        vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1,1))
        tdm = pd.DataFrame(vectorizer.fit_transform(the_matrix.body).toarray())
        tdm.columns=vectorizer.get_feature_names()
        
        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(the_matrix[the_matrix.columns[1]])
        the_matrix[the_matrix.columns[1]] = [label_enc.transform([word])[0] for
                  word in the_matrix[the_matrix.columns[1]]]
    
        #validation stage - Best practice
        cross_val_score(model,tdm,the_matrix[the_matrix.columns[1]],cv=10)
        
        model.fit(tdm, the_matrix[the_matrix.columns[1]])
        
        return vectorizer, model, label_enc
        
    def classify(self, v, m, l, to_predict):
        ############
        ##Classify##
        tdm_pred = v.transform(to_predict)
        
        prediction = m.predict(tdm_pred)
        the_prob = pd.DataFrame(m.predict_proba(tdm_pred))
        the_prob.columns = l.classes_
        the_pred = l.inverse_transform([prediction][0])
        
        return the_pred, the_prob