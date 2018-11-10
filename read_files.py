# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:25:06 2018

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
from pymongo import MongoClient
from textwrap import TextWrapper
from tweepy.utils import import_simplejson
from main_class import classifier_workshop
json = import_simplejson()
from sklearn import svm

framework = classifier_workshop()
    
the_path = 'C:/Users/pathouli/myStuff/academia/columbia/workshop/data/'

the_directories = framework.fetch_directories(the_path)

the_matrix = framework.body_matrix(the_directories, the_path)

from sklearn.neural_network import MLPClassifier
model_in = MLPClassifier(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(5, 2), random_state=1)

v_o, m_o, l_o = framework.model_train(model_in, the_matrix)

auth1 = tweepy.auth.OAuthHandler('','')  
auth1.set_access_token('-','')  
api = tweepy.API(auth1)
class StreamListener(tweepy.StreamListener):  
    status_wrapper = TextWrapper(width=140, initial_indent='', subsequent_indent='')    
    def on_status(self, status): 
        try:     
            the_text = str(self.status_wrapper.fill(status.text))
            the_predicted, prob_val = framework.classify(v_o, m_o, l_o, [the_text])
            print the_text + ": ===> " + the_predicted[0]
        except:            
            pass 

l = StreamListener()  
streamer = tweepy.Stream(auth=auth1, listener=l, timeout=3000)   
setTerms = ["probability","artificial intelligence","trail mix","calculus"]
streamer.filter(None,setTerms)