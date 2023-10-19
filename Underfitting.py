# -*- coding: utf-8 -*-
"""
Created on Mon Oct 9 18:54:50 2023

@author: vitha
"""

import numpy as np 
import pandas as pd 
import re 
import nltk 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_string(str_arg):
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    return cleaned_str # Returning the preprocessed string in tokenized form

import_df = pd.read_csv('D:\\ML\\Flipkart Product Dataset\\flipkart_com-ecommerce_sample.csv')
# Reading relevant data
import_df['product_category_tree'] = import_df['product_category_tree'].apply(lambda x : x.split('>>')[0][2:].strip())
# Category processing. (Check data to understand)
top_fiv_gen = list(import_df.groupby('product_category_tree').count().sort_values(by='uniq_id',ascending=False).head(5).index)
# Taking only top 5 categories for example sake
processed_df = import_df[import_df['product_category_tree'].isin(top_fiv_gen)][['product_category_tree','description']]
# Selecting only relevant columns
processed_df['description'] = processed_df['description'].astype('str').apply(preprocess_string)
# Cleaning strings
cat_list = list(processed_df['product_category_tree'].unique())
# Creating a list of categories for later use
print(cat_list)
# Printing the list of top 5 categories
le = preprocessing.LabelEncoder()
category_encoded=le.fit_transform(processed_df['product_category_tree'])
processed_df['product_category_tree'] = category_encoded

accuracies_train = []
accuracies_test = []
train_sizes = np.linspace(0.1, 1.0, num=10)

for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(processed_df['description'], processed_df['product_category_tree'], test_size=0.2)
    vect = CountVectorizer(stop_words='english')
    X_train_matrix = vect.fit_transform(X_train) 
    clf = MultinomialNB()
    clf.fit(X_train_matrix, y_train)
    accuracies_train.append(clf.score(X_train_matrix, y_train))
    X_test_matrix = vect.transform(X_test) 
    accuracies_test.append(clf.score(X_test_matrix, y_test))
    
plt.plot(train_sizes, accuracies_train, label='Training accuracy')
plt.plot(train_sizes, accuracies_test, label='Test accuracy')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.title('Model performance on training and test sets')
plt.legend()
plt.show()
