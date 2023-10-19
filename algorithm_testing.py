# -*- coding: utf-8 -*-
"""
Created on Mon Oct 9 20:46:50 2023

@author: vitha
"""
from sklearn.metrics import classification_report, roc_auc_score
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

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
accuracies_train=[]
accuracies_test=[] 
# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(processed_df['description'], processed_df['product_category_tree'], test_size=0.2, random_state=42)

# Count Vectorizer
cv = CountVectorizer(stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)


# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
# Classification metrics for each model
# Linear Regression
# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_tfidf, y_train)
y_pred_train_lr = lr.predict(X_train_tfidf)
y_pred_test_lr = lr.predict(X_test_tfidf)

# Evaluation of Linear Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Linear Regression:")
print("Train MSE:", mean_squared_error(y_train, y_pred_train_lr))
print("Test MSE:", mean_squared_error(y_test, y_pred_test_lr))
print("Train MAE:", mean_absolute_error(y_train, y_pred_train_lr))
print("Test MAE:", mean_absolute_error(y_test, y_pred_test_lr))
print("Train R-squared:", r2_score(y_train, y_pred_train_lr))
print("Test R-squared:", r2_score(y_test, y_pred_test_lr))

# Gradient Descent
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
sgd.fit(X_train_tfidf, y_train)
y_pred_train_sgd = sgd.predict(X_train_tfidf)
y_pred_test_sgd = sgd.predict(X_test_tfidf)

# Evaluation of Gradient Descent
print("Gradient Descent:")
print("Train MSE:", mean_squared_error(y_train, y_pred_train_sgd))
print("Test MSE:", mean_squared_error(y_test, y_pred_test_sgd))
print("Train MAE:", mean_absolute_error(y_train, y_pred_train_sgd))
print("Test MAE:", mean_absolute_error(y_test, y_pred_test_sgd))
print("Train R-squared:", r2_score(y_train, y_pred_train_sgd))
print("Test R-squared:", r2_score(y_test, y_pred_test_sgd))

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train_tfidf, y_train)
y_pred_train_rf = rf.predict(X_train_tfidf)
y_pred_test_rf = rf.predict(X_test_tfidf)

# Evaluation of Random Forest Regressor
print("Random Forest Regressor:")
print("Train MSE:", mean_squared_error(y_train, y_pred_train_rf))
print("Test MSE:", mean_squared_error(y_test, y_pred_test_rf))
print("Train MAE:", mean_absolute_error(y_train, y_pred_train_rf))
print("Test MAE:", mean_absolute_error(y_test, y_pred_test_rf))
print("Train R-squared:", r2_score(y_train, y_pred_train_rf))
print("Test R-squared:", r2_score(y_test, y_pred_test_rf))

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

precisions = []
recalls = []
f1_scores = []

# XGBoost Classifier
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_tfidf, y_train)

y_proba_train_xgb = xgb_model.predict_proba(X_train_tfidf)
y_proba_test_xgb = xgb_model.predict_proba(X_test_tfidf)

# Evaluation of XGBoost Classifier
print("XGBoost Classifier:")
y_pred_train_xgb = y_proba_train_xgb.argmax(axis=1)
y_pred_test_xgb = y_proba_test_xgb.argmax(axis=1)

precisions.append(precision_score(y_test, y_pred_test_xgb, average='weighted'))
recalls.append(recall_score(y_test, y_pred_test_xgb, average='weighted'))
f1_scores.append(f1_score(y_test, y_pred_test_xgb, average='weighted'))

print("Precision:", precisions[-1])
print("Recall:", recalls[-1])
print("F1-score:", f1_scores[-1])
print("AUC-ROC:", roc_auc_score(y_test, y_proba_test_xgb, multi_class='ovr'))

