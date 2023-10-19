# -*- coding: utf-8 -*-
"""
Created on Sat Oct 7 15:25:13 2023

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
    '''
    input: str_arg --> Takes string to clean
    output: cleaned_str --> Gives back cleaned string
    This fuction cleans the text in the mentioned ways as comments after the line.This has been copied from some other kernel.

    '''
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    
    return cleaned_str # Returning the preprocessed string in tokenized form

'''
This code block is for reading and cleaning data.

'''
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
# Encoding the product category

'''
This code block is for spliting train test data

'''
X_train, X_test, y_train, y_test = train_test_split(processed_df['description'],processed_df['product_category_tree'],test_size=0.2)

'''
This code block is for converting the training data to vectorized form

'''
vect = CountVectorizer(stop_words = 'english')
# Removing stop words
X_train_matrix = vect.fit_transform(X_train) 
# Converting the train data

'''
This code block is for training vectorized data and predicting & scoring test data

'''
clf=MultinomialNB()
# Defining model
clf.fit(X_train_matrix, y_train)
# Fitting to multinomial NB model 
print(clf.score(X_train_matrix, y_train))
# Scoring the trained model (Expected to be above 95 percent)
X_test_matrix = vect.transform(X_test) 
# Converting the test data
print (clf.score(X_test_matrix, y_test))
# Scoring for the test data
predicted_result=clf.predict(X_test_matrix)
print(classification_report(y_test,predicted_result))
# Printing score 
Flipkart_data=pd.read_csv("D:\\ML\\Flipkart Product Dataset\\flipkart_com-ecommerce_sample.csv")
Flipkart_data

# %% [code]
Flipkart_data.info()

# %% [code]
## Checking the head of the data

Flipkart_data.head(n=5)

# %% [code]
Flipkart_data.shape

# %% [code]
Flipkart_data.isnull().sum()

# %% [code]
Flipkart_data.columns

# %% [code]
## The Brand column has lots of null values.

plt.figure(figsize =(10,8))
sns.heatmap(Flipkart_data.isnull(),yticklabels=False,cmap='plasma',cbar=True)

# %% [code]
Flipkart_data.duplicated().value_counts()

# %% [code]
#make this column into a datetime type for workability

Flipkart_data['crawl_timestamp']=pd.to_datetime(Flipkart_data['crawl_timestamp'])
Flipkart_data['crawl_timestamp']

# %% [code]
Flipkart_data['crawl_year']=Flipkart_data['crawl_timestamp'].apply(lambda x : x.year)

# %% [code]
Flipkart_data['crawl_year']

# %% [code]
Flipkart_data['Month']=Flipkart_data['crawl_timestamp'].apply(lambda x : x.month)
Flipkart_data['Month']

Flipkart_data['main_category']=Flipkart_data['product_category_tree'].apply(lambda x :x.split('>>')[0][2:len(x.split('>>')[0])-1])

# %% [code]
def secondary_category(value):
    try:
        return value.split('>>')[1][1:len(value.split('>>')[1])-1]
    except IndexError:
        return 'None'       
def tertiary_category(value):
    try:
        return value.split('>>')[2][1:len(value.split('>>')[2])-1]
    except IndexError:
        return 'None'
def quaternary_category(value):
    try:
        return value.split('>>')[3][1:len(value.split('>>')[3])-1]
    except IndexError:
        return 'None'

# %% [code]
Flipkart_data['secondary']=Flipkart_data['product_category_tree'].apply(secondary_category)
Flipkart_data['tertiary']=Flipkart_data['product_category_tree'].apply(tertiary_category)
Flipkart_data['quaternary']=Flipkart_data['product_category_tree'].apply(quaternary_category)

# %% [code]
Flipkart_data.head(n=5)

# %% [code]
### Sales by month;

plt.figure(figsize=(150,10))
temp=Flipkart_data.groupby(by='Month',axis=0).count().plot(kind='bar',legend=False)
plt.ylabel('Sales')

# %% [markdown]
# ## 12th Month of the year has most number of sales.

# %% [code]
Flipkart_data.groupby(by='crawl_year',axis=0).count().plot(kind='bar',legend=False)
plt.ylabel('Sales')

# %% [markdown]
# ## Both the year 2015 and 2016 has almost same amount of sale.

# %% [code]
plt.figure(figsize=(10,10))
Flipkart_data['main_category'].value_counts()[:20].sort_values(ascending=False).plot(kind='barh')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)


# %% [markdown]
# ## Looks like Most of the customer prefer getting Cloths from Flipkart.

# %% [code]
Flipkart_data['main_category'].value_counts()[:10].sort_values(ascending=False)

# %% [code]
plt.figure(figsize=(10,10))
Flipkart_data['secondary'].value_counts()[:20].sort_values(ascending=False).plot(kind='barh')

# %% [markdown]
# ## Ladies prefer more shopping on Flipkart compared to mens.

# %% [code]
Flipkart_data['secondary'].value_counts()[:10].sort_values(ascending=False)

# %% [code]
plt.figure(figsize=(10,10))
Flipkart_data['tertiary'].value_counts()[:20].plot(kind='barh')

# %% [markdown]
# ## And What do ladies buy on Flipakrt, It's Western Wear :(

# %% [code]
Flipkart_data['tertiary'].value_counts()[:10].sort_values(ascending=False)

# %% [code]
plt.figure(figsize=(10,10))
Flipkart_data['quaternary'].value_counts()[:20].plot(kind='barh')

# %% [code]
Flipkart_data['quaternary'].value_counts()[:10].sort_values(ascending=False)

# %% [code]
Flipkart_data['discounted_price'].max()

# %% [markdown]
# ## The max Price of a product lsited in the dataset is 571230.0, it a wrist watch..

# %% [code]
## Discount Percentage
#retail_price
#discounted_price
Flipkart_data['discounted_percentage']=round((Flipkart_data['retail_price']-Flipkart_data['discounted_price'])/Flipkart_data['retail_price']*100,1)


# %% [code]
## that contain the product by category, average discounted percentages and count of each product.

# %% [code]
main_category_discount_percentage=Flipkart_data.groupby('main_category').agg({'discounted_percentage':[np.mean],'main_category':['count']})
main_category_discount_percentage

# %% [code]
main_category_discount_percentage.columns=['_' .join(column) for column in main_category_discount_percentage.columns]
main_category_discount_percentage

# %% [code]
main_category_discount_percentage[main_category_discount_percentage['main_category_count']>50].sort_values(by='main_category_count',ascending=False)

# %% [code]
# Mean of the Discount %age in Automative is largest in the sales record.

plt.figure(figsize=(10,10))
main_category_discount_percentage[main_category_discount_percentage['main_category_count']>50].sort_values(by='discounted_percentage_mean',ascending=False)['discounted_percentage_mean'].plot(kind='barh',legend=False)

# %% [code]
## Checking for the maximum discount in the secondary category

# %% [code]
secondary_discounted_percentage=Flipkart_data.groupby('secondary').agg({'discounted_percentage':[np.mean],'secondary':['count']})
secondary_discounted_percentage

# %% [code]
secondary_discounted_percentage.columns=['_'.join (column) for column in secondary_discounted_percentage.columns]
secondary_discounted_percentage.columns

# %% [code]
plt.figure(figsize=(30,30))
plt.yticks(size=30)
plt.xticks(size=30)
secondary_discounted_percentage[secondary_discounted_percentage['secondary_count']>50].sort_values(by='discounted_percentage_mean',ascending=False)['discounted_percentage_mean'].plot(kind='barh')

# %% [code]
tertiary_discount_percentage=Flipkart_data.groupby('tertiary').agg({'discounted_percentage':[np.mean],'tertiary':['count']})
tertiary_discount_percentage

# %% [code]
tertiary_discount_percentage.columns=['_'.join(column) for column in tertiary_discount_percentage.columns]

# %% [code]
plt.figure(figsize=(20,20))
tertiary_discount_percentage[tertiary_discount_percentage['tertiary_count']>50].sort_values(by='discounted_percentage_mean',ascending=False)['discounted_percentage_mean'].plot(kind='barh')
plt.xticks(size=20)
plt.yticks(size=20)