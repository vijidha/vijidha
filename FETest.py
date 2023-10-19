# -*- coding: utf-8 -*-
"""
Created on Sat Oct 7 15:25:13 2023

@author: vitha
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

st.set_page_config(layout='wide', page_title='Startup Analysis')
df = pd.read_csv('D:\\ML\\Flipkart Product Dataset\\flipkart_com-ecommerce_sample.csv', encoding='utf-8')

df['crawl_timestamp'] = pd.to_datetime(df['crawl_timestamp'], errors='coerce')
df['month'] = df['crawl_timestamp'].dt.month
df['year'] = df['crawl_timestamp'].dt.year

colors = ["#E78CAE", "#926580", "#926580", "#707EA0", "#34495E"]
custom_palette = sns.color_palette(colors)


def load_overall_analysis():
    st.title('Overall Analysis')

    # total invested amount
    total = round(df['discounted_price'].sum() / 10000000, 2)
    # max amount infused in a startup
    max_funding = df.groupby('product_name')['discounted_price'].max().sort_values(ascending=False).head(1).values[0] / 100000
    # avg ticket size
    avg_funding = df.groupby('product_name')['discounted_price'].sum().mean() / 100000
    # total funded startup
    num_startups = df['product_name'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total', str(total) + 'Cr')
    with col2:
        st.metric('Max', str(max_funding) + 'Cr')
    with col3:
        st.metric('Avg', str(round(avg_funding)) + 'Cr')
    with col4:
        st.metric('Product Names', num_startups)

    col1, col2 = st.columns(2)
    with col1:
        st.header('MoM graph')
        selected_option = st.selectbox('Select Type', ['Total', 'count'])
        if selected_option == 'Total':
            temp_df = df.groupby(['year', 'month'])['discounted_price'].sum().reset_index()
        else:
            temp_df = df.groupby(['year', 'month'])['discounted_price'].count().reset_index()

        temp_df['x_axis'] = temp_df['month'].astype('str') + '_' + temp_df['year'].astype('str')

        # Create plot
        fig5, ax = plt.subplots()
        ax.plot(temp_df['x_axis'], temp_df['discounted_price'])

        # Set plot labels and title
        ax.set_xlabel('Month-Year')
        ax.set_ylabel('Total Amount' if selected_option == 'Total' else 'Transaction Count')
        ax.set_title('Month-on-Month Analysis')

        # Display plot in Streamlit
        st.pyplot(fig5)

        with col2:
            st.header('Top sectors')
            sector_option = st.selectbox('select Type ', ['total', 'count'])
        if sector_option == 'total':
            tmp_df = df.groupby(['product_category_tree'])['discounted_price'].sum().sort_values(ascending=False).head(5)
        else:
            tmp_df = df.groupby(['product_category_tree'])['discounted_price'].count().sort_values(ascending=False).head(5)

        # Create plot
        fig6, ax = plt.subplots()
        ax.barh(tmp_df.index, tmp_df.values, color=custom_palette)

        # Set plot labels and title
        ax.set_xlabel('Total Amount' if sector_option == 'total' else 'Transaction Count')
        ax.set_title('Top 5 Sectors by ' + ('Total Amount' if sector_option == 'total' else 'Transaction Count'))

        # Display plot in Streamlit
        st.pyplot(fig6)

def load_startup_analysis():
    st.title('Startup Analysis')
    product_name_list = df['product_name'].unique().tolist()
    product_name = st.selectbox('Select Product Name', product_name_list)

    # filter data by selected startup
    selected_startup = df[df['product_name'] == product_name].reset_index(drop=True)

    col1, col2 = st.columns(2)
    with col1:
        st.header('Startup Overview')
        st.write(selected_startup.iloc[0])

    with col2:
        st.header('Investment Distribution')
        # Create plot
        fig7, ax = plt.subplots()
        ax.pie(selected_startup['discounted_price'], labels=selected_startup['crawl_timestamp'], autopct='%1.1f%%', startangle=90)

        # Set plot title
        ax.set_title('Investment Distribution')

        # Display plot in Streamlit
        st.pyplot(fig7)

    st.header('Feature Engineering')
    selected_startup = selected_startup[['discounted_price', 'product_rating']]
    selected_startup = selected_startup.dropna()

    # Label encoding for categorical column
    le = LabelEncoder()
    selected_startup['product_rating'] = le.fit_transform(selected_startup['product_rating'])

    # Feature scaling
    sc = StandardScaler()
    X = sc.fit_transform(selected_startup)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X[:, 1:], X[:, 0], test_size=0.2, random_state=42)

    st.header('Model Building')
    model_option = st.selectbox('Select Model', ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'])

    if model_option == 'Linear Regression':
        # Create model and fit to training data
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Generate predictions on testing data
        y_pred = model.predict(X_test)

        # Evaluate model using various metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Display model results
        st.write('Linear Regression Results:')
        st.write('R^2 Score:', round(r2, 2))
        st.write('Mean Squared Error:', round(mse, 2))
        st.write('Mean Absolute Error:', round(mae, 2))

    elif model_option == 'Random Forest':
        # Define hyperparameters to tune
        # Define hyperparameters to tune
        params = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }

    # Create random forest regressor and perform grid search to find best hyperparameters
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, params, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

    # Generate predictions on testing data using best model
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)

    # Evaluate model using various metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Display model results
        st.write('Random Forest Results:')
        st.write('R^2 Score:', round(r2, 2))
        st.write('Mean Squared Error:', round(mse, 2))
        st.write('Mean Absolute Error:', round(mae, 2))

    elif model_option == 'Gradient Boosting':
        # Define hyperparameters to tune
        params = {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 10],
            'subsample': [0.5, 0.8, 1],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
        # Create gradient boosting regressor and perform grid search to find best hyperparameters
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, params, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
    
        # Generate predictions on testing data using best model
        best_gb = grid_search.best_estimator_
        y_pred = best_gb.predict(X_test)
    
        # Evaluate model using various metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    
        # Display model results
        st.write('Gradient Boosting Results:')
        st.write('R^2 Score:', round(r2, 2))
        st.write('Mean Squared Error:', round(mse, 2))
        st.write('Mean Absolute Error:', round(mae, 2))
    
    elif model_option == 'XGBoost':
        # Define hyperparameters to tune
        params = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.5],
            'subsample': [0.5, 0.7, 1],
            'colsample_bytree': [0.5, 0.7, 1]
        }
    
        # Create XGBoost regressor and perform grid search to find best hyperparameters
        xgb_reg = XGBRegressor()
        grid_search = GridSearchCV(xgb_reg, params, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
    
        # Generate predictions on testing data
        y_pred = grid_search.predict(X_test)
    
        # Evaluate model using various metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    
        # Display model results
        st.write('XGBoost Regression Results:')
        st.write('R^2 Score:', round(r2, 2))
        st.write('Mean Squared Error:', round(mse, 2))
        st.write('Mean Absolute Error:', round(mae, 2))

if __name__ == '__main__':
    load_overall_analysis()