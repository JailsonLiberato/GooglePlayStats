# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
import re
import time
import datetime
from sklearn import svm
from sklearn.svm import LinearSVC

class MainClass:

    def __init__(self):
        df = pd.read_csv('files/googleplaystore.csv')
        df = self.execute_pre_process(df)
        self.execute_linear_regression(df)
        
    def change_size(self, size):
        if 'M' in size:
            x = size[:-1]
            x = float(x)*1000000
            return(x)
        elif 'k' == size[-1:]:
            x = size[:-1]
            x = float(x)*1000
            return(x)
        else:
            return None

    def type_cat(self, types):
        if types == 'Free':
            return 0
        else:
            return 1

    def installs_clean(self, installs):
        if installs == '0':
            return 0
        else:
            installs = installs.replace('+','').replace(',','').replace('Free', '0')
        return installs

    def price_clean(self, price):
        if price == '0':
            return 0
        else:
            price = price[1:]
            try:
                price = price.replace('$','').replace('+', '').replace(',','')
                price = float(price)
            except ValueError as identifier:
                price = 0.0
            return price

    def sizes_clean(self, sizes):
        if sizes == '0':
            return 0
        else:
            sizes = sizes.replace('M', '').replace('Varies with device', '0')
        return sizes
    
    def execute_pre_process(self, df):
        # The best way to fill missing values might be using the median instead of mean.
        df['Rating'] = df['Rating'].fillna(df['Rating'].median())

        # Before filling null values we have to clean all non numerical values & unicode charachters 
        replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
        for i in replaces:
            df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))
            df['Android Ver'] = df['Android Ver'].astype(str).apply(lambda x : x.replace(i, ''))

        regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']
        for j in regex:
            df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))
            df['Android Ver'] = df['Android Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))

        df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace(',', '').replace('.', '').replace(',', '',0)).astype(float).astype(int)
        df['Current Ver'] = df['Current Ver'].fillna(df['Current Ver'].median())

        df['Android Ver'] = df['Android Ver'].astype(str).apply(lambda x : x.replace(',', '').replace('.', '').replace(',', '',0)).astype(float).astype(int)
        df['Android Ver'] = df['Android Ver'].fillna(df['Android Ver'].median())

        df['Category'].unique()

        i = df[df['Category'] == '1.9'].index
        df.loc[i]

        df = df.drop(i)

        df = df[pd.notnull(df['Last Updated'])]
        df = df[pd.notnull(df['Content Rating'])]

        # App values encoding
        le = preprocessing.LabelEncoder()
        df['App'] = le.fit_transform(df['App'])
        # This encoder converts the values into numeric values

        # Category features encoding
        category_list = df['Category'].unique().tolist() 
        category_list = ['cat_' + word for word in category_list]
        df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)

        # Genres features encoding
        le = preprocessing.LabelEncoder()
        df['Genres'] = le.fit_transform(df['Genres'])

        # Encode Content Rating features
        le = preprocessing.LabelEncoder()
        df['Content Rating'] = le.fit_transform(df['Content Rating'])

        # Price cealning
        df['Price'] = df['Price'].apply(lambda x : x.strip('$'))

        # Installs cealning
        df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))

        # Type encoding
        df['Type'] = pd.get_dummies(df['Type'])


        # Last Updated encoding
        df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))


        # Size cealning
        df['Size'] = df['Size'].apply(lambda x: x.strip('M').strip('k'))
        df[df['Size'] == 'Varies with device'] = 0

        df = df.drop(columns=['Category'])

        # df.to_csv('results.csv')

        features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']
        features.extend(category_list)
        self.x = df[features]
        self.y = df['Rating']
        
        return df
        

    def execute_linear_regression(self, df):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=10)
        
        lab_enc = preprocessing.LabelEncoder()
        y_train_encoded = lab_enc.fit_transform(y_train)
        y_test_encoded = lab_enc.fit_transform(y_test)
           
        # Init - Test others parameters
            
        for this_C in [1,3,5,20,80,120,160,200]:
            clf2 = LinearSVC(C=this_C).fit(x_train,y_train_encoded)
            scoretrain = clf2.score(x_train,y_train_encoded)
            scoretest  = clf2.score(x_test,y_test_encoded)
            print("Linear SVM value of C:{}, training score :{:2f} , Test Score: {:2f} \n".format(this_C,scoretrain,scoretest))
            
        # End - Test others parameters

    def evaluation_matrix(self, y_true, y_predict):
        print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
        print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
        print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))

    def evaluation_matrix_dict(self, y_true, y_predict, name = 'Linear - Integer'):
        dict_matrix = {}
        dict_matrix['Series Name'] = name
        dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
        dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
        dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
        return dict_matrix

main = MainClass()