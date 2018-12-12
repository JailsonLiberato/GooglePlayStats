# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import random
import matplotlib.pyplot as plt

class MainClass:

    def __init__(self):
        df = pd.read_csv('files/googleplaystore.csv')
        self.executePreProcessamento(df)
        self.executeLinearRegression(df)
        
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

    def executePreProcessamento(self, df):
        CategoryString = df["Category"]
        categoryVal = df["Category"].unique()
        categoryValCount = len(categoryVal)
        category_dict = {}
        for i in range(0,categoryValCount):
            category_dict[categoryVal[i]] = i
        df["Category_c"] = df["Category"].map(category_dict).astype(int)
        #print(df["Category_c"])

        #filling Size which had NA
        df["Size"] = df["Size"].map(self.change_size)
        df.Size.fillna(method = 'ffill', inplace = True)

        #Cleaning no of installs classification 
        df['Installs'] = df['Installs'].map(self.installs_clean).astype(int)
        #print(df['Installs'])

        df['Type'] = df['Type'].map(self.type_cat)
        #print(df['Type'])

        #Cleaning of content rating classification
        RatingL = df['Content Rating'].unique()
        RatingDict = {}
        for i in range(len(RatingL)):
            RatingDict[RatingL[i]] = i
        df['Content Rating'] = df['Content Rating'].map(RatingDict).astype(int)
        #print(df['Content Rating'])

        df.drop(labels = ['Last Updated','Current Ver','Android Ver','App'], axis = 1, inplace = True)

        #Cleaning of genres
        GenresL = df.Genres.unique()
        GenresDict = {}
        for i in range(len(GenresL)):
            GenresDict[GenresL[i]] = i
        df['Genres_c'] = df['Genres'].map(GenresDict).astype(int)
        #print(df['Genres_c'])

        
        df['Price'] = df['Price'].map(self.price_clean).astype(float)


        #print(df['Price'])
        #df.info()

        df['Reviews'] = df['Reviews'].map(self.change_size)
        df.Reviews.fillna(method = 'ffill', inplace = True)
        #df['Reviews'] = df['Reviews'].astype(int)
        #print(df['Reviews'])
        
        df.info()



    def executeLinearRegression(self, df):
        X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1)
        y = df.Rating
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        model = LinearRegression()
        model.fit(X_train,y_train)
        Results = model.predict(X_test)

    def Evaluationmatrix(self, y_true, y_predict):
        print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
        print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
        print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))

    def Evaluationmatrix_dict(self, y_true, y_predict, name = 'Linear - Integer'):
        dict_matrix = {}
        dict_matrix['Series Name'] = name
        dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
        dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
        dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
        return dict_matrix

main = MainClass()