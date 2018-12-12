# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

class MainClass:

    def __init__(self):
        print("Início programa.")
        df = pd.read_csv('files/playstore_pre_processed.csv')
        df.info()

main = MainClass()