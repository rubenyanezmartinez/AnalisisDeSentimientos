# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import os
for dirname, _, filenames in os.walk('/content/datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data_training = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/Corona_NLP_train.csv", encoding='ISO-8859-1', engine='python')

data_training.head()

data_training.shape

data_training.info()

data_training.isnull().sum()

data_training.Sentiment.unique()

"""**Datos de pruebas**"""

data_testing = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/Corona_NLP_test.csv", encoding='ISO-8859-1', engine='python')

data_testing.head()

data_testing.shape

data_testing.info()

data_testing.isnull().sum()

data_testing.Sentiment.unique()

"""**DATOS DE PRUEBAS VS DATOS DE ENTRENAMIENTO**"""

values_name = ['Train rows', 'Test rows']

values = [len(data_training), len(data_testing)]

plt.figure(figsize=(8,2))

plt.title("Number of rows")

ax = sns.barplot(x=values_name, y=values, palette='Blues_d')

"""**DATOS SIN ETIQUETAR**"""

tweets_covid = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/tweets_covid.csv", encoding='ISO-8859-1', engine='python')

tweets_covid.head()

tweets_covid.shape

tweets_covid.info()

tweets_covid.isnull().sum()

"""Gráfica del número de tweets diarios"""

dates = []

for date in tweets_covid['created_at'].unique():
    dates.append(date[:10])

sorted(dates)

unique_dates = []

for date in dates:
    if date not in unique_dates:
        unique_dates.append(date)

number_tweets_day = []

for date in unique_dates:
    number_tweets_day.append(dates.count(date))

data_for_dataframe = {'dates': [29, 30, 31, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                       'num_of_tweets': number_tweets_day}

df = pd.DataFrame(data=data_for_dataframe)
g = sns.relplot(x="dates", y="num_of_tweets", kind="line", data=df)