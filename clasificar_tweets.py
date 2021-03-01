# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from unicodedata import normalize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc 
from sklearn.naive_bayes import BernoulliNB


FILE_PATH = '/content/drive/My Drive/Colab Notebooks/Dataset/'

"""# Funciones y métodos

**Bolsa de palabras:**
"""

def bag_of_words (training_data, testing_data):

  tfidf = TfidfVectorizer()

  training_bag = tfidf.fit_transform(training_data)
  testing_bag = tfidf.transform(testing_data)

  return tfidf, training_bag, testing_bag

"""**Entrenar al modelo y realizar las predicciones:**"""

def train_and_classify (classifier, training_tweets, training_sentiments, unlabeled_tweets):

  #Create model
  classifier.fit(training_tweets, training_sentiments)
  #Predict using model
  predictions = classifier.predict(unlabeled_tweets) 

  return predictions

"""# Datos que se van a emplear

**Datos sin etiquetar:**
"""

unlabeled_data = pd.read_csv(FILE_PATH + "data_tweets_processed.csv", encoding='ISO-8859-1', engine='python')

unlabeled_data

"""**Datos de entrenamiento:**"""

training_data = pd.read_csv(FILE_PATH + "data_train_processed.csv", encoding='ISO-8859-1', engine='python')

training_data

"""**Datos de pruebas:**"""

testing_data = pd.read_csv(FILE_PATH + "data_test_processed.csv", encoding='ISO-8859-1', engine='python')

testing_data

"""# Previo

**Crear bolsa de palabras:**
"""

tfidf, training_bag, unlabeled_bag = bag_of_words(training_data['Tweet'].values.astype(str), unlabeled_data['Tweet'].values.astype(str))

"""# SVM

**Crear clasificador:**
"""

support_vector_machine = svm.SVC()

"""**Entrenar al clasificador con los datos de entrenamiento y predecir las etiquetas de los datos sin sentimiento:**"""

predictions = train_and_classify(support_vector_machine, training_bag, training_data['Sentiment'], unlabeled_bag)

"""**Añadir el sentimiento a los datos no etiquetados:**"""

unlabeled_data['Sentiment'] = predictions
labeled_data = unlabeled_data
labeled_data

"""**Ordenar los datos en función de la fecha de publicación:**"""

labeled_data = labeled_data.sort_values(by=['Date'])
labeled_data

"""**Dejar en la columna fecha sólo la información sobre el año, el mes y el día:**"""

labeled_data['Date'] = labeled_data['Date'].apply(lambda date: date[0:10])
labeled_data = labeled_data.reset_index(drop = True)
labeled_data

"""**Guardar nuevo dataset ya etiquetado:**"""

labeled_data.to_csv(FILE_PATH + 'labeled_final_data.csv',index=False)

"""**Conseguir los sentimientos de los tweets por día:**"""

unique_dates = labeled_data['Date'].unique()

number_tweets_negative = list()
number_tweets_neutral = list()
number_tweets_positive = list()

for date in unique_dates:
  sentiments_by_date = list(labeled_data.loc[labeled_data['Date'] == date, 'Sentiment'])

  negative_sentiments = sentiments_by_date.count('Negative')
  neutral_sentiments = sentiments_by_date.count('Neutral')
  positive_sentiments = sentiments_by_date.count('Positive')

  number_tweets_negative.append(negative_sentiments)
  number_tweets_neutral.append(neutral_sentiments)
  number_tweets_positive.append(positive_sentiments)

unique_dates_without_year = [x for x in range (33)]

"""**Mostrar gráfico con número de tweets según el sentimiento por día:**"""

plt.plot(unique_dates_without_year, number_tweets_negative, label='Negative')
plt.plot(unique_dates_without_year, number_tweets_neutral, label='Neutral')
plt.plot(unique_dates_without_year, number_tweets_positive, label='Positive')

plt.legend()
plt.title('Number of tweets: Sentiments and Days')
plt.show()