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
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc 


FILE_PATH = '/content/drive/My Drive/Colab Notebooks/Dataset/'

"""# Funciones y métodos

**Bolsa de palabras:**
"""

def bag_of_words (training_data, testing_data):

  tfidf = TfidfVectorizer()

  training_bag = tfidf.fit_transform(training_data)
  testing_bag = tfidf.transform(testing_data)

  return tfidf, training_bag, testing_bag

"""**Entrenar al modelo para luego poder realizar las predicciones:**"""

def train_model (classifier, training_tweets, training_sentiments, testing_tweets):

  #Create model
  classifier.fit(training_tweets, training_sentiments)
  #Predict using model
  predictions = classifier.predict(testing_tweets) 

  return predictions

"""**Obtener información de las métricas:**"""

def get_metrics (true_labels, predicted_labels):

  parameters = {}
  information = ''

  #Add metrics into dic
  parameters['accuracy'] = np.round(metrics.accuracy_score(true_labels,predicted_labels), 4)
  parameters['precision'] = np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'), 4)
  parameters['recall'] = np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'), 4)
  parameters['score'] = np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'), 4)

  #Save metrics into string variable
  information += 'Accuracy:' + str(parameters['accuracy']) + '\n'
  information += 'Precision:' + str(parameters['precision']) + '\n'
  information += 'Recall:' + str(parameters['recall']) + '\n'
  information += 'F1 Score:' + str(parameters['score']) + '\n'

  return list(parameters.values()), information

"""**Mostrar matriz de confusión:**"""

def show_confusion_matrix (true_labels, predicted_labels, classes=['Negative', 'Neutral', 'Positive']):

  total_classes = len(classes)
  level_labels = [total_classes*[0], list(range(total_classes))]

  cm = metrics.confusion_matrix(true_labels,predicted_labels, classes)
  cm_frame = pd.DataFrame(cm, pd.MultiIndex([['Predicted:'], classes], level_labels), pd.MultiIndex([['Actual:'], classes], level_labels)) 
  
  return str(cm_frame)

def show_classification_report(true_labels, predicted_labels, classes=['Negative', 'Neutral', 'Positive']):

  report = metrics.classification_report(true_labels, predicted_labels, classes) 
    
  return report

"""**Mostrar métricas de rendimiento del modelo:**"""

def show_model_performance_metrics(true_labels, predicted_labels, classes=['Negative', 'Neutral', 'Positive']):

  information = ''

  #Model performaces metrics
  information += 'Model Performance metrics:' + '\n'
  information += ('-'*30) + '\n'
  parameters, information_metrics = get_metrics(true_labels, predicted_labels)
  information += information_metrics + '\n'

  #Model classification report
  information += '\nModel Classification report:' + '\n'
  information += ('-'*30) + '\n'
  report = show_classification_report(true_labels, predicted_labels, classes)
  information +=  report + '\n'

  #Confusion matrix
  information += '\nPrediction Confusion Matrix:' + '\n'
  information += ('-'*30) + '\n'
  information += show_confusion_matrix(true_labels, predicted_labels, classes) + '\n'
    
  return parameters, report, information

"""# Datos que se van a emplear

**Datos sin etiquetar:**
"""

unlabeled_data = pd.read_csv(FILE_PATH + "data_tweets_processed.csv", encoding='ISO-8859-1', engine='python')

unlabeled_data

"""**Datos de entrenamiento:**"""

training_data = pd.read_csv(FILE_PATH + "data_train_processed.csv", encoding='ISO-8859-1', engine='python')

training_data

training_data_without = training_data[training_data.Sentiment != 'Neutral']
training_data_without

"""**Datos de pruebas:**"""

testing_data = pd.read_csv(FILE_PATH + "data_test_processed.csv", encoding='ISO-8859-1', engine='python')

testing_data

testing_data_without = testing_data[testing_data.Sentiment != 'Neutral']
testing_data_without

"""# Previo

**Crear bolsa de palabras:**
"""

tfidf, training_bag, testing_bag = bag_of_words(training_data['Tweet'].values.astype(str), testing_data['Tweet'].values.astype(str))

tfidf_without, training_bag_without, testing_bag_without = bag_of_words(training_data_without['Tweet'].values.astype(str), testing_data_without['Tweet'].values.astype(str))

"""# Random Forest

**Crear clasificador:**
"""

random_forest_classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

random_forest_classifier_without = RandomForestClassifier(n_estimators=106, n_jobs=-1)

"""**Entrenar al clasificador con los datos:**"""

rfc_tfidf_predictions = train_model(random_forest_classifier, training_bag, training_data['Sentiment'], testing_bag)

rfc_tfidf_predictions_without = train_model(random_forest_classifier_without, training_bag_without, training_data_without['Sentiment'], testing_bag_without)

"""**Mostrar parametros resultantes del modelo:**"""

parameters_random_forest, report_random_forest, information_random_forest = show_model_performance_metrics(testing_data['Sentiment'], rfc_tfidf_predictions, ['Negative', 'Neutral', 'Positive'])
print(information_random_forest)

parameters_random_forest_without, report_random_forest_without, information_random_forest_without = show_model_performance_metrics(testing_data_without['Sentiment'], rfc_tfidf_predictions_without, ['Negative', 'Positive'])
print(information_random_forest_without)

"""# KNeighbors

**Crear clasificador:**

*Número de vecinos = 5*
"""

neigh_5_classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

neigh_5_classifier_without = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

"""*Número de vecinos = 10*"""

neigh_10_classifier = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)

neigh_10_classifier_without = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)

"""*Número de vecinos = 15*"""

neigh_15_classifier = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)

neigh_15_classifier_without = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)

"""*Número de vecinos = 20*"""

neigh_20_classifier = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)

"""**Entrenar al clasificador con los datos:**

*Número de vecinos = 5*
"""

neigh_5_tfidf_predictions = train_model(neigh_5_classifier, training_bag, training_data['Sentiment'], testing_bag)

neigh_5_tfidf_predictions_without = train_model(neigh_5_classifier_without, training_bag_without, training_data_without['Sentiment'], testing_bag_without)

"""*Número de vecinos = 10*"""

neigh_10_tfidf_predictions = train_model(neigh_10_classifier, training_bag, training_data['Sentiment'], testing_bag)

neigh_10_tfidf_predictions_without = train_model(neigh_10_classifier_without, training_bag_without, training_data_without['Sentiment'], testing_bag_without)

"""*Número de vecinos = 15*"""

neigh_15_tfidf_predictions = train_model(neigh_15_classifier, training_bag, training_data['Sentiment'], testing_bag)

neigh_15_tfidf_predictions_without = train_model(neigh_15_classifier_without, training_bag_without, training_data_without['Sentiment'], testing_bag_without)

"""*Número de vecinos = 20*"""

neigh_20_tfidf_predictions = train_model(neigh_20_classifier, training_bag, training_data['Sentiment'], testing_bag)

"""**Mostrar parámetros resultantes del modelo:**

*Número de vecinos = 5*
"""

parameters_neigh_5, report_neigh_5, information_neigh_5 = show_model_performance_metrics(testing_data['Sentiment'], neigh_5_tfidf_predictions, ['Negative', 'Neutral', 'Positive'])
print(information_neigh_5)

parameters_neigh_5_without, report_neigh_5_without, information_neigh_5_without = show_model_performance_metrics(testing_data_without['Sentiment'], neigh_5_tfidf_predictions_without, ['Negative', 'Positive'])
print(information_neigh_5_without)

"""*Número de vecinos = 10*"""

parameters_neigh_10, report_neigh_10, information_neigh_10 = show_model_performance_metrics(testing_data['Sentiment'], neigh_10_tfidf_predictions, ['Negative', 'Neutral', 'Positive'])
print(information_neigh_10)

parameters_neigh_10_without, report_neigh_10_without, information_neigh_10_without = show_model_performance_metrics(testing_data_without['Sentiment'], neigh_10_tfidf_predictions_without, ['Negative', 'Positive'])
print(information_neigh_10_without)

"""*Número de vecinos = 15*"""

parameters_neigh_15, report_neigh_15, information_neigh_15 = show_model_performance_metrics(testing_data['Sentiment'], neigh_15_tfidf_predictions,['Negative', 'Neutral', 'Positive'])
print(information_neigh_15)

parameters_neigh_15_without, report_neigh_15_without, information_neigh_15_without = show_model_performance_metrics(testing_data_without['Sentiment'], neigh_15_tfidf_predictions_without, ['Negative', 'Positive'])
print(information_neigh_15_without)

"""*Número de vecinos = 20*"""

parameters_neigh_20, report_neigh_20, information_neigh_20 = show_model_performance_metrics(testing_data['Sentiment'], neigh_20_tfidf_predictions,['Negative', 'Neutral', 'Positive'])
print(information_neigh_20)

"""# SVM

**Crear clasificador:**
"""

support_vector_machine = svm.SVC()

support_vector_machine_without = svm.SVC()

"""**Entrenar al clasificador con los datos:**"""

support_vector_machine_tfidf_predictions = train_model(support_vector_machine, training_bag, training_data['Sentiment'], testing_bag)

support_vector_machine_tfidf_predictions_without = train_model(support_vector_machine_without, training_bag_without, training_data_without['Sentiment'], testing_bag_without)

"""**Mostrar parámetros resultantes del modelo:**"""

parameters_support_vector_machine, report_support_vector_machine, information_support_vector_machine = show_model_performance_metrics(testing_data['Sentiment'], support_vector_machine_tfidf_predictions, ['Negative', 'Neutral', 'Positive'])
print(information_support_vector_machine)

parameters_support_vector_machine_without, report_support_vector_machine_without, information_support_vector_machine_without = show_model_performance_metrics(testing_data_without['Sentiment'], support_vector_machine_tfidf_predictions_without, ['Negative', 'Positive'])
print(information_support_vector_machine)

"""# Naive Bayes

*Reducir el número de instancias, en caso contrario el modelo devuelve un error*
"""

training_data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/data_train_processed.csv", encoding='ISO-8859-1', engine='python')
training_data = training_data.loc[0:31000]

tfidf, training_bag, testing_bag = bag_of_words(training_data['Tweet'].values.astype(str), testing_data['Tweet'].values.astype(str))

training_data_without = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/data_train_processed.csv", encoding='ISO-8859-1', engine='python')
training_data_without = training_data_without.loc[0:31000]

tfidf_without, training_bag_without, testing_bag_without = bag_of_words(training_data_without['Tweet'].values.astype(str), testing_data_without['Tweet'].values.astype(str))

"""**Crear clasificador:**"""

naive_bayes_classifier = BernoulliNB()

naive_bayes_classifier_without = BernoulliNB()

"""**Entrenar al clasificador con los datos:**"""

naive_bayes_tfidf_predictions = train_model(naive_bayes_classifier, training_bag.toarray(), training_data['Sentiment'], testing_bag.toarray())

naive_bayes_tfidf_predictions_without = train_model(naive_bayes_classifier_without, training_bag_without.toarray(), training_data_without['Sentiment'], testing_bag_without.toarray())

"""**Mostrar parámetros resultantes del modelo:**"""

parameters_naive_bayes, report_naive_bayes, information_naive_bayes = show_model_performance_metrics(testing_data['Sentiment'], naive_bayes_tfidf_predictions,['Negative', 'Neutral', 'Positive'])
print(information_naive_bayes)

parameters_naive_bayes_without, report_naive_bayes_without, information_naive_bayes_without = show_model_performance_metrics(testing_data_without['Sentiment'], naive_bayes_tfidf_predictions_without, ['Negative', 'Positive'])
print(information_naive_bayes_without)