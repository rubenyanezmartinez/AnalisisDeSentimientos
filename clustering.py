# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

import os

"""# Extraer palabras más repetidas

Función que extraer todas las palabras de los tweets y el número que aparecen
"""

def extract_words (data):
    tweets = list(data)
    return pd.Series(' '.join(tweets).lower().split()).value_counts()

unlabeled_data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/data_tweets_processed.csv", encoding='ISO-8859-1', engine='python')

words = extract_words(unlabeled_data['Tweet'])
words[:20]

"""# Creación de TF-IDF

Para poder llevar a cabo este proceso es necesario convertir los textos de los tweets ya procesados en una matriz TF-IDF. Esta matriz se basa en la representación de documentos bolsa de palabras que para cada palabra de cada documento indica la frecuencia con la que se presenta.
"""

vectorizer = TfidfVectorizer(min_df=4, max_features = 10000)
list_of_tweets = list(unlabeled_data['Tweet'])
vz = vectorizer.fit_transform(list_of_tweets)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

"""# K-MEANS

El K-Means es un método de agrupamiento, que tiene como objetivo la partición de un conjunto de n observaciones en k grupos en el que cada observación pertenece al grupo cuyo valor medio es más cercano.

En este proyecto se utiliza una implementación concreta de la librería sklearn, llamada MiniBatchKMeans, que es mucho más eficiente a la hora de procesar datos.

Para aplicar clustering, lo primero es conocer el número de clusters más optimo, para esto se emplea la estadística WCSS que se define como la suma de la distancia al cuadrado entre cada miento del grupo y su centroide. Gráficamente se puede ver la relación entre el número de clusters y la WCSS, y el número de clusters óptimos es donde se comienza a estabilizar WCSS.

Almacenamos en listas los datos obtenidos de ejecutar el algoritmos con diferente número de clusters.
"""

wcss = []
list_number_of_clusters = []
list_sorted_centroids = []
list_terms = []

for number_of_clusters in range(1, 15):  
  kmeans_model = MiniBatchKMeans(n_clusters=number_of_clusters, init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
  kmeans = kmeans_model.fit(vz)
  kmeans_clusters = kmeans.predict(vz)
  list_number_of_clusters.append(number_of_clusters)
  kmeans_distances = kmeans.transform(vz)
  sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
  list_sorted_centroids.append(sorted_centroids)
  terms = vectorizer.get_feature_names()
  list_terms.append(terms)
  wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss, 'o-')
plt.title('K-means')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

"""Elegimos el número óptimo de clusters en función de la gráfica y mostramos los resultados."""

best_point = 5

number_of_clusters = list_number_of_clusters[best_point]
sorted_centroids = list_sorted_centroids[best_point]
terms = list_terms[best_point]

for i in range(number_of_clusters):
    print("Cluster %d:" % i)
    for j in sorted_centroids[i, :10]:
        print(' %s' % terms[j])
    print()

"""# Topic Modelling

El Topic Modelling abarca un conjunto de técnicas que buscan encontrar una serie de temas ocultos dentro de un conjunto de documentos. Teniendo los textos ya procesados se aplica una de las técnicas más conocidas, denominada LDA (Latent Dirichlet Allocation).

Función para mostrar los topics
"""

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_index, topic in enumerate(model.components_):
        print("\nTopic %d:" % topic_index)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

"""Crear vector de palabras"""

count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(unlabeled_data['Tweet'])

"""Indicar número de topics y palabras que se van a usar"""

number_topics = 5
number_words = 10

lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
print_topics(lda, count_vectorizer, number_words)

number_topics = 10

lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
print_topics(lda, count_vectorizer, number_words)

number_topics = 15

lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
print_topics(lda, count_vectorizer, number_words)