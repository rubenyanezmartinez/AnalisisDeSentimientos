# -*- coding: utf-8 -*-
import nltk
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from unicodedata import normalize
from nltk.stem import SnowballStemmer
import seaborn as sns


nltk.download('stopwords')
nltk.download('punkt')

import os

"""Extraer el número de palabras, número de letras y sentimiento de cada uno de los tweets.

Método para datos con la columna sentimiento
"""

def data_len(data):
    
    tweets = []
    sentiments = []
    letters = []
    words = []
    
    for i, tweet in data.iterrows():
        tweet_split = tweet.Tweet.split()
        
        sentiments.append(tweet.Sentiment)
        tweets.append(tweet.Tweet)
        letters.append(len(tweet.Tweet))
        words.append(len(tweet_split))
    
    data['Tweet'] = tweets
    data['Sentiment'] = sentiments
    data['Words'] = words
    data['Letters'] = letters
    
    return data

"""Método para datos sin la columna sentimiento"""

def data_len_only_words(data):
    
    tweets = []
    words = []
    
    for i, tweet in data.iterrows():
        tweet_split = tweet.Tweet.split()
        
        tweets.append(tweet.Tweet)
        words.append(len(tweet_split))
    
    data['Tweet'] = tweets
    data['Words'] = words
    
    return data

"""Métodos para mostrar mediante un gráfico el número de palabras de lso tweets, de manera general o por sentimientos."""

def graphic_words (data):
    sns.boxenplot(x=data)
    
def graphic_words_sentiment(data_x, data_y):
    sns.boxenplot(x=data_x, y=data_y)

"""# Preprocesamiento de los datos

**Método que realiza las siguientes transformaciones a los datos:**

*	Eliminar enlaces, como los que contengan las expresiones: ‘http://’, ‘https://’, ‘.com’, ‘.es’, ‘.net’, ‘.org’, ‘.uk’ y ‘.us’
*	Eliminar menciones (@).
*	Eliminar hashtags (#).
*	Eliminar retweets (rt).
*	Eliminar caracteres duplicados, para que no tengan tanto peso en el resultado final.
*	Eliminar risas como puede ser "jajajaja", ya que pueden ser indicativo de más de un sentimiento dependiendo del contexto.
*	Eliminar caracteres especiales que se encuentren en las letras, como pueden ser los acentos.
*	Eliminar caracteres que no sean alfabéticos.
*	Sustituir los tabuladores por espacios en blanco. 
*	Sustituir los saltos de línea por espacios en blanco.
*	Eliminar varios espacios en blanco, sustituyendo por un sólo espacio en blanco.
*	Eliminar espacios en blanco antes y después del tweet.
*	Eliminar las STOPWORDS, palabras que necesitan ir con otras palabras para que aporten valor. Como pueden ser las preposiciones.
*	Reducir las palabras a su raíz.
"""

def preprocessing_tweets (data):
    tweets = []
    sentiments = []
    
    for i, tweet in data.iterrows():
        words_cleaned = ""
        tweet_clean = tweet.Tweet.lower()
        
        words_cleaned = " ".join(
            [
                word for word in tweet_clean.split()
                
                if 'http://' not in word
                and 'https://' not in word
                and '.com' not in word
                and '.es' not in word
                and '.net' not in word
                and '.org' not in word
                and '.uk' not in word
                and '.us' not in word
                and not word.startswith('@')
                and not word.startswith('#')
                and word != 'rt'
            ])
        
        
        tweet_clean = re.sub(r'\b([jh]*[aeiou]*[jh]+[aeiou]*)*\b',"",words_cleaned)
        tweet_clean = re.sub(r'(.)\1{2,}',r'\1',tweet_clean)
        tweet_clean = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize( "NFD", tweet_clean), 0, re.I)
        tweet_clean = re.sub("[^a-zA-Z]"," ",tweet_clean)
        tweet_clean = re.sub("\t", " ", tweet_clean)
        tweet_clean = re.sub(" +", " ",tweet_clean) 
        tweet_clean = re.sub("^ ", "", tweet_clean)
        tweet_clean = re.sub(" $", "", tweet_clean)
        tweet_clean = re.sub("\n", "", tweet_clean)
        
        words_cleaned=""
        stemmed =""
        
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        
        tokens = word_tokenize(tweet_clean)
        
        words_cleaned =[word for word in tokens if not word in stop_words]
        stemmed = " ".join([stemmer.stem(word) for word in words_cleaned])
        
        
    
        sentiments.append(tweet.Sentiment)
        tweets.append(stemmed)
    
    data['Tweet'] = tweets
    data['Sentiment'] = sentiments
    data.loc[:,['Sentiment','Tweet']]
    
    return data

def preprocessing_tweets_without_tags (data):
    tweets = []
    dates = []
    
    for i, tweet in data.iterrows():
        words_cleaned = ""
        tweet_clean = tweet.Tweet.lower()
        
        words_cleaned = " ".join(
            [
                word for word in tweet_clean.split()
                
                if 'http://' not in word
                and 'https://' not in word
                and '.com' not in word
                and '.es' not in word
                and '.net' not in word
                and '.org' not in word
                and '.uk' not in word
                and '.us' not in word
                and not word.startswith('@')
                and not word.startswith('#')
                and word != 'rt'
            ])
        
        
        tweet_clean = re.sub(r'\b([jh]*[aeiou]*[jh]+[aeiou]*)*\b',"",words_cleaned)
        tweet_clean = re.sub(r'(.)\1{2,}',r'\1',tweet_clean)
        tweet_clean = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize( "NFD", tweet_clean), 0, re.I)
        tweet_clean = re.sub("[^a-zA-Z]"," ",tweet_clean)
        tweet_clean = re.sub("\t", " ", tweet_clean)
        tweet_clean = re.sub(" +", " ",tweet_clean) 
        tweet_clean = re.sub("^ ", "", tweet_clean)
        tweet_clean = re.sub(" $", "", tweet_clean)
        tweet_clean = re.sub("\n", "", tweet_clean)
        
        words_cleaned=""
        stemmed =""
        
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        
        tokens = word_tokenize(tweet_clean)
        
        words_cleaned =[word for word in tokens if not word in stop_words]
        stemmed = " ".join([stemmer.stem(word) for word in words_cleaned])
        
        
    
        dates.append(tweet.Date)
        tweets.append(stemmed)
    
    data['Tweet'] = tweets
    data['Date'] = dates
    data.loc[:,['Date','Tweet']]
    
    return data

"""# DATOS DE ENTRENAMIENTO"""

data_training = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/Corona_NLP_train.csv", encoding='ISO-8859-1', engine='python')

data_training.info()

data_training = data_training.drop(['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1)

data_training = data_training.rename(columns={'OriginalTweet': 'Tweet'})

data_training.Sentiment.unique()

"""Eliminar etiquetas de sentimientos extremos, y pasarlas a Positivo o Negativo."""

data_training['Sentiment'] = data_training['Sentiment'].replace('Extremely Negative', 'Negative')
data_training['Sentiment'] = data_training['Sentiment'].replace('Extremely Positive', 'Positive')

data_training.Sentiment.unique()

"""# DATOS DE TESTING"""

data_testing = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/Corona_NLP_test.csv", encoding='ISO-8859-1', engine='python')

data_testing.info()

data_testing = data_testing.drop(['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1)

data_testing = data_testing.rename(columns={'OriginalTweet': 'Tweet'})

data_testing.Sentiment.unique()

"""Eliminar etiquetas de sentimientos extremos, y pasarlas a Positivo o Negativo."""

data_testing['Sentiment'] = data_testing['Sentiment'].replace('Extremely Negative', 'Negative')
data_testing['Sentiment'] = data_testing['Sentiment'].replace('Extremely Positive', 'Positive')

data_testing.Sentiment.unique()

"""# DATOS SIN ETIQUETAR"""

tweet_covid = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Dataset/tweets_covid.csv", encoding='ISO-8859-1', engine='python')

tweet_covid.info()

tweet_covid = tweet_covid.rename(columns={'created_at': 'Date'})
tweet_covid = tweet_covid.rename(columns={'text': 'Tweet'})

tweet_covid.info()

"""# Recopilación de información ANTES"""

data_training.Sentiment.value_counts()

plt.title('Data Training')
plt.bar(['Positive','Negative','Neutral'],[18046, 15398,7713], color=['green','red','blue'])

data_testing.Sentiment.value_counts()

plt.title('Data Testing')
plt.bar(['Positive','Negative','Neutral'],[1633, 1546,619], color=['green','red','blue'])

data_training = data_len(data_training)
data_training

graphic_words(data_training['Words'])

graphic_words_sentiment(data_training['Words'], data_training['Sentiment'])

data_testing = data_len(data_testing)
data_testing

graphic_words(data_testing['Words'])

graphic_words_sentiment(data_testing['Words'], data_testing['Sentiment'])

tweet_covid = data_len_only_words(tweet_covid)
tweet_covid

graphic_words(tweet_covid['Words'])

"""# Limpiado de datos: Preprocesar"""

data_training_cleaned = preprocessing_tweets(data_training)
data_training_cleaned.loc[:,['Sentiment','Tweet']]

data_testing_cleaned = preprocessing_tweets(data_testing)
data_testing_cleaned.loc[:,['Sentiment','Tweet']]

data_tweets_cleaned = preprocessing_tweets_without_tags(tweet_covid)
data_tweets_cleaned.loc[:,['Date','Tweet']]
data_tweets_cleaned = data_tweets_cleaned[data_tweets_cleaned.Tweet != '']

"""# Guardar datos preprocesados"""

data_training_final = data_training_cleaned.loc[:,['Sentiment','Tweet']]
data_training_final
data_training_final.to_csv('/content/drive/My Drive/Colab Notebooks/Dataset/data_train_processed.csv',index=False)

data_testing_final = data_testing_cleaned.loc[:,['Sentiment','Tweet']]
data_testing_final
data_testing_final.to_csv('/content/drive/My Drive/Colab Notebooks/Dataset/data_test_processed.csv',index=False)

data_tweets_final = data_tweets_cleaned.loc[:,['Date','Tweet']]
data_tweets_final
data_tweets_final.to_csv('/content/drive/My Drive/Colab Notebooks/Dataset/data_tweets_processed.csv',index=False)

"""# Recopilación de información DESPUES"""

data_training_final = data_len(data_training_final)
data_training_final

graphic_words(data_training_final['Words'])

graphic_words_sentiment(data_training_final['Words'], data_training_final['Sentiment'])

data_testing_final = data_len(data_testing_final)
data_testing_final

graphic_words(data_testing_final['Words'])

graphic_words_sentiment(data_testing_final['Words'], data_testing_final['Sentiment'])

data_tweets_final = data_len_only_words(data_tweets_final)
data_tweets_final

graphic_words(data_tweets_final['Words'])