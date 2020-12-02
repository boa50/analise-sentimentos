import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import plots

# from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# def config_gpu():
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = InteractiveSession(config=config)

def prepare_data(dataset_path):
    random_state = 50

    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(dataset_path, encoding="ISO-8859-1", names=column_names)
    df = df.drop(columns=['id', 'date', 'flag', 'user'])

    # plots.histogram(df['target'], title='Distribuição dos dados')

    df.loc[df['target'] == 4, 'target'] = 1

    # print(df.head())
    # print(df.tail())

    X = df['text']
    y = df['target']

    PONTUACOES = re.compile('(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])')
    HTTP = re.compile('https?://[^\s/$.?#]*.[^\s]*')
    USER = re.compile('^@(\w){1,15}')

    X = [HTTP.sub('', line.lower()) for line in X]
    X = [USER.sub('', line.lower()) for line in X]
    X = [PONTUACOES.sub("", line.lower()) for line in X]

    cv = CountVectorizer()
    X = cv.fit_transform(df['text'])

    # print(X.shape)

    X_train, X_res, y_train, y_res = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_res, y_res, test_size=0.5, random_state=random_state)

    # print('Treino')
    # print(X_train.shape)
    # print('Validação')
    # print(X_val.shape)
    # print('Teste')
    # print(X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    # config_gpu()    
    
    # dataset_path = 'app/dataset/dataset.csv'
    dataset_path = 'app/dataset/dataset-simple.csv'

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(dataset_path)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    plots.show_metrics(y_test, y_pred)