import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import plots
import bert
import utils

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def words_remove(token):
    token = re.sub('https?://[^\s/$.?#]*.[^\s]*', '', token)
    token = re.sub('@(\w){1,15}', '', token)
    token = re.sub('(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])', '', token)

    return token

def tokens_prepare(tokens):
    STOPWORDS = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    tokens_prepared = []

    for word, tag in pos_tag(tokens):
        word = words_remove(word)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemmatizer.lemmatize(word, pos)
        token = token.lower()

        if len(token) > 1 and token not in STOPWORDS:
            tokens_prepared.append(token)

    return tokens_prepared

def tokenize(text):
    tokenizer = TweetTokenizer(reduce_len=True)
    tokens = tokenizer.tokenize(text)
    tokens = tokens_prepare(tokens)

    return tokens

def vecotrize_bow(X):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(X)

def vectorize_tfidf(X):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, tokenizer=tokenize)
    return vectorizer.fit_transform(X)

def prepare_xy(dataset_path):
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(dataset_path, encoding="ISO-8859-1", names=column_names)
    df = df.drop(columns=['id', 'date', 'flag', 'user'])

    # plots.histogram(df['target'], title='Distribuição dos dados')

    df.loc[df['target'] == 4, 'target'] = 1

    # print(df.head())
    # print(df.tail())

    X = df['text']
    y = df['target']

    return X, y

def prepare_data(dataset_path, vectorization='bow'):
    # random_state = 50

    # column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    # df = pd.read_csv(dataset_path, encoding="ISO-8859-1", names=column_names)
    # df = df.drop(columns=['id', 'date', 'flag', 'user'])

    # # plots.histogram(df['target'], title='Distribuição dos dados')

    # df.loc[df['target'] == 4, 'target'] = 1

    # # print(df.head())
    # # print(df.tail())

    # X = df['text']
    # y = df['target']

    X, y = prepare_xy(dataset_path)

    if vectorization == 'bow':
        X = vecotrize_bow(X)
    elif vectorization == 'tfidf':
        X = vectorize_tfidf(X)
    else:
        raise ValueError('Método não implementado!')

    # print(X.shape)

    # X_train, X_res, y_train, y_res = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # X_test, X_val, y_test, y_val = train_test_split(X_res, y_res, test_size=0.5, random_state=random_state)

    X_train, X_test, X_val, y_train, y_test, y_val = utils.split_data(X, y)

    # print('Treino')
    # print(X_train.shape)
    # print('Validação')
    # print(X_val.shape)
    # print('Teste')
    # print(X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def show_wordcloud(dataset_path):
    X, _ = prepare_xy(dataset_path)
    positive = []
    negative = []
    for i, x in enumerate(X):
        if i < len(X)/2:
            negative.append(' '.join(tokenize(x)))
        else:
            positive.append(' '.join(tokenize(x)))

    plots.wordcloud(positive)
    plots.wordcloud(negative)

if __name__ == '__main__':
    config_gpu()    
    
    # dataset_path = 'app/dataset/dataset.csv'
    dataset_path = 'app/dataset/dataset-simple.csv'

    # show_wordcloud(dataset_path)

    # X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(dataset_path, vectorization='tfidf')

    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    # plots.show_metrics(y_test, y_pred)

    # import tensorflow as tf
    # from transformers import *
    # from transformers import BertTokenizer, TFBertForSequenceClassification

    # bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    X, y = prepare_xy(dataset_path)
    bert.train(X, y)

    # inputs = []
    # attention_masks = []

    # for x in X:
    #     tokenized = bert_tokenizer.encode_plus(
    #         x,
    #         add_special_tokens=True,
    #         max_length=128,
    #         padding='max_length',
    #         return_attention_mask=True)

    #     inputs.append(tokenized['input_ids'])
    #     attention_masks.append(tokenized['attention_mask'])

    # inputs = np.asarray(inputs)
    # attention_masks = np.array(attention_masks)
    # labels = np.array(y)

    # train_inp,val_inp,train_label,val_label,train_mask,val_mask = train_test_split(inputs, labels, attention_masks, test_size=0.2)

    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

    # bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])


    # history=bert_model.fit([train_inp,train_mask],train_label,batch_size=16,epochs=4,validation_data=([val_inp,val_mask],val_label))