import os
import pickle
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier

from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import utils

def save_vectors(X, save_path):
    print('Salvando o vetor...')
    pickle.dump((X), open(save_path, 'wb'))
    print('Vetor salvo :)')

def load_vectors(load_path):
    print('Carregando o vetor salvo...')
    X = pickle.load(open(load_path, 'rb'))
    print('Vetor carregado :)')

    return X

def words_remove(token):
    token = re.sub('https?://[^\s/$.?#]*.[^\s]*', '', token)
    token = re.sub('@(\w){1,15}', '', token)
    token = re.sub('(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])', '', token)

    return token

def tokens_prepare(tokens):
    # STOPWORDS = stopwords.words('english')
    STOPWORDS = []
    lemmatizer = WordNetLemmatizer()
    tokens_prepared = []

    for token, tag in pos_tag(tokens):
        token = words_remove(token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemmatizer.lemmatize(token, pos)
        token = token.lower()

        if len(token) > 1 and token not in STOPWORDS:
            tokens_prepared.append(token)

    return tokens_prepared

def tokenize(text):
    tokenizer = TweetTokenizer(reduce_len=True)
    tokens = tokenizer.tokenize(text)
    tokens = tokens_prepare(tokens)

    return tokens

def vecotrize_bow(X, save=True, save_path='bow.pickle'):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    if save:
        save_vectors(X, save_path)

    return X

def vectorize_tfidf(X, save=True, save_path='tfidf.pickle'):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, tokenizer=tokenize)
    X = vectorizer.fit_transform(X)

    if save:
        save_vectors(X, save_path)

    return X

def prepare_data(dataset_path, vectorization='bow', verbose=False):
    X, y = utils.prepare_xy(dataset_path)

    bow_path = 'app/saves/embeddings/bow.pickle'
    tfidf_path = 'app/saves/embeddings/tfidf.pickle'

    if vectorization == 'bow':
        if os.path.isfile(bow_path):
            X = load_vectors(bow_path)
        else:
            X = vecotrize_bow(X, save_path=bow_path)
    elif vectorization == 'tfidf':
        if os.path.isfile(tfidf_path):
            X = load_vectors(tfidf_path)
        else:
            X = vectorize_tfidf(X, save_path=tfidf_path)
    else:
        raise ValueError('Método não implementado!')

    X_train, X_test, X_val, y_train, y_test, y_val = utils.split_data(X, y)

    if verbose:
        print('Treino')
        print(X_train.shape)
        print('Validação')
        print(X_val.shape)
        print('Teste')
        print(X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train(classifier, X_train, y_train):
    if classifier == 'complement_nb':
        clf = ComplementNB()
    elif classifier == 'random_forest':
        clf = RandomForestClassifier(n_jobs=-1, random_state=50)
    else:
        raise ValueError('Método não implementado!')

    clf.fit(X_train, y_train)

    path = 'app/saves/clfs/' + classifier + '.pickle'
    pickle.dump((clf), open(path, 'wb'))

    return clf

def load_classifier(classifier):
    print('Carregando o classificador...')
    path = 'app/saves/clfs/' + classifier + '.pickle'
    clf = pickle.load(open(path, 'rb'))
    print('Classificador carregado :)')

    return clf