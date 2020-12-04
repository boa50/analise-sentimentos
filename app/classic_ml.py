import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import utils

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

def prepare_data(dataset_path, vectorization='bow', verbose=False):
    X, y = utils.prepare_xy(dataset_path)

    if vectorization == 'bow':
        X = vecotrize_bow(X)
    elif vectorization == 'tfidf':
        X = vectorize_tfidf(X)
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