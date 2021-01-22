import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from sklearn.metrics import confusion_matrix, classification_report

from classic_ml import tokenize

def histogram(dados, title='', xlabel='', ylabel=''):
    plt.hist(dados)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def get_labels():
    return['Negativa', 'Positiva']

def plot_confusion_matrix(y_test, y_pred, conf_matrix=None):
    if conf_matrix is None:
        conf_matrix = confusion_matrix(y_test, y_pred)
    
    labels = get_labels()

    print(conf_matrix)

    fig, ax = plt.subplots(figsize=(16, 9))

    sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size":40})
    ax.set_title("Matriz de Confusão", fontsize=22)
    ax.set_ylabel('Classe Verdadeira', fontsize=20)
    ax.set_xlabel('Classe Predita', fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)
    ax.set_xticklabels(labels, fontsize=20)

    plt.show()

def show_metrics(y, y_pred):
    plot_confusion_matrix(y, y_pred)
    print(classification_report(y, y_pred, target_names=get_labels()))

def wordcloud(data, color='black', title=''):
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          max_words=100,
                          width=2500,
                          height=2000
                         ).generate(' '.join(data))
    plt.figure(figsize = (13, 13))
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.show()

def wordclouds(X):
    positive = []
    negative = []
    for i, x in enumerate(X):
        if i < len(X)/2:
            negative.append(' '.join(tokenize(x)))
        else:
            positive.append(' '.join(tokenize(x)))

    wordcloud(negative, title='Textos Negativos')
    wordcloud(positive, title='Textos Positivos', color='white')