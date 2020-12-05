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

def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    labels = get_labels()

    print(conf_matrix)

    fig, ax = plt.subplots(figsize=(16, 9))

    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    ax.set_title("Matriz de Confusão", fontsize=20)
    ax.set_ylabel('Classe Verdadeira', fontsize=15)
    ax.set_xlabel('Classe Predita', fontsize=15)
    plt.show()

def show_metrics(y, y_pred):
    plot_confusion_matrix(y, y_pred)
    print(classification_report(y, y_pred, target_names=get_labels()))

def wordcloud(data, color='black', title=''):
    wordcloud = WordCloud(stopwords = STOPWORDS,
                          background_color = color,
                          width = 2500,
                          height = 2000
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

    wordcloud(positive, title='Palavras de textos positivos')
    wordcloud(negative, title='Palavras de textos negativos')