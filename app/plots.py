import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

def histogram(dados, title='', xlabel='', ylabel=''):
    plt.hist(dados)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def get_labels():
    return['Negativa', 'Positiva']

def plot_confusion_matrix(y_test, y_pred):
    # conf_matrix = confusion_matrix([np.argmax(x) for x in y_test], [np.argmax(x) for x in y_pred])
    conf_matrix = confusion_matrix(y_test, y_pred)
    labels = get_labels()

    fig, ax = plt.subplots(figsize=(16, 9))

    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    ax.set_title("Matriz de Confusão", fontsize=20)
    ax.set_ylabel('Classe Verdadeira', fontsize=15)
    ax.set_xlabel('Classe Predita', fontsize=15)
    plt.show()

def show_metrics(y, y_pred):
    plot_confusion_matrix(y, y_pred)
    print(classification_report(y, y_pred, target_names=get_labels()))