import numpy as np
import matplotlib.pyplot as plt

def histogram(dados, title='', xlabel='', ylabel=''):
    plt.hist(dados)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()