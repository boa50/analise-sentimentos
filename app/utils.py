import pandas as pd
from sklearn.model_selection import train_test_split

import plots

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def prepare_xy(dataset_path, histogram_plot=False):
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(dataset_path, encoding="ISO-8859-1", names=column_names)
    df = df.drop(columns=['id', 'date', 'flag', 'user'])

    if histogram_plot:
        plots.histogram(df['target'], title='Distribuição dos dados')

    df.loc[df['target'] == 4, 'target'] = 1

    X = df['text']
    y = df['target']

    return X, y

def split_data(X, y, attention_masks=None):
    random_state = 50
    first_split = 0.2
    second_split = 0.5

    if attention_masks is None:
        X_train, X_res, y_train, y_res = train_test_split(X, y, test_size=first_split, random_state=random_state)
        X_test, X_val, y_test, y_val = train_test_split(X_res, y_res, test_size=second_split, random_state=random_state)

        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        X_train, X_res, y_train, y_res, mask_train, mask_res = train_test_split(X, y, attention_masks, test_size=first_split)
        X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(X_res, y_res, mask_res, test_size=second_split)

        return X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test