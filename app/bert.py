import os.path
import pickle
import numpy as np
from tensorflow import keras
from transformers import BertTokenizer, TFBertForSequenceClassification

import utils

model_save_path = 'app/saves/model/best_model.h5'
emeddings_root_path = 'app/saves/embeddings/'
inputs_path = emeddings_root_path + 'inputs.pickle'
mask_path = emeddings_root_path + 'mask.pickle'
label_path = emeddings_root_path + 'label.pickle'

def save_embeddings(inputs, attention_masks, y):
    print('Salvando os embeddings...')

    pickle.dump((inputs), open(inputs_path, 'wb'))
    pickle.dump((attention_masks), open(mask_path, 'wb'))
    pickle.dump((y), open(label_path, 'wb'))

    print('Embeddings salvos :)')

def load_embeddings():
    print('Carregando os embeddings salvos...')

    inputs = pickle.load(open(inputs_path, 'rb'))
    attention_masks = pickle.load(open(mask_path, 'rb'))
    y = pickle.load(open(label_path, 'rb'))

    print('Embeddings carregados :)')

    return inputs, attention_masks, y

def tokenize(X, y, split=True, save=True):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    inputs = []
    attention_masks = []

    for x in X:
        tokenized = tokenizer.encode_plus(
                    x,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    return_attention_mask=True)

        inputs.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])

    inputs = np.asarray(inputs)
    attention_masks = np.array(attention_masks)
    y = np.array(y)

    if save:
        save_embeddings(inputs, attention_masks, y)

    if split:
        X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test = utils.split_data(inputs, y, attention_masks)
        return X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test
    else:
        return inputs, attention_masks, y

def model_compile():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)

    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model

def train(X, y):
    if os.path.isfile(inputs_path):
        inputs, attention_masks, y = load_embeddings()
        X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test = utils.split_data(inputs, y, attention_masks)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test = tokenize(X, y)

    model = model_compile()

    callbacks = [keras.callbacks.ModelCheckpoint(
                    filepath=model_save_path,
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    verbose=1),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    verbose=0,
                    mode="min")]

    history = model.fit(
        [X_train, mask_train],
        y_train,
        batch_size=16,
        epochs=100,
        validation_data=([X_val, mask_val], y_val),
        callbacks=callbacks)

    pickle.dump((history.history), open('app/saves/model/history.pickle', 'wb'))

    return history

def load_model():
    model = model_compile()
    model.load_weights(model_save_path)

    return model