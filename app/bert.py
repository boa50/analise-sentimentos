import numpy as np
import tensorflow as tf
from transformers import *
from transformers import BertTokenizer, TFBertForSequenceClassification

import utils

def train(X, y):
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    inputs = []
    attention_masks = []

    for x in X:
        tokenized = bert_tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True)

        inputs.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])

    inputs = np.asarray(inputs)
    attention_masks = np.array(attention_masks)
    labels = np.array(y)

    train_inp,val_inp, _, train_label,val_label, _, train_mask,val_mask, _ = utils.split_data(inputs, labels, attention_masks)#train_test_split(inputs, labels, attention_masks, test_size=0.2)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)

    bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    history = bert_model.fit(
        [train_inp,train_mask],
        train_label,
        batch_size=16,
        epochs=4,
        validation_data=([val_inp,val_mask],val_label))

    return history