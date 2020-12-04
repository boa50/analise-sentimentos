from sklearn.naive_bayes import MultinomialNB

import utils
import plots
import classic_ml
import bert

if __name__ == '__main__':
    # utils.config_gpu()    
    
    # dataset_path = 'app/dataset/dataset.csv'
    dataset_path = 'app/dataset/dataset-simple.csv'

    # X, _ = utils.prepare_xy(dataset_path)
    # plots.wordclouds(X)

    # X_train, X_val, X_test, y_train, y_val, y_test = classic_ml.prepare_data(dataset_path, vectorization='bow')

    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    # plots.show_metrics(y_test, y_pred)


    ### BERT
    # X, y = utils.prepare_xy(dataset_path)
    # history = bert.train(X, y)

    # inputs, attention_masks, y = bert.load_embeddings()
    # _, _, X_test, _, _, y_test, _, _, mask_test = utils.split_data(inputs, y, attention_masks)
    # model = bert.load_model()

    # preds = model.predict([X_test, mask_test], batch_size=16)
    # y_pred = preds.logits.argmax(axis=1)

    # plots.show_metrics(y_test, y_pred)