import utils
import plots
import classic_ml
import bert

if __name__ == '__main__':
    dataset_path = 'app/dataset/dataset.csv'
    # dataset_path = 'app/dataset/dataset-simple.csv'

    ### Wordcloud
    X, _ = utils.prepare_xy(dataset_path)
    plots.wordclouds(X)


    ### Classic ml
    # X_train, X_val, X_test, y_train, y_val, y_test = classic_ml.prepare_data(dataset_path, vectorization='bow')

    # clf = classic_ml.train('complement_nb', X_train, y_train)
    # clf = classic_ml.train('random_forest', X_train, y_train)

    # clf = classic_ml.load_classifier('complement_nb')
    # clf = classic_ml.load_classifier('random_forest')

    # y_pred = clf.predict(X_test)
    # plots.show_metrics(y_test, y_pred)


    ### BERT
    # utils.config_gpu()
    # X, y = utils.prepare_xy(dataset_path)
    # history = bert.train(X, y)

    # inputs, attention_masks, y = bert.load_embeddings()
    # _, _, X_test, _, _, y_test, _, _, mask_test = utils.split_data(inputs, y, attention_masks)
    # model = bert.load_model()

    # preds = model.predict([X_test, mask_test], batch_size=16)
    # y_pred = preds.logits.argmax(axis=1)

    # plots.show_metrics(y_test, y_pred)

    # plots.plot_confusion_matrix([], [], [[70904, 9419], [8796, 70881]])