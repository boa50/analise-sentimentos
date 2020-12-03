from sklearn.model_selection import train_test_split

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
        X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(X, y, attention_masks, test_size=second_split)

        return X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test