import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp


def get_sample_weights(y):
    """
    calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.
    params:
        y: class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = compute_class_weight('balanced', np.unique(y), y)

    print("real class weights are {}".format(class_weights), np.unique(y))
    print("value_counts", np.unique(y, return_counts=True))
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i]

    return sample_weights


def prepare_data(data, start_col, end_col, feature_idx):
    list_features = list(data.loc[:, start_col:end_col].columns)
    print('Total number of features', len(list_features))
    x_train, x_test, y_train, y_test = \
        train_test_split(data.loc[:, start_col:end_col].values,
                         data['labels'].values,
                         train_size=0.8,
                         test_size=0.2,
                         random_state=2,
                         shuffle=True,
                         stratify=data['labels'].values)

    if 0.7*x_train.shape[0] < 2500:
        train_split = 0.8
    else:
        train_split = 0.7
    # train_split = 0.7
    print('train_split =', train_split)
    x_train = x_train[:-1]
    y_train = y_train[:-1]
    x_train, x_cv, y_train, y_cv = \
        train_test_split(x_train,
                         y_train,
                         train_size=train_split,
                         test_size=1-train_split,
                         random_state=2,
                         shuffle=True,
                         stratify=y_train)
    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = mm_scaler.fit_transform(x_train)
    x_cv = mm_scaler.transform(x_cv)
    x_test = mm_scaler.transform(x_test)

    x_train = x_train[:, feature_idx]
    x_cv = x_cv[:, feature_idx]
    x_test = x_test[:, feature_idx]
    x_train.copy()
    print("Shape of x, y train/cv/test {} {} {} {} {} {}".
          format(x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape))
    _labels, _counts = np.unique(y_train, return_counts=True)
    print("percentage of\n\tclass 0 = {}\n\tclass 1 = {}\n\tclass 2 = {}".format(
        _counts[0]/len(y_train) * 100,
        _counts[1]/len(y_train) * 100,
        _counts[2]/len(y_train) * 100))

    sample_weights = get_sample_weights(y_train)
    one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
    y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
    y_cv = one_hot_enc.transform(y_cv.reshape(-1, 1))
    y_test = one_hot_enc.transform(y_test.reshape(-1, 1))
    dim = int(np.sqrt(x_train.shape[1]))
    x_data = [np.stack((reshape_as_image(x, dim, dim),) * 3, axis=-1) for x in [x_train, x_cv, x_test]]
    x_train = x_data[0]
    x_cv = x_data[1]
    x_test = x_data[2]
    print("final shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_test, y_test, x_cv, y_cv, x_train, y_train, sample_weights


def train(model, x_train, y_train, params, x_cv, y_cv, mcp, rlp, es, sample_weights):
    model.fit(x_train,
              y_train,
              epochs=params['epochs'],
              verbose=1,
              batch_size=64,
              shuffle=True,
              # validation_split=0.3,
              validation_data=(x_cv, y_cv),
              callbacks=[mcp, rlp],
              sample_weight=sample_weights)
