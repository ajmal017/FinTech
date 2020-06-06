
import tensorflow as tf
from keras import backend as K
from keras.utils import get_custom_objects
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Flatten
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback


def condition(i, conf_mat):
    return tf.less(i, 3)


def f1_weighted(y_true, y_pred):
    y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)  # can use conf_mat[0, :], tf.slice()
    # precision = TP/TP+FP, recall = TP/TP+FN
    rows, cols = conf_mat.get_shape()
    size = y_true_class.get_shape()[0]
    precision = tf.constant([0, 0, 0])  # change this to use rows/cols as size
    recall = tf.constant([0, 0, 0])
    class_counts = tf.constant([0, 0, 0])

    def get_precision(idx):
        print("prec check", conf_mat, conf_mat[idx, idx], tf.reduce_sum(conf_mat[:, idx]))
        precision[idx].assign(conf_mat[idx, idx] / tf.reduce_sum(conf_mat[:, idx]))
        recall[idx].assign(conf_mat[idx, idx] / tf.reduce_sum(conf_mat[idx, :]))
        tf.add(idx, 1)
        return idx, conf_mat, precision, recall

    def tf_count(i):
        elements_equal_to_value = tf.equal(y_true_class, i)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        class_counts[i].assign(count)
        tf.add(i, 1)
        return count

    i = tf.constant(3)
    i, conf_mat = tf.while_loop(condition, get_precision, [i, conf_mat])

    i = tf.constant(3)
    c = lambda i: tf.less(i, 3)
    b = tf_count(i)
    tf.while_loop(c, b, [i])

    weights = tf.math.divide(class_counts, size)
    numerators = tf.math.multiply(tf.math.multiply(precision, recall), tf.constant(2))
    denominators = tf.math.add(precision, recall)
    f1s = tf.math.divide(numerators, denominators)
    weighted_f1 = tf.reduce_sum(tf.math.multiply(f1s, weights))
    return weighted_f1


def calculate_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def calculate_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall
    """

    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
    # tf.Print(conf_mat, [conf_mat], "confusion_matrix")
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def create_model_cnn():
    get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})

    params = {'batch_size': 60,
              'conv2d_layers': {'conv2d_do_1': 0.0,
                                'conv2d_filters_1': 30,
                                'conv2d_kernel_size_1': 2,
                                'conv2d_mp_1': 2,
                                'conv2d_strides_1': 1,
                                'kernel_regularizer_1': 0.0,
                                'conv2d_do_2': 0.01,
                                'conv2d_filters_2': 10,
                                'conv2d_kernel_size_2': 2,
                                'conv2d_mp_2': 2,
                                'conv2d_strides_2': 2,
                                'kernel_regularizer_2': 0.0,
                                'layers': 'two'},
              'dense_layers': {'dense_do_1': 0.07,
                               'dense_nodes_1': 100,
                               'kernel_regularizer_1': 0.0,
                               'layers': 'one'},
              'epochs': 3000,
              'lr': 0.001,
              'optimizer': 'adam',
              'input_dim_1': 15,
              'input_dim_2': 15,
              'input_dim_3': 3}

    model = Sequential()

    print("Training with params {}".format(params))
    # (batch_size, timesteps, data_dim)
    # x_train, y_train = get_data_cnn(df, df.head(1).iloc[0]["timestamp"])[0:2]
    conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                           params["conv2d_layers"]["conv2d_kernel_size_1"],
                           strides=params["conv2d_layers"]["conv2d_strides_1"],
                           kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                           padding='valid', activation="relu", use_bias=True,
                           kernel_initializer='glorot_uniform',
                           input_shape=(params['input_dim_1'],
                                        params['input_dim_2'], params['input_dim_3']))
    model.add(conv2d_layer1)

    if params["conv2d_layers"]['conv2d_mp_1'] == 1:
        model.add(MaxPool2D(pool_size=2))

    model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))

    if params["conv2d_layers"]['layers'] == 'two':
        conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                               params["conv2d_layers"]["conv2d_kernel_size_2"],
                               strides=params["conv2d_layers"]["conv2d_strides_2"],
                               kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                               padding='valid', activation="relu", use_bias=True,
                               kernel_initializer='glorot_uniform')
        model.add(conv2d_layer2)
        if params["conv2d_layers"]['conv2d_mp_2'] == 1:
            model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(params['conv2d_layers']['conv2d_do_2']))

    model.add(Flatten())

    model.add(Dense(params['dense_layers']["dense_nodes_1"], activation='relu'))

    model.add(Dropout(params['dense_layers']['dense_do_1']))

    if params['dense_layers']["layers"] == 'two':
        model.add(Dense(params['dense_layers']["dense_nodes_2"], activation='relu',
                        kernel_regularizer=params['dense_layers']["kernel_regularizer_1"]))
        model.add(Dropout(params['dense_layers']['dense_do_2']))

    model.add(Dense(3, activation='softmax'))

    if params["optimizer"] == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=params["lr"])
    elif params["optimizer"] == 'sgd':
        optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
    elif params["optimizer"] == 'adam':
        optimizer = optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)


    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])
    # from keras.utils.vis_utils import plot_model use this too for diagram with plot
    model.summary(print_fn=lambda x: print(x + '\n'))

    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=100,
                       min_delta=0.0001)
    rlp = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.02,
                            patience=10,
                            verbose=1,
                            mode='min',
                            min_delta=0.001,
                            cooldown=1,
                            min_lr=0.0001)
    mcp = ModelCheckpoint("test",
                          monitor='val_loss',
                          verbose=0,
                          save_best_only=False,
                          save_weights_only=False,
                          mode='min')

    return model, params, es, rlp, mcp
