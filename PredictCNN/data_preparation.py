from tqdm import tqdm
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def create_labels(data, window_size=11):
    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 0, HOLD => 2
    params :
      df => Data frame with data
      col_name => name of column which should be used to determine strategy
    returns : numpy array with integer codes for labels with
            size = total-(window_size)+1
    """
    buy = 1
    sell = 0
    hold = 2
    row_counter = 0
    total_rows = len(data)
    labels = np.zeros(total_rows)
    labels[:] = np.nan
    print("Calculating labels")
    #pbar = tqdm(total_rows)
    bought_prices = []
    total_profit = 0

    for row_counter in tqdm(range(total_rows - 1)):
        chg = (data.iloc[row_counter + 1]['Close'] - data.iloc[row_counter]['Close']) / data.iloc[row_counter]['Close']
        if chg > 0.02:
            labels[row_counter] = buy
        elif chg < -0.02:
            labels[row_counter] = sell
        else:
            labels[row_counter] = hold

    return labels


def select_features(data, num_features=225, top_k=350, start_col='Open', end_col='eom_26'):
    """
    num_features should be a perfect square
    """
    data_batch = data.copy()
    list_features = list(data_batch.loc[:, start_col:end_col].columns)
    mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
    x_train = mm_scaler.fit_transform(data_batch.loc[:, start_col:end_col].values)
    y_train = data_batch['labels'].values

    select_k_best = SelectKBest(f_classif, k=top_k)
    select_k_best.fit(x_train, y_train)
    selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)

    select_k_best = SelectKBest(mutual_info_classif, k=top_k)
    select_k_best.fit(x_train, y_train)
    selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

    common = list(set(selected_features_anova).intersection(selected_features_mic))
    if len(common) < num_features:
        raise Exception(
            'number of common features found {} < {} required features. Increase "topK"'.format(len(common),
                                                                                                num_features))

    printable_features = "\n".join(["\t".join(common[i:i + 10]) for i in range(0, len(common), 10)])

    print("common selected featues:\n{}".format(printable_features))

    feat_idx = []
    for c in common:
        feat_idx.append(list_features.index(c))
    feat_idx = sorted(feat_idx[0:num_features])
    return feat_idx, start_col, end_col
