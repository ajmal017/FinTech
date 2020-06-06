from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score


def check_baseline(pred, y_test):
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", str((holds / len(y_test) * 100)))


def evaluate(x_test, y_test):
    model = load_model("test")
    test_res = model.evaluate(x_test, y_test, verbose=0)
    print("keras evaluate=", test_res)

    pred = model.predict(x_test)
    pred_classes = np.argmax(pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    check_baseline(pred_classes, y_test_classes)
    conf_mat = confusion_matrix(y_test_classes, pred_classes)
    print(conf_mat)

    # ax = sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
    # ax.xaxis.set_ticks_position('top')
    f1_weighted = f1_score(y_test_classes,
                           pred_classes,
                           labels=None,
                           average='weighted',
                           sample_weight=None)
    print("F1 score (weighted)", f1_weighted)
    print("F1 score (macro)", f1_score(y_test_classes,
                                       pred_classes,
                                       labels=None,
                                       average='macro',
                                       sample_weight=None))
    print("F1 score (micro)", f1_score(y_test_classes,
                                       pred_classes,
                                       labels=None,
                                       average='micro',
                                       sample_weight=None))
    print("cohen's Kappa", cohen_kappa_score(y_test_classes, pred_classes))

    prec = []
    for i, row in enumerate(conf_mat):
        prec.append(np.round(row[i]/np.sum(row), 2))
        print("precision of class {} = {}".format(i, prec[i]))
    print("precision avg", sum(prec)/len(prec))
