from clean_commit import collect_labels_ver2
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVR
from evaluation_dir import evaluation_metrics
from split_train_test import convert_label
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values


def load_features(ids, path_data):
    df = pd.read_csv(path_data)
    new_df = list()
    for id in ids:
        new_df.append(df.loc[df['commit_id'] == id])
    df = pd.concat(new_df)
    df = replace_value_dataframe(df=df)
    ids, features = df[:, :1], df[:, 4:]
    return ids, features


def loading_variable_path(pname):
    f = open(pname, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def split_folding_data(X, y, n_folds):
    sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=0)  # random_state = 0 -- default setting
    for train_index, test_index in sss.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        return (X_train, y_train), (X_test, y_test)


def baseline_algorithm(train, test, algorihm):
    X_train, y_train = train
    X_test, y_test = test
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    if algorihm == 'svr_rbf':
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        y_pred = model.fit(X_train, y_train).predict(X_test)
    elif algorihm == 'svr_poly':
        model = SVR(kernel='poly', C=1e3, degree=2)
        y_pred = model.fit(X_train, y_train).predict(X_test)
    elif algorihm == 'lr':
        model = LogisticRegression()
        y_pred = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    elif algorihm == 'svm':
        model = svm.SVC(probability=True).fit(X_train, y_train)
        y_pred = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    else:
        print('You need to give the correct algorithm name')

    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))


if __name__ == '__main__':
    project = 'openstack'
    path_data = '../output/' + project
    path_label = '../labels/' + project + '_ids_label.txt'
    ids, labels = loading_variable_path(pname='../variables/' + project + '_ids.pkl'), convert_label(
        loading_variable_path(pname='../variables/' + project + '_labels.pkl'))
    ids, features = load_features(ids=ids, path_data='../labels/' + project + '.csv')
    training, testing = split_folding_data(X=features, y=labels, n_folds=5)
    # baseline_algorithm(train=training, test=testing, algorihm='svr_rbf')
    baseline_algorithm(train=training, test=testing, algorihm='svr_poly')
    # baseline_algorithm(train=training, test=testing, algorihm='lr')
    # baseline_algorithm(train=training, test=testing, algorihm='svm')
