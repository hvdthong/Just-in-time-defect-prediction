import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVR
from evaluation_dir import evaluation_metrics
from split_train_test_yasu import load_yasu_data
from sklearn import linear_model
from ultis import load_file
import pandas as pd
import numpy as np


def loading_variable(pname):
    f = open('../variables/' + pname + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values


def loading_data(project):
    train, test = loading_variable(project + '_train'), loading_variable(project + '_test')
    dictionary = (loading_variable(project + '_dict_msg'), loading_variable(project + '_dict_code'))
    return train, test, dictionary


def get_features(data):
    # return the features of yasu data
    return data[:, 11:32]


def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2].flatten().tolist()


def get_label(data):
    data = data[:, 3:4].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data


def load_df_yasu_data(path_data, path_file):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes, new_ids, new_labels, new_features = list(), list(), list(), list()
    cnt_noexits = 0
    for i in range(0, len(ids)):
        try:
            data = load_file(path_file=path_file + '/' + ids[i] + '.diff')
            indexes.append(i)
        except FileNotFoundError:
            print('File commit id no exits', ids[i], cnt_noexits)
            cnt_noexits += 1
    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)


def load_yasu_data(project, duration, period, path_file):
    if project == 'openstack' and period == 'long':
        train_path_data = '../yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.7.csv'
        test_path_data = '../yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.8.csv'
        train, test = load_df_yasu_data(train_path_data, path_file), load_df_yasu_data(test_path_data, path_file)
        return train, test


def baseline_algorithm(train, test, algorihm):
    _, y_train, X_train = train
    _, y_test, X_test = test
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
    elif algorihm == 'ridge':
        model = linear_model.Ridge()
        y_pred = model.fit(X_train, y_train).predict(X_test)
    else:
        print('You need to give the correct algorithm name')

    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))


if __name__ == '__main__':
    project, duration, period = 'openstack', 'three-month', 'long'
    # load training/testing data
    path_file = '../output/' + project
    train, test = load_yasu_data(project=project, duration=duration, period=period, path_file=path_file)
    baseline_algorithm(train=train, test=test, algorihm='lr')
    # baseline_algorithm(train=train, test=test, algorihm='svr_rbf')
    # baseline_algorithm(train=train, test=test, algorihm='svm')
    # baseline_algorithm(train=train, test=test, algorihm='ridge')
