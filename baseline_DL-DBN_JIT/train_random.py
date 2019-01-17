import pickle
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn import preprocessing
import torch
from LR import LR
from train import mini_batches, mini_batches_update, eval
import torch.nn as nn


def convert_label(data):
    return np.array([0 if d == 'False' else 1 for d in data])


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values


def loading_variable_path(pname):
    f = open(pname, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def feature_scaling(data):
    features, labels = data
    features = preprocessing.scale(features)
    return (features, labels)


def load_features(ids, path_data):
    df = pd.read_csv(path_data)
    new_df = list()
    for id in ids:
        new_df.append(df.loc[df['commit_id'] == id])
    df = pd.concat(new_df)
    df = replace_value_dataframe(df=df)
    ids, features = df[:, :1], df[:, 10:]
    return ids, features


def split_folding_data(X, y, n_folds):
    sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=0)  # random_state = 0 -- default setting
    for train_index, test_index in sss.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        return (X_train, y_train), (X_test, y_test)


def train_DBN(train_data, test_data, hidden_units, num_epochs_DBN=50, num_epochs_LR=100):
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    # training DBN model
    #################################################################################################
    # dbn_model = DBN(visible_units=train_features.shape[1],
    #                 hidden_units=[20, hidden_units],
    #                 k=5,
    #                 learning_rate=0.01,
    #                 learning_rate_decay=True,
    #                 xavier_init=True,
    #                 increase_to_cd_k=False,
    #                 use_gpu=False)
    # dbn_model.train_static(train_features, train_labels, num_epochs=num_epochs_DBN, batch_size=32)
    # # Finishing the training DBN model
    # print('---------------------Finishing the training DBN model---------------------')
    # # using DBN model to construct features
    # train_features, _ = dbn_model.forward(train_features)
    # test_features, _ = dbn_model.forward(test_features)
    ##################################################################################################

    # training LR model
    ##################################################################################################
    if len(train_labels.shape) == 1:
        num_classes = 1
    else:
        num_classes = train_labels.shape[1]
    # lr_model = LR(input_size=hidden_units, num_classes=num_classes)
    lr_model = LR(input_size=train_features.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.00001)
    steps = 0
    batches_test = mini_batches(X=test_features, Y=test_labels)
    for epoch in range(1, num_epochs_LR + 1):
        # building batches for training model
        batches_train = mini_batches_update(X=train_features, Y=train_labels)
        for batch in batches_train:
            x_batch, y_batch = batch
            if torch.cuda.is_available():
                x_batch, y_batch = torch.tensor(x_batch).cuda(), torch.cuda.FloatTensor(y_batch)
            else:
                x_batch, y_batch = torch.tensor(x_batch).float(), torch.tensor(y_batch).float()

            optimizer.zero_grad()
            predict = lr_model.forward(x_batch)
            loss = nn.BCELoss()
            loss = loss(predict, y_batch)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 10 == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

        print('Epoch: %i ---Training data' % (epoch))
        acc, prc, rc, f1, auc_ = eval(data=batches_train, model=lr_model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        print('Epoch: %i ---Testing data' % (epoch))
        acc, prc, rc, f1, auc_ = eval(data=batches_test, model=lr_model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))


if __name__ == '__main__':
    project = 'openstack'
    # project = 'qt'
    path_data = '../output/' + project
    path_label = '../labels/' + project + '_ids_label.txt'
    ids, labels = loading_variable_path(pname='../variables_ver1/' + project + '_ids.pkl'), convert_label(
        loading_variable_path(pname='../variables_ver1/' + project + '_labels.pkl'))
    ids, features = load_features(ids=ids, path_data='../labels/' + project + '.csv')
    train, test = split_folding_data(X=features, y=labels, n_folds=5)
    train, test = feature_scaling(data=train), feature_scaling(data=test)
    hidden_units, num_epochs_DBN, num_epochs_LR = 12, 5, 500
    train_DBN(train_data=train, test_data=test, hidden_units=hidden_units, num_epochs_DBN=num_epochs_DBN,
              num_epochs_LR=num_epochs_LR)
