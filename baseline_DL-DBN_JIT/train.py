from DBN import DBN
import torch
from load_data_yasu import load_yasu_data
from sklearn import preprocessing
from LR import LR
import numpy as np
import math
import random
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    return acc, prc, rc, f1, auc_


def eval(data, model):
    with torch.no_grad():
        model.eval()  # since we use drop out
        all_predict, all_label = list(), list()
        for batch in data:
            x, y = batch
            if torch.cuda.is_available():
                x = torch.tensor(x).cuda()
            else:
                x = torch.tensor(x).float()

            if torch.cuda.is_available():
                predict = model.forward(x).cpu().detach().numpy().tolist()
            else:
                predict = model.forward(x).detach().numpy().tolist()
            all_predict += predict
            all_label += y.tolist()
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_pred=all_predict, y_true=all_label)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        return acc, prc, rc, f1, auc_


def feature_scaling(data):
    ids, labels, features = data
    features = preprocessing.scale(features)
    return (ids, labels, features)


def mini_batches_update(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X, mini_batch_Y = shuffled_X[indexes], shuffled_Y[indexes]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def train_DBN(train_data, test_data, hidden_units, num_epochs_DBN=50, num_epochs_LR=100):
    train_ids, train_labels, train_features = train_data
    test_ids, test_labels, test_features = test_data

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
    # loading parameters for the project
    ################################################################################3
    project, duration, period = 'openstack', 'three-month', 'long'
    # project, duration, period = 'openstack', 'three-month', 'short'
    # project, duration, period = 'qt', 'three-month', 'long'
    # project, duration, period = 'qt', 'three-month', 'short'
    ################################################################################3

    # load training/testing data
    ################################################################################3
    path_file = '../output/' + project
    train, test = load_yasu_data(project=project, duration=duration, period=period, path_file=path_file)
    train, test = feature_scaling(data=train), feature_scaling(data=test)
    hidden_units, num_epochs_DBN, num_epochs_LR = 12, 1, 200
    train_DBN(train_data=train, test_data=test, hidden_units=hidden_units, num_epochs_DBN=num_epochs_DBN,
              num_epochs_LR=num_epochs_LR)
