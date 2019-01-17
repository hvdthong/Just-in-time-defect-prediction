from clean_commit import loading_variable
from split_train_test import info_label
from parameters import read_args
from padding import padding_message, padding_commit_code, mapping_dict_msg, mapping_dict_code
import os, datetime
from model_defect_combine_ftr import DefectNetCombine
import torch
import torch.nn as nn
import math
import numpy as np
import random
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


def padding_data(data, dictionary, params, type):
    if type == 'msg':
        pad_msg = padding_message(data=data, max_length=params.msg_length)
        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dictionary)
        return pad_msg
    elif type == 'code':
        pad_code = padding_commit_code(data=data, max_line=params.code_line, max_length=params.code_length)
        pad_code = mapping_dict_code(pad_code=pad_code, dict_code=dictionary)
        return pad_code
    else:
        print('Your type is incorrect -- please correct it')
        exit()


def loading_data(project):
    train, test = loading_variable(project + '_train'), loading_variable(project + '_test')
    dictionary = (loading_variable(project + '_dict_msg'), loading_variable(project + '_dict_code'))
    return train, test, dictionary


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


def mini_batches(X_msg, X_code, X_ftr, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_X_ftr, shuffled_Y = X_msg, X_code, X_ftr, Y

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_X_ftr = shuffled_X_ftr[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_X_ftr, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch_X_ftr = shuffled_X_ftr[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_X_ftr, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_update(X_msg, X_code, X_ftr, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_X_ftr, shuffled_Y = X_msg, X_code, X_ftr, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X_msg, mini_batch_X_code, mini_batch_X_ftr = shuffled_X_msg[indexes], shuffled_X_code[indexes], \
                                                                shuffled_X_ftr[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_X_ftr, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


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
            pad_msg, pad_code, pad_ftr, labels = batch

            if torch.cuda.is_available():
                pad_msg, pad_code, pad_ftr, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(pad_ftr), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, pad_ftr, labels = torch.tensor(pad_msg).long(), torch.tensor(
                    pad_code).long(), torch.tensor(pad_ftr).float(), torch.tensor(labels).float()

            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code, pad_ftr).cpu().detach().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code, pad_ftr).detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_pred=all_predict, y_true=all_label)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        return acc, prc, rc, f1, auc_


def train_model_mini_batches_update(train, test, dictionary, params):
    #####################################################################################################
    # training model using 50% of positive and 50% of negative data in mini batch
    #####################################################################################################
    ids_train, labels_train, features_train, msg_train, code_train = train
    ids_test, labels_test, features_test, msg_test, code_test = test
    # preprocessing features
    features_train, features_test = preprocessing.scale(features_train), preprocessing.scale(features_test)
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))

    print('Training data')
    info_label(labels_train)
    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')
    print('Testing data')
    info_label(labels_test)
    pad_msg_test = padding_data(data=msg_test, dictionary=dict_msg, params=params, type='msg')
    pad_code_test = padding_data(data=code_test, dictionary=dict_code, params=params, type='code')

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels_train.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels_train.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = DefectNetCombine(args=params, num_ftr=features_train.shape[1])
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps = 0

    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, X_ftr=features_test, Y=labels_test)
    for epoch in range(1, params.num_epochs + 1):
        # building batches for training model
        batches_train = mini_batches_update(X_msg=pad_msg_train, X_code=pad_code_train, X_ftr=features_train,
                                            Y=labels_train)
        for batch in batches_train:
            pad_msg, pad_code, pad_ftr, labels = batch

            if torch.cuda.is_available():
                pad_msg, pad_code, pad_ftr, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(pad_ftr), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, pad_ftr, labels = torch.tensor(pad_msg).long(), torch.tensor(
                    pad_code).long(), torch.tensor(pad_ftr).float(), torch.tensor(labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code, pad_ftr)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

        print('Epoch: %i ---Training data' % (epoch))
        acc, prc, rc, f1, auc_ = eval(data=batches_train, model=model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        print('Epoch: %i ---Testing data' % (epoch))
        acc, prc, rc, f1, auc_ = eval(data=batches_test, model=model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        if epoch % 5 == 0:
            save(model, params.save_dir, 'epoch', epoch)


if __name__ == '__main__':
    # project: parameters
    ###########################################################################################
    # project = 'openstack'
    project = 'qt'
    ###########################################################################################
    train, test, dictionary = loading_data(project=project)
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    train_model_mini_batches_update(train=train, test=test, dictionary=dictionary, params=input_option)
