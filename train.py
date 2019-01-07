from clean_commit import loading_variable
from split_train_test import info_label
from parameters import read_args
from padding import dictionary_commit, padding_message, padding_commit_code, mapping_dict_msg, mapping_dict_code
from ultis import mini_batches, mini_batches_update, mini_batches_undersampling
import os, datetime
from model_defect import DefectNet
import torch
import torch.nn as nn
from evaluation import eval


def loading_data(project):
    train, test = loading_variable(project + '_train'), loading_variable(project + '_test')
    dictionary = (loading_variable(project + '_dict_msg'), loading_variable(project + '_dict_code'))
    return train, test, dictionary


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


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


def running_train(batches_train, batches_test, model, params):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps = 0
    for epoch in range(1, params.num_epochs + 1):
        for batch in batches_train:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

        print('Epoch: %i ---Training data' % (epoch))
        acc, prc, rc = eval(data=batches_train, model=model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f' % (acc, prc, rc))

        print('Epoch: %i ---Testing data' % (epoch))
        acc, prc, rc = eval(data=batches_test, model=model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f' % (acc, prc, rc))
        save(model, params.save_dir, 'epoch', epoch)


def train_model(train, test, dictionary, params):
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
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

    # building batches
    batches_train = mini_batches(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)

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
    model = DefectNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    running_train(batches_train=batches_train, batches_test=batches_test, model=model, params=params)


def train_model_mini_batches_update(train, test, dictionary, params):
    #####################################################################################################
    # training model using 50% of positive and 50% of negative data in mini batch
    #####################################################################################################
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))

    # print('Training data')
    # info_label(labels_train)
    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')
    # print('Testing data')
    # info_label(labels_test)
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
    model = DefectNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps = 0

    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)
    for epoch in range(1, params.num_epochs + 1):
        # building batches for training model
        batches_train = mini_batches_update(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
        for batch in batches_train:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
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


def train_model_mini_batches_undersampling(train, test, dictionary, params):
    #####################################################################################################
    # training model using under sampling technique to solve the imbalanced problem
    #####################################################################################################
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))

    # print('Training data')
    # info_label(labels_train)
    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')
    # print('Testing data')
    # info_label(labels_test)
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
    model = DefectNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps = 0

    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)
    for epoch in range(1, params.num_epochs + 1):
        # building batches for training model
        batches_train = mini_batches_undersampling(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
        for batch in batches_train:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
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


def custom_loss(y_pred, y_true, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (y_true * torch.log(y_pred)) + weights[0] * ((1 - y_true) * torch.log(1 - y_pred))
    else:
        loss = y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
    return torch.neg(torch.mean(loss))


def train_model_loss(project, train, test, dictionary, params):
    #####################################################################################################
    # training model using penalized classification technique (modify loss function)
    #####################################################################################################
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))

    # print('Training data')
    # info_label(labels_train)
    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')
    # print('Testing data')
    # info_label(labels_test)
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
    model = DefectNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps = 0

    # building batches
    batches_train = mini_batches(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)

    for epoch in range(1, params.num_epochs + 1):
        for batch in batches_train:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
            if project == 'openstack':
                loss = custom_loss(y_pred=predict, y_true=labels, weights=[0.1, 1])
                loss.backward()
                optimizer.step()
            elif project == 'qt':
                print('We need to find the weights for negative and positive labels later')
                exit()
            else:
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


def train_model_loss_undersampling(project, train, test, dictionary, params):
    #####################################################################################################
    # training model using penalized classification technique (modify loss function) and under sampling technique
    #####################################################################################################
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))

    # print('Training data')
    # info_label(labels_train)
    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')
    # print('Testing data')
    # info_label(labels_test)
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
    model = DefectNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps = 0

    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)
    for epoch in range(1, params.num_epochs + 1):
        # building batches for training model
        batches_train = mini_batches_undersampling(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
        for batch in batches_train:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
            if project == 'openstack':
                loss = custom_loss(y_pred=predict, y_true=labels, weights=[0.1, 1])
                loss.backward()
                optimizer.step()
            elif project == 'qt':
                print('We need to find the weights for negative and positive labels later')
                exit()
            else:
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
    project = 'openstack'
    train, test, dictionary = loading_data(project=project)
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    # train_model(train=train, test=test, dictionary=dictionary, params=input_option)
    # train_model_mini_batches_update(train=train, test=test, dictionary=dictionary, params=input_option)
    # train_model_mini_batches_undersampling(train=train, test=test, dictionary=dictionary, params=input_option)
    # train_model_loss(project=project, train=train, test=test, dictionary=dictionary, params=input_option)
    train_model_loss_undersampling(project=project, train=train, test=test, dictionary=dictionary, params=input_option)
