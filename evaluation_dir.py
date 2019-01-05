import torch
from parameters import read_args
from ultis import mini_batches
from model_defect import DefectNet
from clean_commit import loading_variable
from padding import padding_message, padding_commit_code, mapping_dict_msg, mapping_dict_code
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluation_metrics(y_true, y_pred):
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    return acc, prc, rc, f1


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


def construct_model(data, params):
    training, testing, dictionary = data
    ids_train, labels_train, msg_train, code_train = training
    ids_test, labels_test, msg_test, code_test = testing
    dict_msg, dict_code = dictionary

    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')

    pad_msg_test = padding_data(data=msg_test, dictionary=dict_msg, params=params, type='msg')
    pad_code_test = padding_data(data=code_test, dictionary=dict_code, params=params, type='code')

    batches_train = mini_batches(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels_train.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels_train.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = DefectNet(params)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, batches_test, batches_train


def eval_dir(dir, data, model, params):
    model.load_state_dict(torch.load(dir))
    model.eval()  # since we use drop out
    all_predict, all_label = list(), list()
    for batch in data:
        pad_msg, pad_code, labels = batch
        if torch.cuda.is_available():
            pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
        else:
            pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
        if torch.cuda.is_available():
            predict = model.forward(pad_msg, pad_code).cpu().detach().numpy().tolist()
        else:
            predict = model.forward(pad_msg, pad_code).detach().numpy().tolist()
        all_predict += predict
        all_label += labels.tolist()
    all_predict = [1 if p >= 0.5 else 0 for p in all_predict]
    acc, prc, rc, f1 = evaluation_metrics(y_pred=all_predict, y_true=all_label)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f' % (acc, prc, rc, f1))
    return acc, prc, rc


if __name__ == '__main__':
    project = 'openstack'
    training, testing, dictionary = loading_data(project=project)

    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    input_option.filter_sizes = [int(k) for k in input_option.filter_sizes.split(',')]

    model, data_test, data_train = construct_model(data=(training, testing, dictionary), params=input_option)
    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        dir = './snapshot/' + input_option.datetime + '/epoch_' + str(epoch) + '.pt'
        print('--Epoch: %i' % epoch)
        eval_dir(dir=dir, data=data_test, model=model, params=input_option)
