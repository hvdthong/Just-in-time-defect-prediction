from clean_commit import loading_variable
from split_train_test import info_label
from parameters import read_args
from padding import dictionary_commit, padding_message, padding_commit_code, mapping_dict_msg, mapping_dict_code
from ultis import mini_batches
import os, datetime, torch
from model_defect import DefectNet


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


def train_model(train, test, dictionary, params):
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))

    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')

    pad_msg_test = padding_data(data=msg_test, dictionary=dict_msg, params=params, type='msg')
    pad_code_test = padding_data(data=code_test, dictionary=dict_code, params=params, type='code')

    # building batches
    batches_train = mini_batches(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)

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

    model = DefectNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    exit()
    new_train = (labels_train, pad_msg_train, pad_code_train)
    new_test = (labels_test, pad_msg_test, pad_code_test)
    print(labels_train.shape)
    print(pad_msg_train.shape)
    print(pad_code_train.shape)
    print(labels_test.shape)
    print(pad_msg_test.shape)
    print(pad_code_test.shape)


if __name__ == '__main__':
    project = 'openstack'
    train, test, dictionary = loading_data(project=project)
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    train_model(train=train, test=test, dictionary=dictionary, params=input_option)
