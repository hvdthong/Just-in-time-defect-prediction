from clean_commit import loading_variable, saving_variable
from padding import dictionary_commit
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, KFold


def convert_label(data):
    return np.array([0 if d == 'False' else 1 for d in data])


def info_label(data):
    pos = [d for d in data if d == 1]
    neg = [d for d in data if d == 0]
    print('Positive: %i -- Negative: %i' % (len(pos), len(neg)))


def get_index(data, index):
    return [data[i] for i in index]


def folding_data(pad_msg, pad_code, labels, ids, n_folds):
    sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=0)  # random_state = 0 --default setting
    for train_index, test_index in sss.split(pad_msg, labels):
        pad_msg_train, pad_msg_test = get_index(data=pad_msg, index=train_index), get_index(data=pad_msg,
                                                                                            index=test_index)
        pad_code_train, pad_code_test = get_index(data=pad_code, index=train_index), get_index(data=pad_code,
                                                                                               index=test_index)
        labels_train, labels_test = labels[train_index], labels[test_index]
        ids_train, ids_test = get_index(data=ids, index=train_index), get_index(data=ids, index=test_index)
        train = (ids_train, labels_train, pad_msg_train, pad_code_train)
        test = (ids_test, labels_test, pad_msg_test, pad_code_test)
        dict_msg, dict_code = dictionary_commit(data=pad_msg_train, type_data='msg'), dictionary_commit(
            data=pad_code_train, type_data='code')
        return train, test, dict_msg, dict_code


def folding_data_authordate(pad_msg, pad_code, labels, ids, n_folds):
    kf = KFold(n_splits=n_folds, random_state=0)
    indexes = list(kf.split(pad_msg))
    train_index, test_index = indexes[len(indexes) - 1]

    pad_msg_train, pad_msg_test = get_index(data=pad_msg, index=train_index), get_index(data=pad_msg,
                                                                                        index=test_index)
    pad_code_train, pad_code_test = get_index(data=pad_code, index=train_index), get_index(data=pad_code,
                                                                                           index=test_index)
    labels_train, labels_test = labels[train_index], labels[test_index]
    info_label(data=labels_train)
    info_label(data=labels_test)
    ids_train, ids_test = get_index(data=ids, index=train_index), get_index(data=ids, index=test_index)
    train = (ids_train, labels_train, pad_msg_train, pad_code_train)
    test = (ids_test, labels_test, pad_msg_test, pad_code_test)
    dict_msg, dict_code = dictionary_commit(data=pad_msg_train, type_data='msg'), dictionary_commit(
        data=pad_code_train, type_data='code')
    return train, test, dict_msg, dict_code


if __name__ == '__main__':
    # project = 'openstack'
    project = 'qt'
    messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    ids, labels = loading_variable(project + '_ids'), convert_label(loading_variable(project + '_labels'))
    info_label(data=labels)
    print('Number of instances in commit message %i and commit code %i ' % (len(messages), len(codes)))
    print('Labels: %i' % (len(labels)))
    train, test, dict_msg, dict_code = folding_data(pad_msg=messages, pad_code=codes, labels=labels, ids=ids, n_folds=5)
    saving_variable(project + '_train', train)
    saving_variable(project + '_test', test)
    saving_variable(project + '_dict_msg', dict_msg)
    saving_variable(project + '_dict_code', dict_code)

    # dict_msg, dict_code = dictionary_commit(data=messages, type_data='msg'), dictionary_commit(data=codes,
    #                                                                                            type_data='code')
    # print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))
    # pad_msg, pad_code = padding_message(data=messages, max_length=256), padding_commit_code(data=codes, max_line=10,
    #                                                                                         max_length=512)
    # pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    # pad_code = mapping_dict_code(pad_code=pad_code, dict_code=dict_code)
    # print('Shape of commit messages: ', pad_msg.shape)
    # print('Shape of commit code: ', pad_code.shape)
    # data = (pad_msg, pad_code, labels, ids)

    # project = 'openstack'
    # messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    # ids, labels = loading_variable(project + '_ids'), loading_variable(project + '_labels')
    # info_label(data=labels)
    # print('Number of instances in commit message %i and commit code %i ' % (len(messages), len(codes)))
    # print('Labels: %i' % (len(labels)))
    # train, test, dict_msg, dict_code = folding_data_authordate(pad_msg=messages, pad_code=codes, labels=labels, ids=ids,
    #                                                            n_folds=9)
    # saving_variable(project + '_train', train)
    # saving_variable(project + '_test', test)
    # saving_variable(project + '_dict_msg', dict_msg)
    # saving_variable(project + '_dict_code', dict_code)

    # project = 'openstack'
    # messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    # ids, labels = loading_variable(project + '_ids'), loading_variable(project + '_labels')
    # info_label(data=labels)
    # print('Number of instances in commit message %i and commit code %i ' % (len(messages), len(codes)))
    # print('Labels: %i' % (len(labels)))
    # train, test, dict_msg, dict_code = folding_data(pad_msg=messages, pad_code=codes, labels=labels, ids=ids, n_folds=5)
    # train, test, dict_msg, dict_code = folding_data_authordate(pad_msg=messages, pad_code=codes, labels=labels, ids=ids,
    #                                                            n_folds=9)
    # saving_variable(project + '_train', train)
    # saving_variable(project + '_test', test)
    # saving_variable(project + '_dict_msg', dict_msg)
    # saving_variable(project + '_dict_code', dict_code)

    # project = 'openstack'
    # messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    # ids, labels = loading_variable(project + '_ids'), loading_variable(project + '_labels')
    # info_label(data=labels)
    # print('Number of instances in commit message %i and commit code %i ' % (len(messages), len(codes)))
    # print('Labels: %i' % (len(labels)))
    # train, test, dict_msg, dict_code = folding_data(pad_msg=messages, pad_code=codes, labels=labels, ids=ids, n_folds=5)
    # saving_variable(project + '_train', train)
    # saving_variable(project + '_test', test)
    # saving_variable(project + '_dict_msg', dict_msg)
    # saving_variable(project + '_dict_code', dict_code)
