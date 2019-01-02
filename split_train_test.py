from clean_commit import loading_variable, saving_variable
from padding import dictionary_commit, padding_message, padding_commit_code
from padding import mapping_dict_msg, mapping_dict_code
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def convert_label(data):
    return np.array([0 if d == 'False' else 1 for d in data])


def info_label(data):
    pos = [d for d in data if d == 1]
    neg = [d for d in data if d == 0]
    print('Positive: %i -- Negative: %i' % (len(pos), len(neg)))


def get_index(data, index):
    return [data[i] for i in index]


def folding_data(pad_msg, pad_code, labels, ids, n_folds):
    sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=0)  # random_state = 0 -- default setting
    for train_index, test_index in sss.split(pad_msg, labels):
        pad_msg_train, pad_msg_test = pad_msg[train_index], pad_msg[test_index]
        pad_code_train, pad_code_test = pad_code[train_index], pad_code[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        ids_train, ids_test = get_index(data=ids, index=train_index), get_index(data=ids, index=test_index)
        train = (ids_train, labels_train, pad_msg_train, pad_code_train)
        test = (ids_test, labels_test, pad_msg_train, pad_code_train)
        return train, test


if __name__ == '__main__':
    project = 'openstack'
    messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    ids, labels = loading_variable(project + '_ids'), convert_label(loading_variable(project + '_labels'))
    info_label(data=labels)
    print('Number of instances in commit message %i and commit code %i ' % (len(messages), len(codes)))
    print('Labels: %i' % (len(labels)))
    dict_msg, dict_code = dictionary_commit(data=messages, type_data='msg'), dictionary_commit(data=codes,
                                                                                               type_data='code')
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))
    pad_msg, pad_code = padding_message(data=messages, max_length=256), padding_commit_code(data=codes, max_line=10,
                                                                                            max_length=512)
    pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    pad_code = mapping_dict_code(pad_code=pad_code, dict_code=dict_code)
    print('Shape of commit messages: ', pad_msg.shape)
    print('Shape of commit code: ', pad_code.shape)
    data = (pad_msg, pad_code, labels, ids)
    train, test = folding_data(pad_msg=pad_msg, pad_code=pad_code, labels=labels, ids=ids, n_folds=5)
    saving_variable(project + '_train', train)
    saving_variable(project + '_test', test)
