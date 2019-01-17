from clean_commit import clean_message, clean_code, saving_variable
from parser_commit import info_commit
from sklearn import preprocessing
from padding import dictionary_commit
import pandas as pd
from split_train_test_yasu import replace_value_dataframe, get_ids, get_label
from ultis import load_file
import numpy as np


def load_msg_code_feature(data, path_file):
    ids, labels, features = data
    messages, codes = info_commit(ids=ids, path_file=path_file)
    messages, codes = clean_message(data=messages), clean_code(data=codes)
    return (ids, labels, preprocessing.scale(features), messages, codes)


def loading_dictionary(data):
    ids, labels, features, messages, codes = data
    dict_msg, dict_code = dictionary_commit(data=messages, type_data='msg'), dictionary_commit(data=codes,
                                                                                               type_data='code')
    return dict_msg, dict_code


def get_features(data):
    # return the features of yasu data
    return np.concatenate((data[:, 5:33], data[:, 36:37]), axis=1)


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
        train_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.7.csv'
        test_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.8.csv'
        train, test = load_df_yasu_data(train_path_data, path_file), load_df_yasu_data(test_path_data, path_file)
        return train, test
    elif project == 'openstack' and period == 'short':
        train_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.7.csv'
        test_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.8.csv'
        train, test = load_df_yasu_data(train_path_data, path_file), load_df_yasu_data(test_path_data, path_file)
        return train, test
    elif project == 'qt' and period == 'long':
        train_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.9.csv'
        test_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.10.csv'
        train, test = load_df_yasu_data(train_path_data, path_file), load_df_yasu_data(test_path_data, path_file)
        return train, test
    elif project == 'qt' and period == 'short':
        train_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.9.csv'
        test_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.10.csv'
        train, test = load_df_yasu_data(train_path_data, path_file), load_df_yasu_data(test_path_data, path_file)
        return train, test
    else:
        print('Wrong command')
        exit()


if __name__ == '__main__':
    # similar to split_train_test_yasu.py file, but we will add features extracted from yasu data + commit code + commit message
    # data description
    ################################################################################
    # STRATA_PER_YEAR.2 -> six - month periods
    # STRATA_PER_YEAR.4 -> three - month periods

    # local -> Short - period
    # all -> Long - periods
    ################################################################################
    # setting project, long, and short period
    # project, duration, period = 'openstack', 'three-month', 'long'
    # project, duration, period = 'openstack', 'three-month', 'short'
    # project, duration, period = 'qt', 'three-month', 'long'
    project, duration, period = 'qt', 'three-month', 'short'

    ################################################################################
    # load training/testing data
    path_file = './output/' + project
    train, test = load_yasu_data(project=project, duration=duration, period=period, path_file=path_file)
    train, test = load_msg_code_feature(data=train, path_file=path_file), load_msg_code_feature(data=test,
                                                                                                path_file=path_file)
    dict_msg, dict_code = loading_dictionary(data=train)
    saving_variable(project + '_train', train)
    saving_variable(project + '_test', test)
    saving_variable(project + '_dict_msg', dict_msg)
    saving_variable(project + '_dict_code', dict_code)
