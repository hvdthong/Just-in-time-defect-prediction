from clean_commit import collect_labels_ver2, info_commit, clean_code, clean_message
import pandas as pd
from clean_commit import saving_variable
from sklearn.model_selection import StratifiedShuffleSplit
from padding import dictionary_commit
import numpy as np


def convert_label(data):
    return np.array([0 if d == 'False' else 1 for d in data])


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values


def load_features(ids, path_data):
    df = pd.read_csv(path_data)
    new_df = list()
    for id in ids:
        new_df.append(df.loc[df['commit_id'] == id])
    df = pd.concat(new_df)
    df = replace_value_dataframe(df=df)
    ids, features = df[:, :1], df[:, 5:]
    return ids, features


def get_index(data, index):
    return [data[i] for i in index]


def folding_data(data, n_folds):
    ids, labels, features, messages, codes = data
    sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=0)  # random_state = 0 --default setting
    for train_index, test_index in sss.split(messages, labels):
        ids_train, ids_test = get_index(data=ids, index=train_index), get_index(data=ids, index=test_index)
        labels_train, labels_test = labels[train_index], labels[test_index]
        ftr_train, ftr_test = features[train_index], features[test_index]
        msg_train, msg_test = get_index(data=messages, index=train_index), get_index(data=messages, index=test_index)
        code_train, code_test = get_index(data=codes, index=train_index), get_index(data=codes, index=test_index)

        train = (ids_train, labels_train, ftr_train, msg_train, code_train)
        test = (ids_test, labels_test, ftr_test, msg_test, code_test)
        dict_msg, dict_code = dictionary_commit(data=msg_train, type_data='msg'), dictionary_commit(data=code_train,
                                                                                                    type_data='code')
        return train, test, dict_msg, dict_code


if __name__ == '__main__':
    # project = 'openstack'
    project = 'qt'
    path_data = './output/' + project
    path_label = './labels/' + project + '_ids_label.txt'

    ids, labels = collect_labels_ver2(path_label=path_label)
    # ids, labels = ids[:500], labels[:500]
    messages, codes = info_commit(ids=ids, path_file=path_data)
    ids, features = load_features(ids=ids, path_data='./labels/' + project + '.csv')
    messages, codes = clean_message(data=messages), clean_code(data=codes)
    data = (ids, convert_label(labels), features, messages, codes)
    train, test, dict_msg, dict_code = folding_data(data=data, n_folds=5)

    saving_variable(project + '_train', train)
    saving_variable(project + '_test', test)
    saving_variable(project + '_dict_msg', dict_msg)
    saving_variable(project + '_dict_code', dict_code)
