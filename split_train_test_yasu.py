import pandas as pd
from baseline.load_data import replace_value_dataframe
from clean_commit import loading_variable, saving_variable
from parser_commit import info_commit
from padding import dictionary_commit


def get_features(data):
    # return the features of yasu data
    return data[:, 5:32]


def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2]


def get_label(data):
    data = data[:, 3:4]
    data = [1 if d > 0 else 0 for d in data]
    return data


def load_df_yasu_data(path_data):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    return (ids, labels, features)


def load_yasu_data(project, duration, period):
    if project == 'openstack' and period == 'long':
        train_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.7.csv'
        test_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.local.8.csv'
        train, test = load_df_yasu_data(train_path_data), load_df_yasu_data(test_path_data)
        return train, test

def loading_msg_code(data, path_file):
    ids, labels, features = data
    messages, codes = info_commit(ids=ids, path_file=path_file)
    return (ids, labels, messages, codes)


def loading_dictionary(data):
    ids, labels, messages, codes = data
    dict_msg, dict_code = dictionary_commit(data=messages, type_data='msg'), dictionary_commit(data=codes,
                                                                                               type_data='code')
    return dict_msg, dict_code


if __name__ == '__main__':
    # data description
    ################################################################################
    # STRATA_PER_YEAR.2 -> six - month periods
    # STRATA_PER_YEAR.4 -> three - month periods

    # local -> Short - period
    # all -> Long - periods
    ################################################################################
    project, duration, period = 'openstack', 'three-month', 'long'
    load_yasu_data(project=project, duration=duration, period=period)
    # load training/testing data
    train, test = load_yasu_data(project=project, duration=duration, period=period)
    path_file = './output/' + project
    train, test = loading_msg_code(data=train, path_file=path_file), loading_msg_code(data=test, path_file=path_file)
    dict_msg, dict_code = loading_dictionary(data=train)
    saving_variable(project + '_train', train)
    saving_variable(project + '_test', test)
    saving_variable(project + '_dict_msg', dict_msg)
    saving_variable(project + '_dict_code', dict_code)
