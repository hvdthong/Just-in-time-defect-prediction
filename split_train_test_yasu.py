import pandas as pd
from baseline.load_data import replace_value_dataframe


def get_features(data):
    # return the features of yasu data
    return data[:, 5:32]


def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2]


def get_label(data):
    return data


def load_df_yasu_data(path_data):
    data = pd.read_csv(path_data)


def load_yasu_data(project, duration, period):
    if project == 'openstack' and period == 'long':
        train_path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.7.csv'
        train_data = pd.read_csv(path_data)
        data = list()
        for i in range(0, 5):
            path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.' + str(i) + '.csv'
            data.append(pd.read_csv(path_data))
        data_new = pd.concat(data)
        data_new = replace_value_dataframe(df=data_new)
        ids, features = get_ids(data=data_new), get_features(data=data_new)
        return ids, features




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