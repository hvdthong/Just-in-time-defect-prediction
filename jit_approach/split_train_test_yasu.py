import pandas as pd





def load_yasu_data(project, duration, period):
    if project == 'openstack' and duration == '6-month' and period == 'long':
        data = list()
        for i in range(0, 5):
            path_data = './yasu_replication_data/' + project + '.STRATA_PER_YEAR.2.all.' + str(i) + '.csv'
            df = pd.read_csv(path_data)
            data.append(df)
        data = pd.concat(data)
        print(len(data))


if __name__ == '__main__':
    # data description
    ################################################################################
    # STRATA_PER_YEAR.2 -> six - month periods
    # STRATA_PER_YEAR.4 -> three - month periods

    # local -> Short - period
    # all -> Long - periods
    ################################################################################
    project, duration, period = 'openstack', '6-month', 'long'
    load_yasu_data(project=project, duration=duration, period=period)