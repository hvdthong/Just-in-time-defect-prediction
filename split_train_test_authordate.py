from baseline.load_data import loading_variable_path, convert_label
import pandas as pd


def split_data_authordate(ids, path_data):
    df = pd.read_csv(path_data)
    new_df = list()
    for id in ids:
        new_df.append(df.loc[df['commit_id'] == id])
    new_df = pd.concat(new_df)
    new_df = new_df.sort_values(by=['author_date'])
    return new_df


if __name__ == '__main__':
    project = 'openstack'
    ids, labels = loading_variable_path(pname='../variables_ver1/' + project + '_ids.pkl'), convert_label(
        loading_variable_path(pname='../variables_ver1/' + project + '_labels.pkl'))
    print(len(ids), len(labels))
    split_data_authordate(ids=ids, path_data='../labels/' + project + '.csv')
