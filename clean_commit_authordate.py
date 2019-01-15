from baseline.load_data import loading_variable_path, convert_label
import pandas as pd
from parser_commit import info_commit
from clean_commit import clean_message, clean_code, saving_variable


def sorted_authordate(ids, path_data):
    # sorted data based on author date and return the ids and labels
    df = pd.read_csv(path_data)
    new_df = list()
    for id in ids:
        new_df.append(df.loc[df['commit_id'] == id])
    new_df = pd.concat(new_df)
    new_df = new_df.sort_values(by=['author_date']).fillna(0)
    ids, labels = list(new_df['commit_id'].values), list(new_df['bugcount'].values)
    labels = [1 if (int(l) > 0) else 0 for l in labels]
    return ids, labels


if __name__ == '__main__':
    project = 'openstack'
    ids, labels = loading_variable_path(pname='./variables_ver1/' + project + '_ids.pkl'), convert_label(
        loading_variable_path(pname='./variables_ver1/' + project + '_labels.pkl'))
    path_data = './labels/' + project + '.csv'
    ids, lables = sorted_authordate(ids=ids, path_data=path_data)

    path_file = './output/' + project
    messages, codes = info_commit(ids=ids, path_file=path_file)
    print(len(ids), len(labels), len(messages), len(codes))
    messages, codes = clean_message(data=messages), clean_code(data=codes)

    saving_variable(project + '_messages', messages)
    saving_variable(project + '_codes', codes)
    saving_variable(project + '_labels', labels)
    saving_variable(project + '_ids', ids)