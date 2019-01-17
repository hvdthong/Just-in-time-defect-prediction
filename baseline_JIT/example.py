from load_data import loading_variable_path
from split_train_test import convert_label


if __name__ == '__main__':
    project = 'openstack'
    ids, labels = loading_variable_path(pname='../variables_ver1/' + project + '_ids.pkl'), convert_label(
        loading_variable_path(pname='../variables_ver1/' + project + '_labels.pkl'))
    print(len(ids), len(labels))
    for i, l in zip(ids, labels):
        if l == 1:
            print(i, l)