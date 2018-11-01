from ultis import load_file, commit_id


def load_commit_message(ids):
    print('hello')


if __name__ == '__main__':
    path_labels = './labels/openstack.csv'
    data_labels = load_file(path_file=path_labels)
    ids = commit_id(data=data_labels)
    print(len(ids))