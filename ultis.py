def load_file(path_file):
    lines = list(open(path_file, "r").readlines())
    lines = [l.strip() for l in lines]
    return lines


def commit_id(data):
    ids = list()
    for i in range(1, len(data)):
        split_ = data[i].split(',')
        ids.append(split_[0])
    return ids


if __name__ == '__main__':
    path_labels = './labels/openstack.csv'
    data_labels = load_file(path_file=path_labels)
    ids = commit_id(data=data_labels)
    print(len(ids))
