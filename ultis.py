import os

def load_file(path_file):
    lines = list(open(path_file, 'r', encoding='utf8', errors='ignore').readlines())
    lines = [l.strip() for l in lines]
    return lines


def print_label(data):
    print(data[0])


def get_label(data):
    labels = list()
    for i in range(1, len(data)):
        split_i = data[i].split(',')
        if split_i[2] == '':
            labels.append('False')
        else:
            labels.append('True')
    return labels


def commit_id(data):
    ids = list()
    for i in range(1, len(data)):
        split_ = data[i].split(',')
        ids.append(split_[0])
    return ids


def dict_label(commit_ids, labels):
    dicts = list()
    for c, l in zip(commit_ids, labels):
        dictionary = dict()
        dictionary['id'] = c
        dictionary['label'] = l
        dicts.append(dictionary)
    return dicts


def write_file(path_file, data):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)

    if not os.path.exists(path_):
        os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for line in data:
            # write line to output file
            out_file.write(str(line))
            out_file.write("\n")
        out_file.close()


if __name__ == '__main__':
    # project = 'openstack'
    project = 'qt'
    path_labels = './labels/' + project + '.csv'
    data_labels = load_file(path_file=path_labels)
    ids, labels = commit_id(data=data_labels), get_label(data=data_labels)
    # dict_label(commit_ids=ids, labels=labels)
    print(len(ids), len(labels))
    print_label(data=data_labels)

    valid_ids = load_file(path_file='./labels/' + project + '_ids.txt')
    print(len(valid_ids))
    data = list()
    for i, l in zip(ids, labels):
        if i in valid_ids:
            print(i, l)
            data.append(i + '\t' + l)
    write_file(path_file='./labels/' + project + '_ids_label.txt', data=data)



