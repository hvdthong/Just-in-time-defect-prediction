from clean_commit import collect_labels_ver2, info_commit, clean_code, clean_message


if __name__ == '__main__':
    project = 'openstack'
    # project = 'qt'
    path_data = './output/' + project
    path_label = './labels/' + project + '_ids_label.txt'

    ids, labels = collect_labels_ver2(path_label=path_label)
    messages, codes = info_commit(ids=ids, path_file=path_data)

    print(len(ids), len(labels), len(messages), len(codes))
    exit()
    messages, codes = clean_message(data=messages), clean_code(data=codes)