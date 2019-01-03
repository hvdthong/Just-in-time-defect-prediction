from clean_commit import loading_variable

if __name__ == '__main__':
    project = 'openstack'
    train, test = loading_variable(project + '_train'), loading_variable(project + '_test')
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
    print(len(ids_train), len(labels_train), len(msg_train), len(code_train))
    print(len(ids_test), len(labels_test), len(msg_test), len(code_test))

