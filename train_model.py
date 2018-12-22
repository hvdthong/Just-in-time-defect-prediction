from clean_commit import loading_variable

if __name__ == '__main__':
    project = 'openstack'
    messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    print(len(messages), len(codes))