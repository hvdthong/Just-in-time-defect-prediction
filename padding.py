from clean_commit import loading_variable


def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return ' '.join([line_split[i] for i in range(max_length)])
    else:
        return line


def padding_message(data, max_length):
    new_data = list()
    for d in data:
        new_data.append(padding_length(line=d, max_length=max_length))
    return new_data


def padding_multiple_length(lines, max_length):
    return [padding_length(line=l, max_length=max_length) for l in lines]


def padding_commit_code_length(data, max_length):
    return [padding_multiple_length(lines=commit, max_length=max_length) for commit in data]


def padding_commit_code_line(data, max_line, max_length):
    new_data = list()
    for d in data:
        if len(d) == max_line:
            new_data.append(d)
        elif len(d) > max_line:
            new_data.append(d[:max_line])
        else:
            num_added_line = max_line - len(d)
            for i in range(num_added_line):
                d.append(('<NULL> ' * max_length).strip())
            new_data.append(d)
    return new_data


def padding_commit_code(data, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    padding_line = padding_commit_code_line(padding_length, max_line=max_line, max_length=max_length)
    return padding_line


def dictionary_commit(data, type_data):
    # create dictionary for commit message
    lists = list()
    if type_data == 'msg':
        for m in data:
            lists += m.split()
    elif type_data == 'code':
        for d in data:
            for l in d:
                lists += l.split()
    else:
        print('You need to give an correct data type')
        exit()
    lists = list(sorted(list(set(lists))))
    lists.append('<UNK>')
    lists.append('<NULL>')
    new_dict = dict()
    for i in range(len(lists)):
        new_dict[lists[i]] = i
    return new_dict


if __name__ == '__main__':
    project = 'openstack'
    messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    print(len(messages), len(codes))
    dict_msg, dict_code = dictionary_commit(data=messages, type_data='msg'), dictionary_commit(data=codes,
                                                                                               type_data='code')
    print(len(dict_msg), len(dict_code))
    exit()
    padding_message(data=messages, max_length=256)
    padding_commit_code(data=codes, max_line=10, max_length=512)
