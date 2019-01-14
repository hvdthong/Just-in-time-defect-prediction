from jit_approach.ultis import load_file
from os import listdir
from os.path import isfile, join
import re


def load_commit_msg(path_file):
    msg = ' '.join(load_file(path_file=path_file))
    new_msg = dict()
    new_msg['title'], new_msg['desc'] = re.compile('<title>(.*?)</title>', re.DOTALL).findall(msg)[0], \
                                        re.compile('<message>(.*?)</message>', re.DOTALL).findall(msg)[0]
    return new_msg


def diff_file_index(code):
    file_index = [i for i, c in enumerate(code) if c.startswith('diff')]
    return file_index


def diff_code(diff_file):
    file, diff = diff_file[0], list()
    diff = list()
    for i in range(1, len(diff_file)):
        if diff_file[i].startswith('-') or diff_file[i].startswith('+'):
            if not (diff_file[i].startswith('---') or diff_file[i].startswith('+++')):
                diff.append(diff_file[i])
    return file, diff


def load_commit_code(path_file):
    code = load_file(path_file=path_file)
    indexes = diff_file_index(code=code)
    diffs = list()
    for i in range(0, len(indexes)):
        dict = {}
        if i == len(indexes) - 1:
            file, diff = diff_code(code[indexes[i]:])
        else:
            file, diff = diff_code(code[indexes[i]:indexes[i + 1]])
        dict['file'] = file
        dict['diff'] = diff
        diffs.append(dict)
    return diffs


def info_commit(ids, path_file):
    messsages, codes = list(), list()
    for i in range(0, len(ids)):
        codes.append(load_commit_code(path_file=path_file + '/' + ids[i] + '.diff'))
        messsages.append(load_commit_msg(path_file=path_file + '/' + ids[i] + '.msg'))
    return messsages, codes


def get_ids(data):
    files = list()
    for i in range(1, len(data)):
        files.append(data[i].split('.')[0])
    files = sorted(files)
    return list(set(files))


if __name__ == '__main__':
    project = 'openstack'
    # project = 'qt'
    path_data = './output/' + project
    ids = get_ids([f for f in listdir(path_data) if isfile(join(path_data, f))])
    messages, codes = info_commit(ids=ids, path_file=path_data)
    print(len(ids), len(messages), len(codes))

    # write_file(path_file='./labels/' + project + '_ids.txt', data=ids)