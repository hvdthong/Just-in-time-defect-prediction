from os import listdir
from os.path import isfile, join
from parser_commit import get_ids, info_commit
import nltk
from ultis import load_file
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords


def clean_message(data):
    return [' '.join(word_tokenize(d['title'] + ' ' + d['desc'])).lower() for d in data]


def clean_code_line(line):
    for p in string.punctuation:
        if line.startswith('+#') or line.startswith('-#'):
            line = line[2:].replace(p, ' ' + p + ' ')
        elif line.startswith('+') or line.startswith('-') or line.startswith('#'):
            line = line[1:].replace(p, ' ' + p + ' ')
        elif line == '':
            pass
        else:
            line = line.replace(p, ' ' + p + ' ')
    return line


def clean_code(data, project):
    new_diffs = list()
    for diff in data:
        new_diff = list()
        for file_ in diff:
            file_diff = file_['diff']
            lines = list()
            for line in file_diff:
                line = clean_code_line(line=line)
                lines.append(' '.join(word_tokenize(line)))
            new_diff.append(' '.join(lines).strip())
        new_diffs.append(new_diff)
    print(len(new_diffs))
    exit()
    return new_diffs


def collect_labels(path_data, path_label):
    valid_ids = get_ids([f for f in listdir(path_data) if isfile(join(path_data, f))])
    ids, labels = [l.split('\t')[0] for l in load_file(path_file=path_label)], [l.split('\t')[1] for l in
                                                                                load_file(path_file=path_label)]
    labels_valid_ids = [labels[ids.index(v_id)] for v_id in valid_ids if v_id in ids]
    return valid_ids, labels_valid_ids


def collect_labels_ver2(path_label):
    ids, labels = [l.split('\t')[0] for l in load_file(path_file=path_label)], [l.split('\t')[1] for l in
                                                                                load_file(path_file=path_label)]
    return ids, labels


if __name__ == '__main__':
    project = 'openstack'
    # project = 'qt'
    path_data = './output/' + project
    path_label = './labels/' + project + '_ids_label.txt'

    # ids, labels = collect_labels(path_data=path_data, path_label=path_label)
    ids, labels = collect_labels_ver2(path_label=path_label)

    messages, codes = info_commit(ids=ids, path_file=path_data)
    print(len(ids), len(labels), len(messages), len(codes))
    messages, codes = clean_message(data=messages[:500]), clean_code(data=codes[:500], project=project)
