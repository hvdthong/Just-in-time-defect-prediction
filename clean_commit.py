from os import listdir
from os.path import isfile, join
from parser_commit import get_ids, info_commit
import nltk
from ultis import load_file


def clean_message(data):
    new_data = [' '.join(nltk.word_tokenize(d['title'] + ' ' + d['desc'])).lower() for d in data]
    return new_data


def clean_code_line_openstack(code_line):
    if not (code_line[1:].startswith('#')):
        code_line = code_line[1:].replace("'", ' ').replace(',', ' ').replace('=', ' = ').replace('.', ' ') \
            .replace(':', ' ').replace('"', ' ').replace('[', ' ').replace(']', ' ').replace('(', ' ') \
            .replace(')', ' ').replace('&', ' & ').replace('"', ' ').replace(':', ' ').replace('{', ' ') \
            .replace('}', ' ').replace('#', ' ').replace('/', ' ')
        return nltk.word_tokenize(code_line)
    else:
        return ''


def clean_code(data, project):
    new_diffs = list()
    for diff in data:
        new_diff = list()
        for file_ in diff:
            file_diff = file_['diff']
            lines = list()
            for line in file_diff:
                if project == 'openstack':
                    code_line = clean_code_line_openstack(code_line=line)
                    lines.append(' '.join(code_line))
                    # if not (line.startswith('+#') or line.startswith('-#')):
                    #     file_diff = nltk.word_tokenize(line[1:].replace("'", ' ').replace(',', ' '))
                    #     lines.append(file_diff)

            new_diff.append(' '.join(lines).strip())
        new_diffs.append(new_diff)

    # new_diffs = [clean_code_line_openstack(line) for line in file_['diff'] for file_ in diff for diff in data if
    #              project == 'openstack']
    print(len(new_diffs))
    exit()
    return new_diffs


def collect_labels(path_data, path_label):
    valid_ids = get_ids([f for f in listdir(path_data) if isfile(join(path_data, f))])
    ids, labels = [l.split('\t')[0] for l in load_file(path_file=path_label)], [l.split('\t')[1] for l in
                                                                                load_file(path_file=path_label)]
    labels_valid_ids = [labels[ids.index(v_id)] for v_id in valid_ids if v_id in ids]
    return valid_ids, labels_valid_ids


if __name__ == '__main__':
    project = 'openstack'
    # project = 'qt'
    path_data = './output/' + project
    path_label = './labels/' + project + '_ids_label.txt'
    ids, labels = collect_labels(path_data=path_data, path_label=path_label)

    messages, codes = info_commit(ids=ids, path_file=path_data)
    print(len(ids), len(labels), len(messages), len(codes))

    messages, codes = clean_message(data=messages), clean_code(data=codes, project=project)
