from parser_commit import get_ids, info_commit
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def statistic_msg(data, type):
    if type == 'Title':
        new_data = [d['title'] for d in data]
        new_data = [len(d.split()) for d in data]

        plt.hist(new_data)
        plt.title(type)
        plt.xlabel("Length")
        plt.ylabel("Frequency")

    exit()



if __name__ == '__main__':
    # project = 'openstack'
    project = 'qt'
    path_data = './output/' + project
    ids = get_ids([f for f in listdir(path_data) if isfile(join(path_data, f))])
    messages, codes = info_commit(ids=ids, path_file=path_data)
    print(len(ids), len(messages), len(codes))

    statistic_msg(data=messages, type='Title')
