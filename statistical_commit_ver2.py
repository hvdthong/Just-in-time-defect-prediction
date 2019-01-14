from clean_commit import loading_variable
import matplotlib.pyplot as plt
from statistics import mean, stdev


def statistic_msg(data):
    data = [len(d.split()) for d in data]
    plt.hist(data)
    plt.title('Message')
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()
    return mean(data), stdev(data)


def statistic_code(data, check):
    if check == 'File':
        data = [len(d) for d in data]
        plt.hist(data, bins=range(0, 40))
        plt.title(check)
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.show()
        return mean(data), stdev(data)
    elif check == 'Length':
        new_data = list()
        for d in data:
            for f in d:
                new_data.append(len(f.split()))
        plt.hist(new_data, bins=range(0, 750))
        plt.title(check)
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.show()
        return mean(new_data), stdev(new_data)


if __name__ == '__main__':
    project = 'openstack'
    messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    print(type(messages), type(codes))
    mean_, std_ = statistic_msg(data=messages)
    print(mean_, std_)

    mean_, std_ = statistic_code(data=codes, check='File')
    print(mean_, std_)
    mean_, std_ = statistic_code(data=codes, check='Length')
    print(mean_, std_)
