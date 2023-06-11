import os

from environments import BASE_PATH

dataset_base = os.path.join(BASE_PATH, "train_result", "whole_process")


def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = list(filter(None, lines))
    return lines


def write_file(path, data):
    with open(path, "w") as f:
        if type(data) == str:
            f.write(data)
            return
        for item in data:
            f.write(' '.join(str(i) for i in item) + "\n")


