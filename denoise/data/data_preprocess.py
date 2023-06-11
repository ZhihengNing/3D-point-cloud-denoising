import os
import shutil

from environments import BASE_PATH


# 此文件针对于划分数据集给出了建议
# 以及可恶的Linux中对于百分号出现而产生的25
def divide_data(target_dir, dataset_txt_path, data_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    with open(dataset_txt_path, "r") as f:
        dataset_name = f.readlines()
        dataset_name = [line.strip() for line in dataset_name]
        dataset_name = list(filter(None, dataset_name))

    data_files = os.listdir(data_dir)

    count = 0
    for item in data_files:
        if item.rstrip(".clean_xyz") in dataset_name or item.rstrip(".xyz") in dataset_name:
            print(item)
            count += 1
            abs_path = os.path.join(data_dir, item)
            shutil.copy(abs_path, target_dir)
    print(count)


def noise_train():
    str_files = ["training", "validation"]
    for item in str_files:
        target_dir = os.path.join(BASE_PATH, "data", "pointCleanNetNoiseTrainingSet")
        dataset_txt_path = os.path.join(BASE_PATH, "data", f"noise_removal_{item}set.txt")
        data_dir = os.path.join(BASE_PATH, "data", "pointCleanNetDataset")

        divide_data(target_dir, dataset_txt_path, data_dir)


def outliers_train():
    str_files = ["training", "validation"]
    for item in str_files:
        target_dir = os.path.join(BASE_PATH, "data", "pointCleanNetOutliersTrainingSet")
        dataset_txt_path = os.path.join(BASE_PATH, "data", f"outliers_removal_{item}set.txt")
        data_dir = os.path.join(BASE_PATH, "data", "pointCleanNetOutliersTrainingSet")

        divide_data(target_dir, dataset_txt_path, data_dir)


def noise_eval():
    target_dir = os.path.join(BASE_PATH, "data", "pointCleanNetNoiseTestSet")
    dataset_txt_path = os.path.join(BASE_PATH, "data", f"noise_removal_testset.txt")
    data_dir = os.path.join(BASE_PATH, "data", "pointCleanNetDataset")

    divide_data(target_dir, dataset_txt_path, data_dir)


def outliers_eval():
    target_dir = os.path.join(BASE_PATH, "data", "pointCleanNetOutliersTestSet")
    dataset_txt_path = os.path.join(BASE_PATH, "data", f"outliers_removal_testset.txt")
    data_dir = os.path.join(BASE_PATH, "data", "pointCleanNetOutliersTestSet")

    divide_data(target_dir, dataset_txt_path, data_dir)


def judge_noise_data_set():
    data_set_path = os.path.join(BASE_PATH, "data", "pointCleanNetDataset")
    files = os.listdir(data_set_path)
    count = 0
    set1 = set()
    for item in files:
        if item.endswith(".xyz"):
            set1.add(item.rstrip(".xyz"))
            count += 1

    print(count)
    data_set_path = os.path.join(BASE_PATH, "data", "pointCleanNetNoiseTrainingSet")
    files = os.listdir(data_set_path)
    count = 0
    set2 = set()
    for item in files:
        if item.endswith(".xyz"):
            count += 1
            set2.add(item.rstrip(".xyz"))
    print(count)

    for item in set1 - set2:
        print(item)


def add_postfix():
    path = os.path.join(BASE_PATH, "data", "outliers_removal_testset.txt")
    with open(path) as f:
        lines = f.readlines()
        point_cloud_names = [line.strip() for line in lines]

        point_cloud_names = list(filter(None, point_cloud_names))

        length = len(point_cloud_names)

        for index in range(length):
            point_cloud_names.append(point_cloud_names[index] + "_outliers_gaussian")
            point_cloud_names.append(point_cloud_names[index] + "_outliers_uniform")
        point_cloud_names = point_cloud_names[length:]
        print(point_cloud_names)

    with open(path, "w") as f:
        for item in point_cloud_names:
            f.write(item.strip() + "\n")


def replace():
    ppp = os.path.join(BASE_PATH, "data", "wholeProcessDataSet")
    files = os.listdir(ppp)

    for index, item in enumerate(files):
        path = os.path.join(ppp, item)
        print(path)
        new_path = path.replace("%25.", "%.")
        os.rename(path, new_path)
        # index=path.find(".")
        # new_path=path[:index-1]+"%"+path[index:]
        # os.rename(path,new_path)


if __name__ == '__main__':
    # noise_eval()
    # add_postfix()
    replace()
