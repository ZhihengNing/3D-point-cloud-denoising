# 得到某个文件所在文件夹
def search_file_in_dir(dirPath, fileName):
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file == fileName:
                return root


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