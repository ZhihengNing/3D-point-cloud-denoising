import os

# 得到某个文件所在文件夹
def search_file_in_dir(dirPath, fileName):
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file == fileName:
                return root