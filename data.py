import csv
import sys
import os
import yolo
import utils
import pandas as pd
import numpy as np
import time

# 将一个文件夹下所有的视频处理完后写入csv文件
def process_videos_store_in_csv(video_directory_path, csv_file, frames=10):
    if os.path.exists(csv_file):
        # 视频数据处理
        video_datas = yolo.yolo_detect_from_video_directory(video_directory_path, frames)
        print('in store funcation:', video_datas)
        # 打开一个csv文件用于写入
        # file = open(csv_file, 'w', encoding='utf-8-sig', newline='')
        file = open(csv_file, 'w', encoding='GB18030', newline='')
        # 在csv文件中进行追加
        # file = open(csv_file, 'a', encoding='utf8', newline='')

        # 获取csv的writer用于写入数据
        writer = csv.writer(file)
        # 列表字段名
        labels = utils.get_labels(video_datas)
        labels.append('tag')
        # print(labels)
        # 构建表头
        writer.writerow(labels)

        # 数据处理
        rows = utils.get_rows(video_datas)
        # print(rows)

        # 写入每一行数据
        writer.writerows(rows)
        print('数据处理完毕，并存入{}中'.format(os.path.basename(csv_file)))

    else:
        print("{} is not exist! in data.py".format(video_directory_path))
        sys.exit(1)


# 向指定的csv中追加数据
def add_data_to_csv(video_directory_path, csv_file, frames=10):
    if os.path.exists(csv_file):
        # 视频数据处理
        video_datas = yolo.yolo_detect_from_video_directory(video_directory_path, frames)
        print('in store funcation:', video_datas)
        # 打开一个csv文件用于写入，追加
        file = open(csv_file, 'a', encoding='GB18030', newline='')
        # file = open(csv_file, 'a', encoding='utf-8-sig', newline='')
        write = csv.writer(file)

        rows = utils.get_rows(video_datas)
        # 写入数据
        write.writerows(rows)
        print('数据追加完毕，存入{}中.'.format(os.path.basename(csv_file)))
    else:
        print("{} is not exist! in data.py".format(video_directory_path))
        sys.exit(1)


# 从数据集中读取某一类型的数据，返回data，target
def get_datas(datasets_path):
    datas = pd.read_csv(datasets_path, encoding='gb18030')
    tag = datas['tag']
    data = datas.iloc[:, 1:-1]
    data = np.array(data)
    tag = np.array(tag)
    # print('data.py:', data)

    return data, tag


if __name__ == '__main__':
    video_directory = './video/sport/1'
    csv_file = './datasets/sport4.csv'
    frames = 20
    # start = time.time()
    # process_videos_store_in_csv(video_directory, csv_file, frames)
    # end = time.time()
    # print('Process all videos in {} and store in {}, cost time: {}'.format(video_directory, csv_file, end - start))

    # get_datas(csv_file)
    # get_datas(csv_file)
    # 追加tag=0的视频向量
    video_directory_0 = './video/sport/0/v3'
    add_data_to_csv(video_directory_0, csv_file, frames)

    # 追加./video/sport/0/v1中的视频
    # video_directory_0_v1 = './video/sport/0/v1'
