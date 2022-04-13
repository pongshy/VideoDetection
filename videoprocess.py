import ffmpeg
import numpy as np
import random
import cv2
import sys
import os
import datetime
import utils

store_path = './out/'


def judgeEscape(s):
    linux_escape = '/'
    win_escape = '\\'
    if linux_escape in s:
        return '/'
    elif win_escape in s:
        return '\\'
    else:
        print('filename error!')
        sys.exit(1)


# 获取指定地址视频的总帧数下平均6帧的数据，并在./out下保存为jpg格式
def get_video_frame_store(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        # print(probe)
        # print(probe['streams'][0])
        # print('1:', probe['streams'][1])
        # print(len(probe))
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)

        total_duration = float(video_stream['duration'])
        total_frame = int(video_stream['nb_frames'])
        width = video_stream['width']

        escape = judgeEscape(file_path)
        videoname = ''
        # 处理获取视频名称
        if escape == '/':
            videoname = file_path[file_path.rfind('/') + 1 : file_path.rfind('.')]
        elif escape == '\\':
            videoname = file_path[file_path.rfind('\\') + 1 : file_path.rfind('.')]

        print(videoname)
        # 视频时长
        print('total_duration:', total_duration)
        # 总帧数
        print('total_frame:', total_frame)
        # 宽度
        print('width:', width)
        # 所要获取的帧数
        framenums = 6
        # print(isinstance(total_frame, str))
        intervals = int(total_frame // framenums)
        interval_list = [(i + 1) * intervals for i in range(framenums)]
        print(interval_list)
        # 获取时间戳
        now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        cnt = 1
        images = []
        for frame_num in interval_list:
            print('frame_num:', frame_num)
            out, err = (
                ffmpeg
                    .input(file_path)
                    .filter('select', 'gte(n, {})'.format(frame_num - 1))
                    .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                    .run(capture_stdout=True)
            )
            if not out:
                print('Wrong...')
            else:
                # filename = store_path + videoname + '_{0}_{1}.jpg'.format(now_time, cnt)
                cnt += 1
                image_array = np.asarray(bytearray(out), dtype='uint8')
                # <class 'numpy.ndarray'>
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                # print(type(image))
                images.append(image)
                # print('Get Frame: {}'.format(filename))
                # 显示抽出的帧
                # cv2.imshow('frame', image)
                # cv2.waitKey()
                # 保存图片
                # cv2.imwrite(filename, image)
                # print('Store Frame: {}'.format(filename))
        return images
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)


# 外部调用——通过视频地址获取视频所在帧的图像数据
def get_one_video_frames(file_path):
    videoname = os.path.basename(file_path)
    videos_images = dict()

    videos_images[videoname] = get_one_video_frames_inself(file_path)
    return videos_images


# 内部使用通过视频地址获取视频所在帧
def get_one_video_frames_inself(file_path):
    images = get_video_frame_store(file_path)
    return images


# 处理一个文件夹下所有的视频
def get_videos_frames_from_directory(video_directory_path):
    fileordir = os.listdir(video_directory_path)

    videos_images = dict()
    escape = judgeEscape(video_directory_path)
    for tmp in fileordir:
        video_path = video_directory_path + escape + tmp
        videoname = os.path.basename(video_path)
        if os.path.exists(video_path) and os.path.isfile(video_path):
            print(video_path)
            print(videoname)
            images = get_one_video_frames_inself(video_path)
            videos_images[videoname] = images
    return videos_images





if __name__ == '__main__':
    # 单个处理
    # file_path = './video/a.mp4'
    # file_path = 'D:\\university\\GraduationDesign\\try\\SVM\\video\\d.mp4'
    # videos_images =  get_one_video_frames(file_path)
    # print(len(videos_images))
    # print(videos_images)

    # 批量处理
    video_directory_path = './video/v2'
    # video_directory_path = 'D:\\university\\GraduationDesign\\try\\SVM\\video'
    video_images = get_videos_frames_from_directory(video_directory_path)
    print(len(video_images))
    print(len(video_images['a.mp4']))
