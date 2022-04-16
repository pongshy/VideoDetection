import sys



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


def get_file_name(file_path):
    escape = judgeEscape(file_path)
    videoname = ''
    # 处理获取视频名称
    if escape == '/':
        videoname = file_path[file_path.rfind('/') + 1: file_path.rfind('.')]
    elif escape == '\\':
        videoname = file_path[file_path.rfind('\\') + 1: file_path.rfind('.')]

    return videoname


# 获取列表字段
def get_labels(video_datas):
    labels = ['videoname']
    # labels = list()
    first_key = ''
    for key in video_datas.keys():
        first_key = key
        break
    tmp_labels = list(video_datas[first_key].keys())
    # print(tmp_labels)
    labels.extend(tmp_labels)
    # print(labels)
    return labels


# 获取每个视频的名字与向量 ['a.mp4', 0.11, 0, 0, ...]
def get_rows(video_datas):
    datas = []
    for key, value in video_datas.items():
        tmpList = list()
        tmpList.append(key)
        tmpList.extend(list(value.values()))
        datas.append(tmpList)
    return datas