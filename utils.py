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