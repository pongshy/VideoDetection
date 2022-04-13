import datetime
import cv2
import numpy as np
import os
import time
import utils






if __name__ == '__main__':
    s = 'D:\\university\\GraduationDesign\\try\\SVM\\video\\a.mp4'
    print(s)
    index = s.rfind('\\')
    print(s.rfind('\\'))
    print(s[index + 1:])
    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print(now_time)
    print(utils.get_file_name(s))

    print(os.path.basename(s))

    dict_nums = dict(zip(['a', 'b', 'c'], [0] * 3))
    print(dict_nums)