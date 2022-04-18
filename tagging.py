import numpy as np
import train
import yolo
import utils


def tag(video_path, model_path='./model/sport4.pickle', frames=5):
    # 获取到训练好的classifier
    clsf = train.get_model(model_path)
    images = yolo.yolo_detect_from_video(video_path, frames)
    data = utils.get_rows(images, False)

    data = np.array(data)
    # svm classifier得出的结果
    pred = clsf.predict(data)

    print('tag:', pred)
    return pred



if __name__ == '__main__':
    model_path = './model/sport4.pickle'
    # video_path = './video/test/单板滑雪进阶指南 - 1.单板滑雪进阶指南(Av935605929,P1).mp4'
    video_path = 'D:\\university\\GraduationDesign\\try\\SVM\\video\\test\\足球史上值得反复观看的100粒进球，别说你全看过！ - 1.17(Av980715928,P1).mp4'
    tag(video_path, model_path)
