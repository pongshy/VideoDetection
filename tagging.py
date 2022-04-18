import numpy as np
import train
import yolo
import utils

def tag(model_path, video_path, frames=5):
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
    video_path = './video/test/大二在校Vlog｜晚课 校园散步 充实又快乐～ - 1.大二在校Vlog｜晚课 校园散步 充实又快乐～(Av725361265,P1).mp4'
    tag(model_path, video_path)
