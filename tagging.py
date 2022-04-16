import numpy as np
import train
import yolo
import utils

def tag(model_path, video_path):
    clsf = train.get_model(model_path)
    images = yolo.yolo_detect_from_video(video_path)
    data = utils.get_rows(images, False)

    data = np.array(data)
    # svm classifier得出的结果
    pred = clsf.predict(data)

    print('tag:', pred)
    return pred



if __name__ == '__main__':
    model_path = './model/sport2.pickle'
    video_path = './video/test/2022新款单板滑雪板推荐——男款篇 - 1.20211101 测评一(Av251472235,P1).mp4'
    tag(model_path, video_path)
