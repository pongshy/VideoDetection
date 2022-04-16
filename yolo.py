import cv2
import numpy as np
import os
import time
import datetime
import utils
import videoprocess
import csv
import sys


# 读取文件夹中的每一个视频，并处理各视频的物体出现概率
def yolo_detect_from_video_directory(video_direcotry,
                                     label_path='./cfg/coco.names',
                                     config_path='./cfg/yolov3.cfg',
                                     weights_path='./cfg/yolov3.weights',
                                     confidence_thre=0.5,
                                     nms_thre=0.3,
                                     jpg_quality=80):
    print('directory:', video_direcotry)
    # 加载类别标签文件
    LABELS = open(label_path).read().strip().split("\n")

    nclass = len(LABELS)

    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')

    # 加载模型配置和权重文件
    print('从硬盘中加载YOLO......')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # 获取yolo输出层的名字
    ln = net.getLayerNames()
    ln = [ln[x - 1] for x in net.getUnconnectedOutLayers()]

    # 通过视频处理，获取指定帧数的图片（默认为6张）
    # 载入图片并获取其维度

    video_images_directory = videoprocess.get_videos_frames_from_directory(video_direcotry)
    # print('1: ', len(video_images_directory))
    # 所要返回的变量
    video_appear_object_probability = {}
    # 遍历这个字典
    for video_name_key in video_images_directory.keys():
        images = video_images_directory[video_name_key]
        # print('images len:', len(images))

        sum = 0
        # 一个视频中各物体出现的次数
        object_nums = dict(zip(LABELS, [0] * len(LABELS)))
        for img in images:
            # 这里可以直接由自定义的视频截取帧的方法返回的image，放入处理
            # <class 'numpy.ndarray'>
            # print(type(img))
            (H, W) = img.shape[:2]
            # print(H, W)

            # 将图片构成一个blob，设置图片尺寸，然后执行一次
            # yolo前馈网络计算，最终获取边界框和相应概率
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            # 显示预测所花费的时间
            print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))

            # 初始化边界框，置信度（概率）以及类别
            boxes = []
            confidences = []
            classIds = []

            # 迭代每个输出层，总共三个
            for output in layerOutputs:
                # 迭代每个检测
                for detection in output:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]

                    # 只保留置信度大于某值的边界框
                    if confidence > confidence_thre:
                        # 将边界框的坐标还原至与原图片项匹配
                        # yolo返回的是边界框的中心以及边界框的宽度和高度
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # 计算边界框的左上角位置
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # 更新边界框，置信度（概率）以及类别
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIds.append(classId)

            # 使用非极大值抑制方法抑制弱、重叠边界框
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
            objects = dict(zip(LABELS, [0] * len(LABELS)))
            # 确保至少一个边界框
            if len(idxs) > 0:
                # 迭代每个边界框
                print(idxs.flatten())
                for i in idxs.flatten():
                    # 提取边界框坐标
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # 绘制边界框以及在左上角添加类别标签
                    color = [int(c) for c in COLORS[classIds[i]]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = '{}: {:.3f}'.format(LABELS[classIds[i]], confidences[i])
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # 获取每个检测的图片，出现的物品种类次数
                    object_nums[LABELS[classIds[i]]] += 1
                    sum += 1
                    # 输出结果图片
                    # now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

                    # cv2.imwrite('./out/res/{}_{}.jpg'.format('tt', now_time), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        #  print(object_nums)
        #     print(sum)
        if sum > 0:
            video_appear_object_probability[video_name_key] = {}
            objects_probability = dict(zip(LABELS, [0] * len(LABELS)))
            for key in object_nums.keys():
                objects_probability[key] = object_nums[key] / sum
            # print('video_name:', video_name)
            video_appear_object_probability[video_name_key] = objects_probability
            # print('video:', video_name_key)

    print(len(video_appear_object_probability))
    print(video_appear_object_probability)
    return video_appear_object_probability


# 读取一个视频，并处理出该视频的物体出现概率
def yolo_detect_from_video(video_path,
                           label_path='./cfg/coco.names',
                           config_path='./cfg/yolov3.cfg',
                           weights_path='./cfg/yolov3.weights',
                           confidence_thre=0.5,
                           nms_thre=0.3,
                           jpg_quality=80):
    print(video_path)
    # 加载类别标签文件
    LABELS = open(label_path).read().strip().split("\n")

    nclass = len(LABELS)

    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')

    # 通过视频处理，获取指定帧数的图片（默认为6张）
    # 载入图片并获取其维度
    video_name = os.path.basename(video_path)
    video_images = videoprocess.get_one_video_frames(video_path)
    images = video_images[video_name]
    # print(len(images))

    # 加载模型配置和权重文件
    print('从硬盘中加载YOLO......')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # 获取yolo输出层的名字
    ln = net.getLayerNames()
    ln = [ln[x - 1] for x in net.getUnconnectedOutLayers()]

    sum = 0
    # 一个视频中各物体出现的次数
    object_nums = dict(zip(LABELS, [0] * len(LABELS)))
    tmp = []
    for img in images:
        # 这里可以直接由自定义的视频截取帧的方法返回的image，放入处理
        print(type(img))  # <class 'numpy.ndarray'>
        (H, W) = img.shape[:2]
        # print(H, W)

        # 将图片构成一个blob，设置图片尺寸，然后执行一次
        # yolo前馈网络计算，最终获取边界框和相应概率
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # 显示预测所花费的时间
        print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))

        # 初始化边界框，置信度（概率）以及类别
        boxes = []
        confidences = []
        classIds = []

        # 迭代每个输出层，总共三个
        for output in layerOutputs:
            # 迭代每个检测
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                # 只保留置信度大于某值的边界框
                if confidence > confidence_thre:
                    # 将边界框的坐标还原至与原图片项匹配
                    # yolo返回的是边界框的中心以及边界框的宽度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # 计算边界框的左上角位置
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # 更新边界框，置信度（概率）以及类别
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIds.append(classId)

        # 使用非极大值抑制方法抑制弱、重叠边界框
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
        objects = dict(zip(LABELS, [0] * len(LABELS)))
        # 确保至少一个边界框
        if len(idxs) > 0:
            # 迭代每个边界框
            print(idxs.flatten())
            for i in idxs.flatten():
                # 提取边界框坐标
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # 绘制边界框以及在左上角添加类别标签
                color = [int(c) for c in COLORS[classIds[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.3f}'.format(LABELS[classIds[i]], confidences[i])
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # 获取每个检测的图片，出现的物品种类次数
                object_nums[LABELS[classIds[i]]] += 1
                sum += 1
                # 输出结果图片
                # now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

                # cv2.imwrite('./out/res/{}_{}.jpg'.format('tt', now_time), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    # print(object_nums)
    # print(sum)
    objects_probability = dict(zip(LABELS, [0] * len(LABELS)))
    for key in object_nums.keys():
        objects_probability[key] = object_nums[key] / sum
    # print('video_name:', video_name)
    video_appear_object_probability = {video_name: objects_probability}
    print(video_appear_object_probability)
    return video_appear_object_probability



# using yolo3 to realize image detection
def yolo_detect(pathIn='',
                label_path='./cfg/coco.names',
                config_path='./cfg/yolov3.cfg',
                weights_path='./cfg/yolov3.weights',
                confidence_thre=0.5,
                nms_thre=0.3,
                jpg_quality=80):
    """
    pathIn：原始图片的路径
    pathOut：结果图片的路径
    label_path：类别标签文件的路径
    config_path：模型配置文件的路径
    weights_path：模型权重文件的路径
    confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
    nms_thre：非极大值抑制的阈值，默认为0.3
    jpg_quality：设定输出图片的质量，范围为0到100，默认为80，越大质量越好
    """

    # 加载类别标签文件
    LABELS = open(label_path).read().strip().split("\n")
    print(len(LABELS))
    print(LABELS)

    nclass = len(LABELS)

    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')

    # 载入图片并获取其维度
    img = cv2.imread(pathIn)
    # 这里可以直接由自定义的视频截取帧的方法返回的image，放入处理
    print(type(img))    # <class 'numpy.ndarray'>
    (H, W) = img.shape[:2]

    # 加载模型配置和权重文件
    print('从硬盘中加载YOLO......')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # 获取yolo输出层的名字
    ln = net.getLayerNames()
    # print(ln)
    # print(ln[253])
    # t = net.getUnconnectedOutLayers()
    # for i in t:
    #     print(i - 1)
    # print(t)
    # print(t[0])
    ln = [ln[x - 1] for x in net.getUnconnectedOutLayers()]
    print(ln)

    # 将图片构成一个blob，设置图片尺寸，然后执行一次
    # yolo前馈网络计算，最终获取边界框和相应概率
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # 显示预测所花费的时间
    print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))

    # 初始化边界框，置信度（概率）以及类别
    boxes = []
    confidences = []
    classIds = []

    # 迭代每个输出层，总共三个
    for output in layerOutputs:
        # 迭代每个检测
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            # 只保留置信度大于某值的边界框
            if confidence > confidence_thre:
                # 将边界框的坐标还原至与原图片项匹配
                # yolo返回的是边界框的中心以及边界框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # 计算边界框的左上角位置
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # 更新边界框，置信度（概率）以及类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)


    # 使用非极大值抑制方法抑制弱、重叠边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
    # 一个图像中各物体出现的次数
    object_nums = dict(zip(LABELS, [0] * len(LABELS)))
    print(object_nums)
    # 确保至少一个边界框
    if len(idxs) > 0:
        # 迭代每个边界框
        for i in idxs.flatten():
            # 提取边界框坐标
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 绘制边界框以及在左上角添加类别标签
            color = [int(c) for c in COLORS[classIds[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.3f}'.format(LABELS[classIds[i]], confidences[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - text_h - baseline), (x+ text_w, y), color, -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # 获取每个检测的图片，出现的物品种类次数
            object_nums[LABELS[classIds[i]]] += 1

    print(object_nums)

    # 输出结果图片
    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    cv2.imwrite('./out/res/{}_{}.jpg'.format(utils.get_file_name(pathIn), now_time), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])






if __name__ == '__main__':
    # yolo_detect('./out/a_20220412201110_1.jpg')
    # yolo_detect('./out/c_20220412232540_2.jpg')
    # yolo_detect('./out/d_20220413000320_5.jpg')
    # yolo_detect_from_video('./video/明天.mp4')
    # yolo_detect_from_video_directory('./video/v2')
    # video_directory = './video/v1'
    # process_videos_store_in_csv(video_directory, './datasets/test.csv')
    # video_path = './video/concert/tmp.flv'
    # yolo_detect(video_path)

    # 图像检测
    video_tmp_path = 'out/ttt1.jpg'
    yolo_detect(video_tmp_path)