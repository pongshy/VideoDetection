# encoding: utf-8
from flask import Flask, request
import json
import tagging

app = Flask(__name__)

model_path = 'D:\\university\\GraduationDesign\\try\\SVM\\model\\sport4.pickle'

# 调用模型给视频分类并返回分类结果
@app.route('/analysis', methods=['POST'])
def analyse():
    # 默认返内容
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': False}
    # 判断传入参数是否为空
    if request.args is None:
        return_dict['return_code'] = '400'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的参数
    get_data = request.get_data()
    get_data = json.loads(get_data)

    video_path = get_data['vpath']

    tag = tagging.tag(video_path, model_path)

    return_dict['result'] = tag.tolist()[0]

    return json.dumps(return_dict, ensure_ascii=False)



if __name__ == '__main__':
    app.run(port=8001, debug=True)