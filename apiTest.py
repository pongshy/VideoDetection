# encoding=utf-8
from flask import Flask, request
import json

app = Flask(__name__)


# 只接受Get方法访问
@app.route('/test', methods=['GET'])
def testGet():
    # 默认返内容
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': False}
    # 判断传入参数是否为空
    if request.args is None:
        return_dict['return_code'] = '400'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的参数
    get_data = request.args.to_dict()
    name = get_data.get('name')
    age = get_data.get('age')
    # 对参数进行处理
    s = '{0}今年{1}岁'.format(name, age)
    return_dict['result'] = s

    return json.dumps(return_dict, ensure_ascii=False)


# 只接受Post方法请求
@app.route('/postts', methods=['POST'])
def testPost():
    # 默认返内容
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': False}
    # 判断传入参数是否为空
    if request.args is None:
        return_dict['return_code'] = '400'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的参数
    get_data = request.get_data()
    # 将json格式的参数转为python对象
    get_data = json.loads(get_data)
    name = get_data['name']
    age = get_data['age']
    s = '{0}今年{1}岁'.format(name, age)
    return_dict['result'] = s

    return json.dumps(return_dict, ensure_ascii=False)


if __name__ == '__main__':
    app.run(port=8000, debug=True)