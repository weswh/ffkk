#!/home/veily/anaconda3/envs/feet3d/bin/python3
import requests
import threading
# import main_version5
# import main_version2
import time
from flask import Flask, request, jsonify
from wsgiref.simple_server import make_server  
import subprocess

app = Flask(__name__)

def handle_aiwork(code):

    # _code = 'test'
    _code = code

    time_start = time.time()



    subprocess.call(['bash', 'test.sh', code])  # 输出：Hello World


    # 分析工作
    # feet3d = main_version5.Feet(_code, 'False', "all")
    # feet3d.run()
    time_end = time.time()

    print('ai时间：', time_end - time_start)

    # url = 'http://localhost:3000/api/aiwork/finish/' + code
    # data = {'status': 4}
    # response = requests.put(url, json=data)
    # if response.status_code == 200:
    #   print('请求成功')
    # else:
    #   print('请求失败，状态码:', response.status_code)
    return

@app.route('/api/ping')
def handle_ping():
    return "pang"

@app.route('/api/startwork', methods=['POST'])
def handle_startwork():
    data = request.json
    code = data.get('code', None)

    print('code', code)
    # 这里异步开始处理分析数据


    result = jsonify({ 'code': 0 }) 

    try:
      handle_aiwork(code)
    #   param = "test/01"
    #   thread = threading.Thread(target=handle_aiwork, args=(code,))
    #   thread.start()
    except ValueError as err:
        # 处理ValueError异常的代码
        print(err)
        # result = jsonify({ 'code': 1, 'msg': err }) 


    # 在这里处理 code 信息
    return result

if __name__ == '__main__':
    app.run(port=3001)
    # httpd = make_server('127.0.0.1', 3001, app)  
    # httpd.serve_forever()