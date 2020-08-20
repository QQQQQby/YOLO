from gevent import pywsgi
from flask import Flask, request
import base64
import os

from util.loaders import read_classes, get_color_dict, read_anchors
from models import YOLO

app = Flask(__name__)


def after_request(response):
    # JS前端跨域支持
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


app.after_request(after_request)


# 这里默认的是get请求方式
@app.route('/detection/', methods=['POST'])
def upload():
    f = request.files["image"]
    print("Receive image:", f.filename)
    in_path = os.path.join("./data/images", f.filename)
    out_path = os.path.join("./data/images", f.filename.split(".")[0] + "_pred.jpg")
    if not os.path.exists(out_path):
        f.save(in_path)
        model.detect_image(
            in_path,
            0.5,
            0.5,
            color_dict,
            do_show=False,
            output_path=out_path
        )
    with open(out_path, 'rb') as f:
        image = base64.b64encode(f.read())
    return image


if __name__ == '__main__':
    classes = read_classes("./data/coco-ch.names")
    color_dict = get_color_dict(classes, "./data/colors")
    anchors = read_anchors("./data/anchors")
    model = YOLO(classes,
                 model_load_path="./models/yolov3.pth",
                 anchors=anchors,
                 device_ids="0")
    server = pywsgi.WSGIServer(('0.0.0.0', 34560), app)
    # server = make_server('0.0.0.0', 34556, app)
    server.serve_forever()
    app.run(threaded=False)
