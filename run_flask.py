# coding: utf-8

from gevent import pywsgi
from flask import Flask, request
import threading
import torch
import time
import base64
import os

from util.loaders import read_classes, get_color_dict, read_anchors
from models import YOLO


def after_request(response):
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


app = Flask(__name__)
app.after_request(after_request)
out_dir = "./data/images"
model_path = "./models/yolov3.pth"

@app.route('/detection/', methods=['POST'])
def upload():
    global model_lock, model, model_use_time
    model_lock.acquire()
    if model.backbone is None:
        model.backbone = torch.load(model_path).cuda()
    f = request.files["image"]
    print("Receive image:", f.filename)
    in_path = os.path.join(out_dir, f.filename)
    out_path = os.path.join(out_dir, f.filename.split(".")[0] + "_pred.jpg")
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
    model_use_time = time.time()
    model_lock.release()
    with open(out_path, 'rb') as f:
        image = base64.b64encode(f.read())
    return image


def cuda_memory_control(max_time, time_gap):
    global model_lock, model, model_use_time
    while True:
        model_lock.acquire()
        if time.time() - model_use_time >= max_time and model.backbone is not None:
            model.backbone = None
            torch.cuda.empty_cache()
            print("Cache cleaned.")
        model_lock.release()
        time.sleep(time_gap)


def run_server():
    server = pywsgi.WSGIServer(('0.0.0.0', 34560), app, log=app.logger)
    server.serve_forever()
    # app.run(threaded=False)


if __name__ == '__main__':
    classes = read_classes("./data/coco-ch.names")
    color_dict = get_color_dict(classes, "./data/colors")
    anchors = read_anchors("./data/anchors")
    model = YOLO(classes,
                 model_load_path=model_path,
                 anchors=anchors,
                 device_ids="0")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_use_time = time.time()
    model_lock = threading.Lock()
    threading.Thread(target=cuda_memory_control, args=(60, 30)).start()
    threading.Thread(target=run_server).start()
    print("Server is successfully started.")
