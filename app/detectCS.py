from flask import Flask, request, jsonify
import json

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from werkzeug.utils import secure_filename
from os import path
import time
import os, sys, getopt
import base64
import random
import numpy as np

app = Flask(__name__)
app.debug = True
#临时文件保存目录，将客户端上传的文件先临时保存，识别后自动删除。
uploadPath = './uploads'
#图像识别物体种类字典表
nameDict = {"head" : "头", "helmet" : "安全帽", "person" : "人", "digger" : "挖掘机", "smoke" : "烟", "truck" : "卡车"}

@app.route('/api/tn/v1/cs', methods=['post'], strict_slashes=False)
def tn_v1_cs():
    if 'image' not in request.files.keys():  #二进制格式
        data = request.get_data()
        json_data = json.loads(data.decode("utf-8"))
        base64Image = json_data.get("image")
        img_decode = base64.b64decode(base64Image)
        img_np_ = np.frombuffer(img_decode, np.uint8)
        img = cv2.imdecode(img_np_, cv2.COLOR_RGB2BGR)  # 转为opencv格式
        now_milli_time = int(time.time() * 1000)
        filename = str(now_milli_time) + '_' + str(random.randint(100000,999999)) + '.jpg'
        cv2.imwrite(uploadPath + '/' + filename, img)  # 存储路径
    else:   #表单上传形式
        f=request.files['image']
        now_milli_time = int(time.time() * 1000)
        filename = str(now_milli_time) + '_' + secure_filename(f.filename)
        f.save(path.join(uploadPath, filename))

    items = detect(uploadPath + '/' + filename)
    ret = {"results" : items}
    ret = json.dumps(ret)

    #删除临时文件
    os.remove(uploadPath + '/' + filename)
    #返回JSON数据
    return (ret)

def detect(source=False):
    weights = './weights/best.pt'
    device = ''
    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.45
    agnostic_nms = False
    augment = False
    classes = None

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                items = []
                for *xyxy, conf, cls in reversed(det):
                    print(f'{names[int(cls)]}')
                    print(f'{conf:.2f}')
                    print(xyxy)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    w = x2 - x1
                    h = y2 - y1
                    print(x1)
                    print(y1)
                    print(w)
                    print(h)
                    location = {"left" : x1, "top" : y1, "width" : w, "height" : h}
                    item = {"name" : nameDict.get(f'{names[int(cls)]}'), "score" : f'{conf:.2f}', "location" : location}
                    items.append(item)

            print(items)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

    print(f'Done. ({time.time() - t0:.3f}s)')
    return items

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hH:P:", ["host=", "port="])
    except getopt.GetoptError:
        print
        'detectCs.py -h <host> -p <port>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('detectCs.py -H <host> -P <port>')
            sys.exit()
        elif opt in ("-H", "--host"):
            host = arg
        elif opt in ("-P", "--port"):
            port = arg
    # 这里指定了地址和端口号。
    app.run(host=host, port=port)