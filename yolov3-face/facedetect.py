import os
import cv2
import time

import argparse
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import * #import cv2, import numpy as np, import torch

import tkinter as tk
from PIL import Image, ImageTk
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/coco80cls.names', help='*.names path')
parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='weights path')
parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
opt = parser.parse_args()
print(opt)
img_size, weights, cfg = opt.img_size, opt.weights, opt.cfg
device = torch_utils.select_device(opt.device)
names = load_classes(opt.names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
model = Darknet(cfg, img_size)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()
ispredict = False
imgtmp = None
imgtk = None
fpsmean = []
def predict():
    global cap
    global model
    global img_size
    global canvas
    global imgtmp
    global imgtk
    global ispredict
    global strinfo
    global fpsmean
    if not ispredict:
        return
    start_time = time.time()
    s = ""
    ret, img0 = cap.read()
    #detect processing
    with torch.no_grad():
        img = letterbox(img0, new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1) #BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for c in det[:, -1].unique():
                    if c!=0:
                        continue
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss\n' % (n, names[int(c)])  # add to string
                for *xyxy, conf, cls in det:
                    if cls!=0:
                        continue
                    label = '%s %.2f' % (names[int(cls)],conf)
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])
                    s += "%s, centerx:%d, centery:%d\n"%(names[int(cls)], (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2)
    timespent = time.time()-start_time
    fps = 1/timespent
    fpsmean.append(fps)
    if len(fpsmean)==20:
        print(np.mean(fpsmean))
        fpsmean.clear()
    strinfo.set("FPS: %0.2f\n%s"%(fps, s))
    imgcv = cv2.cvtColor(img0, cv2.COLOR_BGR2RGBA)
    imgtmp = Image.fromarray(imgcv)
    imgtk = ImageTk.PhotoImage(image=imgtmp)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    main_window.after(1, predict)
def btn_predict():
    global ispredict
    global btn
    ispredict=not ispredict
    btn['text'] = "停止检测" if ispredict else "开始检测"
    if ispredict:
        predict()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Can't open camera!")
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
main_window = tk.Tk()
main_window.title("行人检测")
main_window.geometry("%dx%d"%(frame_width*1.8, frame_height))
main_window.configure(bg="#FFFFFF")
main_window.resizable(0, 0)
canvas = tk.Canvas(main_window, width=frame_width, height=frame_height, bg="white")
canvas.place(x=0, y=0)
main_window.update()
strinfo=tk.StringVar()
info = tk.Label(main_window, bg="#FFFFFF", textvariable=strinfo, font=('consolas',12), justify="left")
info.place(x=frame_width, y=0)

btn = tk.Button(text="开始检测", command=btn_predict)
btn.place(x=frame_width*1.35, y=frame_height*0.8)

main_window.mainloop()
cap.release()
