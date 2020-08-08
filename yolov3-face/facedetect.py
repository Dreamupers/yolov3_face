import argparse
from models import *
from utils.datasets import *
from utils.utils import * #import cv2, import numpy as np, import torch


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/face_1.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/face.names', help='*.names path')
parser.add_argument('--weights', type=str, default='weights/yolov3-wider_16000.pt', help='weights path')
parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--source', type=str, default='./test.jpg', help='source')  # input file/folder, 0 for webcam

opt = parser.parse_args()
print(opt)
img_size, weights, cfg, source = opt.img_size, opt.weights, opt.cfg, opt.source
device = torch_utils.select_device(opt.device)
names = load_classes(opt.names)
colors = [[0, 255, 0]]
model = Darknet(cfg, img_size)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()
img0 = cv2.imread(source)
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
    pred = model(img)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss, ' % (n, names[int(c)]))
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img0, label="", color=colors[int(cls)])
#imgcv = cv2.cvtColor(img0, cv2.COLOR_BGR2RGBA)
cv2.imwrite('./output/' + source.split('/')[-1], img0)



