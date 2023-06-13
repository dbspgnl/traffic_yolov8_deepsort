# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import pprint

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
from collections import defaultdict
import numpy as np
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
# data_deque = deque([], 64)
json_data = {}
# json_data = deque([], 64)
video_second_frame = 30

deepsort = None

object_counter = {} # leave car
object_counter1 = {} # enter car
area1 = [(40, 520), (5720, 630)] # meabong 검출 영역1 (주 도로)
# area1 = [(40, 520), (4400, 630)] # meabong 검출 영역1 (주 도로) # show 짤려서 임시
area2 = [(2040, 520), (5000, 660)] # meabong 검출 영역2 (합류 도로)
# area2 = [(2040, 520), (4355, 660)] # meabong 검출 영역2 (합류 도로) # show 짤려서 임시
detect_area = [area1, area2] # area1 & area2


def estimatespeed(Location1, Location2):
    #Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    ppm = 7.8 
    d_meters = d_pixel/ppm # defining thr pixels per meter
    time_constant = video_second_frame*3.6 # km/h
    speed = d_meters * time_constant #distance = speed/time
    return int(speed)

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label): # 라벨 색상 변경
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: # car
        color = (13,138,255)
    elif label == 1: # trcuk
        color = (255,156,93)
    elif label == 2:  # bus
        color = (1, 174, 124)
    elif label == 3:  # vehicle
        color = (0, 235, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d): # 라벨(그림 영역) 그리기
    x1,y1 = pt1
    x2,y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2 - d - 2), color, -1, cv2.LINE_AA) # -2 줄이면 올라감
    return img

def UI_box(x, img, color=None, label=None, label_speed=None, label_id=None, line_thickness=None):
    # Plots one bounding box on image img
    img_origin = img
    tl = 0.0005 * (img.shape[0] + img.shape[1]) / 2 # 0.75
    color = color or [random.randint(0, 255) for _ in range(3)] # 컬러링
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) # 센터
    label_speed = " " + str(label_speed) + "km"
    label_id = '[ %d ]' % (label_id)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_label_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0] # 라벨 영역 사이즈
        t_id_size = cv2.getTextSize(label_id, 0, fontScale=tl / 3, thickness=tf)[0] # id 영역 사이즈
        img = draw_border(img, (c1[0], c1[1] - t_label_size[1] -3), (c1[0] + t_label_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img_origin, label_id, (c1[0] + t_label_size[0], c1[1] - 4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img_origin, label_speed, (c1[0] + t_label_size[0] + t_id_size[0], c1[1] - 4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # putText(그림, 글자, 글자 시작점, 폰트, 글자 크기 비율, 글자 색상, 글자 굵기, 글자 선형태, 그림 좌표)


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str


def isInDetectArea(center):
    result = 0
    detect_area_len = len(detect_area)
    for i in range(detect_area_len):
        if (detect_area[i][0][0] < center[0] < detect_area[i][1][0]) and (detect_area[i][0][1] < center[1] < detect_area[i][1][1]):
            result +=1
    return True if result > 0 else False # 최소 하나라도 포함되면 in


def get_speed(positions): # 100 200 110 210 110 210 120 220
    if len(positions) < 2: return 0
    try:
        json_x1 = (positions[-4:][0]+positions[-4:][2])//2 # 마지막 x평균
        json_x2 = (positions[-8:-4][0]+positions[-8:-4][2])//2 # 마지막 바로전 x평균
        json_y1 = (positions[-4:][1]+positions[-4:][3])//2 # 마지막 y평균
        json_y2 = (positions[-8:-4][1]+positions[-8:-4][3])//2 # 마지막 바로전 y평균
        return estimatespeed((json_x1, json_y1), (json_x2, json_y2))
    except:
        return 0
        

def draw_boxes(frame, img, bbox, names, object_id, identities=None, offset=(0, 0)):
    # 검출 영역
    for area in detect_area:
        cv2.rectangle(img, area[0], area[1], [128, 255, 128], 1, cv2.LINE_AA)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(json_data):
        if key in json_data and frame > json_data[key]["now"] + 60:#60프레임이 지날 때까지도 나타나지 않으면 그땐 지우기
            if key not in identities:
                json_data.pop(key) # 제거하면 lost 되서 더 이상 찾을 수가 없음
            

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((x1+x2)/ 2), int((y1+y2)/2))
        id = int(identities[i]) if identities is not None else 0 # get ID of object
        if id not in json_data: # create deque by id 
            # data_deque[id] = deque(maxlen= 64)
            json_data[id] = {
                "id": id,
                "first": frame, 
                "now": frame, 
                "nows": [], 
                "centers": [], 
                "position": [], 
                "during": 0, 
                "class": object_id[i], 
                "label": names[object_id[i]],
                "speed": 0
            }
        # add center to buffer
        # data_deque[id].appendleft(center)
        json_data[id]["now"] = frame
        json_data[id]["nows"].append(frame)
        json_data[id]["position"].extend([x1, y1, x2, y2])
        json_data[id]["during"] = (frame - json_data[id]["first"])
        json_data[id]["centers"].append(center)
        if frame % video_second_frame == 0:
            json_data[id]["speed"] = get_speed(json_data[id]["position"])

    for k,v in json_data.copy().items():
        if frame not in v["nows"]: # nows 연장
            v["nows"].append(frame)
        color = compute_color_for_labels(v["class"])

        # # 진입 조건
        if not isInDetectArea(v["centers"][-1]): # 영역 밖
            if k in json_data: # 근데 데이터리스트에 있으면 지워줌
                json_data.pop(k)
            continue

        # 범위 내 겹친 데이터가 있으면 하나로 처리
        # if v and (k is not id) and abs(v[0][0] - center[0]) < 20 and abs(v[0][1] - center[1]) < 20:
        #     id = k

        if frame == v["now"]: # 현재 프레임과 일치하는 데이터만 라벨링
            UI_box(v["position"][-4:], img, label=v["label"], label_speed=v["speed"], label_id=v["id"], color=color, line_thickness=None)

    return img


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(frame, im0, bbox_xyxy, self.model.names, object_id, identities)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
