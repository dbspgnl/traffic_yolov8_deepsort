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

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
json_data = {}
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
    # defining thr pixels per meter
    # ppm = 8
    ppm = 9.6 # 1m : 9.6pixel
    # frame = 30
    d_meters = d_pixel/ppm
    time_constant = video_second_frame*3.6 # km/h
    #distance = speed/time
    speed = d_meters * time_constant

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
    # tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    img_origin = img
    tl = 0.0005 * (img.shape[0] + img.shape[1]) / 2 # 0.75
    color = color or [random.randint(0, 255) for _ in range(3)] # 컬러링
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) # 센터
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_label_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0] # 라벨 영역 사이즈
        t_id_size = cv2.getTextSize(label_id, 0, fontScale=tl / 3, thickness=tf)[0] # id 영역 사이즈
        
        img = draw_border(img, (c1[0], c1[1] - t_label_size[1] -3), (c1[0] + t_label_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img_origin, label_id, (c1[0] + t_label_size[0], c1[1] - 4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img_origin, label_speed, (c1[0] + t_label_size[0] + t_id_size[0], c1[1] - 4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # putText(그림, 글자, 글자 시작점, 폰트, 글자 크기 비율, 글자 색상, 글자 굵기, 글자 선형태, 그림 좌표)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


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


def draw_boxes(frame, img, bbox, names, object_id, identities=None, offset=(0, 0)):
    # 검출 영역
    for area in detect_area:
        cv2.rectangle(img, area[0], area[1], [128, 255, 128], 1, cv2.LINE_AA)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities and abs(frame - json_data[key]["now"]) > 60:
            # print(json_data[key]["now"])
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        # 좌표
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((x1+x2)/ 2), int((y1+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # 진입 조건
        if not isInDetectArea(center): # 영역 밖
            if id in data_deque: # 근데 데이터리스트에 있으면 지워줌
                data_deque.pop(id)
            continue

        # 범위 안에 데이터가 있으면 그걸로 처리
        for k,v in data_deque.items():
            if v and abs(v[0][0] - center[0]) < 20 and abs(v[0][1] - center[1]) < 20:
                id = k

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen= 64)
            json_data[id] = {
                "first": frame, 
                "now": frame, 
                "position": [], 
                "during": 0, 
                "class": object_id[i], 
                "label": names[object_id[i]],
                "speed": "0km"
            }

        # 만약 frame이 현재랑 똑같으면 스킵
        if json_data[id]["now"] and frame == json_data[id]["now"]: continue   

        # add center to buffer
        data_deque[id].appendleft(center)
        json_data[id]["now"] = frame
        json_data[id]["position"].extend([x1, y1, x2, y2])
        json_data[id]["during"] = (frame - json_data[id]["first"])

        # 방향 측정
        # if len(data_deque[id]) >= 2:
        #     direction = get_direction(data_deque[id][0], data_deque[id][1])
        #     object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
        #     speed_line_queue[id].append(object_speed)
            # if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
            #     if "West" in direction:
            #         if obj_name not in object_counter:
            #             object_counter[obj_name] = 1
            #         else:
            #             object_counter[obj_name] += 1
            #     if "East" in direction:
            #         if obj_name not in object_counter1:
            #             object_counter1[obj_name] = 1
            #         else:
            #             object_counter1[obj_name] += 1
        
        # 라벨 작업
        color = compute_color_for_labels(json_data[id]["class"])
        obj_name = json_data[id]["label"]
        label_speed = ""
        label_id = '[ %d ]' % (id)
        try:
            json_x1 = (json_data[id]["position"][-4:][0]+json_data[id]["position"][-4:][2])//2 # 마지막 x평균
            json_x2 = (json_data[id]["position"][-8:-4][0]+json_data[id]["position"][-8:-4][2])//2 # 마지막 바로전 x평균
            json_y1 = (json_data[id]["position"][-4:][1]+json_data[id]["position"][-4:][3])//2 # 마지막 y평균
            json_y2 = (json_data[id]["position"][-8:-4][1]+json_data[id]["position"][-8:-4][3])//2 # 마지막 바로전 y평균
            check_spped = " " + str(estimatespeed((json_x1, json_y1), (json_x2, json_y2))) + "km"
            label_speed = ""

            if json_data[id]["speed"] == "0km": # 속도가 0이면 초기화해준다.
                label_speed = check_spped
                json_data[id]["speed"] = check_spped
            elif json_data[id]["speed"] is not check_spped: # 속도가 체크랑 다른데
                if frame % video_second_frame == 0: # 비디오 프레임 길이일 때(초당) 속도 갱신
                    label_speed = check_spped
                    json_data[id]["speed"] = check_spped
                else:
                    label_speed = json_data[id]["speed"]
                    
        except:
            pass
        UI_box(box, img, label=obj_name, label_speed=label_speed, label_id=label_id, color=color, line_thickness=None) # 라벨링 박스 그리기
    
        #4. Display Count in top right corner
        # for idx, (key, value) in enumerate(object_counter1.items()): # 영역 카운팅 목록이 있으면
        #     cnt_str = str(key) + ":" +str(value)
        #     cv2.line(img, (width - 500,25), (width,25), [85,45,255], 40)
        #     cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        #     cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
        #     cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)
    
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
