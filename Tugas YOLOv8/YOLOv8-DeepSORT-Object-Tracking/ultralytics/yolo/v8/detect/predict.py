# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import time
from pathlib import Path
import cv2
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque, defaultdict
import numpy as np

# Constants & Globals
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
line_y = 400
vehicle_counter = {'in': defaultdict(int), 'out': defaultdict(int)}
track_history = {}
prev_centers = {}
FPS = 30
PIXEL_TO_METER = 0.05

deepsort = None


def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )


def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    return x_c, y_c, bbox_w, bbox_h


def compute_color_for_labels(label):
    if label == 0:
        return (85, 45, 255)
    elif label == 2:
        return (222, 82, 175)
    elif label == 3:
        return (0, 204, 255)
    elif label == 5:
        return (0, 149, 255)
    else:
        return tuple([int((p * (label ** 2 - label + 1)) % 255) for p in palette])


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf)
    return img


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    cv2.line(img, (0, line_y), (img.shape[1], line_y), (0, 255, 0), 2)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]; x2 += offset[0]; y1 += offset[1]; y2 += offset[1]
        center = (int((x1 + x2) / 2), int((y2 + y2) / 2))
        cy = center[1]
        id = int(identities[i]) if identities is not None else 0
        class_id = object_id[i]
        class_name = names[class_id]

        # Speed estimation
        speed_kmh = 0
        if id in prev_centers:
            dx = center[0] - prev_centers[id][0]
            dy = center[1] - prev_centers[id][1]
            dist = np.sqrt(dx**2 + dy**2)
            speed_kmh = dist * PIXEL_TO_METER * FPS * 3.6
        prev_centers[id] = center

        # Counting logic
        prev_y = track_history.get(id, None)
        if prev_y is not None:
            if prev_y < line_y and cy >= line_y:
                vehicle_counter['in'][class_name] += 1
            elif prev_y > line_y and cy <= line_y:
                vehicle_counter['out'][class_name] += 1
        track_history[id] = cy

        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        data_deque[id].appendleft(center)

        color = compute_color_for_labels(class_id)
        label = f"{id}:{class_name} {int(speed_kmh)}km/h"
        UI_box(box, img, label=label, color=color, line_thickness=2)

        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
            cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, thickness)

    # Display counter
    y_start = 30
    cv2.putText(img, "Vehicles Leaving", (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y = y_start + 30
    for cls, count in vehicle_counter['out'].items():
        cv2.putText(img, f"{cls}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 25

    cv2.putText(img, "Vehicles Entering", (width - 300, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    y = y_start + 30
    for cls, count in vehicle_counter['in'].items():
        cv2.putText(img, f"{cls}: {count}", (width - 300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y += 25

    return img


class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        frame = self.dataset.count if self.webcam else getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        if len(det) == 0:
            return

        xywh_bboxs, confs, oids = [], [], []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
            confs.append([conf.item()])
            oids.append(int(cls))

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        outputs = deepsort.update(xywhs, confss, oids, im0)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

        return f"{len(outputs)} tracks"



@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
