import os
import json
import requests
import sys

from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
import numpy as np

import threading
from PIL import ImageDraw, ImageFont

def add_text_to_left_center(image, text="操作员侧", font_path="./simsun.ttc", color="red"):
    draw = ImageDraw.Draw(image)
    font_size = int(image.height * 0.05)
    font = ImageFont.truetype(font=font_path, size=font_size)
    textwidth, textheight = draw.textsize(text, font)
    x = 100
    y = (image.height - textheight) / 2
    draw.text((x, y), text, fill=color, font=font)
    return image

def add_text_to_right_center(image, text="操作员侧", font_path="./simsun.ttc", color="red"):
    draw = ImageDraw.Draw(image)
    font_size = int(image.height * 0.05)
    font = ImageFont.truetype(font=font_path, size=font_size)
    textwidth, textheight = draw.textsize(text, font)
    x = image.width - textwidth - 100
    y = (image.height - textheight) / 2
    draw.text((x, y), text, fill=color, font=font)
    return image

class ResourcePool:
    def __init__(self, initial_resources):
        self.resources = initial_resources
        self.lock = threading.Lock()

    def get_resource(self):
        with self.lock:
            if not self.resources:
                self.resources.append((YOLO("./best.pt"), YOLO("./best_allocate.pt")))
            return self.resources.pop()

    def return_resource(self, resource):
        with self.lock:
            self.resources.append(resource)

def draw_mask(image, mask_generateds, nms_index):
    mask_with_edges = np.zeros_like(image)
    for index, mask_generated in enumerate(mask_generateds):
        if index not in nms_index:
            continue
        mask_generated = mask_generated.numpy()
        mask_resized = cv2.resize(mask_generated.astype(np.float32), (image.shape[1], image.shape[0]))
        _, binary_mask = cv2.threshold(mask_resized, 0.5, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_with_edges, contours, -1, (0, 0, 255), thickness=3)
    highlighted_image = cv2.addWeighted(image, 0.8, mask_with_edges, 1.0, 0)
    return highlighted_image

def getRectangularArea(masks_data):
    return masks_data.sum().item()

# 更新 isInclusion 函数
def isInclusion(wspot_xyxy, rectangular_xyxy, mode):
    if wspot_xyxy == None or rectangular_xyxy == None:
        return False

    res = False
    x1, y1, x2, y2 = wspot_xyxy
    x3, y3, x4, y4 = rectangular_xyxy
    if (x3<=x1<=x4) and (y3<=y1<=y4):
        wspot_x, wspot_y = (x1 + x2) / 2, (y1 + y2) / 2
        wspot_x_per, wspot_y_per = (wspot_x - x3) / (x4 - x3), (wspot_y - y3) / (y4 - y3)
        if mode == 'M':
            area_dict = [
                {'A3': [0, 3/6], 'B3': [3/6, 6/6]},
                {'C3': [0, 2/6], 'D3': [2/6, 1/2], 'E3': [1/2, 4/6], 'F3': [4/6, 6/6]},
                {'G3': [0, 1/6], 'H3': [1/6, 2/6], 'J3': [2/6, 3/6], 'K3': [3/6, 4/6], 'L3': [4/6, 5/6], 'M3': [5/6, 6/6]}
            ]
        elif mode == 'L':
            area_dict = [
                {'A3': [0, 3/3]},
                {'C3': [0, 2/3], 'D3': [2/3, 3/3]},
                {'G3': [0, 1/3], 'H3': [1/3, 2/3], 'J3': [2/3, 3/3]}
            ]
        elif mode == 'R':
            area_dict = [
                {'B3': [0, 3/3]},
                {'E3': [0, 1/3], 'F3': [1/3, 3/3]},
                {'K3': [0, 1/3], 'L3': [1/3, 2/3], 'M3': [2/3, 3/3]}
            ]
        area_dict_index = 0
        if wspot_y_per <= 0.3:
            area_dict_index = 0
        elif wspot_y_per > 0.6:
            area_dict_index = 2
        else:
            area_dict_index = 1

        for key, val in area_dict[area_dict_index].items():
            if val[0] <= wspot_x_per <= val[1]:
                res = key
                break
    return res

def getAreaDict(r):  # 修改整个函数
    res = {
        'TOP': {
            'area': 23.5,
            'rectangular_area': None,
            'xyxy': None,
            'conf': 0
        },
        'MIDDLE': {
            'area': 19.9,
            'rectangular_area': None,
            'xyxy': None,
            'conf': 0
        },
        'BOTTOM': {
            'area': 28.8,
            'rectangular_area': None,
            'xyxy': None,
            'conf': 0
        }
    }
    for index, box in enumerate(r.boxes):
        box_key = r.names[box.cls.item()]
        rectangular_xyxy = box.xyxy.numpy().tolist()[0]
        if res[box_key]['conf'] < box.conf.item():
            res[box_key]['conf'] = box.conf.item()
            res[box_key]['rectangular_area'] = getRectangularArea(r.masks.data[index])
            res[box_key]['xyxy'] = rectangular_xyxy
    return res

res = []
for _ in range(1):
    res.append((YOLO("./best.pt"), YOLO("./best_allocate.pt")))
rsp = ResourcePool(res)

def getWspotArea(image, mode):
    model_wspot, model_allocate = rsp.get_resource()
    if 'gpu' in sys.argv:
        result_wspot = model_wspot(image, imgsz=2560, device='cuda', nms=True)[0]
        result_allocate = model_allocate(image, imgsz=1280, device='cuda')[0]
    else:
        result_wspot = model_wspot(image, imgsz=2560, device='cpu', nms=True)[0].to('cpu')
        result_allocate = model_allocate(image, imgsz=1280, device='cpu')[0]
    rsp.return_resource((model_wspot, model_allocate))

    area_dict = getAreaDict(result_allocate)

    res_area = []
    res_region = []
    xy_list = []
    for index1, box in enumerate(result_wspot.boxes):
        xy_list.append(box.xyxy.cpu().numpy().tolist()[0])
    nms_index = filter_boxes_custom_nms(xy_list)
    for index, box in enumerate(result_wspot.boxes):
        if index not in nms_index:
            continue
        wspot_xyxy = box.xyxy.cpu().numpy().tolist()[0]
        for key, val in area_dict.items():
            area_index = isInclusion(wspot_xyxy, val['xyxy'], mode)
            if area_index:
                wspot_area = 0.25 * getRectangularArea(result_wspot.masks.data[index]) / val['rectangular_area'] * 0.87 # 修改此处
                if mode != 'M':
                    wspot_area /= 2
                res_area.append(wspot_area)
                res_region.append(area_index)
                break

    # 合成图像
    if result_allocate.masks is not None:
        image_array = draw_mask(result_wspot.orig_img, result_wspot.masks.data, nms_index)
    else:
        image_array = result_wspot.plot(labels=False, boxes=True)
    image = Image.fromarray(image_array[..., ::-1])
    return res_area, res_region, image

# nms
def calculate_iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    iou = intersection_area / float(min(box1_area, box2_area))
    return iou

def filter_boxes_custom_nms(boxes, threshold=0.3):
    keep_indices = []
    boxes = np.array(boxes)

    for i in range(len(boxes)):
        keep = True
        for j in range(len(boxes)):
            if i != j:
                overlap = calculate_iou(boxes[i], boxes[j])
                if overlap > threshold:
                    # If current box is smaller than the other overlapping box, discard it
                    area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                    area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                    if area_i < area_j:
                        keep = False
                        break
        if keep:
            keep_indices.append(i)

    return keep_indices

app = Flask(__name__)

@app.route('/process_json', methods=['GET', 'POST'])
def process_json():
    json_post = json.loads(request.get_data())
    image_path = json_post['image_path']

    res_area, res_region, image_yolo = getWspotArea(image_path, json_post['mode'])
    if json_post['operator'] == 0:
        image_yolo = add_text_to_left_center(image_yolo)
    else:
        image_yolo = add_text_to_right_center(image_yolo)

    # 随机一个 filename
    folder_path, file_name = os.path.split(image_path)
    file_name_without_ext, file_ext = os.path.splitext(file_name)
    new_file_name = file_name_without_ext + '_result' + file_ext
    new_file_path = os.path.join(folder_path, new_file_name)
    image_yolo.save(new_file_path)

    result = {
        'area': res_area, # 面积
        'region': res_region, # 区域
        'image_path': new_file_path
    }

    json_response = json.dumps(result)
    return json_response, 200, {"Content-Type":"application/json"}

@app.route('/', methods=['GET'])
def index():
    # 构建测试 json 包
    # image_base64 = encode_image_base64('./test.jpg')
    image_path = '/data/yaoyulin/model/flask/test.jpg'
    data = {
        'image_path': image_path,
        'mode': 'M',
        'operator': 0,
    }
    json_post = json.dumps(data)
    response = requests.post(f'http://127.0.0.1:20000/process_json', data=json_post)

    # 拿到回传结果解析图像
    json_result = json.loads(response.text)
    print(json_result)
    return 'hello word'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=20000, debug=True)