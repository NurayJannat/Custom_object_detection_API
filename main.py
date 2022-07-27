from urllib import response
from fastapi import FastAPI
import uvicorn
import os
from pydantic import BaseModel
import cv2
import numpy as np

from utils.util_fun import base64_to_image, image_to_base64

from process.detect import detect
from process.list_obj import object_lists
from process.draw_gt_box import draw
from typing import List

app = FastAPI()


class Image_INFO(BaseModel):
    img_base64: str

class Image_INFO_PROC(BaseModel):
    img_base64: str
    x1: int
    y1: int
    x2: int
    y2: int

class CordBase(BaseModel):
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    # score: float

class CordList(BaseModel):
    obj_cords: List[CordBase]

class ReposponseBody(BaseModel):
    is_success: bool
    output_img: str
    obj_cords: List[CordList]


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/object-detection')
def object_detection(image_info: Image_INFO, response_model=ReposponseBody):
    img_base64 = image_info.img_base64
    img = base64_to_image(img_base64)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # encoded_str = "data:image/jpeg;base64,/9j/" + str(encoded_str)
    # print(img.shape)

    boxes, scores = detect(img)

    # print(boxes.shape)
    # print(scores.shape)
    result, score = object_lists(boxes, scores, img.shape)

    print("result: ", result)

    mask = draw(img, result, score)
    # cv2.imwrite("make.jpg", mask)
    encoded_str = image_to_base64(mask)

    if result:
        response_body = {
            'is_success': True,
            "output_img": encoded_str,
            "obj_cords": result

        }
    return response_body


@app.post('/post_proc')
def post_proc(image_info: Image_INFO_PROC):
    img_base64 = image_info.img_base64
    x1 = image_info.x1
    y1 = image_info.y1
    x2 = image_info.x2
    y2 = image_info.y2

    img = base64_to_image(img_base64)

    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)

    # print(img.shape)

    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    # cv2.imwrite("make.jpg", mask)
    out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)

    # cv2.imwrite("./out.png", out)

    encoded_str = image_to_base64(out)
    # encoded_str = "data:image/jpeg;base64,/9j/" + str(encoded_str)

    return {"output_base64": encoded_str}



if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)