import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../external/repository/ildoonet-tf-pose-estimation')))

import io
from fastapi import FastAPI, File, UploadFile
import uvicorn
import face_recognition
from starlette.responses import StreamingResponse
import numpy as np
import cv2

from tf_pose.estimator import TfPoseEstimator, Human, BodyPart, CocoPart
from tf_pose.networks import get_graph_path

app = FastAPI()

@app.post("/image/process")
async def process_image(file: UploadFile = File(...)):
    # アップロードされた画像を読み込む
    image_bytes = await file.read()

    # PIL イメージを numpy 配列に変換
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # OpenCV で画像として読み込む
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 画像をリサイズ
    resized_image, w, h = resize_image(image, 656)

    # 顔のランドマークを取得
    face_landmarks_list = face_recognition.face_landmarks(resized_image)

    e = TfPoseEstimator(get_graph_path("cmu"), target_size=(w, h))
    humans: list[Human] = e.inference(resized_image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    face_landmarks = face_landmarks_list[0]
    human = humans[0]
    
    chin_points = face_landmarks['chin']
    p = chin_points[0]
    q = chin_points[16]
    r = chin_points[8]
    o = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    
    left = int(o[0] - (o[0] - p[0]) * 1.1)
    top = int(o[1] - (r[1] - o[1]) * 1.1)
    right = int(o[0] - (o[0] - q[0]) * 1.1)
    bottom = int(o[1] - (o[1] - r[1]) * 1.01)
    face_height = bottom - top
    face_width = right - left

    cropped_region = resized_image[top: bottom, left: right]
    rankle: BodyPart = human.body_parts.get(CocoPart.RAnkle.value)
    lankle: BodyPart = human.body_parts.get(CocoPart.LAnkle.value)

    ankle_y = -1
    if(rankle):
        ankle_y = rankle.y
    if(lankle and ankle_y < lankle.y):
        ankle_y = lankle.y

    body_left = right + 10
    body_right = body_left + right - left

    if(ankle_y != -1):
        body_bottom = int(top + (ankle_y * h - top) * 1.1)
        body_height = body_bottom - top
        toushin = body_height / face_height

        # cv2.rectangle(resized_image, (body_left, top), (body_right, body_bottom), (0, 0, 255), 2)
        for i in range(0, int(toushin)):
            resized_image[top + i * face_height: top + (i+1) * face_height, body_left: body_right] = cropped_region
            #cv2.rectangle(resized_image, (body_left, top + i * face_height), (body_right, top + (i+1) * face_height), (0, 0, 255), 2)

        last_top = top + int(toushin) * face_height
        last_bottom = body_bottom

        resized_image[last_top: last_bottom, body_left: body_right] = cropped_region[0:last_bottom - last_top, 0:face_width]
    # TfPoseEstimator.draw_humans(resized_image, humans)

    # 画像をバイト形式に変換
    _, buffer = cv2.imencode('.jpg', resized_image)
    io_buf = io.BytesIO(buffer)
    
    # 画像を返す
    return StreamingResponse(io_buf, media_type="image/jpeg")

def resize_image(image, max_size):
    height, width = image.shape[:2]
    if height > width:
        new_height = max_size
        new_width = int((max_size / height) * width)
    else:
        new_width = max_size
        new_height = int((max_size / width) * height)
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image, new_width, new_height

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="debug")
