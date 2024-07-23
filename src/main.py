import io
from fastapi import FastAPI, File, UploadFile
import uvicorn
import face_recognition
from PIL import Image, ImageDraw
from starlette.responses import StreamingResponse
import numpy as np

app = FastAPI()

@app.post("/image/process")
async def process_image(file: UploadFile = File(...)):
    # アップロードされた画像を読み込む
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # PIL イメージを numpy 配列に変換
    image_np = np.array(image)

    # 顔のランドマークを取得
    face_landmarks_list = face_recognition.face_landmarks(image_np)

    # 各顔のランドマークをループして顎を描画
    draw = ImageDraw.Draw(image)
    for face_landmarks in face_landmarks_list:
        chin_points = face_landmarks['chin']
        p = chin_points[0]
        q = chin_points[16]
        r = chin_points[8]
        o = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        
        left = o[0] - (o[0] - p[0])  * 1.1
        top = o[1] - (r[1] - o[1]) * 1.1
        right = o[0] - (o[0] - q[0]) * 1.1
        bottom = o[1] - (o[1] - r[1]) * 1.01

        draw.rectangle(((left, top), (right, bottom)),
                       outline=(255, 0, 0), width=2)

    # 結果の画像をバイトストリームに変換
    img_byte_arr = io.BytesIO()
    image_format = image.format  # 元の画像のフォーマットを取得
    if image_format is None:
        image_format = "JPEG"  # フォーマットが取得できない場合はJPEGにフォールバック
    image.save(img_byte_arr, format=image_format)
    img_byte_arr.seek(0)

    # 結果の画像を表示する
    return StreamingResponse(img_byte_arr, media_type=f"image/{image_format.lower()}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="debug")
