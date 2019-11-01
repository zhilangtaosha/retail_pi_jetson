"""
Miris central processing server using:
 - FastAPI as ASWG server
 - TensorRT as face recognition framework
 - MongoDB for logging essential data
 - FaceSearch with bruteforce/SPTAG (big data)
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging
import time
import json, base64, io
import cv2
import numpy as np
from arcface import ArcFace
from clustering import cluster_raw_faces
# from utils import good_head_angle


class Miris(BaseModel):
    unique_faces: list
    raw_faces: list


# server modules
app = FastAPI()

# Vision modules
face_embed = ArcFace()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/face/log/")
async def log_faces(item: Miris):
    logging.info(f"unique faces: {len(item.unique_faces)}, raw_faces: {len(item.raw_faces)}")
    # process raw faces into unique faces
    feats = []
    raw_faces= []
    s = time.time()
    for r_face in item.raw_faces:
        face_bin = base64.b64decode(r_face['face'])
        face_stream = io.BytesIO(face_bin)
        face_cv = cv2.imdecode(np.fromstring(
            face_stream.read(), np.uint8), 1)
        raw_faces.append(face_cv)
        # yaw, pitch, roll = self.head_pose.inference(face_cv)
    print("decode time", time.time() - s)
    s = time.time()
    feats = face_embed.inference(raw_faces)
    print("arcface time", time.time() - s)
    unique_faces = cluster_raw_faces(feats, item.raw_faces)

    item_dict = {
        'total_upload_unique_faces': len(item.unique_faces),
        'total_upload_raw_faces': len(item.raw_faces),
        'arcface feats': feats.shape,
        'total_unique_faces': len(unique_faces),
    }
    return item_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
