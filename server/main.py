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
from face_analysis import FaceAnalysis
from clustering import cluster_raw_faces
from database import FaceDatabase
from search import bruteforce, unique_people_search
# from utils import good_head_angle


class Miris(BaseModel):
    unique_faces: list
    raw_faces: list


# server modules
app = FastAPI()

# Vision modules
face_embed = ArcFace()
face_analyze = FaceAnalysis()

# database modules
face_database = FaceDatabase()


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
    refined_unique_faces = cluster_raw_faces(feats, item.raw_faces)
    # TODO: test the func below
    uploaded_unique_faces = face_embed.extend_inference(item.unique_faces)
    # print(uploaded_unique_faces[0]['person'][0]['feat'].shape)
    # database search (both refined unique faces and upload unique faces)
    unique_people = unique_people_search(
        uploaded_unique_faces,
        refined_unique_faces,
        face_database,
        0.6 # TODO: move this threshold into config file
    )

    # # face analysis for new people
    # unique_people = face_analyze.inference(unique_people, new_only=True)

    # # log unique people 
    # FaceDatabase.newPeopleLog(unique_people)

    # # extend face database with unidentified people
    # FaceDatabase.newFaces(unique_people)

    item_dict = {
        'total_upload_unique_faces': len(item.unique_faces),
        'total_upload_raw_faces': len(item.raw_faces),
        'arcface feats': feats.shape,
        'total_unique_faces': len(refined_unique_faces),
    }
    return item_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
