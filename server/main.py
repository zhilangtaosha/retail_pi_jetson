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
from datetime import datetime
from arcface import ArcFace
from age_gender import AgeGenderEstimator
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
age_gender_est = AgeGenderEstimator()

# database modules
FDC = FaceDatabase()
# face_list = FDC.loadFaces()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/face/log/")
async def log_faces(item: Miris):
    logging.info(f"unique faces: {len(item.unique_faces)}, raw_faces: {len(item.raw_faces)}")
    print("TIME: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # process raw faces into unique faces
    feats = []
    raw_faces= []
    s = time.time()
    # for r_face in item.raw_faces:
    for i in range(len(item.raw_faces)):
        face_bin = base64.b64decode(item.raw_faces[i]['face'])
        face_stream = io.BytesIO(face_bin)
        face_cv = cv2.imdecode(np.fromstring(
            face_stream.read(), np.uint8), 1)
        raw_faces.append(face_cv)
        item.raw_faces[i]['face'] = face_cv
        # yaw, pitch, roll = self.head_pose.inference(face_cv)
    print("decode time", time.time() - s)
    s = time.time()
    feats = face_embed.inference(raw_faces)
    print("arcface time", time.time() - s)
    refined_unique_faces = cluster_raw_faces(feats, item.raw_faces)
    # TODO: test the func below
    s = time.time()
    uploaded_unique_faces = face_embed.extend_inference(item.unique_faces)
    print("upload face inference time", time.time() - s)
    s = time.time()
    # print(uploaded_unique_faces[0]['person'][0]['feat'].shape)
    # database search (both refined unique faces and upload unique faces)
    known_people, new_people = unique_people_search(
        uploaded_unique_faces,
        refined_unique_faces,
        # face_list,
        FDC.loadFaces(),
        0.6 # TODO: move this threshold into config file
    )
    print("search time", time.time() - s)

    print("known people", len(known_people))
    print("new people", len(new_people))

    # # face analysis for new people
    new_people = age_gender_est.extend_inference(new_people)
    if len(new_people):
        print("age", new_people[0]['age'])
        print("gender", new_people[0]['gender'])

    # # extend face database with unidentified people
    FDC.addNewFaces(new_people)

    # # log unique people 
    # FaceDatabase.newPeopleLog(known_people, new_people)


    item_dict = {
        'total_upload_unique_faces': len(item.unique_faces),
        'total_upload_raw_faces': len(item.raw_faces),
        'arcface feats': feats.shape,
        'total_unique_faces': len(refined_unique_faces),
    }
    return item_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
