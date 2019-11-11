"""
Miris central processing server using:
 - FastAPI as ASWG server
 - TensorRT as face recognition framework
 - MongoDB for logging essential data
 - FaceSearch with SPTAG
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
from search_ANN import ANN
from head_pose_est import HeadPoseEst
from face_detection_naive import FaceDetection
from utils import good_head_angle


class Miris(BaseModel):
    unique_faces: list
    raw_faces: list
    time: float


# server modules
app = FastAPI()

# Vision modules
face_embed = ArcFace()
face_detect = FaceDetection()
age_gender_est = AgeGenderEstimator()
headpose = HeadPoseEst()

# database modules
FDC = FaceDatabase()
vector_db = ANN()
# face_list = FDC.loadFaces()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/face/log/")
async def log_faces(item: Miris):
    logging.info(f"unique faces: {len(item.unique_faces)}, raw_faces: {len(item.raw_faces)}")
    # print("TIME: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("UPLOAD TIME: ", item.time)
    print("RECEIVE TIME: ", time.time())
    # process raw faces into unique faces
    feats = []
    raw_faces = []
    s = time.time()
    # remove bad data from raw faces
    for i in range(len(item.raw_faces)):
        face_bin = base64.b64decode(item.raw_faces[i]['face'])
        face_stream = io.BytesIO(face_bin)
        face_cv = cv2.imdecode(np.fromstring(
            face_stream.read(), np.uint8), 1)
        raw_faces.append(face_cv)
        item.raw_faces[i]['face'] = face_cv

    print("good face before", len(raw_faces))
    raw_faces, item.raw_faces = headpose.remove_bad_pose(raw_faces, item.raw_faces)
    print("good face after", len(raw_faces))
    raw_faces, item.raw_faces = face_detect.multi_inference(raw_faces, item.raw_faces)
    print("decode time", time.time() - s)
    s = time.time()
    feats = face_embed.inference(raw_faces)
    print("arcface time", time.time() - s)
    refined_unique_faces = cluster_raw_faces(feats, item.raw_faces, len(feats))
    s = time.time()
    # database search 
    known_people, new_people = vector_db.face_search(refined_unique_faces)
    print("search time", time.time() - s)

    print("known people", len(known_people))
    print("new people", len(new_people))

    # # face analysis for new people
    new_people = age_gender_est.extend_inference(new_people)
    if len(new_people):
        print("age", new_people[0]['age'])
        print("gender", new_people[0]['gender'])

    # # extend face database with unidentified people
    FDC.addNewFacesTree(new_people, vector_db)
    # TODO: update old customer with new faces, based on some criteria
    # FDC.updateKnownFaces(known_people)

    # # log unique people 
    FDC.newPeopleLog(known_people, new_people)


    item_dict = {
        'total_upload_raw_faces': len(item.raw_faces),
        'known_people': len(known_people),
        'new people': len(new_people),
    }
    # time.sleep(4)
    return item_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
