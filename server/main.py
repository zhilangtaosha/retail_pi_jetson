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
        # bbs = face_detect.inference(face_cv)
        # face_crop = face_cv
        # h, w, _ = face_cv.shape
        # if len(bbs):
        #     print("found face: ", bbs)
        #     x0 = int(max(bbs[0][0], 0))
        #     y0 = int(max(bbs[0][1], 0))
        #     x1 = int(min(bbs[0][2], w))
        #     y1 = int(min(bbs[0][3], h))
        #     face_crop = face_cv[y0:y1, x0:x1, :]
        # else:
        #     print(i, "faces not found")
        # raw_faces.append(face_crop)
        raw_faces.append(face_cv)
        item.raw_faces[i]['face'] = face_cv
        # item.raw_faces[i].update(
        #     {
        #         'face_crop': face_crop
        #     }
        # )
        # yaw, pitch, roll = self.head_pose.inference(face_cv)
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
    FDC.newPeopleLog(known_people, new_people)


    item_dict = {
        'total_upload_unique_faces': len(item.unique_faces),
        'total_upload_raw_faces': len(item.raw_faces),
        'arcface feats': feats.shape,
        'total_unique_faces': len(refined_unique_faces),
    }
    # time.sleep(4)
    return item_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
