"""
a protoptype application of the FQC (face-queue-clustering) algorithm
4 main parts: face detection, face recognition, task scheduler, face clustering

"""
import cv2
import numpy as np
import configparser
import os
import ast
import time
import base64
# from bson import ObjectId
# from pymongo import MongoClient
from datetime import datetime
from arcface import FaceEmbedding
from face_det import FaceDetection
from clustering import face_clustering
from utils import b64_encode, face_marginalize
from xnet import Xnet
# from head_pose_est import HeadPoseEst
# from face_search import bruteforce


class FaceQueueClustering(object):
    def __init__(self, parent=None):
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.config = config
        self.face_detect = FaceDetection(config)
        self.face_embed = FaceEmbedding(config)
        # standby phase, delayed face detection
        self.standby_detection_delay = float(config["TASK_SCHEDULER"]['Standby_detection_delay'])
        self.standby_max_analysis = int(config["TASK_SCHEDULER"]['Standby_max_analysis'])
        self.clustering_upload_delay = float(config["TASK_SCHEDULER"]['Clustering_upload_delay'])
        self.max_img_per_person = int(config["UPLOAD"]['Max_img_per_person'])
        self.face_lap_min_var = float(config["FACE_CONSOLIDATION"]['Laplacian_min_variance'])
        self.face_margin_w_ratio = float(config["FACE_CONSOLIDATION"]['Face_margin_w_ratio'])
        self.face_margin_h_ratio = float(config["FACE_CONSOLIDATION"]['Face_margin_h_ratio'])
        self.xnet = Xnet(config)
        self.unprocess_face_queue = []
        self.face_queue = []
        self.feat_queue = []
        # self.standby_mode = True
        self.last_detected_face_time = time.time()
        self.last_clustering_time = time.time()
        # self.num_cams = int(config["CAMERA"]['Num_cams'])
        # self.cam_ids = ast.literal_eval(config["CAMERA"]['Id'])

    def serve(self):
        # TODO: multi-cam central processing
        # cams = [
        #     cv2.VideoCapture(i)
        #     for i in self.cam_ids
        # ]

        cam = cv2.VideoCapture(0)
        cam.set(3, 1920)
        cam.set(4, 1080)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        while(True):
            ret, frame = cam.read()
            print(frame.shape)
            if not ret:
                # dead cam
                cam = cv2.VideoCapture(0)
                time.sleep(3.000) # some delay to init cam
                # cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
                continue
            if (time.time() - self.last_detected_face_time) > self.standby_detection_delay:
                # TODO: clustering and upload mode
                st = time.time()
                self.standby_serve(frame)
                print("standby", time.time() - st)
            else:
                st = time.time()
                self.active_serve(frame)
                print("active", time.time() - st)
            if (time.time() - self.last_clustering_time) > self.clustering_upload_delay:
                st = time.time()
                self.cluster_upload()
                print("clustering", time.time() - st)

        return 0

    def standby_serve(self, frame):
        """
        1 face detection + n face analysis from queue
        """
        self.process_queue()
        self.detect_queue(frame)
        return 0

    def active_serve(self, frame):
        """
        1 face detection, save face to memory
        TODO: memory management, e.g. save to disk before OOM
        """
        self.detect_queue(frame)
        return 0

    def detect_queue(self, frame):
        face_bboxes = self.face_detect.inference(frame)
        if len(face_bboxes) == 0:
            return 1
        # TODO: blurry analysis, head angle estimate
        self.last_detected_face_time = time.time()
        # print(face_bboxes)
        for b in face_bboxes:
            if (b[3] > b[1]) and (b[2] > b[0]):
                crop = frame[b[1]:b[3], b[0]:b[2], :]
                # detect blurr, headpose est
                # TODO: head pose est, move to separate function/class
                blur_face = cv2.resize(crop, (112, 112))
                blur_face_var = cv2.Laplacian(blur_face, cv2.CV_64F).var()
                if blur_face_var > self.face_lap_min_var:
                    self.unprocess_face_queue.append(
                        {
                            'crop': crop,
                            'time': self.last_detected_face_time
                        }
            )
        return 0

    def process_queue(self):
        """
        face embed, age, gender recognition
        TODO: memory management
        """
        for i in range(self.standby_max_analysis):
            if len(self.unprocess_face_queue) == 0:
                break
            face = self.unprocess_face_queue.pop(0)
            # print(face)
            input_face = face_marginalize(face['crop'], self.face_margin_w_ratio, self.face_margin_h_ratio)
            face_feature = self.face_embed.inference(input_face)
            self.face_queue.append(
                {
                    'crop': face['crop'],
                    'time': face['time'],
                    # 'feat': face_feature
                }
            )
            self.feat_queue.append(face_feature)
        return 0

    def cluster_upload(self):
        """
        cluster face queue into different people
        send several imgs and info per person to server
        # TODO: better time handling (timezone, cam location based, etc)
        """
        if len(self.feat_queue) == 0:
            print("no human")
            self.last_clustering_time = time.time()
            return 1
        print("cluster size", len(self.feat_queue), len(self.face_queue))
        labels = face_clustering(self.feat_queue)
        class_ids = np.unique(labels)
        unique_faces = []
        print("unique ", class_ids)
        print(labels)
        for cli in class_ids:
            # noise
            if cli == -1:
                continue
            if len(labels) > 1:
                cli_feat_ids = np.asarray(np.where(labels==cli))
                cli_feat_ids = np.squeeze(cli_feat_ids)
                sample_size = cli_feat_ids.shape[0]
                # print(sample_size, self.max_img_per_person)
                # TODO handle sample_size < max_img_per_person
                num_upload_imgs = min(self.max_img_per_person, sample_size)
                chosen_ids = np.unique(
                    np.random.choice(
                        sample_size,
                        num_upload_imgs,
                        replace=False
                    )
                )
            else:
                cli_feat_ids = np.asarray([0])
                chosen_ids = np.asarray([0])
            unique_faces.append(
                {
                    'faces': [
                        b64_encode(self.face_queue[cli_feat_ids[i]]['crop'])
                        for i in chosen_ids
                    ],
                    'time': [
                        self.face_queue[i]['time']
                        for i in cli_feat_ids
                    ]
                }
            )
        print("num of unique people: ", len(unique_faces))

        self.xnet.log_face(unique_faces)
        # cleanup garbage
        self.feat_queue = []
        self.face_queue = []
        self.last_clustering_time = time.time()
        return 0


if __name__ == '__main__':
    fqc = FaceQueueClustering()
    fqc.serve()
