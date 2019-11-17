"""
v1.0, main changes including:
 - periodical server upload, 
 - FQC only when hardware are free, 
 - change to camera max resolution in active mode / multi-threading face detection during active capture
4 main parts: face detection, face recognition, task scheduler, face clustering
"""
import cv2
import numpy as np
import configparser
import os
import ast
import time
import base64
import psutil
import asyncio
import aiohttp
from datetime import datetime
from arcface import FaceEmbedding
from face_det import FaceDetection
from clustering import face_clustering
from utils import b64_encode, face_marginalize, face_crop, resize_keepres
from xnet import Xnet


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
        self.task_await = float(config["TASK_SCHEDULER"]['Task_await'])
        self.xnet_timeout = float(self.config["XNET"]['Timeout'])
        self.cam_width = int(config["CAMERA"]['Width'])
        self.cam_height = int(config["CAMERA"]['Height'])
        self.ROI = ast.literal_eval(config["CAMERA"]["ROI"])
        self.img_upload_width = int(config["UPLOAD"]['Img_width'])
        self.img_upload_height = int(config["UPLOAD"]['Img_height'])
        self.cam = cv2.VideoCapture(0)
        self.set_cam_params()
        self.xnet = Xnet(config)
        self.unprocess_face_queue = []
        self.face_queue = []
        self.feat_queue = []
        self.upload_ufq = []
        # self.standby_mode = True
        self.last_detected_face_time = time.time()
        self.last_clustering_time = time.time()
        # self.num_cams = int(config["CAMERA"]['Num_cams'])
        # self.cam_ids = ast.literal_eval(config["CAMERA"]['Id'])

    def read_frame(self):
        s = time.time()
        ret, frame = self.cam.read()

        print("read cam time", time.time() - s)
        # print(frame.shape)
        if not ret:
            # dead cam
            self.cam = cv2.VideoCapture(0)
            self.set_cam_params()
            time.sleep(3.000) # some delay to init cam
            # cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
            return None
        r = self.ROI
        frame = frame[r[0]:r[2], r[1]:r[3], :]
        # cv2.imwrite("frame.jpg", frame)
        return frame

    def serve(self):
        while(True):
            if (time.time() - self.last_clustering_time) > self.clustering_upload_delay:
                # asyncronous clustering/upload and sensing 
                asyncio.run(self.task_scheduler())
            else:
                asyncio.run(self.sense_process(self.clustering_upload_delay))
        return 0

    async def task_scheduler(self):
        self.cluster_processed_feature()
        await asyncio.gather(
            self.upload(),
            self.sense_process(self.xnet_timeout),
        )
        print("finish upload/sensing block")
        return 0

    async def sense_process(self, duration):
        await asyncio.sleep(self.task_await)
        init_time = time.time()
        while(self.cam.isOpened()):
            print("unprocess: ", len(self.unprocess_face_queue), "processed: ", len(self.face_queue))
            # get RAM info, TODO: save to disk when OOM
            # print("RAM", psutil.virtual_memory()[2])
            if (time.time() - init_time) > duration:
                print("finish sensing block")
                return 0
            if (time.time() - self.last_detected_face_time) > self.standby_detection_delay:
                print("last face detected", time.time() - self.last_detected_face_time)
                frame = self.read_frame()
                if frame is None:
                    continue
                st = time.time()
                self.standby_serve(frame)
                print("standby", time.time() - st)
            else:
                frame = self.read_frame()
                if frame is None:
                    continue
                st = time.time()
                self.active_serve(frame)
                print("active", time.time() - st)

    def set_cam_params(self):
        self.cam.set(3, self.cam_width)
        self.cam.set(4, self.cam_height)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def standby_serve(self, frame):
        """
        1 face detection + n face analysis from queue
        """
        print("standby process detect")
        self.process_queue()
        self.detect_queue(frame, mode='standby')
        return 0

    def active_serve(self, frame, mode='active'):
        """
        1 face detection, save face to memory
        """
        self.detect_queue(frame)
        return 0

    def detect_queue(self, frame, mode='active'):
        face_bboxes = self.face_detect.inference(frame)
        print("faces detected: ", len(face_bboxes))
        if len(face_bboxes) == 0:
            return 1
        self.last_detected_face_time = time.time()
        # if mode == 'standby':
        #     return 0
        # print(face_bboxes)
        for b in face_bboxes:
            if (b[3] > b[1]) and (b[2] > b[0]):
                # detect blurr, headpose est
                # crop = frame[b[1]:b[3], b[0]:b[2], :]
                # crop = face_marginalize(crop, self.face_margin_w_ratio, self.face_margin_h_ratio)
                crop = face_crop(frame, b, 0, self.face_margin_h_ratio)
                blur_face = cv2.resize(crop, (112, 112))
                blur_face_var = cv2.Laplacian(blur_face, cv2.CV_64F).var()
                if blur_face_var > self.face_lap_min_var:
                    self.unprocess_face_queue.append(
                        {
                            # 'crop': cv2.resize(
                            #     crop, 
                            #     (
                            #         self.img_upload_height,
                            #         self.img_upload_width
                            #     )
                            # ),
                            'crop': resize_keepres(crop, self.img_upload_height, self.img_upload_width),
                            'time': self.last_detected_face_time
                        }
                    )
                    # cv2.imwrite("face.jpg", crop)
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

    def cluster_processed_feature(self):
        """
        cluster face queue into different people
        send several imgs and info per person to server
        # TODO: separate into cluster and upload functions
        """
        self.unique_faces = []
        # clustering
        if len(self.feat_queue) > 0:
            # print("no human")
            # self.last_clustering_time = time.time()
            # return 1
            print("cluster size", len(self.feat_queue), len(self.face_queue))
            labels = face_clustering(self.feat_queue)
            class_ids = np.unique(labels)
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
                self.unique_faces.append(
                    {
                        'faces': [
                            b64_encode(self.face_queue[cli_feat_ids[i]]['crop'])
                            for i in chosen_ids
                        ],
                        'time': [
                            self.face_queue[i]['time']
                            for i in cli_feat_ids
                        ],
                    }
                )
            print("num of unique people: ", len(self.unique_faces))

        self.last_clustering_time = time.time()
        # cleanup garbage
        self.feat_queue = []
        self.face_queue = []
        # convert b64 unprocess_face_queue
        self.upload_ufq = [
            {
                'face': b64_encode(ufq['crop']),
                'time': ufq['time'],
            }
            for ufq in self.unprocess_face_queue
        ]
        self.unprocess_face_queue = []

    async def upload(self):
        # upload to server
        # st = time.time()
        await self.xnet.log_face(self.unique_faces, self.upload_ufq)
        # print("upload time: ", time.time() - st)
        self.upload_ufq = []
        return 0


if __name__ == '__main__':
    fqc = FaceQueueClustering()
    fqc.serve()
