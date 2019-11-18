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
# from arcface import FaceEmbedding
from face_detection import FaceDetection
# from clustering import face_clustering
from utils import b64_encode, face_marginalize, face_crop, resize_keepres
from xnet import Xnet


class FaceQueueClustering(object):
    def __init__(self, parent=None):
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.config = config
        self.face_detect = FaceDetection(config)
        # self.face_embed = FaceEmbedding(config)
        # standby phase, delayed face detection
        self.standby_detection_delay = float(config["TASK_SCHEDULER"]['Standby_detection_delay'])
        self.standby_max_analysis = int(config["TASK_SCHEDULER"]['Standby_max_analysis'])
        self.clustering_upload_delay = float(config["TASK_SCHEDULER"]['Clustering_upload_delay'])
        self.max_img_per_person = int(config["UPLOAD"]['Max_img_per_person'])
        self.face_lap_min_var = float(config["FACE_CONSOLIDATION"]['Laplacian_min_variance'])
        # self.face_margin_w_ratio = float(config["FACE_CONSOLIDATION"]['Face_margin_w_ratio'])
        # self.face_margin_h_ratio = float(config["FACE_CONSOLIDATION"]['Face_margin_h_ratio'])
        self.face_margin = float(config["FACE_CONSOLIDATION"]['Face_margin'])
        self.face_min_w = int(config["FACE_CONSOLIDATION"]['Face_min_width'])
        self.face_min_h = int(config["FACE_CONSOLIDATION"]['Face_min_height'])
        self.face_min_ratio = float(config["FACE_CONSOLIDATION"]['Face_min_ratio'])
        self.task_await = float(config["TASK_SCHEDULER"]['Task_await'])
        self.xnet_timeout = float(self.config["XNET"]['Timeout'])
        self.cam_width = int(config["CAMERA"]['Width'])
        self.cam_height = int(config["CAMERA"]['Height'])
        self.frame_rate = int(config["CAMERA"]['Frame_rate'])
        self.flip_method = int(config["CAMERA"]['Flip_method'])
        self.ROI = ast.literal_eval(config["CAMERA"]["ROI"])
        self.img_upload_width = int(config["UPLOAD"]['Img_width'])
        self.img_upload_height = int(config["UPLOAD"]['Img_height'])
        self.cam_source = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                self.cam_width,
                self.cam_height,
                self.frame_rate,
                self.flip_method,
                # int(self.cam_width / 6),
                # int(self.cam_height / 6),
                self.cam_width,
                self.cam_height,
            )
        )
        self.cam = cv2.VideoCapture(self.cam_source, cv2.CAP_GSTREAMER)
        # self.set_cam_params()
        time.sleep(1.000) # some delay to init cam
        self.xnet = Xnet(config)
        self.unprocess_face_queue = []
        self.face_queue = []
        self.feat_queue = []
        self.upload_ufq = []
        # self.standby_mode = True
        self.last_detected_face_time = time.time()
        self.last_clustering_time = time.time()

    def read_frame(self):
        s = time.time()
        ret, frame = self.cam.read()

        print("read cam time", time.time() - s)
        # print(frame.shape)
        if not ret:
            # dead cam
            self.cam = cv2.VideoCapture(self.cam_source)
            self.set_cam_params()
            time.sleep(3.000) # some delay to init cam
            return None
        r = self.ROI
        frame = frame[r[0]:r[2], r[1]:r[3], :]
        # cv2.imwrite("frame.jpg", frame)
        return frame

    def serve(self):
        while(True):
            if (time.time() - self.last_clustering_time) > self.clustering_upload_delay:
                # asyncronous clustering/upload and sensing 
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.task_scheduler())
                # asyncio.run(self.task_scheduler())
            else:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.sense_process(self.clustering_upload_delay))
                # asyncio.run(self.sense_process(self.clustering_upload_delay))
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

            frame = self.read_frame()
            if frame is None:
                continue
            st = time.time()
            # self.active_serve(frame)
            self.detect_queue(frame)
            print("active", time.time() - st)

    def set_cam_params(self):
        self.cam.set(3, self.cam_width)
        self.cam.set(4, self.cam_height)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def detect_queue(self, frame):
        face_bboxes = self.face_detect.inference(frame)
        print("faces detected: ", len(face_bboxes))
        if len(face_bboxes) == 0:
            return 1
        self.last_detected_face_time = time.time()
        for b in face_bboxes:
            b[0] = int(max(0, b[0]))
            b[1] = int(max(0, b[1]))
            b[2] = int(min(self.cam_width, b[2]))
            b[3] = int(min(self.cam_height, b[3]))
            if (b[3]-b[1] < self.face_min_h) or (b[2]-b[0] < self.face_min_w):
                continue
            else:
                print("face too small")
            if (b[2]-b[0])/(b[3]-b[1]) < self.face_min_ratio:
                continue
            else:
                print("face wrong ratio")
            # detect blurr, headpose est
            main_face, main_head, hf_box = face_crop(frame, b, self.face_margin)
            blur_face = cv2.resize(main_face, (112, 112))
            blur_face_var = cv2.Laplacian(blur_face, cv2.CV_64F).var()
            if blur_face_var > self.face_lap_min_var:
                head_crop, resize_scale = resize_keepres(
                    main_head, 
                    self.img_upload_width,
                    self.img_upload_height
                )
                hf_box = [hf * resize_scale for hf in hf_box]

                self.unprocess_face_queue.append(
                    {
                        'crop': head_crop,
                        'face_crop': hf_box,
                        'time': self.last_detected_face_time
                    }
                )
                # cv2.imwrite(f"face_crop{time.time()}.jpg", main_head)
        return 0

    def cluster_processed_feature(self):
        """
        cluster face queue into different people
        send several imgs and info per person to server
        """
        # convert b64 unprocess_face_queue
        self.upload_ufq = [
            {
                'face': b64_encode(ufq['crop']),
                'crop_box': ufq['face_crop'],
                'time': ufq['time'],
            }
            for ufq in self.unprocess_face_queue
        ]
        self.unprocess_face_queue = []

    async def upload(self):
        # upload to server
        # st = time.time()
        await self.xnet.log_face(self.upload_ufq)
        # print("upload time: ", time.time() - st)
        self.upload_ufq = []
        return 0


if __name__ == '__main__':
    fqc = FaceQueueClustering()
    fqc.serve()
