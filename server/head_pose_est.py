import tensorrt as trt
import cv2
import numpy as np
import os
import base64, io
import time
import configparser
import common
import ast
from scipy.special import softmax
from utils import good_head_angle, blurry_face

class HeadPoseEst(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.model_path = self.config["HEAD_POSE"]['Model_path']
        self.img_size = int(self.config["HEAD_POSE"]['Img_size'])
        self.batch_size = int(self.config["HEAD_POSE"]['Batch_size'])
        self.feat_size = int(self.config["HEAD_POSE"]['Feat_size'])
        self.angle_min = ast.literal_eval(self.config["HEAD_POSE"]["Angle_min"])
        self.angle_max = ast.literal_eval(self.config["HEAD_POSE"]["Angle_max"])
        self.face_lap_min_score = float(self.config["HEAD_POSE"]['Lap_min'])
        self.TRT_LOGGER = trt.Logger()
        self.engine = self.getEngine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)


    def getEngine(self):
        if os.path.exists(self.model_path):
            print("Reading engine from file {}".format(self.model_path))
            with open(self.model_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print("TensorRT engine file not found")
            return None

    def convert_angle(self, feat):
        """
        convert numpy angle features into [-180, 180] angle
        """
        sm_feat = softmax(feat)
        idx_tensor = np.arange(self.feat_size)
        return np.sum(sm_feat * idx_tensor) * 3 - 99


    def inference_batch(self, images, batch_size):
        """
        multi-batch inference
        return list of [yaw, pitch, roll] for each input image
        """
        img_batch = np.zeros((batch_size, 3, self.img_size, self.img_size))
        for i, img in enumerate(images):
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize
            img = (img - 127.5) / 128
            img = img.transpose(2, 0, 1)
            img_batch[i] = img

        img_batch = np.array(img_batch, dtype=np.float32, order='C')
        self.inputs[0].host = img_batch
        fb = common.do_inference(
            self.context, 
            bindings=self.bindings, 
            inputs=self.inputs, 
            outputs=self.outputs, 
            stream=self.stream, 
            batch_size=batch_size
        )
        # TODO make sure yaw, pitch, roll order is 0-1-2 (or make min/max threshold based on this exact model)
        angles = [
            {
                'yaw': self.convert_angle(
                    fb[0][i*self.feat_size:(i+1)*self.feat_size]
                ),
                'pitch': self.convert_angle(
                    fb[1][i*self.feat_size:(i+1)*self.feat_size]
                ),
                'roll': self.convert_angle(
                    fb[2][i*self.feat_size:(i+1)*self.feat_size]
                ),
            }
            for i in range(batch_size)
        ]
        return angles

    def inference(self, images):
        """
        get images list of arbitrary length, 
        separate into small enough batches 
        and doing batch inference
        """
        if len(images) == 0:
            return np.asarray([])
        ret_angles = []
        queue_length = int(len(images)/self.batch_size)
        # within batch size
        for i in range(queue_length):
            batch_imgs = images[i*self.batch_size:(i+1)*self.batch_size]
            ret_angles += self.inference_batch(batch_imgs, self.batch_size)

        # handle batch margin
        margin = -int(len(images)%self.batch_size)
        if margin == 0:
            return ret_angles
        batch_imgs = images[margin:]
        ret_angles += self.inference_batch(batch_imgs, len(batch_imgs))
        return ret_angles

    def remove_bad_pose(self, raw_faces, raw_faces_info):
        """
        remove face with bad angle
        """
        if not raw_faces:
            return [], []
        blur_idx_check = np.zeros(len(raw_faces))
        hp_idx_check = np.zeros(len(raw_faces))
        for i, face in enumerate(raw_faces):
            if not blurry_face(face, self.face_lap_min_score):
                blur_idx_check[i] = 1
        head_angles = self.inference(raw_faces)
        for i, angle in enumerate(head_angles):
            if good_head_angle(angle, self.angle_min, self.angle_max):
                hp_idx_check[i] = 1
            else:
                print(angle)
        hp_idx_check = hp_idx_check * blur_idx_check
        good_face = [
            face for i, face in enumerate(raw_faces) 
            if hp_idx_check[i]
        ]
        good_face_info = [
            face for i, face in enumerate(raw_faces_info) 
            if hp_idx_check[i]
        ]
        return good_face, good_face_info