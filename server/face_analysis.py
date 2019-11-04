"""
TODO
Face analysis on opencv images
Gender, age, emotion modules
"""

import tensorrt as trt
import cv2
import numpy as np
import os
import time
import configparser
import common


class FaceAnalysis(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.model_path = self.config["ARCFACE"]['Model_path']
        self.img_size = int(self.config["ARCFACE"]['Img_size'])
        self.batch_size = int(self.config["ARCFACE"]['Batch_size'])
        self.feat_size = int(self.config["ARCFACE"]['Feat_size'])
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

    def inference(self, unique_people, new_only=False):
        """
        face analysis on list of people,
        new_only for inference on unidentified peopel only
        """
        return 0