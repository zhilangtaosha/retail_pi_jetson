"""
Arcface face embedding model with mobilenet-V2 backend
Int8 quantized model, take float input image
"""
import cv2
import configparser
import numpy as np
import tensorflow as tf

class FaceEmbedding(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.model_path = self.config["ARCFACE"]["Model_path"]
        self.num_threads = int(self.config["ARCFACE"]["Num_threads"])
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.interpreter.set_num_threads(self.num_threads)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_blob = self.input_details[0]['index']
        self.output_blob = self.output_details[0]['index']

    def inference(self, img):
        h, w, _ = img.shape
        if h * w == 0:
            print("empty face error")
            return None
        img = cv2.resize(img, (112, 112))
        img = np.expand_dims(img, axis=0)
        img = img.astype("float32")
        self.interpreter.set_tensor(self.input_blob, img)
        self.interpreter.invoke()
        embed = self.interpreter.get_tensor(self.output_blob)
        return embed