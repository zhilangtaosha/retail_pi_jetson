"""
SSD face detection with mobilenet backend
Int8 quantized model, take uint8 input image
"""
import cv2
import configparser
import numpy as np
import tensorflow as tf

class FaceDetection(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.model_path = self.config["FACE_DET"]["Model_path"]
        self.num_threads = int(self.config["FACE_DET"]["Num_threads"])
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.interpreter.set_num_threads(self.num_threads)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_blob = self.input_details[0]['index']
        self.threshold = 0.5
        self.output_blob = {
            'box': self.output_details[0]['index'],
            'class': self.output_details[1]['index'],
            'score': self.output_details[2]['index'],
            'num': self.output_details[3]['index'],
        }


    def inference(self, img):
        h, w, _ = img.shape
        img = cv2.resize(img, (320, 320))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.uint8)
        self.interpreter.set_tensor(self.input_blob, img)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_blob['box'])
        classes = self.interpreter.get_tensor(self.output_blob['class'])
        scores = self.interpreter.get_tensor(self.output_blob['score'])
        num = self.interpreter.get_tensor(self.output_blob['num'])

        valid_boxes = []
        for i in range(int(num)):
            t, l, b, r = boxes[0][i]
            # cid = int(classes[0][i])
            score = scores[0][i]
            # print(score)
            if score > self.threshold:
                x0 = max(int(l*w), 0)
                y0 = max(int(t*h), 0)
                x1 = min(int(r*w), w)
                y1 = min(int(b*h), h)
                valid_boxes.append([x0, y0, x1, y1])
        return valid_boxes
