import tensorrt as trt
import cv2
import numpy as np
import os
import base64, io
import time
import configparser
import ast
import common


class FaceDetection(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.model_path = self.config["FACE_DET"]['Model_path']
        self.img_width = int(self.config["FACE_DET"]['Img_width'])
        self.img_height = int(self.config["FACE_DET"]['Img_height'])
        self.img_width_sm = int(self.config["FACE_DET"]['Img_width_sm'])
        self.img_height_sm = int(self.config["FACE_DET"]['Img_height_sm'])
        self.batch_size = int(self.config["FACE_DET"]['Batch_size'])

        self.receptive_field_list = ast.literal_eval(self.config["FACE_DET"]["RFL"])
        self.receptive_field_stride = ast.literal_eval(self.config["FACE_DET"]["RFS"])
        self.bbox_small_list = ast.literal_eval(self.config["FACE_DET"]["BSL"])
        self.bbox_large_list = ast.literal_eval(self.config["FACE_DET"]["BLL"])
        self.receptive_field_center_start = ast.literal_eval(self.config["FACE_DET"]["RFCS"])
        self.num_output_scales = int(self.config["FACE_DET"]["NOS"])
        self.constant = ast.literal_eval(self.config["FACE_DET"]["Constant"])

        self.score_threshold = int(self.config["FACE_DET"]["Score_threshold"])
        self.top_k = int(self.config["FACE_DET"]["Top_k"])
        self.NMS_threshold = int(self.config["FACE_DET"]["NMS_threshold"])

        self.TRT_LOGGER = trt.Logger()
        self.engine = self.getEngine()
        self.output_shapes = []
        self.input_shapes = []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.input_shapes.append(tuple([self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
            else:
                self.output_shapes.append(tuple([self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
        self.input_shape = self.input_shapes[0]
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

    def inference_batch(self, input_batch, batch_size):
        """
        multi-batch inference
        return numpy array
        """
        self.inputs[0].host = input_batch
        outputs = common.do_inference(
            self.context, 
            bindings=self.bindings, 
            inputs=self.inputs, 
            outputs=self.outputs, 
            stream=self.stream, 
            batch_size=self.batch_size
        )

        outputs = [
            np.squeeze(output.reshape(shape)) 
            for output, shape in zip(outputs, self.output_shapes)
        ]
        return None


    def pre_process_batch(self, images):
        """
        put images into canvas
        put canvas into batch
        handle batch margin (images not fit into 1 batch)
        """
        # put imgs on canvas
        col = self.img_width / self.img_width_sm
        row = self.img_height / self.img_height_sm
        img_canvas_num = col * row
        canvas_num = np.ceil(len(images)/img_canvas_num)
        batch_num = np.ceil(canvas_num / self.batch_size)
        input_batchs = np.zeros((batch_num, self.batch_size, self.img_height*self.img_width*3))
        canvas = np.zeros((self.img_height_sm*row, self.img_width_sm*col, 3))
        for batch in range(len(batch_num)):
            for b in range(self.batch_size):
                for i in range(row):
                    for j in range(col):
                        img = images[
                            batch*img_canvas_num*self.batch_size
                            + b*img_canvas_num
                            + i*col
                            + j
                        ]
                        canvas[
                            i*self.img_height_sm:(i+1)*self.img_height_sm, 
                            j*self.img_width_sm:(j+1)*self.img_width_sm, 
                            :
                        ] = img
                ip_canvas = cv2.resize(canvas, (self.img_height, self.img_width))
                ip_canvas = ip_canvas.transpose([0, 3, 1, 2])
                ip_canvas = np.array(ip_canvas, dtype=np.float32, order='C')
                input_batchs[batch, b] = ip_canvas.copy()

        return input_batchs


    def inference(self, images):
        """
        get images list of arbitrary length, separate into small enough 
        batches and doing batch inference
        """
        input_batchs = self.pre_process_batch(images)
        for img_batch in input_batchs:
            outputs = self.inference_batch(img_batch, self.batch_size)
        return None
