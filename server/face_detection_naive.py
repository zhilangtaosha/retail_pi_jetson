import tensorrt as trt
import cv2
import numpy as np
import os
import base64, io
import time
import configparser
import ast
import common
from utils import NMS


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
        self.canvas_col = int(self.config["FACE_DET"]['Canvas_col'])
        self.canvas_row = int(self.config["FACE_DET"]['Canvas_row'])
        self.batch_size = int(self.config["FACE_DET"]['Batch_size'])

        self.receptive_field_list = ast.literal_eval(self.config["FACE_DET"]["RFL"])
        self.receptive_field_stride = ast.literal_eval(self.config["FACE_DET"]["RFS"])
        self.bbox_small_list = ast.literal_eval(self.config["FACE_DET"]["BSL"])
        self.bbox_large_list = ast.literal_eval(self.config["FACE_DET"]["BLL"])
        self.receptive_field_center_start = ast.literal_eval(self.config["FACE_DET"]["RFCS"])
        self.num_output_scales = int(self.config["FACE_DET"]["NOS"])
        self.constant = ast.literal_eval(self.config["FACE_DET"]["Constant"])

        self.score_threshold = float(self.config["FACE_DET"]["Score_threshold"])
        self.top_k = int(self.config["FACE_DET"]["Top_k"])
        self.NMS_threshold = float(self.config["FACE_DET"]["NMS_threshold"])

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

    def inference(self, image):
        """
        get images list of arbitrary length, separate into small enough 
        batches and doing batch inference
        """
        skip_scale_branch_list = []
        if image.ndim != 3 or image.shape[2] != 3:
            print('Only RGB images are supported.')
            return None
        input_height = self.input_shape[2]
        input_width = self.input_shape[3]
        input_batch = np.zeros((1, input_height, input_width, self.input_shape[1]), dtype=np.float32)
        left_pad = 0
        top_pad = 0
        if image.shape[0] / image.shape[1] > input_height / input_width:
            resize_scale = input_height / image.shape[0]
            input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
            left_pad = int((input_width - input_image.shape[1]) / 2)
            input_batch[0, :, left_pad:left_pad + input_image.shape[1], :] = input_image
        else:
            resize_scale = input_width / image.shape[1]
            input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
            top_pad = int((input_height - input_image.shape[0]) / 2)
            input_batch[0, top_pad:top_pad + input_image.shape[0], :, :] = input_image

        input_batch = input_batch.transpose([0, 3, 1, 2])
        input_batch = np.array(input_batch, dtype=np.float32, order='C')
        self.inputs[0].host = input_batch

        outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=self.engine.max_batch_size)
        outputs = [np.squeeze(output.reshape(shape)) for output, shape in zip(outputs, self.output_shapes)]

        bbox_collection = []
        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = np.squeeze(outputs[i * 2])
            bbox_map = np.squeeze(outputs[i * 2 + 1])

            RF_center_Xs = np.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
            RF_center_Xs_mat = np.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = np.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
            RF_center_Ys_mat = np.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat
            x_rb_mat[x_rb_mat > input_width] = input_width
            y_rb_mat = y_rb_mat
            y_rb_mat[y_rb_mat > input_height] = input_height

            select_index = np.where(score_map > self.score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((
                    x_lt_mat[select_index[0][idx], select_index[1][idx]] - left_pad,
                    y_lt_mat[select_index[0][idx], select_index[1][idx]] - top_pad,
                    x_rb_mat[select_index[0][idx], select_index[1][idx]] - left_pad,
                    y_rb_mat[select_index[0][idx], select_index[1][idx]] - top_pad,
                    score_map[select_index[0][idx], select_index[1][idx]]
                ))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > self.top_k:
            bbox_collection = bbox_collection[0:self.top_k]
        bbox_collection_np = np.array(bbox_collection, dtype=np.float32)
        bbox_collection_np = bbox_collection_np / resize_scale

        final_bboxes = NMS(bbox_collection_np, self.NMS_threshold)
        # final_bboxes_ = []
        # for i in range(final_bboxes.shape[0]):
        #     final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3], final_bboxes[i, 4]))
        final_bboxes_ = [
            (
                final_bboxes[i, 0], final_bboxes[i, 1], 
                final_bboxes[i, 2], final_bboxes[i, 3], 
                final_bboxes[i, 4]
            )
            for i in range(final_bboxes.shape[0])
        ]
        return final_bboxes_


    def multi_inference(self, images, images_info):
        """
        inference on multiple images
        put images on canvas
        """
        ret_imgs = []
        ret_imgs_info = []
        if not images:
            return images, images_info
        img_num = len(images)
        img_per_canvas = self.canvas_col*self.canvas_row
        canvas_num = int(np.ceil(img_num/img_per_canvas))
        canvas_width = self.canvas_col * self.img_width_sm
        canvas_height = self.canvas_row * self.img_height_sm
        canvas = np.zeros((
            self.img_height_sm*self.canvas_row, 
            self.img_width_sm*self.canvas_col, 
            3
        )) 
        # print(canvas.shape)
        for c in range(canvas_num):
            # paste img on canvas
            for i in range(self.canvas_col):
                for j in range(self.canvas_row):
                    img_id = int((c*img_per_canvas+i*self.canvas_row+j) % img_num)
                    # print(c, i, j, img_id, images[img_id].shape)
                    canvas[
                        j*self.img_height_sm:(j+1)*self.img_height_sm,
                        i*self.img_width_sm:(i+1)*self.img_width_sm,
                        :
                    ] = images[img_id]
            boxes = self.inference(canvas)
            # indexing output boxes on canvas to original image id
            for bid, box in enumerate(boxes):
                x0 = int(max(box[0], 0))
                y0 = int(max(box[1], 0))
                x1 = int(min(box[2], canvas_width))
                y1 = int(min(box[3], canvas_height))
                i = np.floor(x0/self.img_width_sm)
                j = np.floor(y0/self.img_height_sm)
                img_id = int(c*img_per_canvas+i*self.canvas_row+j)
                if img_id < img_num:
                    face_crop = canvas[y0:y1, x0:x1, :]
                    cv2.imwrite(f"images/debug/{c}_{bid}.jpg", face_crop)
                    # print(face_crop.shape)
                    ret_imgs.append(face_crop)
                    ret_imgs_info.append(images_info[img_id])
                    ret_imgs_info[-1].update(
                        {
                            'face_crop': face_crop
                        }
                    )
        return ret_imgs, ret_imgs_info