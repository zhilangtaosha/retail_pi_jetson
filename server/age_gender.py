"""
Face analysis on opencv images
Gender, age, emotion modules
Gender result is 0-female, 1-male
"""

import tensorrt as trt
import cv2
import numpy as np
import os
import time
import configparser
import common


class AgeGenderEstimator(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.model_path = self.config["AG"]['Model_path']
        self.img_size = int(self.config["AG"]['Img_size'])
        self.batch_size = int(self.config["AG"]['Batch_size'])
        self.feat_size = int(self.config["AG"]['Feat_size'])
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

    def inference_batch(self, images, batch_size):
        """
        multi-batch inference
        """
        img_batch = np.zeros((batch_size, 3, self.img_size, self.img_size))
        for i, img in enumerate(images):
            img = cv2.resize(img, (self.img_size, self.img_size))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        fb = fb[0]
        atts = [
            {
                'gender': np.argmax(
                    fb[
                        i*self.feat_size:i*self.feat_size+2
                    ]
                ),
                'age': np.sum(
                    np.argmax(
                        fb[
                            i*self.feat_size+2:i*self.feat_size+202
                        ].reshape((100, 2)), 
                        axis=1
                    )
                )
            }
            for i in range(batch_size)
        ]
        return atts

    
    def inference(self, images):
        """
        get images list of arbitrary length, 
        separate into small enough batches 
        and doing batch inference
        """
        if len(images) == 0:
            return np.asarray([])
        # bs = self.batch_size
        ret_atts = []
        # ret_feats = np.zeros((len(images), self.feat_size))
        queue_length = int(len(images)/self.batch_size)
        # within batch size
        for i in range(queue_length):
            batch_imgs = images[i*self.batch_size:(i+1)*self.batch_size]
            # ret_feats[i*bs:(i+1)*bs, :] = self.inference_batch(batch_imgs, self.batch_size)
            ret_atts += self.inference_batch(batch_imgs, self.batch_size)

        # handle batch margin
        margin = -int(len(images)%self.batch_size)
        if margin == 0:
            return ret_atts
        batch_imgs = images[margin:]
        # ret_feats[margin:] = self.inference_batch(batch_imgs, len(batch_imgs))
        ret_atts += self.inference_batch(batch_imgs, len(batch_imgs))
        return ret_atts

    def extend_inference(self, new_people):
        """
        extend each person in people list with age, gender infomation
        """
        if not new_people:
            return []
        faces = []
        fids = []
        for i, p in enumerate(new_people):
            for f in p['person']:
                faces.append(f['face'])
                fids.append(i)
        atts = self.inference(faces)
        last_fid = -1
        same_fid = []
        for i, fid in enumerate(fids):
            if last_fid != fid:
                # update last same fid list
                if len(same_fid): 
                    avg_gender = round(sum([
                            atts[f]['gender'] 
                            for f in same_fid
                        ]) / len(same_fid)
                    )
                    avg_age = round(sum([
                            atts[f]['age'] 
                            for f in same_fid
                        ]) / len(same_fid)
                    )
                    new_people[last_fid].update(
                        {
                            'gender': avg_gender,
                            'age': avg_age,
                        }
                    )
                same_fid = [i]
                last_fid = fid
            else:
                same_fid.append(i)
        # add last person
        if len(same_fid): 
            avg_gender = round(sum([
                    atts[f]['gender'] 
                    for f in same_fid
                ]) / len(same_fid)
            )
            avg_age = round(sum([
                    atts[f]['age'] 
                    for f in same_fid
                ]) / len(same_fid)
            )
            new_people[last_fid].update(
                {
                    'gender': avg_gender,
                    'age': avg_age,
                }
            )
        return new_people