import tensorrt as trt
import cv2
import numpy as np
import configparser
import io, base64, os
import common


class ArcFace(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.model_path = self.config["ARCFACE_R50"]['Model_path']
        self.img_size = int(self.config["ARCFACE_R50"]['Img_size'])
        self.batch_size = int(self.config["ARCFACE_R50"]['Batch_size'])
        self.feat_size = int(self.config["ARCFACE_R50"]['Feat_size'])
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


    def inference_batch(self, images):
        batch_size = len(images)
        img_batch = np.zeros((self.batch_size, 3, self.img_size, self.img_size))
        for i, img in enumerate(images):
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img[..., ::-1] # BGR to RGB
            img = img.transpose(2, 0, 1)
            img = np.array(img, dtype=np.float32)
            img = (img - 127.5) / 128
            img_batch[i] = img

        img_batch = np.array(img_batch, dtype=np.float32, order='C')
        self.inputs[0].host = img_batch
        feat_batch = common.do_inference(
            self.context, 
            bindings=self.bindings, 
            inputs=self.inputs, 
            outputs=self.outputs, 
            stream=self.stream, 
            batch_size=batch_size
        )
        feats = np.asarray([
            self.l2_norm_numpy(
                feat_batch[0][i*self.feat_size:(i+1)*self.feat_size]
            )
            for i in range(batch_size)
        ])
        return feats

    def l2_norm_numpy(self, input):
        norm = np.linalg.norm(input)
        output = input / norm
        return output

    def inference(self, images):
        """
        get images list of arbitrary length, separate into small enough 
        batches and doing batch inference
        """
        ret_feats = np.zeros((len(images), self.feat_size))
        if len(images) == 0:
            return ret_feats
        bs = self.batch_size
        queue_length = int(len(images)/self.batch_size)
        # within batch size
        for i in range(queue_length):
            batch_imgs = images[i*self.batch_size:(i+1)*self.batch_size]
            ret_feats[i*bs:(i+1)*bs, :] = self.inference_batch(batch_imgs)

        # handle batch margin
        margin = -int(len(images)%self.batch_size)
        if margin == 0:
            return ret_feats
        batch_imgs = images[margin:]
        ret_feats[margin:] = self.inference_batch(batch_imgs)
        return ret_feats

    def extend_inference(self, unique_faces):
        """
        extend unique faces list with embedded feature for each face
        """
        if not unique_faces:
            return []
        faces = []
        fids = []
        for i, uf in enumerate(unique_faces):
            for face in uf['faces']:
                face_bin = base64.b64decode(face)
                face_stream = io.BytesIO(face_bin)
                face_cv = cv2.imdecode(np.fromstring(
                    face_stream.read(), np.uint8), 1)
                faces.append(face_cv)
                fids.append(i)
        feats = self.inference(faces)
        ret_unique_faces = []
        last_fid = -1
        same_fid = []
        for i, fid in enumerate(fids):
            if last_fid != fid:
                # update last same fid list
                if len(same_fid): 
                    ret_unique_faces.append(
                        {
                            'person': [
                                {
                                    'face': faces[f],
                                    'feat': feats[f],
                                }
                                for f in same_fid
                            ],
                            'time': unique_faces[last_fid]['time']
                        }
                    )
                same_fid = [i]
                last_fid = fid
            else:
                same_fid.append(i)
        # add last person
        if len(same_fid): 
            ret_unique_faces.append(
                {
                    'person': [
                        {
                            'face': faces[f],
                            'feat': feats[f],
                        }
                        for f in same_fid
                    ],
                    'time': unique_faces[last_fid]['time']
                }
            )
        return ret_unique_faces