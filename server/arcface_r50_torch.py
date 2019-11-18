import cv2
import numpy as np
import configparser
import io, base64
from torch2trt import TRTModule
import torch


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
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(self.model_path))

    def inference_batch(self, images):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.rand(self.batch_size, 3, self.img_size, self.img_size)
        for i, img in enumerate(images):
            resized = cv2.resize(img, (112, 112))
            ccropped = resized[..., ::-1]  # BGR to RGB
            # load numpy to tensor
            ccropped = ccropped.transpose(2, 0, 1)
            ccropped = np.array(ccropped, dtype=np.float32)
            ccropped = (ccropped - 127.5) / 128.0
            ccropped = torch.from_numpy(ccropped)
            x[i] = ccropped

        # extract features
        with torch.no_grad():
            feat = self.model(x.to(device))
            feat = feat.cpu()
            features = self.l2_norm(feat)
        features = features.cpu().numpy()
        return features[:len(images)]

    def l2_norm(self, input, axis=1):
        print(input)
        print(input.shape)
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
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