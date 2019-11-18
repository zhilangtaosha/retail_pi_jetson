import torch
import cv2
import numpy as np
import configparser
import time
from model_irse import IR_50

class ArcFace(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")

        model_weights = "/workspace/pretrained_models/torch/backbone_ir50_asia.pth"
        # load model def
        INPUT_SIZE = [112, 112]  # support: [112, 112] and [224, 224]
        self.backbone = IR_50(INPUT_SIZE)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load model weights
        self.backbone.load_state_dict(torch.load(model_weights))
        self.backbone.to(device)
        # set to evaluation mode
        self.backbone.eval()

    def extract_feature(self, img, tta=False):
        """
        run single OpenCV image through trained model to extract 512 embedding feature
        """
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # # resize image to [128, 128]
        # resized = cv2.resize(img, (128, 128))
        # # center crop image
        # a = int((128-112)/2)  # x start
        # b = int((128-112)/2+112)  # x end
        # c = int((128-112)/2)  # y start
        # d = int((128-112)/2+112)  # y end
        # ccropped = resized[a:b, c:d]  # center crop the image
        # ccropped = ccropped[..., ::-1]  # BGR to RGB

        resized = cv2.resize(img, (112, 112))
        ccropped = resized[..., ::-1]  # BGR to RGB

        # flip image horizontally
        flipped = cv2.flip(ccropped, 1)

        # load numpy to tensor
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype=np.float32)
        ccropped = (ccropped - 127.5) / 128.0
        ccropped = torch.from_numpy(ccropped)

        flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
        flipped = np.reshape(flipped, [1, 3, 112, 112])
        flipped = np.array(flipped, dtype=np.float32)
        flipped = (flipped - 127.5) / 128.0
        flipped = torch.from_numpy(flipped)

        # extract features
        with torch.no_grad():
            if tta:
                emb_batch = self.backbone(ccropped.to(device)).cpu() + \
                    self.backbone(flipped.to(device)).cpu()
                features = self.l2_norm(emb_batch)
            else:
                s = time.time()
                feat = self.backbone(ccropped.to(device))
                print("forward: ", time.time() - s)
                s = time.time()
                feat = feat.cpu()
                print("gpu to cpu: ", time.time() - s)
                s = time.time()
                features = self.l2_norm(feat)
                print("l2 norm", time.time() - s)
                s = time.time()
        return features

    def inference(self, face_img, tta=True):
        """
        IR-50_A model
        input 128x128
        output 512
        """
        # resize image to [128, 128]
        # resized = cv2.resize(face_img, (128, 128))

        # center crop image
        # a = int((128-112)/2)  # x start
        # b = int((128-112)/2+112)  # x end
        # c = int((128-112)/2)  # y start
        # d = int((128-112)/2+112)  # y end
        # ccropped = resized[a:b, c:d]  # center crop the image
        resized = cv2.resize(face_img, (112, 112))
        ccropped = resized[..., ::-1]  # BGR to RGB

        # flip image horizontally
        flipped = cv2.flip(ccropped, 1)

        # load numpy to tensor
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype=np.float32)
        ccropped = (ccropped - 127.5) / 128.0

        if tta:
            flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
            flipped = np.reshape(flipped, [1, 3, 112, 112])
            flipped = np.array(flipped, dtype=np.float32)
            flipped = (flipped - 127.5) / 128.0

            # extract features
            crop_output = self.fr_net.infer(inputs={self.fr_input_blob: ccropped})['536']
            flip_output = self.fr_net.infer(inputs={self.fr_input_blob: flipped})['536']
            emb_batch = crop_output + flip_output
            features = self.l2_norm_numpy(emb_batch)
        else:
            crop_output = self.fr_net.infer(inputs={self.fr_input_blob: ccropped})['536']
            features = self.l2_norm_numpy(crop_output)
        return features

    def l2_norm_numpy(self, input):
        norm = np.linalg.norm(input)
        output = input / norm
        return output

    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output
