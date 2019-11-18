import numpy as np
import cv2
import time
# from arcface import ArcFace
from arcface_torch import ArcFace
from utils import cosineDistance
# import torch 

def l2_norm_numpy(input):
    norm = np.linalg.norm(input)
    output = input / norm
    return output

img0 = cv2.imread("images/test/0.jpg")
img1 = cv2.imread("images/test/1.jpg")

af = ArcFace()

s = time.time()
# feat0 = af.inference([img0])[0]
# feat0 = l2_norm_numpy(feat0)
feat0 = af.extract_feature(img0).numpy()
print("inf img 0: ", time.time() - s)
s = time.time()
# feat1 = af.inference([img1])[0]
# feat1 = l2_norm_numpy(feat1)
feat1 = af.extract_feature(img1).numpy()
print("inf img 1: ", time.time() - s)
s = time.time()
# print(feat0)
# print(feat1)

dst = cosineDistance(feat0, feat1)
print("cos dst: ", dst)