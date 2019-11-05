import cv2
import os
import pickle
import numpy as np

def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def cosineDistance(a, b):
    '''return cosine distance of 2 vectors'''
    a = a.flatten()
    b = b.flatten()
    ab = np.matmul(np.transpose(a), b)
    aa = np.sqrt(np.sum(np.multiply(a, a)))
    bb = np.sqrt(np.sum(np.multiply(b, b)))
    ret = 1 - (ab / (aa * bb))
    return ret

def good_head_angle(angles, angle_min, angle_max):
    """
    good head angle would be looking directly to the camera, give or take 
    some degree each
    """
    y = angles['yaw']
    p = angles['pitch']
    r = angles['roll']
    if ((angle_min[0] < y) and (angle_max[0] > y) 
        and (angle_min[1] < p) and (angle_max[1] > p) 
        and(angle_min[2] < r) and (angle_max[2] > r)):
        return True
    return False

def blurry_face(face, lap_min_score):
    blur_face = cv2.resize(face, (112, 112))
    blur_face_var = cv2.Laplacian(blur_face, cv2.CV_64F).var()
    return blur_face_var < lap_min_score
