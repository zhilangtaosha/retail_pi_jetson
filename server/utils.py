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

def NMS(boxes, overlap_threshold):
    '''
    :param boxes: numpy nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :TODO: Use GPU NMS
    '''
    if boxes.shape[0] == 0:
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype != np.float32:
        boxes = boxes.astype(np.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths
    idxs = np.argsort(sc) 

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compare secend highest score boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bo（ box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]

def merge_data(main_dict, merge_dict, keys):
    """
    merge data from 2 list
    """
    # main_dict['time'] += merge_dict['time']
    for key in keys:
        main_dict[key] += merge_dict[key]
    return main_dict
