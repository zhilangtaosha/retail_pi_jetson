import cv2
import base64
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

def good_head_angle(y, p, r, angle_min, angle_max):
    """
    good head angle would be looking directly to the camera, give or take
    some degree each
    """
    if ((angle_min[0] < y) and (angle_max[0] > y)
        and (angle_min[1] < p) and (angle_max[1] > p)
        and(angle_min[2] < r) and (angle_max[2] > r)):
        return True
    return False

def b64_encode(img):
    # ret, buffer = cv2.imencode('.jpg', img, params=(cv2.IMWRITE_JPEG_QUALITY, 30))
    ret, buffer = cv2.imencode('.jpg', img)
    b64_buffer = base64.b64encode(buffer).decode('utf-8')
    return b64_buffer

def face_marginalize(face, margin_w_ratio, margin_h_ratio):
    """
    crop face based on margin ratio
    """
    if (margin_w_ratio >= 0.5) or (margin_h_ratio >= 0.5):
        return face
    h, w, _ = face.shape
    wm = int(w * margin_w_ratio)
    hm = int(h * margin_h_ratio)
    return face[hm:h-hm, wm:w-wm, :]

def face_crop(frame, b, margin):
    """
    crop face from frame with bounding box, margin
    return face crop, head crop, face box in head image
    """
    cam_height, cam_width, _ = frame.shape
    width_margin = int(margin * (b[2]-b[0]))
    height_margin = int(margin * (b[3]-b[1]))
    # b = [int(be) for be in b]
    x0 = max(0, b[0] - width_margin)
    y0 = max(0, b[1] - height_margin)
    x1 = min(cam_width, b[2] + width_margin)
    y1 = min(cam_height, b[3] + height_margin)
    if (b[0]-width_margin) > 0:
        x0_ = (x1-x0) * margin
    else:
        x0_ = 0
    if (b[1]-height_margin) > 0:
        y0_ = (y1-y0) * margin
    else:
        y0_ = 0
    if (b[2] + width_margin) < cam_width:
        x1_ = (x1-x0) * (1-margin)
    else:
        x1_ = x1
    if (b[3] + height_margin) < cam_height:
        y1_ = (y1-y0) * (1-margin)
    else:
        y1_ = y1
    main_head = frame[y0:y1, x0:x1, :]
    main_face = frame[b[1]:b[3], b[0]:b[2], :]
    return main_face, main_head, [x0_, y0_, x1_, y1_]

def resize_keepres(image, input_height, input_width):
    """
    resize image, keeping original resolution (0 padding borders)
    """
    ret_img = np.zeros((input_height, input_width, 3))
    img_height, img_width, _ = image.shape
    if img_height / img_width > input_height / input_width:
        resize_scale = input_height / img_height
        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        ret_img = input_image
        # left_pad = int((input_width - input_image.shape[1]) / 2)
        # ret_img[:, left_pad:left_pad + input_image.shape[1], :] = input_image
    else:
        resize_scale = input_width / image.shape[1]
        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        ret_img = input_image
        # top_pad = int((input_height - input_image.shape[0]) / 2)
        # ret_img[top_pad:top_pad + input_image.shape[0], :, :] = input_image
    return ret_img, resize_scale

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