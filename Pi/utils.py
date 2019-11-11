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

def face_crop(frame, box, margin_w_ratio, margin_h_ratio):
    """
    crop face from frame with bounding box, margin
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    fh, fw, _ = frame.shape
    if (w <= 0) or (h <= 0):
        return None
    margin_w = w * margin_w_ratio
    margin_h = h * margin_h_ratio
    x0 = int(max(box[0]-margin_w, 0))
    y0 = int(max(box[1]-margin_h, 0))
    x1 = int(min(box[2]+margin_w, fw))
    y1 = int(min(box[3]+margin_h, fh))
    return frame[y0:y1, x0:x1, :]

def resize_keepres(image, input_height, input_width):
    """
    resize image, keeping original resolution (0 padding borders)
    """
    ret_img = np.zeros((input_height, input_width, 3))
    img_height, img_width, _ = image.shape
    if img_height / img_width > input_height / input_width:
        resize_scale = input_height / img_height
        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        left_pad = int((input_width - input_image.shape[1]) / 2)
        ret_img[:, left_pad:left_pad + input_image.shape[1], :] = input_image
    else:
        resize_scale = input_width / image.shape[1]
        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        top_pad = int((input_height - input_image.shape[0]) / 2)
        ret_img[top_pad:top_pad + input_image.shape[0], :, :] = input_image
    return ret_img