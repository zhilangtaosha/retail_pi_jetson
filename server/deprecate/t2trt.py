from torch2trt import TRTModule
from torch2trt import torch2trt
print("import torch")
import io
import numpy as np
# import torch.utils.model_zoo as model_zoo
from model_irse import IR_50
import cv2
import time
import torch
# from torch import nn


def arcface(batch_size, img_size, fp16=False):
    # arcface_onnx = "/workspace/pretrained_models/onnx/" + "arcface"  "_b" + str(batch_size) + "_r" + str(img_size) + ".onnx"
    model_weights = "/workspace/pretrained_models/torch/backbone_ir50_asia.pth"
    model_weights_ts = f"/workspace/pretrained_models/torch/arcface_50_b{batch_size}_fp16.pth"
    # load model def
    INPUT_SIZE = [img_size, img_size]  # support: [112, 112] and [224, 224]
    backbone = IR_50(INPUT_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # load model weights
    # print(torch.load(model_weights))
    for param_tensor in backbone.state_dict():
        print(param_tensor, "\t", backbone.state_dict()[param_tensor].size())

    backbone.load_state_dict(torch.load(model_weights))
    print(device)
    backbone.to(device)
    # set to evaluation mode
    backbone.eval()
    # print(backbone)
    # model_trt = TRTModule()
    # testBackbone(backbone)
    x = torch.randn(batch_size, 3, img_size, img_size).cuda()
    # x = torch.randn(batch_size, 3, img_size, img_size).cpu()
    model_trt = torch2trt(backbone, [x], fp16_mode=True, max_batch_size=batch_size)
    torch.save(model_trt.state_dict(), model_weights_ts)
    print(model_trt)
    s = time.time()
    print(model_trt(x).shape)
    print("inf. time: ", time.time() - s)
    return model_trt, x

def test_arcface():
    model_weights = "/workspace/pretrained_models/torch/arcface_50_b1.pth"
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_weights))
    # x = torch.randn(1, 3, 112, 112).cuda()
    img0 = cv2.imread("/app/images/test/0.jpg")
    img1 = cv2.imread("/app/images/test/1.jpg")
    s = time.time()
    feat0 = extract_feature(img0, model_trt)
    print("inf. time: ", time.time() - s)
    print(feat0.shape)
    test_num = 10
    s = time.time()
    for i in range(test_num):
        feat1 = extract_feature(img1, model_trt)
    print("inf. time: ", (time.time() - s)/test_num)
    dst = cosineDistance(feat0.cpu().numpy(), feat1.cpu().numpy())
    print("cos dst: ", dst)

def test_arcface_mb(bs):
    model_weights = f"/workspace/pretrained_models/torch/arcface_50_b{bs}_fp16.pth"
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_weights))
    img0 = cv2.imread("/app/images/test/0.jpg")
    img1 = cv2.imread("/app/images/test/1.jpg")
    s = time.time()
    feat0 = extract_feature_batch([img0 for i in range(bs)], model_trt, batch_size=bs)
    print("inf. time: ", time.time() - s)
    print(feat0.shape)
    test_num = 10
    s = time.time()
    for i in range(test_num):
        feat1 = extract_feature_batch([img1 for i in range(bs)], model_trt, batch_size=bs)
    print("inf. time: ", (time.time() - s)/test_num)
    dst = cosineDistance(feat0.cpu().numpy()[0], feat1.cpu().numpy()[0])
    print("cos dst: ", dst)

def cv2ToImage(img):
    '''convert cv2 image to PIL.Image'''
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def cosineDistance(a, b):
    a = a.flatten()
    b = b.flatten()
    ab = np.matmul(np.transpose(a), b)
    aa = np.sqrt(np.sum(np.multiply(a, a)))
    bb = np.sqrt(np.sum(np.multiply(b, b)))
    return 1 - (ab / (aa * bb))

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def extract_feature(img, backbone, tta=False):
    """
    run single OpenCV image through trained model to extract 512 embedding feature
    """
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    # x = torch.rand((batch_size, 3, 112, 112))
    # extract features
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + \
                backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            # features = l2_norm(backbone(ccropped.to(device)).cpu())
            s = time.time()
            feat = backbone(ccropped.to(device))
            print("forward: ", time.time() - s)
            s = time.time()
            # feat = feat.cpu()
            # print("gpu to cpu: ", time.time() - s)
            # s = time.time()
            features = l2_norm(feat)
            print("l2 norm", time.time() - s)
            s = time.time()
    # print(features.shape)
    # s = time.time()
    # features = np.squeeze(features.cpu().numpy())
    # print("gpu to cpu: ", time.time() - s)
    return features

def extract_feature_batch(imgs, backbone, batch_size=1):
    """
    run single OpenCV image through trained model to extract 512 embedding feature
    """
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.rand(batch_size, 3, 112, 112)
    for i, img in enumerate(imgs):
        resized = cv2.resize(img, (112, 112))
        ccropped = resized[..., ::-1]  # BGR to RGB

        # load numpy to tensor
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        # ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype=np.float32)
        ccropped = (ccropped - 127.5) / 128.0
        ccropped = torch.from_numpy(ccropped)
        x[i] = ccropped

    # extract features
    with torch.no_grad():
        # features = l2_norm(backbone(ccropped.to(device)).cpu())
        s = time.time()
        feat = backbone(x.to(device))
        print("forward: ", time.time() - s)
        s = time.time()
        # feat = feat.cpu()
        print("gpu to cpu: ", time.time() - s)
        s = time.time()
        features = l2_norm(feat)
        print("l2 norm", time.time() - s)
        s = time.time()
    # features = np.squeeze(features.cpu().numpy())
    return features


# arcface(64, 112, fp16=True)
# test_arcface()
test_arcface_mb(64)
