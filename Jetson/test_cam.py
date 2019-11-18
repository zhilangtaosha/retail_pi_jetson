# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    # capture_width=1920,
    # capture_height=1080,
    # display_width=3820,
    # display_height=2464,
    display_width=3264,
    display_height=2464,
    framerate=21,
    flip_method=0,
):
    print(capture_width, capture_height)
    return (
        # f"nvarguscamerasrc ! "
        # # "exposuretimerange='20000000 20000000' ! "
        # # "auto-exposure=1 exposure-time=.0005 ! "
        # "video/x-raw(memory:NVMM), "
        # "width=(int){capture_width}, height=(int){capture_height}, "
        # "format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        # "nvvidconv flip-method={flip_method} ! "
        # "video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        # "videoconvert ! "
        # "video/x-raw, format=(string)BGR ! appsink"

        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    for i in range(10):
        s = time.time()
        r, f = cap.read()
        print(f.shape, time.time() - s)
        # cv2.imwrite(f"cam_pi_{i}.jpg", f)
    # if cap.isOpened():
    #     window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    #     # Window
    #     while cv2.getWindowProperty("CSI Camera", 0) >= 0:
    #         ret_val, img = cap.read()
    #         cv2.imshow("CSI Camera", img)
    #         # This also acts as
    #         keyCode = cv2.waitKey(30) & 0xFF
    #         # Stop the program on the ESC key
    #         if keyCode == 27:
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()
    # else:
    #     print("Unable to open camera")


if __name__ == "__main__":
    show_camera()