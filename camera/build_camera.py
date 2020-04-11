import json
import os

import cv2
import numpy as np

from camera import Camera


def build_camera(cam_id):
    return Camera.Camera(cam_id)
