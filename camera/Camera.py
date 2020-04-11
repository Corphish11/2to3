import numpy as np
import cv2
from numba import njit



class Camera:
    def __init__(self, cam_id):
        self.id = cam_id
