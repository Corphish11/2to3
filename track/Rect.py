import cv2
import numpy as np

from track import BaseTrackObj


class Rect(BaseTrackObj.BaseTrackObj):
    def __init__(self, xmin, ymin, xmax, ymax, frame_index, score):
        super(Rect, self).__init__()
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.frame_index = frame_index
        self.score = score
        self.box = [xmin, ymin, xmax, ymax]
        self.center = np.array([(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2], dtype=np.float32)

    @property
    def area(self):
        width, height = self.xmax - self.xmin, self.ymax - self.ymin
        return width * height

    @property
    def radius(self):
        return self.area ** 0.5

    def output_traj(self):
        return [self.track_id, self.frame_index, self.center[0], self.center[1]]

    def show_in_frame(self, frame):
        xmin, ymin, xmax, ymax, score = self.xmin, self.ymin, self.xmax, self.ymax, self.score
        text_pos = (int((xmin + xmax) / 2), int(ymin - 10))
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0))
        cv2.putText(frame, '{}'.format(self.track_id), text_pos, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255))
        return frame

    def __str__(self):
        return '{} {}'.format(self.center[0], self.center[1])