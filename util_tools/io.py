import cv2

class Mp4VideoWriter:
    def __init__(self, video_path, height, width, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(
            video_path,
            fourcc,
            float(fps),
            (int(width), int(height)), True
        )

    def write(self, frame):
        self.video.write(frame)
