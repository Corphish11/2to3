import os
from imutils import paths
import cv2


def get_fth_frame(video_or_image_dirs, f):
    if os.path.isdir(video_or_image_dirs):
        image_files = list(paths.list_images(video_or_image_dirs))
        sorted_image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
        image_file_name = sorted_image_files[f]
        frame = cv2.imread(image_file_name)

    elif os.path.isfile(video_or_image_dirs):
        cap = cv2.VideoCapture(video_or_image_dirs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, frame = cap.read()
    else:
        raise NotImplementedError('cant deal with such file :{}'.format(video_or_image_dirs))

    return frame
