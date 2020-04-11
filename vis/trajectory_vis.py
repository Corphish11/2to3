import os

import cv2

from util_tools import io
from imutils import paths


def vis_trajectories_in_video(sequence_trajectories, video_or_image_paths, camera, vis=False, out_video_file=None, out_fps=10, cfg=None):
    print('out video: {}'.format(out_video_file))
    if os.path.isfile(video_or_image_paths):
        cap = cv2.VideoCapture(video_or_image_paths)

        if out_video_file:
            height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            out_video = io.Mp4VideoWriter(
                out_video_file, height, width, out_fps)

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255))
            for sequence_trajectory in sequence_trajectories:
                frame = sequence_trajectory.show_fth_pose_in_image(frame, i, camera, cfg)

            if out_video_file:
                out_video.write(frame)

            if vis:
                cv2.imshow('res', frame)
                cv2.waitKey(0)
            i += 1
        cap.release()

    elif os.path.isdir(video_or_image_paths):
        images = sorted(list(paths.list_images(video_or_image_paths)),
                        key=lambda x: int(os.path.basename(x).split('.')[0]))

        if out_video_file:
            first_image_path = images[0]
            first_image = cv2.imread(first_image_path)
            height, width = first_image.shape[:2]
            out_video = io.Mp4VideoWriter(
                out_video_file, height, width, out_fps)

        for i, img in enumerate(images):
            frame = cv2.imread(img)
            cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255))
            for sequence_trajectory in sequence_trajectories:
                frame = sequence_trajectory.show_fth_pose_in_image(frame, i, camera, cfg)
            if out_video_file:
                out_video.write(frame)

            if vis:
                cv2.imshow('res', frame)
                cv2.waitKey(0)
            i += 1

    else:
        raise NotImplementedError('{} should be a video path or images dir, but get {} instead'.format(
            video_or_image_paths, type(video_or_image_paths)
        ))