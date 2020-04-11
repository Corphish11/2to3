import functools

import tqdm
import cv2
from imutils import paths
import numpy as np

from neural_networks import ball_detection
from track import Rect, track_inference
from vis import vis


def detect_ball_for_match(video_or_image_dir, start_frame, end_frame, ball_detector):
    frame_based_result = []

    if os.path.isdir(video_or_image_dir):
        images = sorted(list(paths.list_images(video_or_image_dir)),
                        key=lambda x: int(os.path.basename(x).split('.')[0]))

        for i, img in enumerate(images):
            if start_frame <= i <= end_frame:
                nmsed_result = ball_detector.detection_ball(img)
                frame_rects = list()
                for rect in nmsed_result:
                    xmin, ymin, xmax, ymax, score = rect
                    frame_rects.append(
                        Rect.Rect(xmin, ymin, xmax, ymax, i, score)
                    )
                frame_based_result.append(frame_rects)

    else:
        cap = cv2.VideoCapture(video_or_image_dir)
        i = 0
        while i <= end_frame:
            ret, frame = cap.read()

            if i <= start_frame:
                i += 1
                continue

            if not ret:
                break
            nmsed_result = ball_detector.detection_ball(frame)
            frame_rects = list()
            for rect in nmsed_result:
                xmin, ymin, xmax, ymax, score = rect
                frame_rects.append(
                    Rect.Rect(xmin, ymin, xmax, ymax, i, score)
                )
            frame_based_result.append(frame_rects)
            i += 1
        cap.release()

    return frame_based_result


def detect_reconstruct_and_track_in_range(cameras, videos_or_image_dirs, time_region, cfg):
    start_frame, end_frame = time_region
    print('video total num: {}'.format(end_frame - start_frame + 1))

    view_trajectories = list()
    ball_detector = ball_detection.BallDetector(cfg)
    for i, (camera, video_or_image_dir) in enumerate(zip(cameras, videos_or_image_dirs)):
        frame_base_balls = detect_ball_for_match(video_or_image_dir, start_frame, end_frame, ball_detector)
        tracklet_heads = track_inference.track_balls(frame_base_balls, cfg.ball_delta_frame, cfg.ball_radius_parameter)
        trajectories = track_inference.parse_tracklet_heads(tracklet_heads, drop_short_tracklet=cfg.ball_2d_short_drop,
                                                            short_thresh=cfg.ball_2d_short_thresh)

        vis.vis_trajectories(trajectories, video_or_image_dir, out_fps=10,
                             out_video_file=os.path.join(cfg.ball_vis_dir, 'ball{}_out.mp4'.format(i))
                             )

        view_trajectories.append(trajectories)


def velocity_constraint(sequence_ball_trajectory, velocity_thresh=0, fps=0):
    ball_average_speed = sequence_ball_trajectory.average_speed(fps)
    print('average speed is: {}'.format(ball_average_speed))
    if ball_average_speed > velocity_thresh:
        return True
    else:
        return False


def constraint_select_ball_trajectory(sequence_ball_trajectories, *constraint_func):
    for func in constraint_func:
        sequence_ball_trajectories = filter(
            lambda x: func(x), sequence_ball_trajectories
        )
    return sequence_ball_trajectories

def detect_reconstruct_and_track_high_speed_ball_in_range(cameras, videos_or_image_dirs, time_region, cfg):
    sequence_ball_trajectories = detect_reconstruct_and_track_in_range(
        cameras, videos_or_image_dirs, time_region, cfg
    )


if __name__ == '__main__':
    from Configs import *

    cfg = Config()
    cfg.init_config()

    sequence_ball_trajectories = detect_reconstruct_and_track_in_range(
        cfg.ball_cameras, cfg.ball_videos, (5, 142), cfg
    )
