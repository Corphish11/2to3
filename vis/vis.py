import random
import os

import matplotlib.pyplot as plt
import cv2
from util_tools import io
from imutils import paths

def draw_pose_2d_in_image(aa, kp, idx, is_valid_joint, cfg):
    """
    :param aa:
    :param kp: shape is (25, 2)
    :param idx:
    :param is_valid_joint:
    :return:
    """
    skeleton = cfg.kp_connection

    colors = [(255, 215, 0), (0, 0, 255), (100, 149, 237), (139, 0, 139), (192, 192, 192), (100, 100, 100),
              (255, 255, 255)]

    for i, j in skeleton:
        if is_valid_joint[i-1] and is_valid_joint[j-1]:
            aa = cv2.line(aa, (int(kp[i - 1][0]), int(kp[i - 1][1])), (int(kp[j - 1][0]), int(kp[j - 1][1])),
                          (0, 255, 255), 5)

    for j in range(len(kp)):
        if is_valid_joint[j]:
            aa = cv2.circle(aa, (int(kp[j, 0]), int(kp[j, 1])), 5, colors[idx % 7], -1)
    return aa

def plot_sequence_ball_trajectories(sequence_ball_trajectories, name):
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(90, 90)
    ax.set_xlim3d(-300, 2100)
    ax.set_ylim3d(-300, 1200)
    ax.set_zlim3d(0, 500)

    for i, sequence_ball_trajectory in enumerate(sequence_ball_trajectories):
        color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
        col = '#%02x%02x%02x' % color
        sequence_ball_trajectory.draw_in_3d(ax, col)

    ax.set_title(name)
    plt.show(block=True)

def vis_trajectories(trajectories, video_or_image_paths, vis=False, out_video_file=None, out_fps=10):

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

            for ty, trajectory in enumerate(trajectories):
                if trajectory.get(i):
                    rect = trajectory[i]
                    frame = rect.show_in_frame(frame)

            if out_video_file:
                out_video.write(frame)

            if vis:
                cv2.imshow('res', frame)
                cv2.waitKey(0)
            i += 1
        cap.release()

    elif os.path.isdir(video_or_image_paths):
        images = sorted(list(paths.list_images(video_or_image_paths)), key=lambda x: int(os.path.basename(x).split('.')[0]))

        if out_video_file:
            first_image_path = images[0]
            first_image = cv2.imread(first_image_path)
            height, width = first_image.shape[:2]
            out_video = io.Mp4VideoWriter(
                out_video_file, height, width, out_fps)

        for i, img in enumerate(images):
            frame = cv2.imread(img)

            cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255))

            for ty, trajectory in enumerate(trajectories):
                if trajectory.get(i):
                    rect = trajectory[i]
                    frame = rect.show_in_frame(frame)

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