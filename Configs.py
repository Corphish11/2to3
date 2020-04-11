import os

from camera import build_camera


class Config:
    def __init__(self):
        self.root = os.path.join(os.getcwd(), 'src')
        self.task_name = 'LaoZi'
        self.task_root = os.path.join(self.root,  self.task_name)
        self.camera_parameter = os.path.join(self.root, 'camera_parameter')
        self.ball_vis_dir = os.path.join(self.root, 'ball_vis')
        self.camera_ids = [0]
        self.ball_camera_ids = [0]

        self.use_gpu = True

        # ball_detector config
        self.ball_config_file = '/home/nata/Documents/code/faster_rcnn_r50_fpn_1x_ball_blur.py'
        self.ball_checkpoint_file = '/home/nata/Documents/code/latest.pth'

        # ball tracking parameter
        self.ball_delta_frame = 4
        self.ball_radius_parameter = 1.6

        # track
        self.pose_track_delta_frame = 10
        self.dis_thresh = 5

        self.ball_2d_short_drop = True
        self.ball_2d_short_thresh = 5

        # velocity thresh
        self.velocity_thresh = 500

        # undistort
        self.use_undistort = False

        self.train_mode = True

    def init_config(self):

        self.cameras = [
            build_camera.build_camera(cam_id) for cam_id in
            self.camera_ids
        ]

        self.ball_cameras = [
            build_camera.build_camera(cam_id) for cam_id in
            self.ball_camera_ids
        ]

        self.videos = [
            os.path.join(self.task_root, 'cam{}.mp4'.format(cam_id)) for cam_id in self.camera_ids
        ]

        self.out_videos = [
            os.path.join(self.task_root, 'cam{}_out.mp4'.format(cam_id)) for cam_id in self.camera_ids
        ]

        self.ball_videos = [
            os.path.join(self.task_root, 'cam{}.mp4'.format(cam_id)) for cam_id in self.ball_camera_ids
        ]

        self.fps = 50
