from mmdet.apis import init_detector, inference_detector
from nms.nms import boxes as nms_boxes
import numpy as np


class BallDetector:
    def __init__(self, cfg):
        config_file = cfg.ball_config_file
        checkpoint_file = cfg.ball_checkpoint_file

        if cfg.use_gpu:
            self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        else:
            # sadly to say, currently mmdetection not support cpu version
            self.model = init_detector(config_file, checkpoint_file, device='cpu')

    def detection_ball(self, frame):
        result = inference_detector(self.model, frame)
        nmsed_result = merge_nms_result(result)
        return nmsed_result

def nms_result(result, thresh=0.3):

    if len(result) == 0:
        return None, None

    boxes = result[:, :4]
    boxes[:, 3] -= boxes[:, 1]
    boxes[:, 2] -= boxes[:, 0]
    scores = result[:, 4]
    best_indict = nms_boxes(boxes, scores)
    result[:, 3] += result[:, 1]
    result[:, 2] += result[:, 0]
    nmsed_result = result[best_indict]
    nmsed_result = [r for r in nmsed_result if r[4] > thresh]
    return nmsed_result


def merge_nms_result(result):
    result = np.concatenate(result, axis=0)
    nmsed_result = nms_result(result)
    return nmsed_result

