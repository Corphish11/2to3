import numpy as np


def group_consecutive(a):
    return np.split(a, np.where(np.diff(a) != 1)[0] + 1)


class SequenceTrajectory:
    def __init__(self, start_frame, end_frame, id, unit_num, dimension):

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.id = id

        self.unit_num = unit_num
        self.dimension = dimension

        # coordinate shape is (T, N, D)
        self.coordinates = np.zeros(
            (
                self.end_frame + 1 - self.start_frame, self.unit_num, dimension
            ),
            dtype=np.float32
        )

        self.scores = np.zeros(
            (
                self.end_frame + 1 - self.start_frame,  self.unit_num
            ),
            dtype=np.float32
        )
        self.frame_list = list()

    def load_frame_dict_trajectory(self, trajectory):
        pass

    def post_process(self):
        print('{} --> {}'.format(self.start_frame, self.end_frame))
        for i in range(self.unit_num):
            valid_frame_list = np.where(self.scores[:, i] > 0)[0] + self.start_frame
            frame_groups = group_consecutive(valid_frame_list)
            for j, (p_group, b_group) in enumerate(zip(frame_groups[:-1], frame_groups[1:])):
                pf, bf = p_group[-1], b_group[0]

                p_loc = self.coordinates[pf - self.start_frame, i]
                b_loc = self.coordinates[bf - self.start_frame, i]
                diff = bf - pf + 1
                pf, bf = pf - self.start_frame, bf - self.start_frame

                for k in range(self.dimension):
                    interpolates = np.linspace(p_loc[k], b_loc[k], diff)
                    self.coordinates[pf: bf + 1, i, k] = interpolates