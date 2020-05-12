import numpy as np
from scipy.linalg import svd
import cv2
from solve_pnp import DLT
import pickle


def build_matrix(tracklet_2d, g, P, times):
    M = []
    n = len(times)
    R = []
    start_time = times[0]
    for i in range(n):
        frame_num, x, y = tracklet_2d[i]
        t = (times[i] - start_time) / 50
        M.append([P[0, 0] - x * P[2, 0], P[0, 0] * t - x * P[2, 0] * t,
                  P[0, 1] - x * P[2, 1], P[0, 1] * t - x * P[2, 1] * t,
                  P[0, 2] - x * P[2, 2], P[0, 2] * t - x * P[2, 2] * t
                  ])
        M.append([P[1, 0] - y * P[2, 0], P[1, 0] * t - y * P[2, 0] * t,
                  P[1, 1] - y * P[2, 1], P[1, 1] * t - y * P[2, 1] * t,
                  P[1, 2] - y * P[2, 2], P[1, 2] * t - y * P[2, 2] * t
                  ])

        R.append(
            [
                x * (P[2, 2] * g * t ** 2 / 2 + 1) - (P[0, 2] * g * t ** 2 / 2 + P[0, 3])
            ]
        )

        R.append(
            [y * (P[2, 2] * g * t ** 2 / 2 + 1) - (P[1, 2] * g * t ** 2 / 2 + P[1, 3])]
        )
    return M, R


def svd_solve(A, b):
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    S, U, V = cv2.SVDecomp(A)
    B = U.T.dot(b)
    X = [B[i]/S[i] for i in range(len(S))]
    res = V.T.dot(X)
    return res


def solve_single_tracklet(tracklet_2d, P, fps, g):
    """
    :param tracklet_2d: {t1: (x1, y1), t2: (x2, y2), ...}
    :param P: 通过solve_pnp得到
    :param fps: 视频的fps, cap = cv2.VideoCapture(video) fps = cap.get(cv2.CAP_PROP_FPS)
    :param g: 重力系数
    :return: M2, [X, Y, Z, Vx, Vy, Vz]
    """
    N = len(tracklet_2d)
    M = np.zeros((2 * N, 6), dtype=np.float32)
    # 按照论文中的公式10构建求解矩阵
    # A, b = build_matrix(tracklet_2d, M, fps, g, P)
    A, b = build_matrix(tracklet_2d, g, P, fps)
    # 使用the direct linear transform algorithm 求解，你搜索一下python的相关实现，肯定有的
    res = svd_solve(A, b)
    return res


def convert_tracklet_2d_to_3d(start_p, velocity, times):
    '''
    :param tracklet2d: .
    :param start_p: X, Y, Z
    :param velocity: Vx, Vy, Vz
    :return:
    '''

    pos_3ds = list()
    X0, Y0, Z0 = start_p
    Vx, Vy, Vz = velocity

    start_time = times[0]
    g = -980
    for time in times:
        delat_time = (time - start_time) / 50
        pos_3ds.append(
            [time,
             float((X0 + delat_time * Vx)/100),
             float((Y0 + delat_time * Vy)/100),
             float((Z0 + delat_time * Vz + g * delat_time ** 2 / 2)/100),
             ]
        )
    return pos_3ds

# def pmatrix_test(pmatrix):

#     np.set_printoptions(suppress=True)
#     xyz = [[0, 900, 0,1], [900, 900, 0, 1], [0, 300, 0, 1], [900, 300, 0, 1], [0, 0, 0,1], [900, 0, 0, 1], [0, -300, 0, 1], [900, -300, 0, 1], [0, -900, 0, 1], [900, -900, 0, 1], [0, 0, 224, 1], [900, 0, 224, 1], [0, 0, 124, 1], [900, 0, 124, 1]]
#     uv = [[86, 795], [1817, 795], [381, 578], [1533, 578], [464, 515], [1451, 515], [527, 471], [1388, 471], [615, 408], [1298, 408], [461, 270], [1452, 270], [461, 375], [1452, 375]]
#     error = []
#     i = 0
#     for item in xyz:
#         real_world_point = np.array(item).reshape((-1,1))
#         m_matrix = np.dot(pmatrix, real_world_point)
#         error.append((abs(m_matrix[0] - uv[i][0]), abs(m_matrix[1] - uv[i][1])))
#         print(error[i])
#         i += 1
#     return 0
if __name__ == '__main__':
    tracklet_2ds = list()
    times = list()
    with open('/home/nata/Documents/code/BallTrack/output_traj1.txt', 'rb') as f: 
        new_data = pickle.load(f) 
    for item in new_data: 
        if item[0] == 0: 
            tracklet_2ds.append([item[0], item[2], item[3]])
            times.append(item[1])
    # print(tracklet_2ds)
    # print('\n')
    # print(len(times))
    # print('\n')
    # print(times)
    # P = np.random.rand(3, 4)
    g = -980
    P = DLT()
    # pmatrix_test(P)

    A, b = build_matrix(tracklet_2ds, g, P, times)

    X0, Vx, Y0, Vy, Z0, Vz = svd_solve(A, b)
    V_init = [float(Vx/100), float(Vy/100), float(Vz/100)]
    print(X0/100, Vx/100, Y0/100, Vy/100, Z0/100, Vz/100)
    # X0, Vx, Y0, Vy, Z0, Vz = solve_single_tracklet(tracklet_2ds, P, times, g)

    pos_3ds = convert_tracklet_2d_to_3d((X0, Y0, Z0), (Vx, Vy, Vz), times)
    print('-----------3dtraj---------\n')
    # print(pos_3ds)
    file = open('3d_traj1.txt', 'w')
    file.write(str(V_init))
    file.write('\n')
    for item in pos_3ds:
        file.write(str(item))
        file.write('\n')
    file.close()
