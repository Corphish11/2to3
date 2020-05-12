import numpy as np
from scipy.linalg import svd
import cv2
from solve_pnp import DLT
import pickle
import json
import math


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

def build_matrix_2view(tracklet_2ds_1, traclet_2ds_2, g, P0, P, times):
    M = []
    n = len(times)
    R = []
    start_time = times[0]
    for i in range(n):
        frame_num1, x0, y0 = tracklet_2ds_1[i]
        t = (times[i] - start_time) / 50
        M.append([P0[0, 0] - x0 * P0[2, 0], P0[0, 0] * t - x0 * P0[2, 0] * t,
                  P0[0, 1] - x0 * P0[2, 1], P0[0, 1] * t - x0 * P0[2, 1] * t,
                  P0[0, 2] - x0 * P0[2, 2], P0[0, 2] * t - x0 * P0[2, 2] * t
                  ])
        M.append([P0[1, 0] - y0 * P0[2, 0], P0[1, 0] * t - y0 * P0[2, 0] * t,
                  P0[1, 1] - y0 * P0[2, 1], P0[1, 1] * t - y0 * P0[2, 1] * t,
                  P0[1, 2] - y0 * P0[2, 2], P0[1, 2] * t - y0 * P0[2, 2] * t
                  ])

        R.append(
            [x0 * (P0[2, 2] * g * t ** 2 / 2 + 1) - (P0[0, 2] * g * t ** 2 / 2 + P0[0, 3])]
        )
        R.append(
            [y0 * (P0[2, 2] * g * t ** 2 / 2 + 1) - (P0[1, 2] * g * t ** 2 / 2 + P0[1, 3])]        
        )
    for i in range(n):
        frame_num2, x, y = traclet_2ds_2[i]
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

def convert_tracklet_2d_to_3d(start_p, velocity, times):

    pos_3ds = list()
    X0, Y0, Z0 = start_p
    Vx, Vy, Vz = velocity
    start_time = times[0]
    g = -980 #重力加速度

    #可以根据每一帧的相对时间，算出每一帧球的位置
    for time in times:
        delat_time = (time - start_time) / 50 #这里用的当前帧数-起始帧数，算出来实际的秒数
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

def single_traj_process(traj_dir):
    tracklet_2ds = list()
    times = list()
    with open(traj_dir, 'rb') as f: 
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
    P0, P = DLT()

    A, b = build_matrix(tracklet_2ds, g, P, times)

    X0, Vx, Y0, Vy, Z0, Vz = svd_solve(A, b)
    V_init = [float(Vx/100), float(Vy/100), float(Vz/100)]
    print(X0/100, Vx/100, Y0/100, Vy/100, Z0/100, Vz/100)

    pos_3ds = convert_tracklet_2d_to_3d((X0, Y0, Z0), (Vx, Vy, Vz), times)

    print('-----------3dtraj---------\n')
    # print(pos_3ds)
    file = open('single_3d_traj.txt', 'w')
    file.write(str(V_init))
    file.write('\n')
    for item in pos_3ds:
        file.write(str(item))
        file.write('\n')
    file.close()

def single_traj_process_2view(traj_dir1, traj_dir2):
    tracklet_2ds_1 = list()
    tracklet_2ds_2 = list()
    times = list()
    with open(traj_dir1, 'rb') as f: 
        new_data = pickle.load(f) 
    with open(traj_dir2, 'rb') as f2: 
        new_data2 = pickle.load(f2) 
    i = 0
    j = 0
    while (i < len(new_data)) & (j < len(new_data2)):
        if (new_data[i][0] == 0) & (new_data2[j][0] == 0):
            if new_data[i][1] == new_data2[j][1]:
                tracklet_2ds_1.append([new_data[i][0], new_data[i][2], new_data[i][3]])
                tracklet_2ds_2.append([new_data2[j][0], new_data2[j][2], new_data2[j][3]])
                times.append(new_data[i][1])
                i += 1
                j += 1
            elif new_data[i][1] < new_data2[j][1]:
                i += 1
            elif new_data[i][1] > new_data2[j][1]:
                j += 1
        elif new_data[i][0] == 0:
            j += 1
        elif new_data2[j][0] == 0:
            i += 1
        else:
            i += 1
            j += 1
    # print(tracklet_2ds)
    # print('\n')
    # print(len(times))
    # print('\n')
    # print(times)
    # P = np.random.rand(3, 4)
    print(tracklet_2ds_1)
    print(tracklet_2ds_2)
    g = -980
    P0, P = DLT()

    A, b = build_matrix_2view(tracklet_2ds_1, tracklet_2ds_2, g, P0, P, times)

    X0, Vx, Y0, Vy, Z0, Vz = svd_solve(A, b)
    V_init = [float(Vx/100), float(Vy/100), float(Vz/100)]
    print(X0/100, Vx/100, Y0/100, Vy/100, Z0/100, Vz/100)

    pos_3ds = convert_tracklet_2d_to_3d((X0, Y0, Z0), (Vx, Vy, Vz), times)

    print('-----------3dtraj---------\n')
    # print(pos_3ds)
    file = open('single_3d_traj_2view.txt', 'w')
    file.write(str(V_init))
    file.write('\n')
    for item in pos_3ds:
        file.write(str(item))
        file.write('\n')
    file.close()

    return tracklet_2ds_1, tracklet_2ds_2

def multi_traj_process(traj_dir, parabola_dir):
    tracklet_2ds = list()
    pinyin_json_list = list()
    start_frame = 0
    end_frame = 0
    with open(traj_dir, 'rb') as f: 
        new_data = pickle.load(f)
    with open(parabola_dir,'r') as f2:
        parabola_info = json.load(f2) 
    for parabola in parabola_info:
        times = list()        
        for item in new_data: 
            if (item[0] == parabola["ball_id"]) & (item[1] <= parabola["end_frame"]) & (item[1] >= parabola["start_frame"]): 
                tracklet_2ds.append([item[0], item[2], item[3]])
                times.append(item[1])
        g = -980
        P0, P = DLT()
        A, b = build_matrix(tracklet_2ds, g, P, times)
        X0, Vx, Y0, Vy, Z0, Vz = svd_solve(A, b)  

        V_init = [round(float(Vx/100), 3), round(float(Vy/100), 3), round(float(Vz/100), 3)]
        XYZ_init = [round(float(X0/100), 3), round(float(Y0/100), 3), round(float(Z0/100), 3)]
        ball_velocity = round(math.sqrt(V_init[0] ** 2 + V_init[1] ** 2 + V_init[2] ** 2), 3)
        start_frame = times[0]
        end_frame = parabola["end_frame"]
        start_time = round((start_frame / 50), 2)
        end_time = round((end_frame / 50), 2)
        pinyin_json_list.append({"ks":start_time, "js":end_time, "qy":"未定义", "dz":"二传", "zdz":"", "init_pos":XYZ_init, "velocity":V_init, "qs":ball_velocity})
        pinyin_dump = {"jdlist":pinyin_json_list}
        with open("calculated_parabola.json","w") as f:
            json.dump(pinyin_dump,f,ensure_ascii=False)

if __name__ == '__main__':
    traj_dir = '/home/nata/Documents/code/BallTrack/output_traj.txt'
    parabola_dir = '/home/nata/Documents/code/BallTrack/2to3/parabola_info.json'
    traj_dir_2view_1 = '/home/nata/Documents/code/BallTrack/output_traj0.txt'
    traj_dir_2view_2 = '/home/nata/Documents/code/BallTrack/output_traj1.txt'
    single_traj_process(traj_dir_2view_2)
    # multi_traj_process(traj_dir, parabola_dir)
    # single_traj_process_2view(traj_dir_2view_1,traj_dir_2view_2)
