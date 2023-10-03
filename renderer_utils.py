import traceback

import numpy as np
import os

def hsv2bgr(h, s, v):
    c = s
    h1 = h / 60
    x = c * (1 - abs(h1 % 2 - 1))
    z = np.zeros_like(h)

    dst = np.zeros((h.shape[0], 3))

    vals = [[c, x, z], [x, c, z], [z, c, x], [z, x, c], [x, z, c], [c, z, x]]

    for i in range(6):
        index = np.where((i <= h1) & (h1 < i + 1))  # h1是单通道图像，不是标量
        dst[..., 2][index] = (v - c)[index] + vals[i][0][index]    # b
        dst[..., 1][index] = (v - c)[index] + vals[i][1][index]    # g
        dst[..., 0][index] = (v - c)[index] + vals[i][2][index]    # r

    return dst

def color_points_by_z(points):
    z = points[:, 2]
    z = (z - z.min()) / (z.max() - z.min()) * 240
    colors = hsv2bgr(z, np.ones_like(z), np.ones_like(z))
    return colors

def color_by_z(z):
    z = (z - z.min()) / (z.max() - z.min()) * 240
    colors = hsv2bgr(z, np.ones_like(z), np.ones_like(z))
    return colors


extrinsic_matrix_R = np.array([ 1.11924833, -1.10191008, 1.29169431])
extrinsic_matrix_T = np.array([-0.00703028, 0.13272954, -0.12334772])
intrinsic_matrix = np.array([966.248110091494, 0., 953.315532068699, 0., 967.321780170433, 596.556221513083, 0., 0., 1.]).reshape(3, 3)
distortion_coefficients = np.array([-0.013606865600276, 0.013729062099712, 0., 0.])
extrinsic_matrix = np.array([[-0.00826926, -0.99974942, 0.02080192, -0.00703028],
                             [0.13700393, -0.02173917, -0.99033193, 0.13272954],
                             [0.99053599, -0.00533936, 0.13714937, -0.12334772],
                             [0., 0., 0., 1.]])

def affine(X, matrix):
    n = X.shape[0]
    res = np.concatenate((X, np.ones((n, 1))), axis=-1).T
    res = np.dot(matrix, res).T
    return res[..., :-1]
