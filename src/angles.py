from typing import List, Tuple

import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage as nd


def find_lines(skeleton: np.ndarray, *, visualize=False) -> list:
    skeletons = split_components(skeleton)
    lines_data = []

    for s in skeletons:
        try:
            lines = find_two_lines(s)

            A, B = to_linear(lines)
            angle = ang(A)
            intersection_point = np.linalg.solve(A, B)

            lines_data.append((A, B, intersection_point, angle))
        except Exception as e:
            print(e)

    if visualize:
        visualize_lines(skeleton, lines_data)

    return lines_data


def split_components(skeleton: np.ndarray) -> List[np.ndarray]:
    label_im, nb_labels = nd.label(thicken(skeleton))

    splited = []
    for n in range(1, nb_labels + 1):
        skeleton_n = np.zeros(skeleton.shape)
        skeleton_n[label_im == n] = 255
        splited.append(skeleton_n)

    return splited


def find_two_lines(thicker_skeleton: np.ndarray):
    rhos = np.linspace(0.5, 2.0, 20)
    thetas = np.linspace(np.pi/15, np.pi/5, 20)
    thresholds = np.linspace(80, 200, 20)

    for rho in rhos:
        for theta in thetas:
            for threshold in thresholds:
                lines = cv2.HoughLines(
                    thicker_skeleton.astype(np.uint8),
                    rho=rho,
                    theta=theta,
                    threshold=int(threshold))

                if lines is not None and len(lines) == 2 and angle_threshold(lines):
                    return lines
    raise Exception("Lines not found")


def angle_threshold(lines):
    A, _ = to_linear(lines)
    return ang(A) > 20


def to_linear(lines):
    A = np.zeros((2, 2), dtype=np.float64)
    B = np.zeros(2, dtype=np.float64)
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        phi = lines[i][0][1]

        A[i] = np.cos(phi, dtype=np.float64), np.sin(phi, dtype=np.float64)
        B[i] = rho

    return A, B


def visualize_lines(skeleton, lines_data):
    for A, B, intersection_point, angle in lines_data:
        plt.plot((B[0] / A[0][0], (B[0] - A[0][1] * skeleton.shape[0]) / A[0][0]), (0, skeleton.shape[0]))
        plt.plot((B[1] / A[1][0], (B[1] - A[1][1] * skeleton.shape[0]) / A[1][0]), (0, skeleton.shape[0]))

        ann = plt.annotate(str(int(angle)), (intersection_point[0], intersection_point[1]), color='white')
        ann.set_fontsize(20)
        plt.scatter(intersection_point[0], intersection_point[1], c="y", marker="o")
    plt.imshow(skeleton, cmap="gray")
    plt.show()


def thicken(skeleton: np.ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    skeleton = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    return skeleton


def ang(A):
    dot_prod = np.dot(A[0], A[1])
    angle = np.arccos(dot_prod, dtype=np.float64)
    ang_deg = np.degrees(angle, dtype=np.float64) % 360

    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    else:

        return ang_deg
