import glob
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def load_image(file_path: str) -> np.ndarray:
    return io.imread(file_path, as_gray=True)


def load_images(data_dir: str) -> List[Tuple[str, np.ndarray]]:
    pattern = os.path.join(data_dir, "*.png")
    file_paths = glob.glob(pattern)
    return [(file_path, load_image(file_path)) for file_path in file_paths]


def sample_images(data_dir: str, n: int, *, seed=None) -> List[Tuple[str, np.ndarray]]:
    pattern = os.path.join(data_dir, "*.png")
    rs = np.random.RandomState(seed)
    file_paths = rs.choice(glob.glob(pattern), size=n)
    return [(file_path, load_image(file_path)) for file_path in file_paths]


def visualize_all(img, skeleton, lines_data, write_path=None):

    def show_image(ax, image, title):
        ax.imshow(image, cmap="gray")
        ax.axis('off')
        ax.set_title(title, fontsize=15)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 20), sharex='all', sharey='all')
    ax = axes.ravel()
    fig.tight_layout()
    show_image(ax[0], img, 'original')
    show_image(ax[1], skeleton, 'skeleton')

    for (A, B, intersection_point, angle) in lines_data:
        ax[2].plot((B[0] / A[0][0], (B[0] - A[0][1] * img.shape[0]) / A[0][0]), (0, img.shape[0]))
        ax[2].plot((B[1] / A[1][0], (B[1] - A[1][1] * img.shape[0]) / A[1][0]), (0, img.shape[0]))

        ann = ax[2].annotate(str(int(angle)), (intersection_point[0], intersection_point[1]), color='white')
        ann.set_fontsize(15)
        ax[2].scatter(intersection_point[0], intersection_point[1], c="y", marker="o")

    show_image(ax[2], skeleton, 'angle')

    if write_path:
        plt.savefig(write_path)
    else:
        plt.show()
