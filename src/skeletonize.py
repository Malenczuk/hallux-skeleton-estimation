from copy import deepcopy
from typing import Tuple

from fil_finder import FilFinder2D
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


def skeletonize(img: np.ndarray, *, visualize=False) -> Tuple[FilFinder2D, FilFinder2D]:
    fil = FilFinder2D(img, distance=250 * u.pc, mask=img)
    fil.preprocess_image(flatten_percent=85)
    fil.medskel(verbose=False)
    fil_pruned = deepcopy(fil)

    for _ in range(2):
        fil_pruned.analyze_skeletons(
            prune_criteria='length',
            skel_thresh=100 * u.pix,
            branch_thresh=40 * u.pix,
            max_prune_iter=25)

    if visualize:
        visualize_skeleton(img, fil, fil_pruned)

    return fil, fil_pruned


def visualize_skeleton(img: np.ndarray, fil: FilFinder2D, fil_pruned: FilFinder2D) -> None:

    def show_image(ax, image, title):
        ax.imshow(image, cmap="gray")
        ax.axis('off')
        ax.set_title(title, fontsize=50)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 20), sharex='all', sharey='all')
    ax = axes.ravel()
    show_image(ax[0], img, 'original')
    show_image(ax[1], fil.skeleton, 'skeleton original')
    show_image(ax[2], fil_pruned.skeleton_longpath, 'skeleton pruned')
    fig.tight_layout()
    plt.show()
