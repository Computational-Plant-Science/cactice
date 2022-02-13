from typing import Optional, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import seaborn as sns

from cactice.utils import unit_scale


def plot_grid(grid: np.ndarray, title: Optional[str] = None, cmap='Greens') -> None:
    # plt.figure(figsize=(20, 20))
    values = list(set(np.ravel(grid)))
    labels = np.vectorize(lambda x: str(int(x)) if x != 0 else '')(grid)
    ax = sns.heatmap(
        grid,
        square=True,
        cbar=False,
        annot=labels,
        fmt='',
        cmap=cmap,
        vmin=0,
        vmax=5 if 0 in values else 4,
        alpha=0.5)
    if title: ax.set_title(title)
    plt.show()


def plot_bond_energies(plot: np.ndarray, energies: Dict[str, float], title: Optional[str] = None) -> None:
    plt.figure(figsize=(14, 14))
    labels = np.vectorize(lambda x: str(int(x)) if x != 0 else '')(plot)
    ax = sns.heatmap(
        plot,
        # center=len(class_freqs.keys()) / 2,
        square=True,
        cbar=False,
        annot=labels,
        fmt='',
        cmap="Greens",
        vmin=0,
        vmax=5,
        alpha=0.5)
    scaled_energies = unit_scale(energies)
    for k, v in scaled_energies.items():
        ks = k.split('_')
        i1 = int(ks[0])
        j1 = int(ks[1])
        i2 = int(ks[2])
        j2 = int(ks[3])
        dx = 1 if j1 == j2 else 0
        dy = 1 if i1 == i2 else 0
        p1 = str(int(plot[j1, i1]))
        p2 = str(int(plot[j2, i2]))

        if not (p1 == '0' or p2 == '0'):
            ax.add_patch(ConnectionPatch(
                (i1 + 0.5, j1 + 0.5),
                (i1 + 0.5 + dx, j1 + 0.5 + dy),
                'data',
                'data',
                shrinkA=1,
                shrinkB=1,
                linewidth=(1 - v) * 30,
                color='blue',
                alpha=(1 - v) * 0.2))

    if title: ax.set_title(title)
    plt.show()
