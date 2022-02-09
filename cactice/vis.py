from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def show_plot(plot: np.ndarray, title: Optional[str] = None, cmap='Greens') -> None:
    plt.figure(figsize=(20, 20))
    values = list(set(np.ravel(plot)))
    labels = np.vectorize(lambda x: str(int(x)) if x != 0 else '')(plot)
    ax = sns.heatmap(
        plot,
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
