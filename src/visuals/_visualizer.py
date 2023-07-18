from typing import (
    Any,
    List
)

from pandas import DataFrame

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

sns.set()


class Visualizer:

    @staticmethod
    def correlation_heatmap(
        correlation_df: DataFrame
    ) -> None:

        plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        cmap = sns.diverging_palette(10, 250, as_cmap=True)
        mask = np.zeros_like(correlation_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        sns.heatmap(correlation_df, mask=mask, cmap=cmap, vmax=.6, vmin=-.6, annot=True, annot_kws={'size': 5})

        plt.show()

    @staticmethod
    def boxplot(
        x: List[Any],
        positions: List[int],
        labels: List[str],
        y_label: str
    ) -> None:

        fig, ax = plt.subplots(figsize=(8, 6))

        box = ax.boxplot(
            x=x,
            positions=positions,
            labels=labels,
            widths=6,
            patch_artist=True,
            showfliers=False,
            meanline=False,
            medianprops={
                'linewidth': 2,
                'color': 'crimson'
            }
        )

        box['boxes'][0].set(
            color='black',
            linewidth=1,
            facecolor='mediumaquamarine',
            hatch='x'
        )

        plt.ylabel(y_label)
        plt.xlim(0, 102)
        plt.xticks(fontsize=12)
        plt.tick_params(bottom=True, labelbottom=True)
        plt.grid(False)
        plt.tight_layout()

        plt.show()
