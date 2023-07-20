from typing import (
    Any,
    List
)

from pandas import DataFrame

import numpy as np
from numpy import ndarray

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

    @staticmethod
    def predictive_ability_plot(
        y_train: ndarray,
        y_train_hat: ndarray,
        y_validation: ndarray,
        y_validation_hat: ndarray,
        y_bt: ndarray,
        y_bt_hat: ndarray
    ):
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(hspace=0.5, wspace=0.7)

        ax = plt.subplot(1, 4, (1, 3))

        train_color = 'dodgerblue'
        valid_color = 'darkorange'
        test_color = 'mediumseagreen'

        ax.scatter(
            y_train,
            y_train_hat,
            c=train_color,
            marker="o",
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        ax.scatter(
            y_validation,
            y_validation_hat,
            c=valid_color,
            marker="^",
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        ax.scatter(
            y_bt,
            y_bt_hat,
            facecolor=test_color,
            marker="s",
            alpha=0.5,
            edgecolors='black',
            linewidths=0.5
        )

        _y_all: ndarray = np.hstack(
            (
                y_train.ravel(),
                y_train_hat.ravel(),
                y_validation.ravel(),
                y_validation.ravel(),
                y_bt.ravel(),
                y_bt_hat.ravel())
        ).ravel()

        ax.plot(y_train, y_train, c='black', alpha=0.8, linewidth=1)

        ax.legend(('Training Set', 'Validation Set', 'BT Set', '1:1 line'), loc='best')

        ax.set_xlabel("$tR_{Experimental}$")
        ax.set_ylabel("$tR_{Predicted}$")

        ###############################    Relative Error    ####################################
        if relative_error:
            # Relative Error (%) -Boxplot
            error_test = (abs(y_test - y_blindtest.ravel()) / y_test) * 100  # calculationRelative Error (%)

            ax_box = ax1.inset_axes([0.7, 0.05, 0.4, 0.4])

            box = ax_box.boxplot(error_test, labels=[''], showfliers=False, patch_artist=True, widths=0.1, )
            box['boxes'][0].set(color='black', linewidth=1, facecolor=test_color, alpha=0.8)

            # remove spines
            ax_box.spines["top"].set_visible(False)
            ax_box.spines["right"].set_visible(False)
            ax_box.spines["bottom"].set_visible(False)
            # make y axis closer to the box
            ax_box.spines['left'].set_position(('axes', 0.3))

            # change background color to invisible
            ax_box.set_facecolor('none')
            # x and y label
            ax_box.set_ylabel("Relative Error (%)", fontsize=14)
            # plt.title("")
            plt.grid(False)
            plt.savefig(f'{out_path}/test.svg', dpi=300, bbox_inches='tight', format='svg')
            plt.show()
