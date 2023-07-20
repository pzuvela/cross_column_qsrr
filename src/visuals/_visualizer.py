from typing import (
    Any,
    Dict,
    List,
    Optional
)

import pandas as pd
from pandas import DataFrame

import numpy as np
from numpy import ndarray

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    learning_curve,
    permutation_test_score
)

import shap

from src.enums import (
    FeatureImportanceType,
    MetricType
)
from src.metrics import Metrics

sns.set()

_COLORS: Dict[str, str] = {
    "train_color": "dodgerblue",
    "validation_color": "darkorange",
    "bt_color": "mediumseagreen",
    "y_randomization_color": "dodgerblue"
}


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
    ) -> None:

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(hspace=0.5, wspace=0.7)

        ax = plt.subplot(1, 4, (1, 3))  # noqa

        ax.scatter(
            y_train,
            y_train_hat,
            c=_COLORS["train_color"],
            marker="o",
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        ax.scatter(
            y_validation,
            y_validation_hat,
            c=_COLORS["validation_color"],
            marker="^",
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        ax.scatter(
            y_bt,
            y_bt_hat,
            facecolor=_COLORS["bt_color"],
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
                y_bt_hat.ravel()
            )
        ).ravel()

        lims = [
            np.min(_y_all), np.max(_y_all)
        ]

        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.legend(
            ('Training Set', 'Validation Set', 'BT Set', '1:1 Line'),
            loc='best'
        )

        ax.set_xlabel("$tR_{Experimental}$")
        ax.set_ylabel("$tR_{Predicted}$")

        # Relative Error (%) -Boxplot
        _re_bt: float = Metrics.get_metric(
            y_bt,
            y_bt_hat,
            MetricType.RE
        )

        ax_box = ax.inset_axes([0.7, 0.05, 0.4, 0.4])

        box = ax_box.boxplot(_re_bt, labels=[''], showfliers=False, patch_artist=True, widths=0.1, )
        box['boxes'][0].set(color='black', linewidth=1, facecolor=_COLORS["bt_color"], alpha=0.8)

        ax_box.spines["top"].set_visible(False)
        ax_box.spines["right"].set_visible(False)
        ax_box.spines["bottom"].set_visible(False)
        ax_box.spines['left'].set_position(('axes', 0.3))

        ax_box.set_facecolor('none')
        ax_box.set_ylabel("Relative Error (%)", fontsize=14)

        plt.show()

    @staticmethod
    def residual_plot(
        y_train: ndarray,
        y_train_hat: ndarray,
        y_validation: ndarray,
        y_validation_hat: ndarray,
        y_bt: ndarray,
        y_bt_hat: ndarray
    ) -> None:

        # Calculate residuals
        _residuals_train: ndarray = y_train_hat - y_train
        _residuals_validation: ndarray = y_validation_hat - y_validation
        _residuals_bt: ndarray = y_bt_hat - y_bt

        # Create subplots with adjusted widths
        fig, (ax_res, ax_hist) = plt.subplots(
            1,
            2,
            figsize=(8, 4),
            gridspec_kw={'width_ratios': [3, 1]}
        )

        # Residual plot for train data
        ax_res.axhline(
            y=0,
            color='black',
            alpha=0.8,
            linewidth=0.7,
            label='_nolegend_'
        )
        ax_res.scatter(
            y_train_hat,
            _residuals_train,
            color=_COLORS["train_color"],
            marker='o',
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        ax_res.scatter(
            y_validation_hat,
            _residuals_validation,
            marker='^',
            color=_COLORS["validation_color"],
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        ax_res.scatter(
            y_bt_hat,
            _residuals_bt,
            color=_COLORS["bt_color"],
            marker='s',
            alpha=0.5,
            edgecolors='black',
            linewidths=0.5
        )
        ax_res.set_xlabel('Predicted Retetion Times')
        ax_res.set_ylabel('Residuals')
        ax_res.set_title('Residual Plot')

        ax_hist.axhline(
            y=0,
            color='black',
            alpha=0.5,
            linewidth=0.6
        )
        ax_hist.hist(
            _residuals_train,
            bins=50,
            color=_COLORS["train_color"],
            alpha=0.8,
            orientation='horizontal',
            edgecolor='black'
        )
        ax_hist.hist(
            _residuals_validation,
            bins=50,
            color=_COLORS["validation_color"],
            alpha=0.6,
            orientation='horizontal',
            edgecolor='black'
        )
        ax_hist.hist(
            _residuals_bt,
            bins=50,
            color=_COLORS["bt_color"],
            alpha=0.5,
            orientation='horizontal',
            edgecolor='black'
        )

        ax_res.legend(['Train', 'Validation', 'Test'], loc="best", ncol=3, fontsize=10)

        ax_hist.set_xlabel('Distribution')
        ax_hist.set_title('Residual Histogram')
        ax_hist.yaxis.tick_right()

        plt.tight_layout()

        plt.grid(False)
        plt.show()

    @staticmethod
    def applicability_domain_plot(
        hat_train: ndarray,
        hat_validation: ndarray,
        hat_bt: ndarray,
        res_scaled_train: ndarray,
        res_scaled_validation: ndarray,
        res_scaled_bt: ndarray,
        hat_star: float
    ) -> None:
        
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(hspace=0.5, wspace=0.7)

        ax = plt.subplot()

        ax.scatter(
            hat_train.ravel(),
            res_scaled_train.ravel(),
            c=_COLORS["train_color"],
            marker="o",
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        ax.scatter(
            hat_validation.ravel(),
            res_scaled_validation.ravel(),
            c=_COLORS["validation_color"],
            marker="^",
            alpha=0.5,
            edgecolors='black',
            linewidths=0.6
        )

        ax.scatter(
            hat_bt.ravel(),
            res_scaled_bt.ravel(),
            facecolor=_COLORS["bt_color"],
            marker="s",
            alpha=0.4,
            edgecolors='black',
            linewidths=0.5
        )

        # draw warning limits
        ax.axhline(3, ls=":", color="tomato", lw=1.5)
        ax.axhline(-3, ls=":", color="tomato", lw=1.5)
        ax.axvline(hat_star, ls=":", color="tomato", lw=1.5)

        h_s = f'h* = {hat_star:.2f}'  # h*
        ax.text(hat_star - 0.025, -2.5, h_s, ha='left', va='top', fontsize=12)

        for _i, (_res, _hat) in enumerate(
            zip(
                np.hstack((res_scaled_train.ravel(), res_scaled_validation.ravel(), res_scaled_bt.ravel())).ravel(),
                np.hstack((hat_train.ravel(), hat_validation.ravel(), hat_bt.ravel())).ravel(),
            )
        ):
            if -3 >= _res >= 3 or _hat >= hat_star:
                ax.annotate(f'{_i}', (_hat, _res), fontsize=12)

        ax.legend(('Train', 'Validation', 'Test'), loc=(0.7, 0.8))

        ax.set_ylim(-6, 6)
        ax.yaxis.set_ticks(np.arange(-6, 7, 1))

        ax.set_xlabel("Leverage")
        ax.set_ylabel("Standardized Residuals")

        plt.show()

    @staticmethod
    def y_randomization_plot(
        x_train_all: ndarray,
        y_train_all: ndarray,
        model: Any,
        cv: Any
    ) -> None:

        _score0, _permutation_scores, _p_value = permutation_test_score(
            model,
            x_train_all,
            y_train_all,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_permutations=1000
        )
        print('Average score after 1000 iteration:', -_permutation_scores.mean())

        fig, ax = plt.subplots(figsize=(6, 6))  # noqa

        ax = sns.histplot(
            data=-_permutation_scores,
            color=_COLORS["y_randomization_color"],
            bins=50,
            edgecolor='black',
            alpha=0.7
        )

        ax.axvline(-_score0, ls="--", color="r")

        score_label = f"RMSE on validation data: {-_score0:.2f} min\n(p-value: {_p_value:.3f} )"

        ax.text(0.08, 0.7, score_label, ha='left', va='top', transform=ax.transAxes)

        ax.set_xlabel("RMSE(validation) score")
        _ = ax.set_ylabel("Probability")

        plt.grid(False)

        plt.show()

    @staticmethod
    def learning_curve_plot(
        model: Any,
        x_train_all: ndarray,
        y_train_all: ndarray,
        train_sizes: ndarray
    ):

        _train_sizes, _train_scores, _test_scores = learning_curve(
            estimator=model,
            X=x_train_all,
            y=y_train_all,
            cv=10,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2',
            n_jobs=1
        )

        _train_mean = np.mean(_train_scores, axis=1)
        _train_std = np.std(_train_scores, axis=1)
        _test_mean = np.mean(_test_scores, axis=1)
        _test_std = np.std(_test_scores, axis=1)

        fig, ax = plt.subplots()  # noqa
        plt.plot(
            _train_sizes,
            _train_mean,
            color=_COLORS["train_color"],
            marker='o',
            markersize=5,
            label='Training R2'
        )
        plt.fill_between(
            train_sizes,
            _train_mean + _train_std,
            _train_mean - _train_std,
            alpha=0.15,
            color='blue'
        )

        plt.plot(
            train_sizes,
            _test_mean,
            color=_COLORS["validation_color"],
            marker='+',
            markersize=5,
            linestyle='--',
            label='Validation R2'
        )
        plt.fill_between(
            train_sizes,
            _test_mean + _test_std,
            _test_mean - _test_std,
            alpha=0.15,
            color=_COLORS["validation_color"]
        )

        plt.title('Learning Curve')

        plt.xlabel('Training Data Size')

        plt.ylabel('R2')

        plt.legend(loc='lower right')

        plt.grid(False)

        plt.show()

    @staticmethod
    def feature_importance_plot(
        model: Any,
        x: ndarray,
        y: ndarray,
        column_names: ndarray,
        feature_importance_type: FeatureImportanceType
    ) -> None:

        if feature_importance_type == FeatureImportanceType.MeanImpurityDecrease:

            _importances = model.feature_importances_
            _forest_importances = pd.Series(_importances, index=column_names).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            _forest_importances.plot.barh(color='dodgerblue', ax=ax)
            plt.title("Feature importances using MDI")
            plt.xlabel("Mean decrease in impurity", fontsize=12)
            plt.tight_layout()
            plt.show()

        elif feature_importance_type == FeatureImportanceType.FeaturePermutation:

            from sklearn.inspection import permutation_importance

            _results = permutation_importance(
                model,
                x,
                y,
                n_repeats=100,
                random_state=42,
                n_jobs=2
            )

            _forest_importances = pd.Series(_results.importances_mean, index=column_names).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(8, 6))

            _palette = sns.color_palette("Blues", 15)
            _forest_importances.plot.barh(color=_palette, ax=ax, edgecolor='black')

            ax.set_xlabel("Mean RMSE decrease")

            plt.tight_layout()

            plt.show()

        elif feature_importance_type == FeatureImportanceType.SHAP:

            fig = plt.figure()

            _explainer = shap.explainers.Tree(model, x, feature_names=column_names)
            _shap_values = _explainer(x, )

            _ax1 = fig.add_subplot(121)
            shap.summary_plot(_shap_values, plot_type='bar', feature_names=column_names, show=False)

            _ax2 = fig.add_subplot(122)
            shap.plots.beeswarm(_shap_values, show=False)

            plt.gcf().set_size_inches(12, 4)

            plt.tight_layout()

            plt.show()

        else:

            raise NotImplementedError()

    @staticmethod
    def latent_variable_plot(
        latent_variables: ndarray,
        rmsecvs: ndarray,
        optimal_n_lvs: int,
        y_max: float
    ) -> None:

        fig, ax = plt.subplots(figsize=(6, 4))  # noqa

        plt.plot(
            latent_variables,
            rmsecvs,
            marker='D',
            color='blue',
            mfc='red'
        )

        plt.axvline(
            optimal_n_lvs,
            ymax=y_max,
            linestyle='--',
            color='black'
        )

        plt.xlabel('n(LVs)', fontsize=12)
        plt.ylabel('RMSECV')
        plt.title('n(LVs) Optimization Plot')

        plt.show()

    @staticmethod
    def coefficient_plot(
        coefficients: ndarray,
        column_names: ndarray
    ):
        _coefficients = pd.Series(
            coefficients.ravel(),
            index=column_names
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))  # noqa

        _palette = sns.color_palette("Blues", 15)

        _coefficients.plot.bar(color=_palette, ax=ax)

        plt.axhline(0, color='black', lw=0.3)

        plt.ylabel("Regression coefficients")

        plt.show()
