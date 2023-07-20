from typing import (
    Any,
    Dict,
    Tuple
)

import numpy as np
from numpy import ndarray

import pandas as pd
from pandas import DataFrame

from src.applicability_domain import ApplicabilityDomain
from src.enums import (
    FeatureImportanceType,
    MetricType
)
from src.metrics import Metrics
from src.visuals import Visualizer


def analyze_model(
    model: Any,
    x_train: ndarray,
    x_validation: ndarray,
    x_bt: ndarray,
    y_train: ndarray,
    y_validation: ndarray,
    y_bt: ndarray,
    x_train_all: ndarray,
    y_train_all: ndarray,
    x_all: ndarray,
    y_all: ndarray,
    cv: Any,
    column_names: ndarray,
    b_plot_y_randomization: bool = False,
    b_plot_feature_importances: bool = True,
    title: str = 'QSRR Model'
) -> Tuple[DataFrame, DataFrame]:

    print(
        f"Results Analysis for : {title}"
    )

    # Predictions
    _y_train_hat: ndarray = model.predict(x_train).ravel()
    _y_validation_hat: ndarray = model.predict(x_validation).ravel()
    _y_bt_hat: ndarray = model.predict(x_bt).ravel()

    _predictions_df: DataFrame = pd.DataFrame()
    _predictions_df["y"] = np.hstack((y_train, y_validation, y_bt)).ravel()
    _predictions_df["y_hat"] = np.hstack((_y_train_hat, _y_validation_hat, _y_bt_hat)).ravel()
    _predictions_df["residuals"] = _predictions_df["y_hat"] - _predictions_df["y"]
    _predictions_df["train_test"] = \
        ["Train" for _ in range(len(_y_train_hat))] \
        + ["Validation" for _ in range(len(_y_validation_hat))] \
        + ["BT" for _ in range(len(_y_bt_hat))]

    # Applicability domain
    _hat_star, _hat_train, _hat_validation, _hat_bt, _res_scaled_train, _res_scaled_validation, _res_scaled_bt = \
        ApplicabilityDomain.calculate(
            x_train=x_train,
            y_train=y_train,
            y_train_hat=_y_train_hat,
            x_validation=x_validation,
            y_validation=y_validation,
            y_validation_hat=_y_validation_hat,
            x_bt=x_bt,
            y_bt=y_bt,
            y_bt_hat=_y_bt_hat
        )

    _predictions_df["leverage"] = np.hstack((_hat_train, _hat_validation, _hat_bt)).ravel()
    _predictions_df["scaled_residuals"] = np.hstack(
        (_res_scaled_train, _res_scaled_validation, _res_scaled_bt)
    ).ravel()

    # Metrics
    _metrics_dict: Dict[str, Any] = {
        "r2_train": Metrics.get_metric(y_train, _y_train_hat, MetricType.R2),
        "rmse_train": Metrics.get_metric(y_train, _y_train_hat, MetricType.RMSE),
        "r2_validation": Metrics.get_metric(y_validation, _y_validation_hat, MetricType.R2),
        "rmse_validation": Metrics.get_metric(y_validation, _y_validation_hat, MetricType.RMSE),
        "r2_bt": Metrics.get_metric(y_bt, _y_bt_hat, MetricType.R2),
        "rmse_bt": Metrics.get_metric(y_bt, _y_bt_hat, MetricType.RMSE),
    }
    _metrics_df: DataFrame = pd.DataFrame.from_dict(_metrics_dict, orient="index")

    # Predictive Ability Plot
    Visualizer.predictive_ability_plot(
        y_train=y_train,
        y_train_hat=_y_train_hat,
        y_validation=y_validation,
        y_validation_hat=_y_validation_hat,
        y_bt=y_bt,
        y_bt_hat=_y_bt_hat
    )

    # Residual Plot
    Visualizer.residual_plot(
        y_train=y_train,
        y_train_hat=_y_train_hat,
        y_validation=y_validation,
        y_validation_hat=_y_validation_hat,
        y_bt=y_bt,
        y_bt_hat=_y_bt_hat
    )

    # Applicability Domain Plot
    Visualizer.applicability_domain_plot(
        hat_train=_hat_train,
        hat_validation=_hat_validation,
        hat_bt=_hat_bt,
        res_scaled_train=_res_scaled_train,
        res_scaled_validation=_res_scaled_validation,
        res_scaled_bt=_res_scaled_bt,
        hat_star=_hat_star
    )

    # Y-Randomization Plot
    if b_plot_y_randomization:
        Visualizer.y_randomization_plot(
            model=model,
            cv=cv,
            x_train_all=x_train_all,
            y_train_all=y_train_all
        )

    # Feature Importance Plots
    if b_plot_feature_importances:

        Visualizer.feature_importance_plot(
            model=model,
            x=x_all,
            y=y_all,
            feature_importance_type=FeatureImportanceType.MeanImpurityDecrease,
            column_names=column_names
        )

        Visualizer.feature_importance_plot(
            model=model,
            x=x_all,
            y=y_all,
            feature_importance_type=FeatureImportanceType.FeaturePermutation,
            column_names=column_names
        )

        Visualizer.feature_importance_plot(
            model=model,
            x=x_train_all,
            y=y_train_all,
            feature_importance_type=FeatureImportanceType.SHAP,
            column_names=column_names
        )

    return _metrics_df, _predictions_df
