import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import lightgbm as lgb


def taukendall_complementarity_visualization(
    taukendall_corr: pd.DataFrame, figsize: Tuple[int, int]
):
    plt.figure(figsize=figsize)
    taukendall_corr = taukendall_corr.astype(float)
    mask = np.triu(taukendall_corr)
    ax = sns.heatmap(
        taukendall_corr, vmin=0, vmax=0.7, annot=False, cmap="viridis", mask=mask
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(
        "Complementarity between metrics obtained with the Kendall method",
        fontdict={"fontsize": 12},
        pad=12,
    )
    return


def get_X_Y(
    df_complete: pd.DataFrame,
    target: str,
    MENLI: bool,
    list_model_MENLI: List,
    list_metric_AEM: List,
    list_metric_MENLI: List,
    list_metric_human: List,
) -> dict:
    if MENLI == True:
        df = df_complete[df_complete["Model"].isin(list_model_MENLI)]
        AEM_var = list_metric_AEM + list_metric_MENLI

    else:
        df = df_complete
        AEM_var = list_metric_AEM

    human_metrics = list_metric_human.copy()
    del human_metrics[human_metrics.index(target)]

    Y = df[target]
    X_AEM = df[AEM_var]
    X_human = df[human_metrics]
    X_combined = df[AEM_var + human_metrics]

    return {"Y": Y, "X_AEM": X_AEM, "X_human": X_human, "X_combined": X_combined}


def compute_lgb_reg(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[object, list, float]:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
    model = lgb.LGBMRegressor(num_leaves=50, n_estimators=100)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, y_pred)
    rmse = mse ** (0.5)
    print("RMSE: %.2f" % rmse)

    return model, y_pred, rmse


def compute_cv_lgb(X: pd.DataFrame, Y: pd.DataFrame) -> Dict:
    cv_k = 15
    regressor = lgb.LGBMRegressor(num_leaves=50, n_estimators=200)
    scores = cross_validate(
        regressor,
        X,
        Y,
        cv=cv_k,
        scoring=("neg_root_mean_squared_error"),
        return_train_score=True,
    )
    return {"RMSE": -scores["test_score"].mean()}


def get_scores(
    df_complete: pd.DataFrame,
    metric: str,
    list_model_MENLI: List,
    list_metric_AEM: List,
    list_metric_MENLI: List,
    list_metric_human: List,
):
    for i in range(2):
        if i == 0:
            dic_var = get_X_Y(
                df_complete,
                metric,
                False,
                list_model_MENLI,
                list_metric_AEM,
                list_metric_MENLI,
                list_metric_human,
            )

            print("Prédictions sans MENLI : ")
        else:
            dic_var = get_X_Y(
                df_complete,
                metric,
                True,
                list_model_MENLI,
                list_metric_AEM,
                list_metric_MENLI,
                list_metric_human,
            )
            print()
            print("Prédictions avec MENLI : ")

        target = dic_var["Y"]
        for x in ["X_AEM", "X_human", "X_combined"]:
            X = dic_var[x]
            print(
                "Modèle avec métriques " + "'" + x.replace("X_", "") + "'",
                compute_cv_lgb(X, target),
            )