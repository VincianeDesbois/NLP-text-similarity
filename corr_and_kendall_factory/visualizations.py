import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib


def print_correlation_map(
    df: pd.DataFrame,
    cmap: matplotlib.colors.LinearSegmentedColormap,
    figsize: Tuple[int, int],
    title: str,
    annot: bool,
):
    plt.figure(figsize=figsize)
    corr = df.corr()
    corr = corr.astype(float)
    mask = np.triu(corr)
    ax = sns.heatmap(corr, vmin=-1, vmax=1, annot=annot, cmap=cmap, mask=mask)
    ax.set_title(title, fontdict={"fontsize": 12}, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    return


def print_max_min_heatmap(df: pd.DataFrame) -> Tuple:
    fig, ax = plt.subplots(figsize=(18, 7), facecolor="w", edgecolor="k")
    ax = sns.heatmap(
        df,
        annot=False,
        vmax=1.0,
        vmin=0,
        cmap="viridis",
        cbar=False,
        fmt=".2g",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    return fig, ax


def print_score_heatmap(df_all_ranked: pd.DataFrame):
    viridis_palette = sns.color_palette("viridis", as_cmap=True)
    reversed_palette = viridis_palette.reversed()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="w", edgecolor="k")
    ax = sns.heatmap(df_all_ranked, cmap=reversed_palette, linewidths=0.03)
    for i, label in enumerate(ax.get_xticklabels()):
        if i < 6:
            label.set_color("#663D00")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    return


def taukendall_complementarity_visualization(
    taukendall_corr: pd.DataFrame, figsize: Tuple[int, int], annot: bool
):
    plt.figure(figsize=figsize)
    taukendall_corr = taukendall_corr.astype(float)
    mask = np.triu(taukendall_corr)
    ax = sns.heatmap(
        taukendall_corr, vmin=0, vmax=0.7, annot=annot, cmap="viridis", mask=mask
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(
        "Complementarity between metrics obtained with the Kendall method",
        fontdict={"fontsize": 12},
        pad=12,
    )
    return