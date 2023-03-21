import pandas as pd
import ast
from typing import List


def clean_ann(df_ann: pd.DataFrame, list_models_to_keep: List) -> pd.DataFrame:
    df_ann_clean = (
        df_ann[["Story ID", "Prompt", "Human", "Story", "Model"]]
        .drop_duplicates()
        .copy()
    )
    df_ann_clean = df_ann_clean.loc[df_ann_clean["Model"].isin(list_models_to_keep)]
    return df_ann_clean


def clean_scoring(
    df_scoring: pd.DataFrame, list_columns_name: List, list_models_to_keep: List
) -> pd.DataFrame:
    df_scoring.columns = list_columns_name
    df_scoring = df_scoring.loc[df_scoring["Model"].isin(list_models_to_keep)]
    return df_scoring


def clean_menli(df_menli: pd.DataFrame) -> pd.DataFrame:
    df_menli_clean = df_menli.copy()
    df_menli_clean["Model"] = df_menli_clean["Model"].replace(
        {"BERT": "BertGeneration", "GPT2": "GPT-2", "TDVAE": "TD-VAE"}
    )
    return df_menli_clean


def merge_scoring_menli(
    df_scoring: pd.DataFrame, df_menli_clean: pd.DataFrame
) -> pd.DataFrame:
    df_all_scoring = pd.merge(df_scoring, df_menli_clean, on="Model", how="outer")
    return df_all_scoring


def merge_ann_scoring(
    df_ann: pd.DataFrame, df_all_scoring: pd.DataFrame
) -> pd.DataFrame:
    df_new = df_ann.copy()
    for metric in df_all_scoring.columns[1:]:
        all_metric_values = []
        for model in df_all_scoring["Model"].unique():
            metric_values = ast.literal_eval(
                df_all_scoring.loc[df_all_scoring["Model"] == model, metric].iloc[0]
            )
            all_metric_values += metric_values
        df_new[metric] = all_metric_values
    df_complete = df_new.copy()
    return df_complete