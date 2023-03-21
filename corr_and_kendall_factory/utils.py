from scipy import stats
import scipy.stats as ss
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict


def create_min_max_scaled_data_without_human_model(
    liste_metric: List,
    df_complete: pd.DataFrame,
    liste_human_score: List,
    list_metric_to_keep_min: List,
    list_models_to_keep: List,
) -> pd.DataFrame:
    liste_submetric = [
        x for x in liste_metric if x != "Novelty-1" and x != "Repetition-1"
    ]
    df = (
        df_complete.groupby("Model")[liste_human_score + liste_submetric]
        .mean()
        .reindex(index=list_models_to_keep)
        .reset_index()
    )
    df_without_human = df.loc[df["Model"] != "Human"]

    df_without_human_positive = df_without_human.copy()
    # Case where it is better to have a low score
    for i in list_metric_to_keep_min:
        df_without_human_positive[i] = -df_without_human[i]

    scaler = MinMaxScaler()
    df_min_max_scaled = df_without_human_positive.set_index("Model")
    df_min_max_scaled[df_min_max_scaled.columns] = scaler.fit_transform(
        df_min_max_scaled[df_min_max_scaled.columns]
    )
    return df_min_max_scaled


def create_df_all_ranked(
    mean_complete: pd.DataFrame,
    list_metric_to_keep_max: List,
    list_metric_to_keep_min: List,
) -> pd.DataFrame:
    df_mean_ranked = mean_complete.copy()
    df_max_ranked = df_mean_ranked[list_metric_to_keep_max].rank(ascending=False)
    df_min_ranked = df_mean_ranked[list_metric_to_keep_min].rank(ascending=True)
    df_all_ranked = df_max_ranked.merge(df_min_ranked, on="Model", how="outer")
    return df_all_ranked


def compute_metrics_complementarity(
    df_complete: pd.DataFrame, liste_human_score: List, liste_metric: List
):
    list_metrics = df_complete[liste_human_score + liste_metric].columns.to_list()
    dict_to_complete = create_dict_for_each_story(
        df_complete, liste_human_score, liste_metric
    )
    dict_taukendall_results = create_dict_taukendall_result(
        df_complete, liste_human_score, liste_metric, dict_to_complete
    )
    taukendall_output = create_taukendall_output(
        dict_taukendall_results,
    )
    taukendall_corr = create_taukendall_corr(list_metrics, taukendall_output)
    return taukendall_corr


def create_dict_for_each_story(
    df_complete: pd.DataFrame, liste_human_score: List, liste_metric: List
) -> Dict:
    global_list = []
    liste_columns_result = liste_human_score + liste_metric
    df_result = pd.DataFrame(columns=liste_columns_result, index=range(0, 96))
    dict_to_complete = {}
    for i in range(0, 96):
        key = "Story " + str(i)
        dict_to_complete[key] = {}
    for index, prompt in enumerate(df_complete["Prompt"].unique()):
        part_list = []
        for metric in df_complete[liste_human_score + liste_metric].columns.to_list():
            part_list = df_complete.loc[df_complete["Prompt"] == prompt][
                metric
            ].to_list()
            if (
                metric == "DepthScore"
                or metric == "BaryScore-W"
                or metric == "InfoLM-FisherRao"
            ):
                neg_list = part_list
                for i in range(len(part_list)):
                    neg_list[i] *= -1
                rank_array = ss.rankdata(neg_list)
                rank_list = rank_array.tolist()
                dict_to_complete["Story " + str(index)][metric] = rank_list
                global_list += rank_list
            elif (
                metric != "DepthScore"
                or metric != "BaryScore-W"
                or metric != "InfoLM-FisherRao"
            ):
                rank_array = ss.rankdata(part_list)
                rank_list = rank_array.tolist()
                dict_to_complete["Story " + str(index)][metric] = rank_list
                global_list += rank_list
    return dict_to_complete


def create_dict_taukendall_result(
    df_complete: pd.DataFrame,
    liste_human_score: List,
    liste_metric: List,
    dict_to_complete: Dict,
) -> Dict:
    list_metrics = df_complete[liste_human_score + liste_metric].columns.to_list()
    dict_taukendall_results = {}
    for story in range(0, 96):
        key_1 = "Story " + str(story)
        dict_taukendall_results[key_1] = {}
        for i in range(len(list_metrics)):
            for j in range(i + 1, len(list_metrics)):
                key_2 = list_metrics[i] + " " + list_metrics[j]
                dict_taukendall_results[key_1][key_2] = []
    for key in dict_to_complete.keys():
        for i in range(len(list_metrics)):
            metric_1 = list_metrics[i]
            for j in range(i + 1, len(list_metrics)):
                metric_2 = list_metrics[j]
                taukendall = stats.kendalltau(
                    dict_to_complete[key][metric_1], dict_to_complete[key][metric_2]
                ).correlation
                dict_taukendall_results[key][str(metric_1) + " " + str(metric_2)] += [
                    taukendall
                ]
    return dict_taukendall_results


def create_taukendall_output(
    dict_taukendall_results,
):
    taukendall_output = {}
    for couple in list(dict_taukendall_results["Story 0"].keys()):
        taukendall_output[couple] = {}
        for couple in list(dict_taukendall_results["Story 0"].keys()):
            list_value = []
            for story in list(dict_taukendall_results.keys()):
                value = dict_taukendall_results[story][couple]
                distance_tau = (1 - value[0]) / 2
                list_value += [distance_tau]
            output_value = np.mean(list_value)
            taukendall_output[couple] = output_value
    return taukendall_output


def create_taukendall_corr(list_metrics: List, new_dict: Dict) -> pd.DataFrame:
    taukendall_corr = pd.DataFrame(columns=list_metrics, index=list_metrics)
    for i in new_dict.keys():
        val1 = i.split()[0]
        val2 = i.split()[1]
        if val1 == "ROUGE-1":
            val3 = i.split()[2]
            if val3 == "ROUGE-WE-3":
                taukendall_corr["ROUGE-1 F-Score"]["ROUGE-WE-3 F-Score"] = new_dict[i]
            if val3 == "BERTScore":
                taukendall_corr["ROUGE-1 F-Score"]["BERTScore F1"] = new_dict[i]
            elif val3 != "BERTScore" and val3 != "ROUGE-WE-3":
                taukendall_corr["ROUGE-1 F-Score"][val3] = new_dict[i]
        if val2 == "ROUGE-1":
            taukendall_corr[val1]["ROUGE-1 F-Score"] = new_dict[i]
        if val1 == "ROUGE-WE-3":
            val3 = i.split()[2]
            if val3 == "BERTScore":
                taukendall_corr["ROUGE-WE-3 F-Score"]["BERTScore F1"] = new_dict[i]
            if val3 == "ROUGE-1":
                taukendall_corr["ROUGE-WE-3 F-Score"]["ROUGE-WE-3 F-Score"] = new_dict[
                    i
                ]
            elif val3 != "BERTScore" and val3 != "ROUGE-1":
                taukendall_corr["ROUGE-WE-3 F-Score"][val3] = new_dict[i]
        if val2 == "ROUGE-WE-3":
            taukendall_corr[val1]["ROUGE-WE-3 F-Score"] = new_dict[i]
        if val1 == "BERTScore":
            val3 = i.split()[2]
            if val3 == "ROUGE-WE-3":
                taukendall_corr["BERTScore F1"]["ROUGE-WE-3 F-Score"] = new_dict[i]
            if val3 == "ROUGE-1":
                taukendall_corr["BERTScore F1"]["ROUGE-F1 F-Score"] = new_dict[i]
            elif val3 != "ROUGE-1" and val3 != "ROUGE-WE-3":
                taukendall_corr["BERTScore F1"][val3] = new_dict[i]
        if val2 == "BERTScore":
            taukendall_corr[val1]["BERTScore F1"] = new_dict[i]
        elif (
            val1 != "ROUGE-1"
            and val2 != "ROUGE-1"
            and val1 != "ROUGE-WE-3"
            and val2 != "ROUGE-WE-3"
            and val1 != "BERTScore"
            and val2 != "BERTScore"
        ):
            taukendall_corr[val1][val2] = new_dict[i]
    taukendall_corr.drop(["SummaQA"], axis=0, inplace=True)
    taukendall_corr.drop(["SummaQA"], axis=1, inplace=True)
    return taukendall_corr