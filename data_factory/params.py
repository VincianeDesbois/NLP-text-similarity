from matplotlib.colors import LinearSegmentedColormap

list_columns_name = [
    "Model",
    "Relevance",
    "Coherence",
    "Empathy",
    "Surprise",
    "Engagement",
    "Complexity",
    "BLEU",
    "ROUGE-1 Recall",
    "ROUGE-1 Precision",
    "ROUGE-1 F-Score",
    "ROUGE-2 Recall",
    "ROUGE-2 Precision",
    "ROUGE-2 F-Score",
    "ROUGE-3 Recall",
    "ROUGE-3 Precision",
    "ROUGE-3 F-Score",
    "ROUGE-4 Recall",
    "ROUGE-4 Precision",
    "ROUGE-4 F-Score",
    "ROUGE-L Recall",
    "ROUGE-L Precision",
    "ROUGE-L F-Score",
    "ROUGE-W-1.2 Recall",
    "ROUGE-W-1.2 Precision",
    "ROUGE-W-1.2 F-Score",
    "ROUGE-S Recall",
    "ROUGE-S Precision",
    "ROUGE-S F-Score",
    "ROUGE-SU Recall",
    "ROUGE-SU Precision",
    "ROUGE-SU F-Score",
    "METEOR",
    "chrF",
    "CIDEr",
    "ROUGE-WE-3 Recall",
    "ROUGE-WE-3 Precision",
    "ROUGE-WE-3 F-Score",
    "BERTScore Precision",
    "BERTScore Recall",
    "BERTScore F1",
    "MoverScore",
    "DepthScore",
    "BaryScore-W",
    "BaryScore-SD-10",
    "BaryScore-SD-5",
    "BaryScore-SD-1",
    "BaryScore-SD-0.5",
    "BaryScore-SD-0.1",
    "BaryScore-SD-0.01",
    "BaryScore-SD-0.001",
    "S3-Pyramid",
    "S3-Responsiveness",
    "SummaQA",
    "InfoLM-FisherRao",
    "InfoLM-R-FisherRao",
    "InfoLM-Sim-FisherRao",
    "BARTScore-SH",
    "BARTScore-HS",
    "SUPERT-Golden",
    "BLANC-Golden",
    "Coverage",
    "Density",
    "Compression",
    "Text length",
    "Novelty-1",
    "Novelty-2",
    "Novelty-3",
    "Repetition-1",
    "Repetition-2",
    "Repetition-3",
    "SUPERT-PS",
    "SUPERT-SS",
    "BLANC-Tune-PS",
    "BLANC-Help-PS",
    "BLANC-Tune-SS",
    "BLANC-Help-SS",
    "BARTScore-PS",
    "BARTScore-SP",
]
list_models_to_keep = [
    "Human",
    "BertGeneration",
    "CTRL",
    "GPT-2",
    "RoBERTa",
    "XLNet",
    "Fusion",
    "HINT",
    "TD-VAE",
]
liste_human_score = [
    "Relevance",
    "Coherence",
    "Empathy",
    "Surprise",
    "Engagement",
    "Complexity",
]
liste_metric = [
    "BLEU",
    "ROUGE-1 F-Score",
    "METEOR",
    "chrF",
    "CIDEr",
    "ROUGE-WE-3 F-Score",
    "BERTScore F1",
    "MoverScore",
    "DepthScore",
    "BaryScore-W",
    "S3-Pyramid",
    "InfoLM-FisherRao",
    "BARTScore-SH",
    "SUPERT-Golden",
    "BLANC-Golden",
    "MENLI_x_BERT",
    "MENLI_x_Mover",
    "SummaQA",
    "Novelty-1",
    "Repetition-1",
]
list_metric_to_keep_max = [
    "Relevance",
    "Coherence",
    "Empathy",
    "Surprise",
    "Engagement",
    "Complexity",
    "BLEU",
    "ROUGE-1 F-Score",
    "METEOR",
    "chrF",
    "CIDEr",
    "ROUGE-WE-3 F-Score",
    "BERTScore F1",
    "MoverScore",
    "SUPERT-Golden",
    "S3-Pyramid",
    "SummaQA",
    "BARTScore-SH",
    "BLANC-Golden",
    "MENLI_x_BERT",
    "MENLI_x_Mover",
]
list_metric_to_keep_min = ["DepthScore", "BaryScore-W", "InfoLM-FisherRao"]
cmap_red_green_red = LinearSegmentedColormap.from_list(
    "RedGreenRed", ["firebrick", "greenyellow", "firebrick"]
)