from data_factory.params import list_columns_name
import pandas as pd
import ast

def create_clean_dataframe(df_ann: pd.DataFrame, df_scoring: pd.DataFrame, list_columns_name: list) -> pd.DataFrame:
    """
    This function takes in two dataframes, df_ann and df_scoring, and creates a new, cleaned dataframe df_ann_complete.
    
    Args:
    - df_ann: a pandas dataframe containing information about stories and their annotations
    - df_scoring: a pandas dataframe containing scoring metrics for each model and story
    - list_columns_name: a list of new column names for df_scoring
    
    Returns:
    - df_complete: a pandas dataframe containing cleaned and merged data from df_ann and df_scoring
    """
    
    # Set column names of df_scoring
    df_scoring.columns = list_columns_name
    
    # Create new dataframe with specific columns from df_ann and remove duplicates
    df_ann_part = df_ann[['Story ID', 'Prompt', 'Human', 'Story', 'Model']].drop_duplicates().copy()
    
    # Iterate through each scoring metric and model, and append values to corresponding list
    for metric in list_columns_name[1:]:
        all_metric_values = []
        for model in df_scoring["Model"].unique(): 
            metric_values = ast.literal_eval(df_scoring.loc[df_scoring["Model"] == model, metric].iloc[0])
            all_metric_values += metric_values
        df_ann_part[metric] = all_metric_values
    
    # Copy df_ann_part to create final dataframe
    df_complete = df_ann_part.copy()
    
    return df_complete