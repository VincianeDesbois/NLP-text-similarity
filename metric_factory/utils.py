
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from typing import Tuple, List
import matplotlib.pyplot as plt
import lightgbm as lgb 


def create_taukendall_corr(list_metrics,new_dict):
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
        taukendall_corr["ROUGE-WE-3 F-Score"]["ROUGE-WE-3 F-Score"] = new_dict[i] 
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
      elif val3 != "ROUGE-1" and val3 != "ROUGE-WE-3" : 
        taukendall_corr["BERTScore F1"][val3] = new_dict[i]
    if val2 == "BERTScore":
      taukendall_corr[val1]["BERTScore F1"] = new_dict[i]
    elif val1 != "ROUGE-1" and val2 != "ROUGE-1" and val1 != "ROUGE-WE-3" and val2 != "ROUGE-WE-3" and val1 != "BERTScore" and val2 != "BERTScore":
      taukendall_corr[val1][val2] = new_dict[i] 
  taukendall_corr.drop(["SummaQA"], axis=0, inplace=True)
  taukendall_corr.drop(["SummaQA"], axis=1, inplace=True)
  return taukendall_corr
  
def taukendall_complementarity_visualization(taukendall_corr,figsize ):
  plt.figure(figsize=figsize)
  taukendall_corr =taukendall_corr.astype(float)
  mask = np.triu(taukendall_corr)
  ax = sns.heatmap(taukendall_corr, vmin=0, vmax=0.7, annot=False, cmap="viridis", mask=mask)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  ax.set_title('Complementarity between metrics obtained with the Kendall method', fontdict={'fontsize':12}, pad=12);
  return 


def get_X_Y(df_complete:pd.DataFrame, target: str, MENLI: bool, list_model_MENLI:List, list_metric_AEM:List, list_metric_MENLI:List, list_metric_human:List) -> dict:

  if MENLI == True:
    df = df_complete[df_complete['Model'].isin(list_model_MENLI)]
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

  return {'Y': Y, 'X_AEM': X_AEM, 'X_human': X_human, 'X_combined': X_combined}

def compute_lgb_reg(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[object, list, float] :

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15) 
  model = lgb.LGBMRegressor(num_leaves=50, n_estimators=100)
  model.fit(X_train, Y_train)
  y_pred = model.predict(X_test)

  mse = mean_squared_error(Y_test, y_pred)
  rmse = mse**(0.5)
  print("RMSE: %.2f" % rmse)

  return model, y_pred, rmse

def compute_cv_lgb(X, Y):
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
  return {"RMSE": - scores["test_score"].mean()}

def get_scores(df_complete, metric, list_model_MENLI, list_metric_AEM, list_metric_MENLI, list_metric_human):
  
  for i in range(2):

    if i == 0:
      dic_var = get_X_Y(df_complete, metric, False, list_model_MENLI, list_metric_AEM, list_metric_MENLI, list_metric_human)

      print("Prédictions sans MENLI : ")
    else:
      dic_var = get_X_Y(df_complete, metric, True, list_model_MENLI, list_metric_AEM, list_metric_MENLI, list_metric_human)
      print()
      print("Prédictions avec MENLI : ")
    
    target = dic_var['Y']
    for x in ['X_AEM', 'X_human', 'X_combined']:
      X = dic_var[x]
      print('Modèle avec métriques ' + "'" + x.replace('X_', '') + "'", compute_cv_lgb(X, target))

