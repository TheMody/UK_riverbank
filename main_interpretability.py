import numpy as np
import pandas as pd
from tabpfn_extensions import unsupervised
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
import torch
from plot import plot_locations
import matplotlib.pyplot as plt
from tabpfn_extensions import interpretability
from config import *
import pickle
from utils import find_outliers
filepath = "data/CSI_Data_ALL_28022025.csv"
df = pd.read_csv(filepath)
#print(df.head())

filter_features = ["river", "site" ]
input_features = ["recentRain", "estimatedWidth", "estimatedDepth", "waterFlow", "timestamp"]#"long", "lat",
intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph", "nitrate", "ammonia","phosphate", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther"]#
all_features =  input_features + intersting_columns
target_features = ["pollutionEvidenceNone"]

categorical_features_names = ["recentRain","waterFlow","nitrate","ammonia","waterLevel", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther", "pollutionEvidenceNone"] 

ids = {}
for name in categorical_features_names:
    ids[name] = np.unique(np.asarray(df[name]).astype(str))
   # print(ids[name])
    df[name] = [list(ids[name]).index(a) for a in np.asarray(df[name]).astype(str)]

#find all unique river and site combinations
site_river_combinations = df[["river", "site"]].drop_duplicates()

X = []
Y = []
i = 0
with open("outliers.pkl", "rb") as f:
    p = pickle.load(f)
position = 0

for combination in site_river_combinations.values:
    df_filtered = df[df["river"] == combination[0]]
    df_filtered = df_filtered[df_filtered["site"] == combination[1]]

    if len(df_filtered) < MIN_LENGTH_TIMESERIES:
        continue

    i += 1
    print("length of df: ", len(df_filtered))
    appended = df_filtered[all_features].values
    appended = np.concatenate((appended, np.zeros((appended.shape[0],1))), axis=1)
    appended[:, -1] = i

    #concat the p values 
   # appended = np.concatenate((appended, p[i-1].unsqueeze(-1)), axis=1)
   # print("appended: ", appended)
    X.append(appended)
    Y.append(df_filtered[target_features].values)
    #ise the p values as the target

  #  position += len(df_filtered)
    if i> MAX_N_TIMESERIES:
        break


X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
num_samples = len(X)
# Y = p[0:X.shape[0]]
# Y = np.log(Y)
attribute_names = df_filtered[all_features].columns.tolist() +["site_index"]
reg = TabPFNRegressor(n_estimators=4)

p_index = 0
for combination in site_river_combinations.values:
    df_filtered = df[df["river"] == combination[0]]
    df_filtered = df_filtered[df_filtered["site"] == combination[1]]

    if len(df_filtered) < MIN_LENGTH_TIMESERIES:
       # print("skipping combination: ", combination, len(df_filtered))
        continue

    X_current = np.concatenate((X[:p_index],X[p_index+len(df_filtered):]), axis = 0)
    Y_current = np.concatenate((Y[:p_index],Y[p_index+len(df_filtered):]), axis = 0)
    X_test = X[p_index:p_index + len(df_filtered)]
    Y_test = Y[p_index:p_index + len(df_filtered)]

    reg.fit(torch.tensor(X_current), torch.tensor(Y_current))

    outliers = find_outliers(p[p_index:p_index + len(df_filtered)])
    print("number of outliers in ",combination, len(outliers))
    intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph", "nitrate", "ammonia","phosphate", "recentRain","waterFlow","pollutionEvidenceNone"]

    shap_values = None
    if not len(outliers) == 0:
        shap_values = interpretability.shap.get_shap_values(
            estimator=reg,
            test_x=X_test[outliers],
            attribute_names=attribute_names,
            algorithm="permutation",
        )
    plot_locations(df_filtered, combination, intersting_columns, ids,categorical_features_names, p[p_index:p_index + len(df_filtered)], shap_values)
    p_index += len(df_filtered)
    # Create visualization
  #  fig = interpretability.shap.plot_shap(shap_values)


