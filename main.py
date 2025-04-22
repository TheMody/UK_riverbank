import numpy as np
import pandas as pd
from tabpfn_extensions import unsupervised
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
import torch
from plot import plot_locations
import matplotlib.pyplot as plt
import pickle
from config import *
filepath = "data/CSI_Data_ALL_28022025.csv"
df = pd.read_csv(filepath)

#print(df.head())

filter_features = ["river", "site" ]
input_features = ["recentRain", "estimatedWidth", "estimatedDepth", "waterFlow","long", "lat", "timestamp"]
intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph", "nitrate", "ammonia","phosphate", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther"]#
all_features =  input_features + intersting_columns

categorical_features_names = ["recentRain","waterFlow","nitrate","ammonia","waterLevel", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther"] 

ids = []
for name in categorical_features_names:
    ids.append(np.unique(np.asarray(df[name]).astype(str)))
    df[name] = [list(ids[-1]).index(a) for a in np.asarray(df[name]).astype(str)]

# ids_river = np.unique(np.asarray(df["river"]).astype(str))
# df["river"] = [list(ids_river).index(a) for a in np.asarray(df["river"]).astype(str)]

# ids_site= np.unique(np.asarray(df["site"]).astype(str))
# df["site"] = [list(ids_site).index(a) for a in np.asarray(df["site"]).astype(str)]

ids_waterFlow = np.unique(np.asarray(df["waterFlow"]).astype(str))
df["waterFlow"] = [list(ids_waterFlow).index(a) for a in np.asarray(df["waterFlow"]).astype(str)]

ids_recentRain = np.unique(np.asarray(df["recentRain"]).astype(str))
df["recentRain"] = [list(ids_recentRain).index(a) for a in np.asarray(df["recentRain"]).astype(str)]

ids_nitrate = np.unique(np.asarray(df["nitrate"]).astype(str))
df["nitrate"] = [list(ids_nitrate).index(a) for a in np.asarray(df["nitrate"]).astype(str)]

ids_ammonia = np.unique(np.asarray(df["ammonia"]).astype(str))
df["ammonia"] = [list(ids_ammonia).index(a) for a in np.asarray(df["ammonia"]).astype(str)]

ids_waterLevel = np.unique(np.asarray(df["waterLevel"]).astype(str))
df["waterLevel"] = [list(ids_waterLevel).index(a) for a in np.asarray(df["waterLevel"]).astype(str)]

#find all unique river and site combinations
site_river_combinations = df[["river", "site"]].drop_duplicates()

X = []
i = 0
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
   # print("appended: ", appended)
    X.append(appended)
    if i> MAX_N_TIMESERIES:
        break
   # 
   # df_y = df_filtered[target_columns]
#df_filtered = df[input_features+intersting_columns]
#X = X[:1000]
X = np.concatenate(X, axis=0)
print(X.shape)
attribute_names = df_filtered.columns.tolist()

# print(X)
# print(attribute_names)
# plot_locations(df, combination, intersting_columns, ids_nitrate, ids_ammonia, ids_waterLevel)
    # Initialize models
clf = TabPFNClassifier(n_estimators=4)
reg = TabPFNRegressor(n_estimators=4)
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf, tabpfn_reg=reg
)
categorical_features = [(all_features).index(c) for c in categorical_features_names]
model_unsupervised.set_categorical_features(categorical_features)

model_unsupervised.fit(torch.tensor(X))
p = model_unsupervised.outliers(torch.tensor(X), n_permutations=3)

#save p values
with open("outliers.pkl", "wb") as f:
    pickle.dump(p.numpy(), f)

print(p)
p = p.numpy()
i = 0
p_index = 0
for combination in site_river_combinations.values:
    df_filtered = df[df["river"] == combination[0]]
    df_filtered = df_filtered[df_filtered["site"] == combination[1]]

    if len(df_filtered) < MAX_N_TIMESERIES:
        continue

    plot_locations(df_filtered, combination, intersting_columns, ids, p[p_index:p_index + len(df_filtered)])
    p_index += len(df_filtered)

