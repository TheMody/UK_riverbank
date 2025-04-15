import numpy as np
import pandas as pd
from tabpfn_extensions import unsupervised
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
import torch
from plot import plot_locations
import matplotlib.pyplot as plt
from tabpfn_extensions import interpretability

filepath = "data/CSI_Data_ALL_28022025.csv"
df = pd.read_csv(filepath)
#print(df.head())

filter_features = ["river", "site" ]
input_features = ["recentRain", "estimatedWidth", "estimatedDepth", "waterFlow","long", "lat", "timestamp"]
intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph", "nitrate", "ammonia",]#
all_features =  input_features + intersting_columns
target_features = ["phosphate"]

categorical_features_names = ["recentRain","waterFlow","nitrate","ammonia","waterLevel"] 

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
Y = []
i = 0
for combination in site_river_combinations.values:
    df_filtered = df[df["river"] == combination[0]]
    df_filtered = df_filtered[df_filtered["site"] == combination[1]]

    if len(df_filtered) < 20:
        continue
    i += 1
    print("length of df: ", len(df_filtered))
    appended = df_filtered[all_features].values
    appended = np.concatenate((appended, np.zeros((appended.shape[0],1))), axis=1)
    appended[:, -1] = i
   # print("appended: ", appended)
    X.append(appended)
    Y.append(df_filtered[target_features].values)
    if i> 20:
        break
   # 
   # df_y = df_filtered[target_columns]
#df_filtered = df[input_features+intersting_columns]
#X = X[:1000]
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
index_of_nans = np.where(np.isnan(Y))
Y = np.delete(Y, index_of_nans[0], axis=0)
X = np.delete(X, index_of_nans[0], axis=0)
from sklearn.model_selection import train_test_split
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X.shape)
attribute_names = df_filtered[all_features].columns.tolist() +["site_index"]

# print(X)
# print(attribute_names)
# plot_locations(df, combination, intersting_columns, ids_nitrate, ids_ammonia, ids_waterLevel)
    # Initialize models
reg = TabPFNRegressor(n_estimators=4)
#categorical_features = [(all_features).index(c) for c in categorical_features_names]
#reg.set_categorical_features(categorical_features)

reg.fit(torch.tensor(X), torch.tensor(Y))


shap_values = interpretability.shap.get_shap_values(
    estimator=reg,
    test_x=X_test[:50],
    attribute_names=attribute_names,
    algorithm="permutation",
)

# Create visualization
fig = interpretability.shap.plot_shap(shap_values)
