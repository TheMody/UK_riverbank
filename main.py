import numpy as np
import pandas as pd
from tabpfn_extensions import unsupervised
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
import torch
from plot import plot_locations
import matplotlib.pyplot as plt
import pickle
from config import *
from dataset import get_ukriver_dataset

X,_,_,_ = get_ukriver_dataset()
X = np.concatenate(X, axis=0)
print(X.shape)
#attribute_names = df_filtered.columns.tolist()

# print(X)
# print(attribute_names)
# plot_locations(df, combination, intersting_columns, ids_nitrate, ids_ammonia, ids_waterLevel)
    # Initialize models
clf = TabPFNClassifier(n_estimators=4)
reg = TabPFNRegressor(n_estimators=4)
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf, tabpfn_reg=reg
)
categorical_features = [(all_features).index(c) for c in categorical_features_names if c in all_features]
model_unsupervised.set_categorical_features(categorical_features)

model_unsupervised.fit(torch.tensor(X))
p = model_unsupervised.outliers(torch.tensor(X), n_permutations=3)

#save p values
with open("outliers.pkl", "wb") as f:
    pickle.dump(p.numpy(), f)

print(p)
p = p.numpy()

# p_index = 0
# for combination in site_river_combinations.values:
#     df_filtered = df[df["river"] == combination[0]]
#     df_filtered = df_filtered[df_filtered["site"] == combination[1]]

#     if len(df_filtered) < MAX_N_TIMESERIES:
#         continue

#     plot_locations(df_filtered, combination, intersting_columns, ids,categorical_features_names, p[p_index:p_index + len(df_filtered)])
#     p_index += len(df_filtered)

