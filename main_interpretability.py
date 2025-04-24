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
from dataset import get_ukriver_dataset
X,Y, site_river_combinations, ids = get_ukriver_dataset()
with open("outliers_100.pkl", "rb") as f:
    p = pickle.load(f)
num_samples = len(X)
# Y = p[0:X.shape[0]]
# Y = np.log(Y)
#
reg = TabPFNRegressor(n_estimators=4)
#reg.set_categorical_features(categorical_features)

p_index = 0
for i,combination in enumerate(site_river_combinations.values):
    X_current = np.concatenate(X[0:i] + X[i+1:], axis = 0)
    Y_current = np.concatenate(Y[0:i]+ Y[i+1:], axis = 0)
    X_test = X[i]
    Y_test = Y[i]

    reg.fit(torch.tensor(X_current), torch.tensor(Y_current))

    outliers = find_outliers(p[p_index:p_index + len(X_test)])
    print("number of outliers in ",combination, len(outliers))
    intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph", "nitrate", "ammonia","phosphate", "recentRain","waterFlow","pollutionEvidenceNone"]

   # shap_values = None
   # if not len(outliers) == 0:
    shap_values = interpretability.shap.get_shap_values(
        estimator=reg,
        test_x=X_test,
        attribute_names=attribute_names,
        algorithm="permutation",
    )
    plot_locations(X_test,Y_test, combination, intersting_columns, ids,categorical_features_names, p[p_index:p_index + len(X_test)], shap_values)
    p_index += len(X_test)


