
import numpy as np
from config import *
import pandas as pd
import pickle
from utils import impute_timeseries
from tqdm import tqdm

def get_ukriver_dataset(preprocess = False):
    
    df = pd.read_csv(filepath)

    ids = {}
    for name in categorical_features_names:
        ids[name] = np.unique(np.asarray(df[name]).astype(str))
        df[name] = [list(ids[name]).index(a) for a in np.asarray(df[name]).astype(str)]

    #find all unique river and site combinations
    filter_features_combinations = df[filter_features].drop_duplicates()

    X = []
    Y = []
    i = 0
    overall_start = int(df["timestamp"].min())
    overall_end = int(df["timestamp"].max())
    intervall = int(abs(overall_start - overall_end)/100)
    print("preprocessing data")
    for combination in tqdm(filter_features_combinations.values):
        for a,value in enumerate(combination):
            if a == 0:
                df_filtered = df[df[filter_features[a]] == value]
            else:
                df_filtered = df_filtered[df_filtered[filter_features[a]] == value]

        if len(df_filtered) < MIN_LENGTH_TIMESERIES:
            continue
        df_filtered = df_filtered.sort_values(by="timestamp")

        if preprocess:
            df_filtered = impute_timeseries(df_filtered[all_features], overall_start, overall_end, intervall)
        i += 1
       # print("length of df: ", len(df_filtered))
        appended = df_filtered[all_features].values
        appended = np.concatenate((appended, np.zeros((appended.shape[0],1))), axis=1)
        appended[:, -1] = i

        X.append(appended)
        #Y.append(df_filtered[target_features].values)

        if i> MAX_N_TIMESERIES:
            break


    return X,Y, filter_features_combinations, ids