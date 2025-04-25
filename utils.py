import numpy as np
import pandas as pd
from config import *
def find_outliers(p):
    mean_p = np.mean(p)
    std_p = np.std(p)
    # #mark all points that are more than 2 std away from the mean
    outliers = np.where(np.abs(p - mean_p) > 5 * std_p)[0]
    return outliers
from sklearn.preprocessing import MinMaxScaler
def impute_timeseries(df,overall_start, overall_end, intervall = 1000):
    #this will regularize the time series to have regular reporting intervalls and will impute missing values using nearest neighbors
    df.sort_values(by="timestamp")

    interpolated_column = []
    for i in range(overall_start, overall_end, intervall):
        if i in df["timestamp"].values:
            interpolated_column.append(df[df["timestamp"] == i].values[0])
        else:
            #find nearest neighbor
            nearest = df.iloc[(df["timestamp"] - i).abs().argsort()[:1]]    
            if abs(nearest["timestamp"].values[0]-i) < intervall * 2:
                #if the nearest neighbor is within the intervall, use it
                interpolated_column.append(nearest.values[0])
            else:
                interpolated_column.append([NAN_VALUE]*df.columns.shape[0])

        
    interpolated_column = np.array(interpolated_column)
    df_new = pd.DataFrame(interpolated_column, columns=df.columns)
    df_new["timestamp"] =  [i for i in range(overall_start, overall_end, intervall)]
    df_new.fillna(NAN_VALUE,inplace=True)
    return df_new

