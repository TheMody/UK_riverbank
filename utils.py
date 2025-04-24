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

    #normalize all columns except the timestamp
    df_cp = df.copy()
    df = df.drop(columns=["timestamp"])
    scaler = MinMaxScaler()
    normalized_df = scaler.fit_transform(df)
    df = pd.DataFrame(normalized_df, columns=df.columns)
    #add the timestamp column again
    df["timestamp"] = df_cp["timestamp"].values

    start = df["timestamp"].min()
    end = df["timestamp"].max()
    #build a new dataframe with the same columns as df but a intervall number of rows
    new_dict = {}
    # for column in df.columns:
    #     if column == "timestamp":
    #         new_dict[column] = [i for i in range(overall_start, overall_end, intervall)]
    #     else:
    #         
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

            
           # new_dict = interpolated_column
    #now normalize timestamp
    #new_dict["timestamp"] = (new_dict["timestamp"] - np.min(new_dict["timestamp"])) / (np.max(new_dict["timestamp"]) - np.min(new_dict["timestamp"]))

    #print(interpolated_column)
    interpolated_column = np.array(interpolated_column)
    df_new = pd.DataFrame(interpolated_column, columns=df.columns)
    df_new["timestamp"] =  scaler.fit_transform([[i] for i in range(overall_start, overall_end, intervall)])
    df_new.fillna(NAN_VALUE,inplace=True)
    return df_new

