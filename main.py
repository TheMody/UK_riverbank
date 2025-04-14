import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath = "data/CSI_Data_ALL_28022025.csv"
df = pd.read_csv(filepath)
#print(df.head())

filter_features = ["river", "site", "timestamp" ]
input_features = ["recentRain", "estimated_Width", "estimated_Depth", "waterFlow","long", "lat",]
intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "phosphate", "ph", "nitrate", "ammonia"]#

ids_nitrate = np.unique(np.asarray(df["nitrate"]).astype(str))
df["nitrate"] = [list(ids_nitrate).index(a) for a in np.asarray(df["nitrate"]).astype(str)]

ids_ammonia = np.unique(np.asarray(df["ammonia"]).astype(str))
df["ammonia"] = [list(ids_ammonia).index(a) for a in np.asarray(df["ammonia"]).astype(str)]

ids_waterLevel = np.unique(np.asarray(df["waterLevel"]).astype(str))
df["waterLevel"] = [list(ids_waterLevel).index(a) for a in np.asarray(df["waterLevel"]).astype(str)]

#find all unique river and site combinations
site_river_combinations = df[["river", "site"]].drop_duplicates()


#visualize timeseries for a single unique feature
for combination in site_river_combinations.values:

    df_filtered = df[df["river"] == combination[0]]
    df_filtered = df_filtered[df_filtered["site"] == combination[1]]
    #df = df[df["site"] == "Bell Combe"]

    if len(df_filtered) < 10:
        continue
    print("length of df: ", len(df_filtered))
    print("current site", combination[0], combination[1])

    #order list by timestamp
    #df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_filtered = df_filtered.sort_values(by="timestamp")
    #print(df[intersting_columns+["timestamp"]])
    #df = df.set_index("timestamp")

    #visualize all the interesting_columns in one figure
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

    for i,column in enumerate(intersting_columns):
        ax = axes.flat[i]
        ax.set_title(f"Time Series of {column}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(column)
        if column == "nitrate":
            ax.set_yticks(np.arange(0, len(ids_nitrate), 1), ids_nitrate)
            ax.scatter(df_filtered["timestamp"], df_filtered[column], label=column)
        elif column == "ammonia":
            ax.set_yticks(np.arange(0, len(ids_ammonia), 1), ids_ammonia)
            ax.scatter(df_filtered["timestamp"], df_filtered[column], label=column)
        elif column == "waterLevel":
            ax.set_yticks(np.arange(0, len(ids_waterLevel), 1), ids_waterLevel)
            ax.scatter(df_filtered["timestamp"], df_filtered[column], label=column)
        else:
            ax.plot(df_filtered["timestamp"], df_filtered[column], label=column)


    plt.tight_layout()
    try:
        plt.savefig(f"figures/{combination[0]}_{combination[1]}.png")
    except:
        plt.close()
    plt.close()