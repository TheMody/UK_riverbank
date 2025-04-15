import numpy as np
from matplotlib import pyplot as plt
def plot_locations(df, combination,interesting_columns,ids_nitrate, ids_ammonia, ids_waterLevel,p=None):
    #visualize timeseries for a single unique feature
   # plt.title(f"Time Series of {combination[0]} {combination[1]}")
    df_filtered = df
    print(combination)
    #df = df[df["site"] == "Bell Combe"]

    # print("length of df: ", len(df_filtered))
    # print("current site", combination[0], combination[1])

    #order list by timestamp
    #df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_filtered = df_filtered.sort_values(by="timestamp")
    #print(df[intersting_columns+["timestamp"]])
    #df = df.set_index("timestamp")

    #visualize all the interesting_columns in one figure
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

    if p is not None:
        p = p.numpy()
        mean_p = np.mean(p)
        std_p = np.std(p)
        # #mark all points that are more than 2 std away from the mean
        outliers = np.where(np.abs(p - mean_p) > 5 * std_p)[0]
        ax = axes.flat[-1]
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Inverse Likelihood")
        ax.scatter(df_filtered["timestamp"],p)
        ax.scatter(df_filtered["timestamp"].iloc[outliers], p[outliers], color="red", label="Outliers")
        ax.set_yscale("log")

    for i,column in enumerate(interesting_columns):
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
        ax.scatter(df_filtered["timestamp"].iloc[outliers], df_filtered[column].iloc[outliers], color="red", label="Outliers")



    plt.tight_layout()
    plt.show()
    try:
        plt.savefig(f"figures/{combination[0]}_{combination[1]}.png")
    except:
        plt.close()
    plt.close()