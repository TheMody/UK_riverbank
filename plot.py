import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from utils import find_outliers
def plot_locations(df, combination,interesting_columns,ids, categorical_feature_names,p=None, shap_values=None):

    df_filtered = df
    print(combination)


    df_filtered = df_filtered.sort_values(by="timestamp")

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 5, figure=fig, wspace=0.4, hspace=0.4)

    if p is not None:
        # #mark all points that are more than 2 std away from the mean
        outliers = find_outliers(p)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Inverse Likelihood")
        ax.scatter(df_filtered["timestamp"],p)
        ax.scatter(df_filtered["timestamp"].iloc[outliers], p[outliers], color="red", label="Outliers")
        ax.set_yscale("log")

    x,y = 1,0
    for i,column in enumerate(interesting_columns[:]):
        ax = fig.add_subplot(gs[x, y])
        x = x+1
        if x > 3:
            x = 0
            y = y+1
        ax.set_title(f"Time Series of {column}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(column)
        if column in categorical_feature_names:
            ax.set_yticks(np.arange(0, len(ids[column]), 1), ids[column])
            ax.scatter(df_filtered["timestamp"], df_filtered[column], label=column)
        else:
            ax.plot(df_filtered["timestamp"], df_filtered[column], label=column)
        ax.scatter(df_filtered["timestamp"].iloc[outliers], df_filtered[column].iloc[outliers], color="red", label="Outliers")

    if shap_values is not None:
        ax_right = fig.add_subplot(gs[0:4, 3:5])
        ax_right.barh(shap_values.feature_names, np.abs(shap_values.values.mean(axis=0)), color="red")
        ax_right.set_title("SHAP values")
        ax_right.set_xlabel("SHAP value")
        ax_right.set_ylabel("Feature")
        ax_right.set_yticklabels(shap_values.feature_names, rotation=45)
        ax_right.legend()


    plt.tight_layout()
    plt.show()
    try:
        plt.savefig(f"figures/{combination[0]}_{combination[1]}.png")
    except:
        plt.close()
    plt.close()