import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from utils import find_outliers
from config import *
def plot_locations(X,Y, combination,interesting_columns,ids, categorical_feature_names,p=None, shap_values=None):

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 5, figure=fig, wspace=0.4, hspace=0.4)

    if p is not None:
        # #mark all points that are more than 2 std away from the mean
        outliers = find_outliers(p)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title(f"Density Estimation for {combination[0]}_{combination[1]}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Inverse Likelihood")

        print(p.shape)
        print(X[:,all_features.index("timestamp")].shape)
        ax.scatter(X[:,all_features.index("timestamp")],p, color = "yellow")
        ax.scatter(X[outliers,all_features.index("timestamp")], p[outliers], color="red", label="Outliers")
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
        if column in target_features:
            ax.set_yticks(np.arange(0, len(ids[column]), 1), ids[column])
            ax.scatter(X[:,all_features.index("timestamp")], Y[:,target_features.index(column)], label=column)
            ax.scatter(X[outliers,all_features.index("timestamp")], Y[outliers,target_features.index(column)], color="red", label="Outliers")
        else:
            if column in categorical_feature_names:
                ax.set_yticks(np.arange(0, len(ids[column]), 1), ids[column])
                ax.scatter(X[:,all_features.index("timestamp")], X[:,all_features.index(column)], label=column)
            else:
                ax.plot(X[:,all_features.index("timestamp")], X[:,all_features.index(column)], label=column)
            ax.scatter(X[outliers,all_features.index("timestamp")], X[outliers,all_features.index(column)], color="red", label="Outliers")

    if shap_values is not None:
        ax_right = fig.add_subplot(gs[0:4, 3:5])
        ax_right.barh(shap_values.feature_names, np.abs(shap_values.values.mean(axis=0)), color="red")
        ax_right.set_title("SHAP values for pollution evidence")
        ax_right.set_xlabel("SHAP value")
        ax_right.set_ylabel("Feature")
        ax_right.set_yticklabels(shap_values.feature_names)
        ax_right.yaxis.tick_right()
        ax_right.yaxis.set_label_position("right")
        ax_right.invert_xaxis()
        ax_right.legend()


    plt.tight_layout()
    plt.show()
    try:
        plt.savefig(f"newer_figures/{combination[0]}_{combination[1]}.png")
    except:
        plt.close()
    plt.close()