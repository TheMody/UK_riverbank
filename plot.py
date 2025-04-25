import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from utils import find_outliers
from config import *
#expects x to be a 2d array with shape (n_timesteps, n_features)
def plot_preprocessed(X, X_pred, X_pred_margin):
    fig = plt.figure(figsize=(35, 20))
    gs = gridspec.GridSpec(4, 5, figure=fig, wspace=0.4, hspace=0.4)
    timeseries_mask = X[ :, 0] != NAN_VALUE
    timeseries_length = np.sum(timeseries_mask)
    X_pred_margin = np.sqrt(np.exp(X_pred_margin))
    for i in range(X.shape[1]):
        timeseries = X[ :, i]
        ax = fig.add_subplot(gs[i//5, i%5])
        ax.set_title(f"Time Series of {all_features[i]}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(all_features[i])
        ax.plot(range(timeseries_length),timeseries[timeseries_mask], color="blue", label="Original", zorder = 3)

        predicted_timeseries = X_pred[ :, i]
        if i not in categorical_features_indices:
            y_lower = predicted_timeseries - X_pred_margin[ :, i]
            y_upper = predicted_timeseries + X_pred_margin[ :, i]
            ax.fill_between(range(timeseries_length), y_lower[timeseries_mask], y_upper[timeseries_mask], color='lightcoral', alpha=0.4, label='±1 Standard Deviation', zorder=2)
        # else:
        #     y_lower = predicted_timeseries - (1-X_pred_margin[ :, i])*2
        #     y_upper = predicted_timeseries + (1-X_pred_margin[ :, i])*2
        #     ax.fill_between(range(timeseries_length), y_lower[timeseries_mask], y_upper[timeseries_mask], color='lightcoral', alpha=0.4, label='uncertainity region (not really accurate)', zorder=2)
        ax.plot(range(timeseries_length),predicted_timeseries[timeseries_mask], color="red", label="Predicted", zorder = 3)
        ax.legend()
        if i == 19:
            break

    #plt.show()
    plt.savefig(f"newer_figures/lstmperformance.png")
    plt.close()
    

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