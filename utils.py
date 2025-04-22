import numpy as np
def find_outliers(p):
    mean_p = np.mean(p)
    std_p = np.std(p)
    # #mark all points that are more than 2 std away from the mean
    outliers = np.where(np.abs(p - mean_p) > 5 * std_p)[0]
    return outliers