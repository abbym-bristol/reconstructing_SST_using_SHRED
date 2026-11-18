"""evaluation.py"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions.area_weighting import area_weighted_mean

COLOURS = {"SHRED-LSTM":"#ffb000",
           "SHRED-GRU": "#dc267f",
           "SDN": "#1589e8",
           "POD": "#631ff3",
           "RBFI": "#fe5100",
            "Truth": "#000000"
            }


def weighted_mae(y_true, y_pred, weights):
    """Weighted Mean Absolute Error given by:
    
    1/Σw_ij Σ|u_ij - û_ij|w_ij,

    where u is y_true, û is y_pred and w are the corresponding weights.

    Args:
        See evaluate_metrics

    Returns:
        (float): mean MAE value across all weeks in y_pred
    """
    err = y_true - y_pred
    mae_by_week = np.sum(np.abs(err) * weights, axis=1) / np.sum(weights)  # Sum over each week, not across all weeks
    return np.mean(mae_by_week)


def weighted_mse(y_true, y_pred, weights):
    """Weighted Mean Square Error given by:
    
    1/Σw_ij Σ(u_ij - û_ij)**2|w_ij,

    where u is y_true, û is y_pred and w are the corresponding weights.

    Args:
        See evaluate_metrics

    Returns:
        (float): mean MSE value across all weeks in y_pred
        mse_by_week (numpy array): MSE for each week in y_pred
    """
    err = y_true - y_pred
    mse_by_week = np.sum(err**2 * weights, axis=1) / np.sum(weights)  # Sum over each week, not across all weeks
    return np.mean(mse_by_week), mse_by_week


def weighted_r2(y_true, y_pred, weights):
    """Weighted R squared correlation given by:
    
    R2 = 1 - MSE/VAR
    VAR = 1/Σw_ij Σ(u_ij - ū_ij)**2|w_ij
    ū = (Σu_ij * w_ij)/Σw_ij

    where u is y_true, ū is the weighted mean of y_true and w are the corresponding weights.

    Args:
        See evaluate_metrics

    Returns:
        np.mean(r2_by_week) (float): mean R2 value across all weeks in y_pred
        np.mean(var_by_week) (numpy array): mean variance across all weeks in y_pred
    """
    _, mse_by_week = weighted_mse(y_true, y_pred, weights)
    y_bar = np.sum(weights * y_true, axis=1) / np.sum(weights)
    var_by_week = np.sum((y_true - y_bar[:, None])**2 * weights, axis = 1) / np.sum(weights)
    r2_by_week = 1 - mse_by_week/var_by_week
    # mean at the end to treat every week equally
    return np.mean(r2_by_week), np.mean(var_by_week)


def evaluate_metrics(y_true, y_pred, weights):
    """Calculate metrics from test reconstructions

    Args:
        y_true (numpy array): corresponding true y (full-field) for each sequence (week) in test set
        y_pred (numpy array): predicted y (full-field) for each sequence (week) in test set
        weights (numpy array): Area weights corresponding to the predicted full-field shape (y_pred.shape[1])

    Returns:
        None
    """
    # Check if all weights are 1
    label = "unweighted" if np.all(weights == 1) else "area-weighted"

    mse, _ = weighted_mse(y_true, y_pred, weights)
    r2, var = weighted_r2(y_true, y_pred, weights)
    metrics = {
        "MAE": weighted_mae(y_true, y_pred, weights),
        "MSE": mse,
        "R2": r2,
        "variance": var
    }

    print(f"Metrics ({label}):")
    for k,v in metrics.items():
        print(f"{k}:\t{v:.3f}")

    return metrics


def mask_array_by_lat_long(array, lats, longs, area="GS"):
    """Mask an array of data that corresponds to the latitude and longitude coordinates stored in lats and longs,
    to a specified area.

    Args:
        array (numpy array): array of data, of length equal to len(lats) and len(longs)
        lats (numpy array): array of latitudes corresponding to array values
        longs (numpy array): array of longitudes corresponding to array values
        area (str, optional): area to crop to. Current options are "NA" (which will filter to the NA basin), or
            "GS" (Gulf stream region of NA basin). Defaults to "GS".

    Returns:
        masked (numpy array): copy of array with data outside of lat & long masked as NaN.
    """  # noqa: D205
    if area == "GS":
        min_lat, max_lat = 30.0, 50.0
        min_long, max_long = -80.0, -30.0
    elif area == "NA":
        min_lat, max_lat = 0.0, 60.0
        min_long, max_long = -80.0, 0.0
    else:
        ValueError("Area specified is not currently configured.")

    valid_mask = (
        (lats >= min_lat) & (lats <= max_lat) &
        (longs >= min_long) & (longs <= max_long)
    )

    masked = array.copy()
    masked[:, ~valid_mask] = np.nan
    return masked

# ---------- Plotting functions ----------

def plot_area_weighted_mean_SST(y_true, y_pred, area_weights, save_path, test_indices, colour_dict=COLOURS):
    """Area-weighted mean SST over time
    
    Args:
        y_true (numpy_array): ground truth data
        y_pred (dict): keys: model names, values: reconstructions made by model for each item in test set
        area_weights (numpy_array): area weights corresponding to the predicted full-field shape (y_pred.shape[1]) or None
        save_path (str): path to save figure under
        test_indices (dict): {"values": test indices to use, "label": x-axis label to use}
        colour_dict (dict, optional): keys: model names, values: colours specified in hexcode. Defaults to COLOURS.

    Returns:
        None
    """
    truth_sst_mean = area_weighted_mean(data=y_true, area_weights=area_weights)
    sst_mean_dict = {
        test_indices["label"]: test_indices["values"],
        "Truth": truth_sst_mean
    }
    for name, data in y_pred.items():
        test_sst_mean = area_weighted_mean(data=data, area_weights=area_weights)
        sst_mean_dict[name] = test_sst_mean

    sst_mean_df = pd.DataFrame(sst_mean_dict)
    sst_mean_df = sst_mean_df.set_index(test_indices["label"])

    plt.figure()
    for name in y_pred.keys():
        plt.plot(sst_mean_df[name], c=colour_dict[name], ls='-', label=name)
    plt.plot(sst_mean_df["Truth"], c=colour_dict["Truth"], ls='--', label="Truth")
    plt.xticks(rotation=90)
    plt.xlabel(test_indices["label"])
    plt.ylabel("Area-weighted mean SST ($\\degree$C)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.xlim(test_indices["values"][0], test_indices["values"][-1])
    plt.savefig(os.path.join(save_path, "area_weighted_mean_SST.png"), bbox_inches="tight", dpi=300)
    plt.close()


def plot_area_weighted_mean_error(y_true, y_pred, area_weights, save_path, test_indices, colour_dict=COLOURS):
    """Area-weighted mean error in SST over time
    
    Args:
        y_true (numpy_array): ground truth data
        y_pred (dict): keys: model names, values: reconstructions made by model for each item in test set
        area_weights (numpy_array): area weights corresponding to the predicted full-field shape (y_pred.shape[1]) or None
        save_path (str): path to save figure under
        test_indices (dict): {"values": test indices to use, "label": x-axis label to use}
        colour_dict (dict, optional): keys: model names, values: colours specified in hexcode. Defaults to COLOURS.

    Returns:
        diff_dict (dict): keys: model names, values: difference between y_pred[model name] and y_true
    """
    diff_dict = {}
    mean_diff = {}
    for name, data in y_pred.items():
        diff = data - y_true
        mean = area_weighted_mean(data=diff, area_weights=area_weights)
        diff_dict[name] = diff
        mean_diff[name] = np.array(mean)

    sst_diff_dict = {
        test_indices["label"]: test_indices["values"],
    }
    for name, data in mean_diff.items():
        sst_diff_dict[name] = data
    sst_diff_df = pd.DataFrame(sst_diff_dict)
    sst_diff_df = sst_diff_df.set_index(test_indices["label"])

    plt.figure(figsize=(8,8))
    for name, data in sst_diff_df.items():
        plt.plot(data, c=colour_dict[name], label=name)
    plt.xticks(rotation=90)
    plt.hlines(0, test_indices["values"][0], test_indices["values"][-1], colors=['k'], linestyles=['--'], label="Truth")
    plt.xlim(test_indices["values"][0], test_indices["values"][-1])
    plt.ylabel("Area-weighted mean error ($\\degree$C)")
    plt.xlabel(test_indices["label"])
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(os.path.join(save_path, "area_weighted_mean_error.png"), bbox_inches="tight", dpi=300)
    plt.close()

    return diff_dict


def plot_latitude_integrated_mean_error(area, diff_dict, lats, save_path, colour_dict=COLOURS):
    """Latitude-integrated mean error in SST

    Args:
        area (str): area of ocean to use
        
        diff_dict (dict): keys: model names, values: difference between y_pred[model name] and y_true
        lats (numpy array): array of latitudes covered by diff_dict values
        save_path (str): path to save figure under
        colour_dict (dict, optional): keys: model names, values: colours specified in hexcode. Defaults to COLOURS.

    Returns:
        None
    """
    avg_all_time_lat_avg_diffs = {}
    for name, data in diff_dict.items():
        # For each reconstruction at a set "date", find the average difference at each latitude
        latitude_avg_diffs = np.array([pd.Series(diff_at_time).groupby(lats).mean().to_numpy() for diff_at_time in data])

        # For each latitude, calculate the mean difference over all time
        avg_all_time_lat_avg_diffs[name] = np.array([np.mean(latitude_avg_diffs.T[j]) for j in range(latitude_avg_diffs.shape[1])])

    plt.figure(figsize=(8,8))
    for name, vals in avg_all_time_lat_avg_diffs.items():
        plt.plot(vals, np.unique(lats), c=colour_dict[name], label=name)
    plt.vlines(0, min(lats), max(lats), colors=['k'], linestyles=['--'], label="Truth")
    if area == "world":
        plt.ylim(-90, 90)
    elif area == "NA":
        plt.ylim(0, 60)

    plt.xlabel('Latitude-integrated mean error ($\\degree$C)')
    plt.ylabel('Latitude')
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_path, "lat_int_mean_error.png"), bbox_inches="tight", dpi=300)
    plt.close()
