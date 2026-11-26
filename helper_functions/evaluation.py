"""evaluation.py"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions.area_weighting import (
    LatLongGrid,
    area_weighted_mean,
    area_weighted_std,
    infer_resolution,
)

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
        (float): mean R2 value across all weeks in y_pred
    """
    _, mse_by_week = weighted_mse(y_true, y_pred, weights)
    y_bar = np.sum(weights * y_true, axis=1) / np.sum(weights)
    var_by_week = np.sum((y_true - y_bar[:, None])**2 * weights, axis = 1) / np.sum(weights)
    r2_by_week = 1 - mse_by_week/var_by_week
    return np.mean(r2_by_week)  # mean at the end to treat every week equally


def evaluate_metrics(y_true, y_pred, weights, print_results=True):
    """Calculate metrics from test reconstructions

    Args:
        y_true (numpy array): corresponding true y (full-field) for each sequence (week) in test set
        y_pred (numpy array): predicted y (full-field) for each sequence (week) in test set
        weights (numpy array): area weights corresponding to the predicted full-field shape (y_pred.shape[1])
        print_results (bool): whether to print results out

    Returns:
        metrics (dict): dictionary of metrics results
    """
    # Check if all weights are 1
    label = "unweighted" if np.all(weights == 1) else "area-weighted"

    mse, _ = weighted_mse(y_true, y_pred, weights)
    metrics = {
        "MAE": weighted_mae(y_true, y_pred, weights),
        "MSE": mse,
        "R2": weighted_r2(y_true, y_pred, weights)
    }

    if print_results:
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

# TODO - Plot spread on single location plots, by doing IQR shading??? What's the appropriate spread on the global mean timeseries? Is there one?

def plot_area_weighted_mean_SST(y_true, y_pred, area_weights, save_path, test_indices, colour_dict=COLOURS,
                               y_pred_std=None):
    """Area-weighted mean SST over time
    
    Args:
        y_true (numpy_array): ground truth data
        y_pred (dict): keys: model names, values: reconstructions made by model for each item in test set
        area_weights (numpy_array): area weights corresponding to the predicted full-field shape (y_pred.shape[1]) or None
        save_path (str): path to save figure under
        test_indices (dict): {"values": test indices to use, "label": x-axis label to use}
        colour_dict (dict, optional): keys: model names, values: colours specified in hexcode. Defaults to COLOURS.
        y_pred_std (dict, optional): keys: model names, values: standard deviations of area-weighted mean
            reconstructions over test indices. When provided, shades the mean +/- one standard deviation.

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
    for name in y_pred.keys():  # For each model
        plt.plot(sst_mean_df[name], c=colour_dict[name], ls='-', label=name)
        if y_pred_std is not None:
            # spread = area_weighted_std(sigma=y_pred_std[name], area_weights=area_weights)  # If the errors are independent
            spread = area_weighted_mean(y_pred_std[name], area_weights)  
                # This is the upper bound of the spread if the cell errors are strongly/fully correlated
            plt.fill_between(
                sst_mean_df.index,
                sst_mean_df[name].to_numpy() - spread,
                sst_mean_df[name].to_numpy() + spread,
                color=colour_dict[name],
                alpha=0.2,
                label=f"{name} +/- 1 SD",
            )
    plt.plot(sst_mean_df["Truth"], c=colour_dict["Truth"], ls='--', label="Truth")
    plt.xticks(rotation=90)
    plt.xlabel(test_indices["label"])
    plt.ylabel("Area-weighted mean SST ($\\degree$C)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.xlim(test_indices["values"][0], test_indices["values"][-1])
    plt.savefig(os.path.join(save_path, "area_weighted_mean_SST.png"), bbox_inches="tight", dpi=300)
    plt.close()


def plot_area_weighted_mean_error(y_true, y_pred, area_weights, save_path, test_indices, colour_dict=COLOURS, y_pred_std=None):
    """Area-weighted mean error in SST over time
    
    Args:
        y_true (numpy_array): ground truth data
        y_pred (dict): keys: model names, values: reconstructions made by model for each item in test set
        area_weights (numpy_array): area weights corresponding to the predicted full-field shape (y_pred.shape[1]) or None
        save_path (str): path to save figure under
        test_indices (dict): {"values": test indices to use, "label": x-axis label to use}
        colour_dict (dict, optional): keys: model names, values: colours specified in hexcode. Defaults to COLOURS.
        y_pred_std (dict, optional): keys: model names, values: standard deviations of area-weighted mean
            reconstructions over test indices. When provided, shades the mean +/- one standard deviation.

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
        if y_pred_std is not None:
            # spread = area_weighted_std(sigma=y_pred_std[name], area_weights=area_weights)  # If the errors are independent
            spread = area_weighted_mean(y_pred_std[name], area_weights)  
                # This is the upper bound of the spread if the cell errors are strongly/fully correlated
            plt.fill_between(
                sst_diff_df.index,
                sst_diff_df[name].to_numpy() - spread,
                sst_diff_df[name].to_numpy() + spread,
                color=colour_dict[name],
                alpha=0.2,
                label=f"{name} +/- 1 SD",
            )
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


def plot_latitude_integrated_mean_error(area, diff_dict, lats, save_path, colour_dict=COLOURS,
                                       diff_std_dict=None):
    """Latitude-integrated mean error in SST

    Args:
        area (str): area of ocean to use
        diff_dict (dict): keys: model names, values: difference between y_pred[model name] and y_true
        lats (numpy array): array of latitudes covered by diff_dict values
        save_path (str): path to save figure under
        colour_dict (dict, optional): keys: model names, values: colours specified in hexcode. Defaults to COLOURS.
        diff_std_dict (dict, optional): keys: model names, values: standard deviations of the errors at each
            test index and grid cell. Used to shade a conservative spread band.

    Returns:
        None
    """
    avg_all_time_lat_avg_diffs = {}
    avg_all_time_lat_avg_stds = {}
    latitude_values = np.unique(lats)
    for name, data in diff_dict.items():
        # For each reconstruction at a set "date", find the average difference at each latitude
        latitude_avg_diffs = np.array([
            pd.Series(diff_at_time).groupby(lats).mean().reindex(latitude_values).to_numpy()
            for diff_at_time in data
        ])

        # For each latitude, calculate the mean difference over all time
        avg_all_time_lat_avg_diffs[name] = np.mean(latitude_avg_diffs, axis=0)

        if diff_std_dict is not None:
            # Conservative estimate -> not computed each std for each point
            latitude_avg_stds = np.array([
                pd.Series(std_at_time).groupby(lats).mean().reindex(latitude_values).to_numpy()
                for std_at_time in diff_std_dict[name]
            ])
            avg_all_time_lat_avg_stds[name] = np.mean(latitude_avg_stds, axis=0)

    plt.figure(figsize=(8,8))
    for name, vals in avg_all_time_lat_avg_diffs.items():
        plt.plot(vals, latitude_values, c=colour_dict[name], label=name)
        if diff_std_dict is not None:
            spread = avg_all_time_lat_avg_stds[name]
            plt.fill_betweenx(
                latitude_values,
                vals - spread,
                vals + spread,
                color=colour_dict[name],
                alpha=0.2,
                label=f"{name} +/- 1 SD",
            )
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

# ---------- evaluating_sst.py specific helper ----------

def metric_weighting(area, arg_weighting, lats, longs, weights, y_true):
    """Configure weighting for metrics

    Optional to evaluate metrics using area-weighting. 
        - If desired, arg_weighted will be True
        - But training may not have used weights, if not they won't be in meta data and will need to be configured

    Args:
        area (str): area of ocean to use.
        arg_weighting (bool): Whether to use area-weighting for metrics.
        lats (numpy array): array of latitudes, shape len(y_true).
        longs (numpy array): array of longitudes, shape len(y_true).
        weights (numpy array): Area weights used in training,
                    either corresponding to the predicted full-field shape or NONE.
        y_true (numpy_array): ground truth data.
    
    Returns:
        metrics_weights (numpy array): metric weights either using areas or all ones
    """
    if weights is not None and arg_weighting:
        # Case where area-weighting wasn't used for training AND we want to do area-weighted metrics
        metric_weights = weights
    elif weights is not None and not arg_weighting:
        # Case where area-weighting was used for training AND we don't want to do area-weighted metrics
        metric_weights = np.ones(y_true.shape[-1])
    elif weights is None and arg_weighting:
        # Case where area-weighting wasn't used for training AND we want to do area-weighted metrics
        resolution = infer_resolution(lats)
        grid = LatLongGrid(area, resolution)
        metric_weights = grid.get_area_weights(lats, longs)
    else:
        # weights is None and not arg_weighting:
        # Case where area-weighting wasn't used for training AND we don't want to do weighted metrics
        metric_weights = np.ones(y_true.shape[-1])

    return metric_weights

def mean_and_std_locations(test_recons_dict):
    """Get mean and standard deviation across a number of reconstructions at varying sensor locations
    
    Args:
        test_recons_dict (dict): 
            keys: model names, 
            values (dict): keys: locations {loc_i} and
                           values: reconstructions made by model for each item in test set
    
    Returns:
        mean_test_recons (dict): keys: model names, values: mean reconstruction SST, for each item in test set
        std_test_recons (dict):  keys: model names, values: std in reconstruction SST, for each item in test set
    """
    mean_test_recons = {}
    std_test_recons = {}
    for name, data in test_recons_dict.items():
        stacked = np.stack(list(data.values()), axis=0)
        mean_test_recons[name] = np.mean(stacked, axis=0)
        std_test_recons[name] = np.std(stacked, axis=0)
    return mean_test_recons, std_test_recons
