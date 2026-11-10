"""evaluation.py"""

import numpy as np


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


def evaluate_metrics(y_true, y_pred, weights=None):
    """Calculate metrics from test reconstructions

    Args:
        y_true (numpy array): corresponding true y (full-field) for each sequence (week) in test set
        y_pred (numpy array): predicted y (full-field) for each sequence (week) in test set
        weights (numpy array, optional): Area weights corresponding to the predicted full-field shape (y_pred.shape[1]) or None

    Returns:
        None
    """
    label = "unweighted" if weights is None else "area-weighted"
    if weights is None:
        weights = np.ones(y_true.shape[-1])

    mse, _ = weighted_mse(y_true, y_pred, weights)
    metrics = {
        "MAE": weighted_mae(y_true, y_pred, weights),
        "MSE": mse,
        "R2": weighted_r2(y_true, y_pred, weights)
    }

    print(f"Metrics ({label}):")
    for k,v in metrics.items():
        print(f"{k}:\t{v:.3f}")

    return metrics, weights


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
