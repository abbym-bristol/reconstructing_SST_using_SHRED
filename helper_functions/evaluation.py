"""evaluation.py"""

import numpy as np


def weighted_mae(y_true, y_pred, weights):
    """Weighted Mean Absolute Error"""
    err = y_true - y_pred
    mae_by_week = np.sum(np.abs(err) * weights, axis=1) / np.sum(weights)  # Sum over each week, not across all weeks
    return np.mean(mae_by_week)


def weighted_mse(y_true, y_pred, weights):
    """Weighted Mean Square Error"""
    err = y_true - y_pred
    mse_by_week = np.sum(err**2 * weights, axis=1) / np.sum(weights)  # Sum over each week, not across all weeks
    return np.mean(mse_by_week), mse_by_week


def weighted_r2(y_true, y_pred, weights):
    """Weighted R squared correlation"""
    _, mse_by_week = weighted_mse(y_true, y_pred, weights)
    y_bar = np.sum(weights * y_true, axis=1) / np.sum(weights)
    var_by_week = np.sum((y_true - y_bar[:, None])**2 * weights, axis = 1) / np.sum(weights)
    r2_by_week = 1 - mse_by_week/var_by_week
    return np.mean(r2_by_week)  # mean at the end to treat every week equally


def evaluate_metrics(y_pred, y_true, weights=None):
    """Calculate metrics from test reconstructions

    Args:
        y_pred (numpy array): predicted y (full-field) for each sequence in test set
        y_true (numpy array): corresponding true y (full-field) for each sequence in test set
        weights (numpy array): Area weights normalised to have mean 1 over the cells or None

    Returns:
        None
    """
    if weights is None:
        weights = np.ones(y_true.shape[-1])

    metrics = {
        "MAE": weighted_mae(y_true, y_pred, weights),
        "MSE": weighted_mse(y_true, y_pred, weights),
        "R2": weighted_r2(y_true, y_pred, weights)
    }

    label = "area-weighted" if weights is not None else "unweighted"
    print(f"Metrics ({label}):")
    for k,v in metrics.items():
        print(f"{k}:\t{v:.3f}")


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
