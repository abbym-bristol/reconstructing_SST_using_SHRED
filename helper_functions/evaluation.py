"""evaluation.py"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Polygon
from skimage.metrics import structural_similarity as SSI
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from tqdm import tqdm

# from shapely.plotting import plot_polygon


def evaluate_metrics(y_pred, y_true):
    """Calculate metrics from test reconstructions

    Args:
        y_pred (numpy array): predicted y (full-field) for each sequence in test set
        y_true (numpy array): corresponding true y (full-field) for each sequence in test set

    Returns:
        None
    """
    mse = MSE(y_pred=y_pred, y_true=y_true)
    # confirm that sklearn MSE direct calculation is same as doing it by timestep
    assert np.isclose(mse, np.mean([MSE(y_pred=y_pred[i], y_true=y_true[i]) for i in range(len(y_pred))]))

    mae = MAE(y_pred=y_pred, y_true=y_true)
    # confirm that sklearn MAE direct calculation is same as doing it by timestep
    assert np.isclose(mae, np.mean([MAE(y_pred=y_pred[i], y_true=y_true[i]) for i in range(len(y_pred))]))

    r2_sklearn = r2_score(y_pred=y_pred, y_true=y_true)
    r2 = np.mean([r2_score(y_pred=y_pred[i], y_true=y_true[i]) for i in range(len(y_pred))])  # needs hand-holding to compare the correct things
    assert not np.isclose(r2, r2_sklearn)  # confirm that r2 by sklearn directly is very different to the per timestep version

    print(f"RMSE:\t{np.sqrt(mse):.3f}")
    print(f"MSE:\t{mse:.3f}")
    print(f"MAE:\t{mae:.3f}")
    print(f"R2:\t{r2:.3f}")


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
