"""pod.py

Partially adapted from qr_place in https://github.com/Jan-Williams/pyshred/blob/main/processdata.py
"""
import numpy as np
import scipy


def create_pod_test_data(data, sensor_locs, test_lags=None):
    """Create test data for POD reconstructions

    Args:
        data (numpy array): 2D array of data to split of format [[values for time 1], [values for time 2], ...]
        sensor_locs (numpy array): indexes corresponding to the shape of the values for
            time t in data that indicate the sensor locations to use for the model
        test_lags (int, optional): if matching a comparison to SHRED, this will be the number of weeks 
            in each SHRED sequence. Defaults to None.

    Returns:
        data_in (numpy array): data values at sensor_locs
        ground_truth (numpy array):  full-field ground truth
    """
    if test_lags is None:
        # Not matching test dataset to SHRED test dataset
        data_in = np.array([data[i, sensor_locs] for i in range(len(data))])
        ground_truth = data

    else:
        # Want only to test on the end point for each lag so can directly compare SDN and SHRED performance
        data_in = np.array([data[i+test_lags, sensor_locs] for i in range(len(data)-test_lags)])
        ### -1 to have output be at the same time as final sensor measurements
        ground_truth = data[np.array(range(len(data)-test_lags)) + test_lags - 1]

    return data_in, ground_truth


def svd_basis(data_matrix, num_sensors):
    """Compute POD basis, without QR

    Args:
        data_matrix (numpy array): training dataset of shape (N (number of training samples) x m (full-field size))
        num_sensors (int): number of sensors being used

    Returns:
        U_r (numpy array): POD basis of the first num_sensors POD modes
    """
    X_train = data_matrix.T  # invert to shape (m x N)
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    U_r = U[:, :num_sensors] # first num_sensors POD modes

    return U_r


def qr_place(data_matrix, num_sensors):
    """QR placement
    
    Takes a (N x m) data matrix consisting of N samples of an m dimensional state and 
    number of sensors, returns QR placed sensors and U_r for the SVD X = U S V^T

    Args:
        data_matrix (numpy array): training dataset
        num_sensors (int): number of sensors being used

    Returns:
        sensor_locs (numpy array): indexes of the sensor locations to use for the model,
            optimised using QR decomposition.
        U_r (numpy array): POD basis of the first num_sensors POD modes
    """
    U_r = svd_basis(data_matrix, num_sensors)

    q, r, pivot = scipy.linalg.qr(U_r.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]

    return sensor_locs, U_r
