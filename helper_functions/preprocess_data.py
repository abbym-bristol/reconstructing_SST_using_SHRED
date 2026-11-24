"""preprocess_sst_ofam3_shred.py

Helper functions for preprocessing ME4OH data SST OFAM3 data for the SHRED algorithm

Prerequisites: load_sst_ofams3_data.py must be run first, producing NA basin data stored under "Data/OFAM3/NA/"

"""
# Standard imports
import numpy as np
import torch

# Third-party imports
from sklearn.preprocessing import StandardScaler

# Local imports
from helper_functions.noaa_data import crop_to_area, load_data
from helper_functions.ofam3_data import load_files
from helper_functions.pod import qr_place, svd_basis

LAGS = 52  # The OFAM3 SST data is sampled weekly (unknown currently whether these are averaged or just for one day each week)

np.random.seed(42)


def load_dataset(area, dataset):
    """For a given dataset type (OFAM3 or NOAA) load and filter to a specified area (world, NA or GS)
    
    Args:
        area (str): area of ocean to use
        dataset (str): dataset type to use

    Returns:
        data (numpy array): SST data
        lats (numpy array): array of latitudes corresponding to SST values
        longs (numpy array): array of longitudes corresponding to SST values
        dates (numpy array): array of dates corresponding to SST values
    """
    if dataset == "NOAA":
        data, lats, longs, dates = load_data()

        if area != "world":
            data, lats, longs = crop_to_area(temps=data, lats=lats, longs=longs, area=area)

    elif dataset == "OFAM3":
        data, dates, lats, longs = load_files(file_path=f"Data/OFAM3/{area}/")

    print(f"\tNumber of weeks in data: {data.shape[0]}, number of lat/long positions: {data.shape[1]}")

    return data, lats, longs, dates


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurements of high-dimensional state, return Torch dataset
    
    Origin: https://github.com/Jan-Williams/pyshred/blob/main/processdata.py
    """  # noqa: D205
    def __init__(self, X, Y):  # noqa: D107
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):  # noqa: D105
        return self.X[index], self.Y[index]
    
    def __len__(self):  # noqa: D105
        return self.len


def split_ordered_data(data, dates, all_dates=False, train_frac=0.75):
    """Split ordered data into training, validation and test sets according to date

    Args:
        data (2D numpy array): 2D array of data to split of format [[values for time 1], [values for time 2], ...]
        dates (numpy array): array of dates for each time in data
        all_dates (bool, optional): whether to return all the date arrays, if False only test_dates is returned.
        train_frac (float, optional): fraction of training data to use, defaults to 0.75.

    Returns:
        train, val, test (2D numpy arrays): train, val and test splits of data using ratio train_frac:(1-train_frac)/2:(1-train_frac)/2
        train_dates, val_dates (numpy array, optional): arrays of dates for train and val sets; returned only if all_dates is True.
        test_dates (numpy array): array of dates for test set
    """
    if len(data) != len(dates):  # Sanity check
        raise ValueError("data_frames and dates must have the same length")
    
    n_total = len(data)
    n_train = int(np.floor(n_total * train_frac))
    n_remaining = n_total - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    if all_dates: 
        train_dates = dates[:n_train]
        val_dates = dates[n_train:n_train + n_val]
    test_dates = dates[n_train + n_val:]

    if len(test) != n_test:  # Sanity check
        raise ValueError("Test data of wrong length")

    if all_dates:
        return train, val, test, train_dates, val_dates, test_dates
    else:
        return train, val, test, test_dates


def convert_to_anomaly(data):
    """Convert data to anomalies by subtracting the bulk mean value for the training data

    Args:
        data (tuple): tuple of (train, validation, test) data to convert to anomalies

    Returns:
        train_anomaly (Numpy array): train dataset of anomaly values
        val_anomaly (Numpy array): validation dataset of anomaly values
        test_anomaly (Numpy array): test dataset of anomaly values
        train_bulk_mean (Numpy array): bulk mean value of training data for each lat/long grid square
    """
    (train, val, test) = data

    # For each week, get mean SST across all weeks for each lat/long grid point
    train_bulk_mean = np.mean(train, axis=0)

    # Calculate anomalies
    train_anomaly = train - train_bulk_mean
    val_anomaly = val - train_bulk_mean
    test_anomaly = test - train_bulk_mean

    return train_anomaly, val_anomaly, test_anomaly, train_bulk_mean

# ---------- reconstructing_sst.py specific helper ----------

def preprocess_data(anomaly, data, dates, lats, longs, num_sensors, sensor_placement):
    """Preprocess data with varying numbers of static sensors
 
    Args:
        anomaly (Bool): whether or not to use anomalies
        data (numpy array): SST data
        dates (numpy array): array of dates corresponding to SST values
        lats, longs (numpy array): coordinates of each grid point in data
        num_sensors (int): number of synthetic sensors to use
        sensor_placement (str): type of sensor placement to use; choices are random or QR
 
    Returns:
        data_dict (dict): normalised float32 train, val and test data (before sequencing)
        dates_dict (dict): dates that correspond with the train, val and test data in data_dict
        sc (class): MinMaxScaler() class fitted on the training data
        sensor_locations (numpy array): sensor locations corresponding to indices in the full-field data
        sensor_coords (tuple): (longs, lats) of each sensor
        U_r (numpy array): POD basis of the first num_sensors POD modes
        bulk_mean (numpy array or None): training mean if anomaly is True
    """
    print("Preprocessing data...")
    train_data, val_data, test_data, train_dates, val_dates, test_dates = split_ordered_data(data, dates, all_dates=True)
    dates_dict = {"train": train_dates, "val": val_dates, "test": test_dates}
    bulk_mean = None
    if anomaly:
        train_data, val_data, test_data, bulk_mean = convert_to_anomaly((train_data, val_data, test_data))

    # Normalise data. 
    # Cast to float32 here so every model (torch or numpy) sees
    # bit-identical inputs and targets - torch models need float32 anyway.
    sc = StandardScaler()
    sc = sc.fit(train_data)  # Computes min and max from training data only (prevent data leakage)
    data_dict = {
        "train": sc.transform(train_data).astype(np.float32),
        "val": sc.transform(val_data).astype(np.float32),
        "test": sc.transform(test_data).astype(np.float32),
    }
    print(f"\tMean of standardised training data: {np.mean(data_dict["train"]):.2f}")
    print(f"\tStd of standardised training data: {np.std(data_dict["train"]):.2f}")
    print(f"\tMean of standardised test data: {np.mean(data_dict["test"]):.2f}")
    print(f"\tStd of standardised test data: {np.std(data_dict["test"]):.2f}")

    # Sensor placement
    if sensor_placement == "random":
        sensor_locations = np.random.choice(data.shape[1], size=num_sensors, replace=False)
        U_r = svd_basis(train_data, num_sensors)  # Set POD basis
    elif sensor_placement == "QR":
        sensor_locations, U_r = qr_place(train_data, num_sensors)
    else:
        raise ValueError("Incorrect placement of sensors specified")

    sensor_coords=(longs[sensor_locations], lats[sensor_locations])
    print(f"\tSensor location indexes will be: {sensor_locations}")

    return data_dict, dates_dict, sc, sensor_locations, sensor_coords, U_r, bulk_mean
