"""preprocess_sst_ofam3_shred.py

Helper functions for preprocessing ME4OH data SST OFAM3 data for the SHRED algorithm

Prerequisites: load_sst_ofams3_data.py must be run first, producing NA basin data stored under "Data/OFAM3/NA/"

"""
import numpy as np
import torch

LAGS = 52  # The OFAM3 SST data is sampled weekly (unknown currently whether these are averaged or just for one day each week)

np.random.seed(42)


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


def create_shred_sequences(data, sensor_locs, lags=LAGS):
    """Create sequences for input to the SHRED model

    Args:
        data (2D numpy array): 2D array of data to split of format [[values for time 1], [values for time 2], ...]
        sensor_locs (numpy array): indexes corresponding to the shape of the values for time t in data that indicate the sensor locations to use for the model
        lags (int, optional): number of weeks in each sequence, defaults to LAGS (52)

    Returns:
        dataset (TimeSeriesDataset): torch tensor dataset of input sequences and labels (entire data field at the end of each sequence)
    """
    num_sensors = len(sensor_locs)
    data_in = np.zeros((len(data) - lags, lags, num_sensors))
    for i in range(len(data_in)):
        data_in[i] = data[i:i+lags, sensor_locs]  # stores values between i and 52 weeks, at 3 sensor locations
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_in = torch.tensor(data_in, dtype=torch.float32).to(device)

    # +lags-1 to have output be at the same time as final sensor measurements
    data_out = torch.tensor(data[np.array(range(len(data)-lags)) + lags - 1], dtype=torch.float32).to(device)

    dataset = TimeSeriesDataset(data_in, data_out)

    return dataset
