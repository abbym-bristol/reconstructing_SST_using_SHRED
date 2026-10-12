"""preprocess_SDN.py

Preprocess data to produce SDN "sequences"
"""
import numpy as np
import torch


class SDNDataset(torch.utils.data.Dataset):
    """Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurements of high-dimensional state, return Torch dataset

    This is just a rename of the TimeSeriesDataset so that it is clear within code that its an SDN appropriate dataset instead of one for SHRED
    
    Origin of TimeSeriesDataset: https://github.com/Jan-Williams/pyshred/blob/main/processdata.py
    """  # noqa: D205
    def __init__(self, X, Y):  # noqa: D107
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):  # noqa: D105
        return self.X[index], self.Y[index]
    
    def __len__(self):  # noqa: D105
        return self.len


def create_sdn_datasets(data, sensor_locs, test_lags=None):
    """Create datasets for input to the SDN model

    Args:
        data (2D numpy array): 2D array of data to split of format [[values for time 1], [values for time 2], ...]
        sensor_locs (numpy array): indexes corresponding to the shape of the values for
            time t in data that indicate the sensor locations to use for the model
        test_lags (int, optional): if matching a comparison to SHRED, this will be the number of weeks 
            in each SHRED sequence. Defaults to None.

    Returns:
        dataset (SDNDataset): torch tensor dataset of input data (values at sensor_locs) and labels (entire data field at the end of each sequence)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if test_lags is None:
        # Not matching test dataset to SHRED test dataset
        data_in = np.array([data[i, sensor_locs] for i in range(len(data))])
        data_in = torch.tensor(data_in, dtype=torch.float32).to(device)
        data_out = torch.tensor(data, dtype=torch.float32).to(device)

    else:
        # Want only to test on the end point for each lag so can directly compare SDN and SHRED performance
        data_in = np.array([data[i+test_lags, sensor_locs] for i in range(len(data)-test_lags)])
        data_in = torch.tensor(data_in, dtype=torch.float32).to(device)
        ### -1 to have output be at the same time as final sensor measurements
        data_out = torch.tensor(data[np.array(range(len(data)-test_lags)) + test_lags - 1], dtype=torch.float32).to(device)

    dataset = SDNDataset(data_in, data_out)

    return dataset
