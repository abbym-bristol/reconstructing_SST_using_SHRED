"""ofam3_data.py

Helper functions for loading ME4OH data SST OFAM3 data

Prerequisites: Data/load_sst_ofams3_data.py must be run first, producing NA basin data stored under NA_DATA_PATH

"""
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

NA_DATA_PATH = "Data/OFAM3/NA/"
LAGS = 52  # The OFAM3 SST data is sampled weekly (unknown currently whether these are averaged or just for one day each week)

np.random.seed(42)


def load_files(file_path=NA_DATA_PATH):
    """Load files and get data from files that have already been preprocessed, for example those produced
    using load_sst_ofam3_data.py which are stored in "Data/OFAM3/NA/".

    Args:
        file_path (str, optional): path to preprocessed data.
            Defaults to those that have already been preprocessed to contain only
            North Atlantic basin data using load_sst_ofam3_data.py and stored in NA_DATA_PATH.
    Returns:
        sst_data (2D numpy array): SST value at each time for each lat/long across area used
            2D array of format [[array of lat/long SST values for time 1], [array of lat/long SST values for time 2], ...]
        dates (numpy array): array of dates for each time in dataset (from filenames)
        lats (numpy array): array of latitudes corresponding to SST values
        longs (numpy array): array of longitudes corresponding to SST values
    """  # noqa: D205
    # TODO: alter so either saves out reformatted version or loads that if it exists?

    # Drop final file path -> not a full week out from penultimate path.
    file_paths = [f for f in sorted(glob(os.path.join(file_path, "*.csv")))][:-1]

    sst_data = []
    dates = []
    
    for file_path in tqdm(file_paths, desc='Loading data'):
        df = pd.read_csv(file_path)
        if file_path == file_paths[0]:
            lats = df.Latitude.to_numpy()
            longs = df.Longitude.to_numpy()
        base_name = os.path.basename(file_path)
        date_str = os.path.splitext(base_name)[0]
        dates.append(pd.to_datetime(date_str).date())
        sst_data.append(df.SST.to_numpy())

    return np.array(sst_data), np.array(dates), lats, longs


def get_lats_longs(data_path=NA_DATA_PATH):
    """Load files and get latitude and longitude data from files that have already been preprocessed,
    for example those produced using load_sst_ofam3_data.py which are stored in "Data/OFAM3/NA/".

    NOTE: Assumes the SST data is always measured against the same latitudes and longitudes in the preprocessed data files

    Args:
        data_path (str): path to preprocessed data.
            Defaults to those that have already been preprocessed to contain only
            North Atlantic basin data using load_sst_ofam3_data.py and stored in NA_DATA_PATH.
    Returns:
        lats (numpy array): array of latitudes corresponding to SST values
        longs (numpy array): array of longitudes corresponding to SST values
    """  # noqa: D205
    # Drop final file path -> not a full week out from penultimate path.
    file_paths = [f for f in sorted(glob(os.path.join(data_path, "*.csv")))][:-1]

    # Read the first file
    df = pd.read_csv(file_paths[0])
    lats = df.Latitude.to_numpy()
    longs = df.Longitude.to_numpy()

    return lats, longs
