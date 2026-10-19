"""reconstructing_sst.py

Datasets:

NOAA: 
- Weekly mean sea-surface temperature at 1° resolution, as given by the NOAA Optimum Interpolation SST V2 dataset [1]
- specifically, using the version taken from the Williams et al. pyshred implementation [2]

OFAM3: 
- ME4OH full-field dataset at 1/4° resolution generated from the OFAM3 numerical model [3].

Models:
- Static SHRED & SDN from [4, 5]
- gappy POD: Partially adapted from qr_place in [4]
- RBF Interpolation: using scipy implementation

"""

# Standard imports
import argparse
import json
import os
import random

# Third-party imports
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler

# Local imports
from helper_functions import models, noaa_data, ofam3_data
from helper_functions.pod import qr_place, svd_basis
from helper_functions.preprocess_data import (
    TimeSeriesDataset,
    convert_to_anomaly,
    split_ordered_data,
)
from helper_functions.saving import JsonEncoder
from helper_functions.windows import build_windows

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

#-------------------- SET UP --------------------#

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
        data, lats, longs, dates = noaa_data.load_data()

        if area != "world":
            data, lats, longs = noaa_data.crop_to_area(data, lats, longs, area=area)

    elif dataset == "OFAM3":
        data, dates, lats, longs = ofam3_data.load_files(file_path=f"Data/OFAM3/{area}/")

    print(f"Number of weeks in data: {data.shape[0]}, number of lat/long positions: {data.shape[1]}")

    return data, lats, longs, dates


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
    print(f"Mean of standardised training data: {np.mean(data_dict["train"])}")
    print(f"Std of standardised training data: {np.std(data_dict["train"])}")
    print(f"Mean of standardised test data: {np.mean(data_dict["test"])}")
    print(f"Mean of standardised test data: {np.std(data_dict["test"])}")

    # Sensor placement
    if sensor_placement == "random":
        sensor_locations = np.random.choice(data.shape[1], size=num_sensors, replace=False)
        U_r = svd_basis(train_data, num_sensors)  # Set POD basis
    elif sensor_placement == "QR":
        sensor_locations, U_r = qr_place(train_data, num_sensors)
    else:
        raise ValueError("Incorrect placement of sensors specified")

    sensor_coords=(longs[sensor_locations], lats[sensor_locations])
    print(f"Sensor location indexes will be: {sensor_locations}")

    return data_dict, dates_dict, sc, sensor_locations, sensor_coords, U_r, bulk_mean


#-------------------- MODELS --------------------#
# All models receive the same windows (from build_windows) and return NORMALISED test
# reconstructions of shape (N, m), in the same row order as windows["test"]["truth"].
# Un-normalising and saving happen once, in main.

def to_torch_dataset(split_windows, input_key, device):
    """Wrap a view of the shared windows as a torch dataset (no re-indexing happens here)
    
    Args:
        TBC
    
    Returns:
        (TimeSeriesDataset): Torch dataset of timeseries data
    """
    X = torch.from_numpy(split_windows[input_key]).to(device)
    Y = torch.from_numpy(split_windows["truth"]).to(device)
    return TimeSeriesDataset(X, Y)


def shred_sdn_reconstructions(windows, hidden_dim, model_type, num_sensors, num_points):
    """Train and validate a SHRED or SDN model on the shared windows, then reconstruct the test split
 
    SHRED uses the sensor history (sensor_seqs); SDN uses only the final step (sensor_now).
    Both are trained, validated and tested on exactly the same target weeks.
 
    Args:
        windows (dict): {train, val, test} windows from build_windows
        hidden_dim (int): hidden dimension for the sequence model
        model_type (str): SHRED-LSTM, SHRED-GRU or SDN
        num_sensors (int): number of synthetic sensors
        num_points (int): size of the full field (m)
 
    Returns:
        test_recons (numpy array): normalised reconstructions, shape (N, m)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO - add mutliple training and saving (interun variability)

    if "SHRED" in model_type:
        input_key = "sensor_seqs"
        model = models.SHRED(num_sensors, num_points, seq_model=model_type.split("-")[-1], hidden_size=hidden_dim,
                            hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    elif model_type == "SDN":
        input_key = "sensor_now"
        model = models.SDN(num_sensors, num_points, l1=350, l2=400, dropout=0.1).to(device)
    else:
        raise ValueError(f"Not a SHRED/SDN model type: {model_type}")

    # Set up dataset
    train_ds = to_torch_dataset(windows["train"], input_key, device)
    val_ds = to_torch_dataset(windows["val"], input_key, device)

    # Train and validate model
    validation_errors = models.fit(model, train_ds, val_ds, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
    print('Validation Reconstruction Error: ', validation_errors)

    # Test
    model.eval()  # make sure dropout is off for testing
    with torch.no_grad():
        X_test = torch.from_numpy(windows["test"][input_key]).to(device)
        test_recons = model(X_test).cpu().numpy()

    return test_recons


def pod_reconstructions(test_windows, num_sensors, sensor_locations, sensor_placement, U_r):
    """Gappy POD reconstruction of the test windows (POD basis comes from the full training data)

    Args:
        test_windows (dict): test windows from build_windows
        num_sensors (int): number of synthetic sensors to use
        sensor_locations (numpy array): sensor locations corresponding to indices in the full-field data
        sensor_placement (str): type of sensor placement to use; choices are random or QR
        U_r (numpy array): POD basis of the first num_sensors POD modes

    Returns:
        test_recons (numpy array): normalised reconstructions, shape (N, m)
    """
    C = np.zeros((num_sensors, len(lats)))
    for i in range(num_sensors):
        C[i, sensor_locations[i]] = 1

    A = C @ U_r

    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: C @ U_r is ill-conditioned (cond={cond:.2e}) "
            f"for placement='{sensor_placement}'.")

    coeffs, *_ = np.linalg.lstsq(A, test_windows["sensor_now"].T, rcond=None)
    test_recons = (U_r @ coeffs).T
    
    return test_recons


def rbfi_reconstructions(test_windows, lats, longs, sensor_coords):
    """RBF interpolation reconstruction of the test windows

    Args:
        test_windows (dict): test windows from build_windows
        lats (numpy array): array of latitudes corresponding to SST values
        longs (numpy array): array of longitudes corresponding to SST values
        sensor_coords (numpy array): Latitude and Longitude coordinates corresponding to sensor_locations

    Returns:
        Saves out the reconstructions produced by the model (and the corresponding ground truth)
        Saves a metadata file for the model, containing
            test_lag_dates (numpy array): lag dates that correspond with the sequenced test data used
    """
    points = np.array([[sensor_coords[0][i], sensor_coords[1][i]] for i in range(len(sensor_coords[0]))])
    full_coords = np.column_stack([longs, lats])

    test_input = test_windows["sensor_now"]
    rbf_recons = np.zeros((len(test_input), len(lats)))
    for j, data in enumerate(test_input):
        interpolator = RBFInterpolator(
            points,
            data,
            kernel='thin_plate_spline',
            smoothing=0.0, # raise this (e.g. 1e-2–1e0) if sensors are noisy
        )
        rbf_recons[j] = interpolator(full_coords)

    return rbf_recons


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    """Split list of strings into a list"""
    return arg.split(',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required (standard)
    parser.add_argument("version", help="Version indicator (str) for saving, e.g. the date in format 051026")
    parser.add_argument("dataset", help="Dataset type to use", choices=["OFAM3", "NOAA"])
    parser.add_argument("models", help="List of model types to train; choices are SHRED-LSTM,SHRED-GRU,SDN,POD,RBFI. Format as comma separated string (with no spaces)", type=list_of_strings)
    parser.add_argument("num_sensors", help="Number of synthetic sensors to use, integer", type=int)

    # Optional - Bool flags
    parser.add_argument("-a", "--anomaly", action="store_true", help="Use anomalies to train if this flag is used")

    # Optional - defaults set
    parser.add_argument("-f", "--filepath", help="File path to save results under",
                        default="Reconstructions/Static/")
    parser.add_argument("-d", "--hidden_dim", help="Hidden dimension for sequence model", type=int, default=128)
    parser.add_argument("-l", "--lags", help="Lags (sequence length in weeks)", default=52, type=int)
    parser.add_argument("-r", "--area", help="Area of ocean to use", default="NA", choices=["world", "NA", "GS"])
    parser.add_argument("-p", "--sensor_placement", help="Type of sensor placement to use; choices are random or QR", choices=["random", "QR"], default="random")

    args = parser.parse_args()

    VALID_MODELS = {"SHRED-LSTM", "SHRED-GRU", "SDN", "POD", "RBFI"}
    unknown = set(args.models) - VALID_MODELS
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(sorted(unknown))}. Choose from {sorted(VALID_MODELS)}")

    # ---------- Shared set up (identical for every model) ----------

    # Load and reformat data
    path = f"{args.dataset}/{args.area}/{args.sensor_placement}/{args.version}_n{args.num_sensors}_l{args.lags}"
    directory = os.path.join(args.filepath, path)
    print("Saving metadata under: ", directory)
    os.makedirs(directory, exist_ok=True)

    data, lats, longs, dates = load_dataset(args.area, args.dataset)
    pp_data, dates_dict, norm_model, sensor_locations, sensor_coords, U_r, bulk_mean = preprocess_data(
        anomaly=args.anomaly, data=data, dates=dates, lats=lats, longs=longs,
        num_sensors=args.num_sensors, sensor_placement=args.sensor_placement)

     # One set of windows for every split; every model takes a view of these
    windows = {split: build_windows(pp_data[split], dates_dict[split], sensor_locations, args.lags)
               for split in ("train", "val", "test")}
 
    # Ground truth is computed and saved once, so it cannot differ between models
    test_ground_truth = norm_model.inverse_transform(windows["test"]["truth"])
    np.save(f"{directory}/truth", test_ground_truth)
 
    metadata = {
        "sensor_locations": list(sensor_locations),
        "sensor_coords": list(sensor_coords),
        "dates": dates_dict,
        "test_lag_dates": windows["test"]["dates"],
    }
    if bulk_mean is not None:
        metadata["bulk_mean"] = bulk_mean
    with open(f"{directory}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, cls=JsonEncoder)
 
    num_points = len(lats)

    # ---------- Model reconstructions ----------
    for model_type in args.models:
        # Storage
        model_directory = os.path.join(directory, model_type)
        print("Saving results under: ", model_directory)
        os.makedirs(model_directory, exist_ok=True)

        # Build sequences
        if model_type in ["SHRED-LSTM", "SHRED-GRU", "SDN"]:
            recons_norm = shred_sdn_reconstructions(windows, hidden_dim=args.hidden_dim, model_type=model_type,
                                                    num_sensors=args.num_sensors, num_points=num_points)
        elif model_type == "POD":
            recons_norm = pod_reconstructions(windows["test"], num_points=num_points, num_sensors=args.num_sensors,
                                              sensor_locations=sensor_locations, sensor_placement=args.sensor_placement, U_r=U_r)
        elif model_type == "RBFI":
            recons_norm = rbfi_reconstructions(windows["test"], lats=lats, longs=longs, sensor_coords=sensor_coords)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        test_recons = norm_model.inverse_transform(recons_norm)
        np.save(f"{model_directory}/recons", test_recons)
 
        rel_err = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
        print(f"{model_type} test relative reconstruction error: {rel_err:.4f}")

"""
References:
[1] NOAA dataset origin, https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html

[2] Jan P Williams, pyshred repository, NOAA SST dataset, https://github.com/Jan-Williams/pyshred/blob/main/Data/SST_data.mat

[3] CSIRO ME4OH OFAM3 dataset, v9, https://data.csiro.au/collection/csiro%3A60826v9

[4] Jan P Williams, pyshred repository, SHRED and SDN model implementations, https://github.com/Jan-Williams/pyshred/blob/main/models.py

[5] Jan P Williams, Olivia Zahn, and J Nathan Kutz. Sensing with shallow recurrent decoder networks. 
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 480(2298), 2024,
https://royalsocietypublishing.org/rspa/article/480/2298/20240054/66770/.
"""
