"""TBC

Datasets:

NOAA: 
- Weekly mean sea-surface temperature at 1° resolution, as given by the NOAA Optimum Interpolation SST V2 dataset [1]
- specifically, using the version taken from the Williams et al. pyshred implementation [2]

OFAM3: 
- ME4OH full-field dataset at 1/4° resolution generated from the OFAM3 numerical model [3].

Models:
- Static SHRED & SDN from [4, 5]
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
from sklearn.preprocessing import MinMaxScaler

# Local imports
from helper_functions import models, noaa_data, ofam3_data
from helper_functions.pod import create_pod_test_data, qr_place, svd_basis
from helper_functions.preprocess_SDN import create_sdn_datasets
from helper_functions.preprocess_SHRED import (
    convert_to_anomaly,
    create_shred_sequences,
    split_ordered_data,
)
from helper_functions.saving import JsonEncoder

np.random.seed(42)
random.seed(42)

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

    print(data.shape, len(longs), len(lats))

    return data, lats, longs, dates


def preprocess_data(anomaly, data, dates, num_sensors, save_path, sensor_placement):
    """Preprocess data with varying numbers of static sensors placed randomly

    Args:
        anomaly (Bool): whether or not to use anomalies
        data (numpy array): SST data
        dates (numpy array): array of dates corresponding to SST values
        num_sensors (int): number of synthetic sensors to use
        save_path (str): file path to save results under
        sensor_placement (str): type of sensor placement to use; choices are random or QR

    Returns:
        data_dict (dict): dictionary contianing train, val and test data (after preprocessing, but before sequencing)
        dates_dict (dict): dictionary containing dates that correspond with the train, val and test data in data_dict
        sc (class): MinMaxScaler() class fitted on the training data
        sensor_locations (numpy array): sensor locations corresponding to indices in the full-field data
        U_r (numpy array): POD basis of the first num_sensors POD modes
    
    Saves a metadata file for the experiment version, containing:
        sensor_locations (see Returns)
        sensor_coords (numpy array): Latitude and Longitude coordinates corresponding to sensor_locations
        dates_dict (see Returns)
    """
    print("Preprocessing data...")
    train_data, val_data, test_data, train_dates, val_dates, test_dates = split_ordered_data(data, dates, all_dates=True)
    dates_dict = {"train": train_dates, "val": val_dates, "test": test_dates}
    if anomaly:
        train_data, val_data, test_data, bulk_mean = convert_to_anomaly((train_data, val_data, test_data))

    # Normalise data
    sc = MinMaxScaler()
    sc = sc.fit(train_data)  # Computes min and max from training data only (prevent data leakage)
    data_dict = {
        "train": sc.transform(train_data),
        "val": sc.transform(val_data),
        "test": sc.transform(test_data)
    }

    # Sensor placement
    if sensor_placement == "random":
        # Set sensor locations
        sensor_locations = np.random.choice(data.shape[1], size=num_sensors, replace=False)
        # Set POD basis
        U_r = svd_basis(train_data, num_sensors)

    elif sensor_placement == "QR":
        sensor_locations, U_r = qr_place(train_data, num_sensors)

    else:
        raise ValueError("Incorrect placement of sensors specified")

    sensor_coords=(longs[sensor_locations], lats[sensor_locations])
    print(f"Sensor location indexes will be: {sensor_locations}")

    # Save metadata
    metadata = {
        "sensor_locations": list(sensor_locations),
        "sensor_coords": list(sensor_coords),
        "dates": dates_dict
    }
    if anomaly:
        metadata["bulk_mean"] = bulk_mean
    with open(f"{save_path}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, cls=JsonEncoder)

    return data_dict, dates_dict, sc, sensor_locations, U_r


def build_sequences(model_type, data_transformed, dates, lags, save_path, sensor_locations):
    """Build sequences for each model type

    Args:
        model_type (str): type of model to train; choices are one of: SHRED, SDN, POD, RBFI
        data_transformed (dict): transformed datasets, stored as {train, val, test}
        dates (dict): corresponding dates, stored as {train, val, test}
        lags (int): sequence length in weeks
        save_path (str): file path to save results under
        sensor_locations (list): sensor locations corresponding to indices in the full-field data
 
    Returns:
        datasets (dict): dictionary containing the sequenced datasets (train, val and test)
    
    Saves a metadata file for the model, containing:
        lag_dates (dict): dictionary containing dates that correspond with the train, val and test data in datasets
    """
    # Build sequences
    if model_type == "SHRED":
        train_dataset = create_shred_sequences(data_transformed["train"], sensor_locations, lags=lags)
        val_dataset = create_shred_sequences(data_transformed["val"], sensor_locations, lags=lags)
        test_dataset = create_shred_sequences(data_transformed["test"], sensor_locations, lags=lags)

        # Get dates corresponding to the predictions made by the model
        train_lag_dates = dates["train"][np.array(range(len(dates["train"])-lags)) + lags - 1]
        val_lag_dates = dates["val"][np.array(range(len(dates["val"])-lags)) + lags - 1]
        test_lag_dates = dates["test"][np.array(range(len(dates["test"])-lags)) + lags - 1]

    elif model_type == "SDN":
        # TODO - verify type of SDN sequences, how much data etc, write comment explaining exactly how the sequences match
        train_dataset = create_sdn_datasets(data_transformed["train"], sensor_locations, test_lags=lags)
        val_dataset = create_sdn_datasets(data_transformed["val"], sensor_locations, test_lags=lags)
        # Generate test set to match SHRED test set
        # (use final sensor results for what each sequence would be if using SHRED)
        test_dataset = create_sdn_datasets(data_transformed["test"], sensor_locations, test_lags=lags)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    train_lag_dates = dates["train"][np.array(range(len(dates["train"])-lags)) + lags - 1]
    val_lag_dates = dates["val"][np.array(range(len(dates["val"])-lags)) + lags - 1]
    test_lag_dates = dates["test"][np.array(range(len(dates["test"])-lags)) + lags - 1]
    lag_dates = {"train": train_lag_dates, "val": val_lag_dates, "test": test_lag_dates}

    # Save metadata
    metadata = {
        "lag_dates": lag_dates
    }
    with open(f"{save_path}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, cls=JsonEncoder)

    return datasets


def recons_truth_save(save_path, test_recons, test_ground_truth):
    """Save reconstructions and ground truth data
    
    Args:
        save_path (tuple): file paths to save results under (experiment directory, model directory)
        test_recons (TBC): 
        test_ground_truth (TBC):
    Returns:
        None
    """
    np.save(f"{save_path[1]}/recons", test_recons)  # save for the model
    if not os.path.isfile(f"{save_path[0]}/truth.npy"):
        np.save(f"{save_path[0]}/truth", test_ground_truth)  # save once for the experiment
    else:
        if not np.load(f"{save_path[0]}/truth.npy") == test_ground_truth:  # check truth was the same for this model
            raise ValueError("Ground truth has changed compared to existing saved file")


def train_shred_sdn(datasets, hidden_dim, model_type, norm_model, num_sensors, save_path, sequence_model):
    """Train, validate and test SHRED or SDN models
    
    Args:
        datasets (dict): dictionary containing the sequenced datasets (train, val and test)
        hidden_dim (int): hidden dimension for the sequence model
        model_type (str): model type to train; one of SHRED, SDN, POD, RBFI
        norm_model (class): MinMaxScaler() class fitted on the training data
        num_sensors (int): number of synthetic sensors to use
        save_path (tuple): file paths to save results under (experiment directory, model directory)
        sequence_model (str): sequence model to use; one of LSTM or GRU

    Returns:
        None

    Saves out the reconstructions produced by the model (and the corresponding ground truth)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO - add mutliple training and saving (interun variability)
    if model_type == "SHRED":
        model = models.SHRED(num_sensors, len(lats), seq_model=sequence_model, hidden_size=hidden_dim,
                            hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    elif model_type == "SDN":
        model = models.SDN(num_sensors, len(lats), l1=350, l2=400, dropout=0.1).to(device)

    validation_errors = models.fit(model, datasets["train"], datasets["val"], batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
    print('Validation Reconstruction Error: ', validation_errors)

    # Test
    test_recons = norm_model.inverse_transform(model(datasets["test"].X).detach().cpu().numpy())
    test_ground_truth = norm_model.inverse_transform(datasets["test"].Y.detach().cpu().numpy())
    print('Test Reconstruction Error: ')
    print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

    recons_truth_save(save_path, test_recons, test_ground_truth)


def pod_reconstructions(data_transformed, dates, lags, num_sensors, save_path, sensor_locations, sensor_placement, U_r):
    """TBC

    Args:
        data_transformed (dict): transformed datasets, stored as {train, val, test}
        dates (dict): corresponding dates, stored as {train, val, test}
        lags (int): sequence length in weeks
        num_sensors (int): number of synthetic sensors to use
        save_path (tuple): file paths to save results under (experiment directory, model directory)
        sensor_locations (numpy array): sensor locations corresponding to indices in the full-field data
        sensor_placement (str): type of sensor placement to use; choices are random or QR
        U_r (numpy array): POD basis of the first num_sensors POD modes

    Returns:
        Saves out the reconstructions produced by the model (and the corresponding ground truth)
        Saves a metadata file for the model, containing
            test_lag_dates (numpy array): lag dates that correspond with the sequenced test data used
    """
    test_input, test_ground_truth = create_pod_test_data(data_transformed["test"], sensor_locations, test_lags=lags)
    test_lag_dates = dates["test"][np.array(range(len(dates["test"])-lags)) + lags - 1]

    C = np.zeros((num_sensors, len(lats)))
    for i in range(num_sensors):
        C[i, sensor_locations[i]] = 1

    for row in C:
        print(row)

    A = C @ U_r

    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: C @ U_r is ill-conditioned (cond={cond:.2e}) "
            f"for placement='{sensor_placement}'.")

    coeffs, *_ = np.linalg.lstsq(A, test_input.T, rcond=None)
    test_recons = (U_r @ coeffs).T

    # Saving
    recons_truth_save(save_path, test_recons, test_ground_truth)
    # Save metadata
    metadata = {
        "test_lag_dates": test_lag_dates
    }
    with open(f"{save_path[1]}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, cls=JsonEncoder)


def rbfi_reconstructions(data_transformed, dates, lags, num_sensors, save_path, sensor_coords, sensor_placement):
    """TBC

    Args:
        data_transformed (dict): transformed datasets, stored as {train, val, test}
        dates (dict): corresponding dates, stored as {train, val, test}
        lags (int): sequence length in weeks
        num_sensors (int): number of synthetic sensors to use
        save_path (tuple): file paths to save results under (experiment directory, model directory)
        sensor_coords (numpy array): Latitude and Longitude coordinates corresponding to sensor_locations
        sensor_placement (str): type of sensor placement to use; choices are random or QR

    Returns:
        Saves out the reconstructions produced by the model (and the corresponding ground truth)
        Saves a metadata file for the model, containing
            test_lag_dates (numpy array): lag dates that correspond with the sequenced test data used
    """
    test_input, test_ground_truth = create_pod_test_data(data_transformed["test"], sensor_locations, test_lags=lags)
    test_lag_dates = dates["test"][np.array(range(len(dates["test"])-lags)) + lags - 1]

    points = np.array([[sensor_coords[0][i], sensor_coords[1][i]] for i in range(len(sensor_coords[0]))])
    full_coords = np.column_stack([longs, lats])

    rbf_recons = np.zeros((len(test_input), len(lats)))
    for j, data in enumerate(test_input):
        interpolator = RBFInterpolator(
            points,
            data,
            kernel='thin_plate_spline',
            smoothing=0.0,      # raise this (e.g. 1e-2–1e0) if sensors are noisy
        )
        rbf_recons[j] = interpolator(full_coords)

    recons_truth_save(save_path, rbf_recons, test_ground_truth)
    # Save metadata
    metadata = {
        "test_lag_dates": test_lag_dates
    }
    with open(f"{save_path[1]}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, cls=JsonEncoder)


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    """Split list of strings into a list"""
    return arg.split(',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required (standard)
    parser.add_argument("version", help="Version indicator (str) for saving, e.g. the date in format 051026")
    parser.add_argument("dataset", help="Dataset type to use", choices=["OFAM3", "NOAA"])
    parser.add_argument("models", help="List of model types to train; choices are SHRED,SDN,POD,RBFI. Format as comma separated string (with no spaces)", type=list_of_strings)
    parser.add_argument("num_sensors", help="Number of synthetic sensors to use, integer", type=int)

    # Optional - Bool flags
    parser.add_argument("-a", "--anomaly", action="store_true", help="Use anomalies to train if this flag is used")
    # parser.add_argument("-e", "--eval", action="store_true", help="Evaluate reconstructions if this flag is used")
    # parser.add_argument("-t", "--train", action="store_true", help="Train models if this flag is used")

    # Optional - defaults set
    parser.add_argument("-f", "--filepath", help="File path to save results under",
                        default="Reconstructions/Static/")
    parser.add_argument("-d", "--hidden_dim", help="Hidden dimension for sequence model", type=int, default=128)
    parser.add_argument("-s", "--sequence_model", help="Sequence model to use", default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("-l", "--lags", help="Lags (sequence length in weeks)", default=52, type=int)
    parser.add_argument("-r", "--area", help="Area of ocean to use", default="NA", choices=["world", "NA", "GS"])
    parser.add_argument("-p", "--sensor_placement", help="Type of sensor placement to use; choices are random or QR", choices=["random", "QR"], default="random")

    args = parser.parse_args()

    # Train models
    # if args.train:
        # print("Mode: Train models")
    
    # Load and reformat data
    path = f"{args.dataset}/{args.area}/{args.sensor_placement}/{args.version}_n{args.num_sensors}_l{args.lags}"
    directory = os.path.join(args.filepath, path)
    print("Saving metadata under: ", directory)
    os.makedirs(directory, exist_ok=True)

    data, lats, longs, dates = load_dataset(args.area, args.dataset)
    pp_data, dates_dict, norm_model, sensor_locations, U_r = preprocess_data(anomaly=args.anomaly, data=data, dates=dates,
                                                                        num_sensors=args.num_sensors, save_path=directory,
                                                                        sensor_placement=args.sensor_placement)

    for model_type in args.models:
        # Storage
        if model_type == "SHRED":
            model_name = f"{model_type}-{args.sequence_model}"
        elif model_type == "SDN":
            model_name = model_type
        model_path = path + f"/{model_name}" 

        # Make experiment folder
        model_directory = os.path.join(args.filepath, model_path)
        print("Saving results under: ", model_directory)
        os.makedirs(model_directory, exist_ok = True)

        # Build sequences
        if model_type in ["SHRED", "SDN"]:
            datasets = build_sequences(model_type, pp_data, dates_dict, args.lags, model_directory, sensor_locations)
            # Train, val and test models
            train_shred_sdn(datasets=datasets, hidden_dim=args.hidden_dim, model_type=model_type, norm_model=norm_model,
                           num_sensors=args.num_sensors, save_path=(directory, model_directory), sequence_model=args.sequence_model)
        elif model_type == "POD":
            pod_reconstructions(data_transformed=pp_data, dates=dates_dict, lags=args.lags,
                                num_sensors=args.num_sensors, save_path=(directory, model_directory),
                                sensor_locations=sensor_locations, sensor_placement=args.sensor_placement, U_r=U_r)
        elif model_type == "RBFI":
            rbfi_reconstructions(data_transformed=pp_data, dates=dates_dict, lags=args.lags,
                                num_sensors=args.num_sensors, save_path=(directory, model_directory),
                                sensor_locations=sensor_locations, sensor_placement=args.sensor_placement)

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
