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

# Local imports
from helper_functions.area_weighting import LatLongGrid, infer_resolution
from helper_functions.preprocess_data import (
    load_dataset,
    preprocess_data,
)
from helper_functions.saving import JsonEncoder
from helper_functions.training import (
    pod_reconstructions,
    rbfi_reconstructions,
    shred_sdn_reconstructions,
)
from helper_functions.windows import build_windows

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

VALID_MODELS = {"SHRED-LSTM", "SHRED-GRU", "SDN", "POD", "RBFI"}

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
    parser.add_argument("-w", "--weighted", action="store_true", help="Use area weighted loss functions in training if this flag is used")

    # Optional - defaults set
    parser.add_argument("-f", "--filepath", help="File path to save results under",
                        default="Reconstructions/Static/")
    parser.add_argument("-d", "--hidden_dim", help="Hidden dimension for sequence model", type=int, default=128)
    parser.add_argument("-l", "--lags", help="Lags (sequence length in weeks)", default=52, type=int)
    parser.add_argument("-r", "--area", help="Area of ocean to use", default="NA", choices=["world", "NA", "GS"])
    parser.add_argument("-p", "--sensor_placement", help="Type of sensor placement to use; choices are random or QR", choices=["random", "QR"], default="random")
    

    args = parser.parse_args()

    unknown = set(args.models) - VALID_MODELS
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(sorted(unknown))}. Choose from {sorted(VALID_MODELS)}")

    # ---------- Shared set up (identical for every model) ----------

    # Load and reformat data
    directory = f"{args.filepath}{args.dataset}/{args.area}/{args.sensor_placement}/{args.version}_n{args.num_sensors}_l{args.lags}"
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

    if args.weighted:
        print("Loss function will be area weighted")
        resolution = infer_resolution(lats)
        grid = LatLongGrid(args.area, resolution)
        weights = grid.get_area_weights(lats, longs)
        # NOTE: not mean-1 normalised so weighted losses will be on a different scale to the unweighted ones
    else:
        weights = None
 
    metadata = {
        "sensor_locations": list(sensor_locations),
        "sensor_coords": list(sensor_coords),
        "dates": dates_dict,
        "test_lag_dates": windows["test"]["dates"],
        "weights": weights
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
                                                    num_sensors=args.num_sensors, num_points=num_points, weights=weights)
        elif model_type == "POD":
            recons_norm = pod_reconstructions(windows["test"], lats=lats, num_sensors=args.num_sensors, sensor_locations=sensor_locations,
                                              sensor_placement=args.sensor_placement, U_r=U_r)
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
