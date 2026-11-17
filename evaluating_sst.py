"""evaluating_sst.py"""
# Standard imports
import argparse
import os
import random

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

# Local imports
from helper_functions import noaa_data, ofam3_data
from helper_functions.area_weighting import area_weighted_mean
from helper_functions.evaluation import evaluate_metrics
from helper_functions.saving import unpack_metadata
from reconstructing_sst import load_dataset

np.random.seed(42)
random.seed(42)

plt.rcParams.update({'font.size': 16})

COLOURS = {"SHRED-LSTM":"#ffb000",
           "SHRED-GRU": "#dc267f",
           "SDN": "#1589e8",
           "POD": "#631ff3",
           "RBFI": "#fe5100",
            "Truth": "#000000"
            }


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

    VALID_MODELS = {"SHRED-LSTM", "SHRED-GRU", "SDN", "POD", "RBFI"}
    unknown = set(args.models) - VALID_MODELS
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(sorted(unknown))}. Choose from {sorted(VALID_MODELS)}")

    # ---------- Load reconstructions ----------
    exp_path = f"{args.dataset}/{args.area}/{args.sensor_placement}/{args.version}_n{args.num_sensors}_l{args.lags}"
    path = os.path.join(args.filepath, exp_path)
    ground_truth = np.load(path+"/truth.npy")
    metadata = unpack_metadata(path+"/metadata.json")
    sensor_locs = metadata["sensor_locations"]
    sensor_coords = metadata["sensor_coords"]
    test_dates = metadata["test_lag_dates"]
    weights = metadata["weights"]

    test_recons = {}
    for name in args.models:
        test_recons[name] = np.load(f"{path}/{name}/recons.npy")

    # ---------- Metrics ----------
    print(f"EXPERIMENT: {exp_path}")
    for name in args.models:
        evaluate_metrics(y_true=ground_truth, y_pred=test_recons[name], weights=weights)

    # ---------- Area-weighted mean SST over time ----------
    truth_sst_mean = area_weighted_mean(data=ground_truth, area_weights=weights)
    TEST_INDICES = {
        "NOAA": {"values": np.arange(len(test_recons[args.models[0]])), "label": "Test index"},
        "OFAM3": {"values": test_dates, "label": "Week date"},
    }
    sst_mean_dict = {
        TEST_INDICES[args.dataset]["label"]: TEST_INDICES[args.dataset]["values"],
        "Truth": truth_sst_mean
    }
    for name in args.models:
        test_sst_mean = area_weighted_mean(data=test_recons[name], area_weights=weights)
        sst_mean_dict[name] = test_sst_mean

    sst_mean_df = pd.DataFrame(sst_mean_dict)
    sst_mean_df = sst_mean_df.set_index(TEST_INDICES[args.dataset]["label"])

    plt.figure()
    for i, name in enumerate(args.models):
        plt.plot(sst_mean_df[name], c=COLOURS[name], ls='-', label=name)
    plt.plot(sst_mean_df["Truth"], c=COLOURS["Truth"], ls='--', label="Truth")
    plt.xticks(rotation=90)
    plt.xlabel(TEST_INDICES[args.dataset]["label"])
    plt.ylabel("Area-weighted mean SST ($\\degree$C)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.xlim(TEST_INDICES[args.dataset]["values"][0], TEST_INDICES[args.dataset]["values"][-1])
    plt.savefig(os.path.join(path, "area_weighted_mean_SST.png"),
                bbox_inches="tight", dpi=300)
    plt.close()
