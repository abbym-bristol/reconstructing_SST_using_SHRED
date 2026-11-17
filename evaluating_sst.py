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
from reconstructing_sst import list_of_strings, load_dataset

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


def plot_area_weighted_mean_SST():
    truth_sst_mean = area_weighted_mean(data=ground_truth, area_weights=weights)
    sst_mean_dict = {
        TEST_INDICES["label"]: TEST_INDICES["values"],
        "Truth": truth_sst_mean
    }
    for name in args.models:
        test_sst_mean = area_weighted_mean(data=test_recons[name], area_weights=weights)
        sst_mean_dict[name] = test_sst_mean

    sst_mean_df = pd.DataFrame(sst_mean_dict)
    sst_mean_df = sst_mean_df.set_index(TEST_INDICES["label"])

    plt.figure()
    for name in args.models:
        plt.plot(sst_mean_df[name], c=COLOURS[name], ls='-', label=name)
    plt.plot(sst_mean_df["Truth"], c=COLOURS["Truth"], ls='--', label="Truth")
    plt.xticks(rotation=90)
    plt.xlabel(TEST_INDICES["label"])
    plt.ylabel("Area-weighted mean SST ($\\degree$C)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.xlim(TEST_INDICES["values"][0], TEST_INDICES["values"][-1])
    plt.savefig(os.path.join(path, "area_weighted_mean_SST.png"), bbox_inches="tight", dpi=300)
    plt.close()


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
    if args.dataset == "NOAA":
        TEST_INDICES = {"values": np.arange(len(test_recons[args.models[0]])), "label": "Test index"}
    elif args.dataset == "OFAM3":
        TEST_INDICES = {"values": test_dates, "label": "Week date"}
    # else not needed, because of args restriction

    plot_area_weighted_mean_SST()

    # ---------- Area-weighted mean error in SST over time ----------
    difference = {}
    mean_diff = {}
    for name in args.models:
        diff = test_recons[name] - ground_truth  # use experiment ground truth
        mean = area_weighted_mean(data=diff, area_weights=weights)
        difference[name] = diff
        mean_diff[name] = np.array(mean)

    sst_diff_dict = {
        TEST_INDICES["label"]: TEST_INDICES["values"],
    }
    for name in args.models:
        sst_diff_dict[name] = mean_diff[name]
    sst_diff_df = pd.DataFrame(sst_diff_dict)
    sst_diff_df = sst_diff_df.set_index(TEST_INDICES["label"])

    plt.figure(figsize=(8,8))
    for name in args.models:
        plt.plot(sst_diff_df[name], c=COLOURS[name], label=name)
    plt.xticks(rotation=90)
    plt.hlines(0, TEST_INDICES["values"][0], TEST_INDICES["values"][-1], colors=['k'], linestyles=['--'], label="Truth")
    plt.xlim(TEST_INDICES["values"][0], TEST_INDICES["values"][-1])
    plt.ylabel("Area-weighted mean error ($\\degree$C)")
    plt.xlabel(TEST_INDICES["label"])
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(os.path.join(path, "area_weighted_mean_error.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # ---------- Latitude-integrated mean error in SST ----------
    _, lats, _, _ = load_dataset(args.area, args.dataset)

    avg_all_time_lat_avg_diffs = {}
    for name in args.models:
        # For each reconstruction at a set "date", find the average difference at each latitude
        latitude_avg_diffs = np.array([pd.Series(diff_at_time).groupby(lats).mean().to_numpy() for diff_at_time in difference[name]])

        # For each latitude, calculate the mean difference over all time
        avg_all_time_lat_avg_diffs[name] = np.array([np.mean(latitude_avg_diffs.T[j]) for j in range(latitude_avg_diffs.shape[1])])

    plt.figure(figsize=(8,8))
    for name, vals in avg_all_time_lat_avg_diffs.items():
        plt.plot(vals, np.unique(lats), c=COLOURS[name], label=name)
    plt.vlines(0, min(lats), max(lats), colors=['k'], linestyles=['--'], label="Truth")
    if args.area == "world":
        plt.ylim(-90, 90)
    elif args.area == "NA":
        plt.ylim(0, 60)

    plt.xlabel('Latitude-integrated mean error ($\\degree$C)')
    plt.ylabel('Latitude')
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(os.path.join(path, "lat_int_mean_error.png"), bbox_inches="tight", dpi=300)
    plt.close()
