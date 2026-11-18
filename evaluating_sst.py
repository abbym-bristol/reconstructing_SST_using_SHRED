"""evaluating_sst.py"""
# Standard imports
import argparse
import os
import random

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from helper_functions.evaluation import (
    evaluate_metrics,
    plot_area_weighted_mean_error,
    plot_area_weighted_mean_SST,
    plot_latitude_integrated_mean_error,
)
from helper_functions.saving import unpack_metadata
from reconstructing_sst import list_of_strings, load_dataset

np.random.seed(42)
random.seed(42)

plt.rcParams.update({'font.size': 16})

# Global variables
SEEDS = [1001, 1002, 1003, 1004, 1006, 2200, 2222, 3345, 23, 42]
VALID_MODELS = {"SHRED-LSTM", "SHRED-GRU", "SDN", "POD", "RBFI"}


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
    parser.add_argument("-s", "--seeds", nargs="+", type=int, default=SEEDS, help="List of seeds to run; if a single seed is given, it's used for all runs")
    

    args = parser.parse_args()

    unknown = set(args.models) - VALID_MODELS
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(sorted(unknown))}. Choose from {sorted(VALID_MODELS)}")

    # ---------- Load reconstructions ----------
    _, lats, _, _ = load_dataset(args.area, args.dataset)

    base_dir = f"{args.dataset}/{args.area}/{args.sensor_placement}/{args.version}_n{args.num_sensors}_l{args.lags}"
    base_path = os.path.join(args.filepath, base_dir)

    for seed in args.seeds:
        path = os.path.join(base_path, f"seed_{seed}")
        ground_truth = np.load(path+"/truth.npy")
        metadata = unpack_metadata(path+"/metadata.json")
        sensor_locs = metadata["sensor_locations"]
        sensor_coords = metadata["sensor_coords"]
        test_dates = metadata["test_lag_dates"]
        weights = metadata["weights"]

        test_recons = {}
        for name in args.models:
            test_recons[name] = np.load(f"{path}/{name}/recons.npy")

        if args.dataset == "NOAA":
                TEST_INDICES = {"values": np.arange(len(test_recons[args.models[0]])), "label": "Test index"}
        elif args.dataset == "OFAM3":
            TEST_INDICES = {"values": test_dates, "label": "Week date"}
        # else not needed, because of args restriction

        # ---------- Metrics ----------
        # Evaluate this seed's results
        print(f"EXPERIMENT (seed={seed}): {path}")
        for name in args.models:
            print(f"MODEL: {name}")
            evaluate_metrics(y_true=ground_truth, y_pred=test_recons[name], weights=weights)

        # ---------- Area-weighted mean SST over time ----------
        plot_area_weighted_mean_SST(y_pred=test_recons, y_true=ground_truth, area_weights=weights,
                                    save_path=path, test_indices=TEST_INDICES)

        # ---------- Area-weighted mean error in SST over time ----------
        difference = plot_area_weighted_mean_error(y_pred=test_recons, y_true=ground_truth, area_weights=weights,
                                                   save_path=path, test_indices=TEST_INDICES)

        # ---------- Latitude-integrated mean error in SST ----------
        plot_latitude_integrated_mean_error(area=args.area, diff_dict=difference, lats=lats, save_path=path)
