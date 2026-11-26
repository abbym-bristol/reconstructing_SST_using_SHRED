"""evaluating_sst.py"""
# Standard imports
import argparse
import os
import random
from glob import glob

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from helper_functions.area_weighting import (
    LatLongGrid,
    area_weighted_mean,
    infer_resolution,
)
from helper_functions.evaluation import (
    evaluate_metrics,
    metric_weighting,
    plot_area_weighted_mean_error,
    plot_area_weighted_mean_SST,
    plot_latitude_integrated_mean_error,
)
from helper_functions.preprocess_data import load_dataset
from helper_functions.saving import unpack_metadata
from reconstructing_sst import list_of_strings

np.random.seed(42)
random.seed(42)

plt.rcParams.update({'font.size': 16})

# Global variable
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
    parser.add_argument("-w", "--weighted", action="store_true", help="Use area weighted metrics in evaluation if this flag is used")

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

    # ---------- Load reconstructions ----------
    _, lats, longs, _ = load_dataset(args.area, args.dataset)

    exp_path = f"{args.dataset}/{args.area}/{args.sensor_placement}/{args.version}_n{args.num_sensors}_l{args.lags}"
    print(f"EXPERIMENT: {exp_path}")
    results_path = os.path.join(args.filepath, exp_path)
    print(results_path)

    loc_paths = [f for f in sorted(glob(os.path.join(results_path, "*/")))]
    # print(loc_paths)
    metrics_dict = {}
    test_recons_dict = {}

    for name in args.models:
        r2 = []
        mae = []
        mse = []
        loc_recons = {}
        for path in loc_paths:
            ground_truth = np.load(path+"/truth.npy")
            metadata = unpack_metadata(path+"/metadata.json")
            sensor_locs = metadata["sensor_locations"]
            sensor_coords = metadata["sensor_coords"]
            test_dates = metadata["test_lag_dates"]
            # TODO - Store weights only once in reconstructing_sst.py, and use same for all locs
            #       Then can have experiment level weight stored in metadata to pull and use here!!!
            weights = metadata["weights"]
            test_recons = np.load(f"{path}/{name}/recons.npy")
            loc_recons[os.path.basename(os.path.normpath(path))] = test_recons

            if args.dataset == "NOAA":
                TEST_INDICES = {"values": np.arange(len(test_recons)), "label": "Test index"}
            elif args.dataset == "OFAM3":
                TEST_INDICES = {"values": test_dates, "label": "Week date"}
            # else not needed, because of args restriction

            # ---------- Metrics ----------
            metric_weights = metric_weighting(area=args.area, arg_weighting=args.weighted,
                                            lats=lats, longs=longs, weights=weights, y_true=ground_truth)
            
            metrics = evaluate_metrics(y_true=ground_truth, y_pred=test_recons, weights=metric_weights, print_results=False)
            r2.append(metrics["R2"])
            mse.append(metrics["MSE"])
            mae.append(metrics["MAE"])
        
        test_recons_dict[name] = loc_recons

        # Combine for all models
        metrics_dict[name] = {
            "R2": f"{np.mean(r2):.4f} +/- {np.std(r2):.4f}",
            "MSE": f"{np.mean(mse):.4f} +/- {np.std(mse):.4f}",
            "MAE": f"{np.mean(mae):.4f} +/- {np.std(mae):.4f}"
            }
        
    # Print for all models
    print(metrics_dict)

    # ---------- Area-weighted evaluations: FOR EACH SENSOR LOC ----------
    # TODO - Store weights only once in reconstructing_sst.py, and use same for all locs
    #       Then can have experiment level weight stored in metadata to pull and use here!!!
    if weights is not None:
        graph_weights = weights  # See todo above: I know that these are the same for all locs so can do this for now
    else:
        resolution = infer_resolution(lats)
        grid = LatLongGrid(args.area, resolution)
        graph_weights = grid.get_area_weights(lats, longs)

    # for path in loc_paths:
    #     test_recons = {}
    #     for model_name, recons in test_recons_dict.items():
    #         test_recons[model_name] = recons[path.split("/")[-1]]
    #     # Area-weighted evaluation -> has to use area weighting regardless of whether the model was trained using weighting
    #     # ---------- Area-weighted mean SST over time ----------
    #     plot_area_weighted_mean_SST(y_pred=test_recons, y_true=ground_truth, area_weights=graph_weights,
    #                                 save_path=path, test_indices=TEST_INDICES)

    #     # ---------- Area-weighted mean error in SST over time ----------
    #     difference = plot_area_weighted_mean_error(y_pred=test_recons, y_true=ground_truth, area_weights=graph_weights,
    #                                                save_path=path, test_indices=TEST_INDICES)

    #     # ---------- Latitude-integrated mean error in SST ----------
    #     plot_latitude_integrated_mean_error(area=args.area, diff_dict=difference, lats=lats, save_path=path)


    # ---------- Area-weighted evaluations: SUMMARY FOR ALL SENSOR LOCS ----------
    mean_test_recons = {}
    std_test_recons = {}
    for name in args.models:
        stacked = np.stack(list(test_recons_dict[name].values()), axis=0)
        mean_test_recons[name] = np.mean(stacked, axis=0)
        std_test_recons[name] = np.std(stacked, axis=0)

    plot_area_weighted_mean_SST(y_pred=mean_test_recons, y_true=ground_truth, area_weights=graph_weights,
                                save_path=results_path, test_indices=TEST_INDICES, y_pred_std=std_test_recons)
    difference = plot_area_weighted_mean_error(y_pred=mean_test_recons, y_true=ground_truth, area_weights=graph_weights,
                                               save_path=results_path, test_indices=TEST_INDICES, y_pred_std=std_test_recons)
