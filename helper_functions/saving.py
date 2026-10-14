"""saving.py"""

import json
import os
from datetime import date, datetime

import numpy as np


class JsonEncoder(json.JSONEncoder):
    """Encodes np and datetime types for json file"""
    def default(self, obj):
        """Default behaviour"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super(JsonEncoder, self).default(obj)


def unpack_metadata(metadata_file):
    """Read json file, unpack metadata dictionary, and restore types
    
    Args:
        metadata_file (str): file path to metadata

    Returns:
        sensor_locations (numpy array): indexes corresponding to the sensor locations in the full lat/long grid array
        sensor_coords (tuple): tuple of arrays (longs[sensor_locations], lats[sensor_locations])
            i.e. longitudes and latitudes at the sensor locations used for the reconstruction.
        test_lag_dates (numpy arrays) dates corresponding to the predictions made by the model
    """
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    for k, v in metadata.items():
        if "date" in k:
            metadata[k] = np.array([date.fromisoformat(string) for string in metadata[k]])
        else:
            metadata[k] = np.array(v)

    return tuple(metadata.values())


def all_equal(iterator):
  """Check all arrays passed in iterator are equal"""
  try:
     iterator = iter(iterator)
     first = next(iterator)
     return all(np.array_equal(first, rest) for rest in iterator)
  except StopIteration:
     return True


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
        saved_truth = np.load(f"{save_path[0]}/truth.npy")
        print(saved_truth)
        print(test_ground_truth)
        if not np.array_equal(saved_truth, test_ground_truth):  # check truth was the same for this model
            raise ValueError("Ground truth has changed compared to existing saved file")
