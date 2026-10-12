"""saving.py"""

import json
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
