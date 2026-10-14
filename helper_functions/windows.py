"""windows.py
 
Decides how sensor inputs line up with full-field targets.
Every model (SHRED, SDN, POD, RBFI) takes a view of these arrays, so their
train/val/test samples are aligned by construction.
"""
import numpy as np
 
 
def build_windows(data, dates, sensor_locations, lags):
    """Window one data split into aligned sensor inputs and full-field targets
 
    Target time t (for every t from lags-1 to the final week, inclusive) has:
        sensor_seqs[i] = data[t-lags+1 : t+1, sensor_locations]  - sensor history ending at t (SHRED)
        sensor_now[i]  = data[t, sensor_locations]               - sensor values at t only (SDN, POD, RBFI)
        truth[i]       = data[t]                                  - full field at t
        dates[i]       = dates[t]
 
    Args:
        data (numpy array): normalised data for one split, shape (T, m)
        dates (numpy array): dates for data, shape (T,)
        sensor_locations (numpy array): sensor indices in the full field
        lags (int): sequence length in weeks
 
    Returns:
        windows (dict): sensor_seqs (N, lags, n_sensors), sensor_now (N, n_sensors),
            truth (N, m), dates (N,), target_idx (N,), where N = T - lags + 1
    """
    if len(data) != len(dates):
        raise ValueError("data and dates must have the same length")
    if len(data) < lags:
        raise ValueError(f"Split has {len(data)} weeks, fewer than lags={lags}")
 
    target_idx = np.arange(lags - 1, len(data))
    sensor_seqs = np.stack([data[t - lags + 1:t + 1, sensor_locations] for t in target_idx])
 
    return {
        "sensor_seqs": sensor_seqs,
        "sensor_now": sensor_seqs[:, -1, :],
        "truth": data[target_idx],
        "dates": np.asarray(dates)[target_idx],
        "target_idx": target_idx,
    }
 