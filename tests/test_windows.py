"""Checks that the shared windows keep sensor inputs and targets aligned.
 
Run with: python -m pytest tests/
"""
import numpy as np
import pytest
 
from helper_functions.windows import build_windows
 
 
@pytest.mark.parametrize("lags", [1, 5, 52])
def test_windows_alignment(lags):
    """Validate that windowed sensor histories remain aligned with their targets.

    This test checks that, for each forecasting target index, the generated
    window contains the correct historical sensor sequence, that the final
    sensor observation matches the last step in each sequence, and that the
    target values at the selected sensor locations are consistent with the
    underlying data.

    Args:
        lags (int): Number of time steps included in each window. The test is
            parameterized over multiple lag values to confirm alignment holds
            across different window lengths.

    Asserts:
        - The number of target samples matches the expected window count.
        - The final target index and date correspond to the last row in the data.
        - `sensor_now` matches the final element of each sensor sequence.
        - `sensor_now` equals the truth values at the chosen sensor locations.
        - Each window sequence contains the correct historical observations in
          chronological order.
        - Output arrays retain the expected float32 dtype.
    """
    rng = np.random.default_rng(0)
    T, m = 120, 30
    data = rng.random((T, m)).astype(np.float32)
    dates = np.arange(T)
    sensors = rng.choice(m, size=4, replace=False)
 
    w = build_windows(data, dates, sensors, lags)
 
    # Every week from lags-1 to the final week is a target (final week included)
    assert len(w["truth"]) == T - lags + 1
    assert w["target_idx"][-1] == T - 1
    assert w["dates"][-1] == dates[-1]
 
    # SDN/POD/RBFI input is exactly the last step of the SHRED sequence
    assert np.array_equal(w["sensor_now"], w["sensor_seqs"][:, -1, :])
 
    # Sensor values at time t equal the truth at the sensor locations at time t
    assert np.array_equal(w["sensor_now"], w["truth"][:, sensors])
 
    # Sequences are the correct history, in order
    for i, t in enumerate(w["target_idx"]):
        assert np.array_equal(w["sensor_seqs"][i], data[t - lags + 1:t + 1, sensors])
        assert np.array_equal(w["truth"][i], data[t])
 
    # No dtype change, so torch.from_numpy gives bit-identical tensors
    assert w["truth"].dtype == np.float32
 
 
def test_too_short_split_raises():
    """Ensure the window builder rejects configurations that are too short.

    This test verifies that building windows with a lag larger than the available
    sequence length raises a ValueError instead of producing invalid data.

    Raises:
        ValueError: If `lags` exceeds the number of observations that can be
            used to form a valid window.
    """
    with pytest.raises(ValueError):
        build_windows(np.zeros((10, 3)), np.arange(10), np.array([0]), lags=52)
 