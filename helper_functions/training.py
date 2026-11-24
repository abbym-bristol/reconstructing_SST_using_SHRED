"""training.py

All models receive the same windows (from build_windows) and return NORMALISED test
reconstructions of shape (N, m), in the same row order as windows["test"]["truth"].
Un-normalising and saving happen once, in main of reconstructing_sst.py.

"""
# Third-party imports
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

# Local imports
from helper_functions import models
from helper_functions.preprocess_data import TimeSeriesDataset


def to_torch_dataset(split_windows, input_key, device):
    """Wrap a view of the shared windows as a torch dataset (no re-indexing happens here)
    
    Args:
        split_windows (dict): windows data for a split (train, val or test) containing:
            sensor_seqs (N, lags, n_sensors), sensor_now (N, n_sensors),
            truth (N, m), dates (N,), target_idx (N,), where N = T - lags + 1
        input_key (str): "sensor_seqs" or "sensor_now" for whether using sequences
            or single timestamp sensor inputs (i.e. SHRED vs SDN models)
        device (str): 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Returns:
        (TimeSeriesDataset): Torch dataset of timeseries data
    """
    X = torch.from_numpy(split_windows[input_key]).to(device)
    Y = torch.from_numpy(split_windows["truth"]).to(device)
    return TimeSeriesDataset(X, Y)


def shred_sdn_reconstructions(windows, hidden_dim, model_type, num_sensors, num_points, weights):
    """Train and validate a SHRED or SDN model on the shared windows, then reconstruct the test split
 
    SHRED uses the sensor history (sensor_seqs); SDN uses only the final step (sensor_now).
    Both are trained, validated and tested on exactly the same target weeks.
 
    Args:
        windows (dict): {train, val, test} windows from build_windows
        hidden_dim (int): hidden dimension for the sequence model
        model_type (str): SHRED-LSTM, SHRED-GRU or SDN
        num_sensors (int): number of synthetic sensors
        num_points (int): size of the full field (m)
        weights (numpy array): Area weights normalised to have mean 1 over the cells or None
 
    Returns:
        test_recons (numpy array): normalised reconstructions, shape (N, m)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO - add mutliple training and saving (interun variability)

    if "SHRED" in model_type:
        input_key = "sensor_seqs"
        model = models.SHRED(num_sensors, num_points, seq_model=model_type.split("-")[-1], hidden_size=hidden_dim,
                            hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    elif model_type == "SDN":
        input_key = "sensor_now"
        model = models.SDN(num_sensors, num_points, l1=350, l2=400, dropout=0.1).to(device)
    else:
        raise ValueError(f"Not a SHRED/SDN model type: {model_type}")

    # Set up dataset
    train_ds = to_torch_dataset(windows["train"], input_key, device)
    val_ds = to_torch_dataset(windows["val"], input_key, device)

    # Train and validate model
    validation_errors = models.fit(model, train_ds, val_ds, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5, weights=weights)
    print('Validation Reconstruction Error: ', validation_errors)

    # Test
    model.eval()  # make sure dropout is off for testing
    with torch.no_grad():
        X_test = torch.from_numpy(windows["test"][input_key]).to(device)
        test_recons = model(X_test).cpu().numpy()

    return test_recons


def pod_reconstructions(test_windows, lats, num_sensors, sensor_locations, sensor_placement, U_r):
    """Gappy POD reconstruction of the test windows (POD basis comes from the full training data)

    Args:
        test_windows (dict): test windows from build_windows
        lats (numpy array): array of latitudes corresponding to full-field shape of reconstructions
        num_sensors (int): number of synthetic sensors to use
        sensor_locations (numpy array): sensor locations corresponding to indices in the full-field data
        sensor_placement (str): type of sensor placement to use; choices are random or QR
        U_r (numpy array): POD basis of the first num_sensors POD modes

    Returns:
        test_recons (numpy array): normalised reconstructions, shape (N, m)
    """
    C = np.zeros((num_sensors, len(lats)))
    for i in range(num_sensors):
        C[i, sensor_locations[i]] = 1

    A = C @ U_r

    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: C @ U_r is ill-conditioned (cond={cond:.2e}) "
            f"for placement='{sensor_placement}'.")

    coeffs, *_ = np.linalg.lstsq(A, test_windows["sensor_now"].T, rcond=None)
    test_recons = (U_r @ coeffs).T
    
    return test_recons


def rbfi_reconstructions(test_windows, lats, longs, sensor_coords):
    """RBF interpolation reconstruction of the test windows

    Args:
        test_windows (dict): test windows from build_windows
        lats (numpy array): array of latitudes corresponding to SST values
        longs (numpy array): array of longitudes corresponding to SST values
        sensor_coords (numpy array): Latitude and Longitude coordinates corresponding to sensor_locations

    Returns:
        rbf_recons (numpy_array): normalised reconstructions, shape (N, m)
    """
    points = np.array([[sensor_coords[0][i], sensor_coords[1][i]] for i in range(len(sensor_coords[0]))])
    full_coords = np.column_stack([longs, lats])

    test_input = test_windows["sensor_now"]
    rbf_recons = np.zeros((len(test_input), len(lats)))
    for j, data in enumerate(test_input):
        interpolator = RBFInterpolator(
            points,
            data,
            kernel='thin_plate_spline',
            smoothing=0.0, # raise this (e.g. 1e-2–1e0) if sensors are noisy
        )
        rbf_recons[j] = interpolator(full_coords)

    return rbf_recons
