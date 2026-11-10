"""Tests for the weighted evaluation metrics and their printed summary.
 
Run with: python -m pytest tests/
"""
import numpy as np
import pytest

from helper_functions.evaluation import (
    evaluate_metrics,
    weighted_mae,
    weighted_mse,
    weighted_r2,
)


def test_weighted_mae():
    """Check that weighted MAE averages absolute errors across rows and cells."""
    y_true = np.array([[1, 3], [3, 6]])
    y_pred = np.array([[1, 1], [3, 3]])
    weights = np.array([3, 2])

    assert weighted_mae(y_true, y_pred, weights) == pytest.approx(1)


def test_weighted_mse_returns_mean_and_per_row_values():
    """Check that weighted MSE returns its overall mean and per-row values."""
    y_true = np.array([[1, 3], [3, 6]])
    y_pred = np.array([[1, 1], [3, 3]])
    weights = np.array([3, 2])

    mean_mse, mse_by_row = weighted_mse(y_true, y_pred, weights)

    assert mean_mse == pytest.approx(2.6)
    np.testing.assert_allclose(mse_by_row, [1.6, 3.6])


def test_weighted_r2():
    """Check that weighted R² is averaged across rows."""
    y_true = np.array([[1, 3], [3, 6]])
    y_pred = np.array([[1, 1], [3, 3]])
    weights = np.array([3, 2])

    assert weighted_r2(y_true, y_pred, weights) == pytest.approx(-2/3)


def test_evaluate_metrics_prints_weighted_results(capsys):
    """Check that evaluate_metrics prints the weighted metrics."""
    y_true = np.array([[1, 3], [3, 6]])
    y_pred = np.array([[1, 1], [3, 3]])
    weights = np.array([3, 2])

    _, weights_out = evaluate_metrics(y_true, y_pred, weights)
    
    np.testing.assert_allclose(weights_out, weights)

    output = capsys.readouterr().out
    assert "Metrics (area-weighted):" in output
    assert "MAE:\t1.000" in output
    assert "MSE:\t2.600" in output
    assert "R2:\t-0.667" in output


test_data = [  # y_true, y_pred, results
    (np.array([[1, 3, 2], [1, 0, 2]]), np.array([[1, 0, 2], [1, 0, 2]]), (0.5, 1.5, -1.25)),
    (np.array([[1, 3], [3, 6]]), np.array([[1, 1], [3, 3]]), (1.25, 3.25, -1)),
]

@pytest.mark.parametrize("y_true,y_pred,results", test_data)
def test_evaluate_metrics_labels_default_weights_unweighted(capsys, y_true, y_pred, results):
    """Check that evaluate_metrics labels its default-weight results unweighted."""
    metrics, weights_out = evaluate_metrics(y_true, y_pred)
    np.testing.assert_allclose(weights_out,  np.ones(y_true.shape[-1]))

    output = capsys.readouterr().out
    assert "Metrics (unweighted):" in output
    assert metrics["MAE"] == pytest.approx(results[0])
    assert metrics["MSE"] == pytest.approx(results[1])
    assert metrics["R2"] == pytest.approx(results[2])
