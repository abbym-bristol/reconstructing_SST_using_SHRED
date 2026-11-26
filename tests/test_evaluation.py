"""Tests for the evaluation.py functions.
 
Run with: python -m pytest tests/

"""
import numpy as np

from helper_functions.evaluation import mean_and_std_locations


def test_mean_and_std():
    """Check that mean_and_std_locations works as expected"""
    test_dict = {
            "model1": {
                "A": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),   # shape (2, 3)
                "B": np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),  # same shape as A
            },
            "model2": {
                "A": np.array([[1.0, 3.0, 3.0], [4.0, 3.0, 6.0]]),   # shape (2, 3)
                "B": np.array([[3.0, 8.0, 9.0], [13.0, 11.0, 12.0]]),  # same shape as A
            }
        }
    mean, std = mean_and_std_locations(test_dict)

    np.testing.assert_array_equal(mean["model1"], np.array([[4., 5., 6.], [7., 8., 9.]]))
    np.testing.assert_array_equal(mean["model2"], np.array([[2., 5.5, 6.], [8.5, 7., 9.]]))

    np.testing.assert_array_equal(std["model1"], np.array([[3., 3., 3.], [3., 3., 3.]]))
    np.testing.assert_array_equal(std["model2"], np.array([[1., 2.5, 3.], [4.5, 4., 3.]]))
