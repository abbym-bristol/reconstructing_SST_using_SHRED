"""evaluation.py"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Polygon
from skimage.metrics import structural_similarity as SSI
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from tqdm import tqdm

# from shapely.plotting import plot_polygon


def evaluate_metrics(y_pred, y_true):
    """Calculate metrics from test reconstructions

    Args:
        y_pred (numpy array): predicted y (full-field) for each sequence in test set
        y_true (numpy array): corresponding true y (full-field) for each sequence in test set

    Returns:
        None
    """
    mse = MSE(y_pred=y_pred, y_true=y_true)
    assert np.isclose(mse, np.mean([MSE(y_pred=y_pred[i], y_true=y_true[i]) for i in range(len(y_pred))]))
    
    rmse = np.sqrt(mse)
    nmse = mse/np.mean(y_true)**2

    mae = MAE(y_pred=y_pred, y_true=y_true)
    assert np.isclose(mae, np.mean([MAE(y_pred=y_pred[i], y_true=y_true[i]) for i in range(len(y_pred))]))

    r2_direct = r2_score(y_pred=y_pred, y_true=y_true)
    r2 = np.mean([r2_score(y_pred=y_pred[i], y_true=y_true[i]) for i in range(len(y_pred))])  # needs hand-holding to compare the correct things
    assert not np.isclose(r2, r2_direct)

    print(f"RMSE:\t{rmse:.3f}")
    print(f"MSE:\t{mse:.3f}")
    print(f"NMSE:\t{nmse:.3f}")
    print(f"MAE:\t{mae:.3f}")
    print(f"R2:\t{r2:.3f}")

    ssi = np.mean([SSI(y_pred[i], y_true[i],
                    data_range=(max(y_pred[i].max(), y_true[i].max()) - min(y_pred[i].min(), y_true[i].min())))
                    for i in range(len(y_pred))])
    print(f"Average SSI:\t{ssi:.3f}")


def ssi_by_image(test_values, true_values, lats, longs, plot_images=False):
    """Calculate SSI values for each test + true set of values
    Based on: https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images

    Args:
        test_values (array): test values, e.g. a reconstruction of SST or OHC produced by SHRED
        true_values (array): true values, e.g. a ground truth of SST or OHC from ME4OH
        lats (numpy array): array of latitudes corresponding to array values
        longs (numpy array): array of longitudes corresponding to array values
        plot_images (bool, optional): whether or not to plot images used, difference plot,
            and masked difference on the test image. Defaults to False.

    Returns:
        ssi_images (list): SSI values for each test/true pair.
        Optional: plots of images used, difference plot, and masked difference on the test image.
    """  # noqa: D205
    ssi_images = []
    for i in range(len(test_values)):
        
        truth = true_values[i]
        test = test_values[i]

        # ISSUE: changing the figure size changes the SSI score.... probably because scatter points no longer overlap + more white is more similar
        plt.scatter(x=longs, y=lats, c=test, cmap='plasma', s=1)
        ax = plt.gca()
        ax.set_xlim([-80, 0])
        ax.set_ylim([0, 60])
        plt.axis('off')
        plt.savefig("images/test.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.scatter(x=longs, y=lats, c=truth, cmap='plasma', s=1)
        ax = plt.gca()
        ax.set_xlim([-80, 0])
        ax.set_ylim([0, 60])
        plt.axis('off')
        plt.savefig("images/truth.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # cv2 reads images as BGR by default, where as plt uses RGB format. So Blue and Red color will get flipped if not converted.
        test_img = cv2.imread("images/test.png")
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        truth_img = cv2.imread("images/truth.png")
        truth_img = cv2.cvtColor(truth_img, cv2.COLOR_BGR2RGB)

        # Convert images to grayscale
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        truth_gray = cv2.cvtColor(truth_img, cv2.COLOR_BGR2GRAY)

        score, diff = SSI(test_gray, truth_gray, full=True)
        diff = (diff * 255).astype("uint8")  # Convert to 8-bit unsigned integers
        # print("Image similarity", score)
        ssi_images.append(score)

        if plot_images:
            # Threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            # mask = np.zeros(test_img.shape, dtype='uint8')
            filled_test = test_img.copy()

            for c in contours:
                area = cv2.contourArea(c)
                if area > 40:
                    # x,y,w,h = cv2.boundingRect(c)
                    # cv2.rectangle(test_img, (x, y), (x + w, y + h), (36,255,12), 2)
                    # cv2.rectangle(truth_img, (x, y), (x + w, y + h), (36,255,12), 2)
                    # cv2.drawContours(mask, [c], 0, (0,255,255), -1)
                    cv2.drawContours(filled_test, [c], 0, (0,255,255), -1)

            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
            for image in [test_img, truth_img, diff, filled_test]:
                plt.imshow(image)
                plt.axis('off')
                plt.show()
                plt.clf()

    os.remove("images/test.png")
    os.remove("images/truth.png")

    return ssi_images


def mask_array_by_lat_long(array, lats, longs, area="GS"):
    """Mask an array of data that corresponds to the latitude and longitude coordinates stored in lats and longs,
    to a specified area.

    Args:
        array (numpy array): array of data, of length equal to len(lats) and len(longs)
        lats (numpy array): array of latitudes corresponding to array values
        longs (numpy array): array of longitudes corresponding to array values
        area (str, optional): area to crop to. Current options are "NA" (which will filter to the NA basin), or
            "GS" (Gulf stream region of NA basin). Defaults to "GS".

    Returns:
        masked (numpy array): copy of array with data outside of lat & long masked as NaN.
    """  # noqa: D205
    if area == "GS":
        min_lat, max_lat = 30.0, 50.0
        min_long, max_long = -80.0, -30.0
    elif area == "NA":
        min_lat, max_lat = 0.0, 60.0
        min_long, max_long = -80.0, 0.0

    valid_mask = (
        (lats >= min_lat) & (lats <= max_lat) &
        (longs >= min_long) & (longs <= max_long)
    )

    masked = array.copy()
    masked[:, ~valid_mask] = np.nan
    return masked


def area_rectangle_sphere(lat_tuple, long_tuple, R=6367449):
    """Determine the area of a rectangle defined using latitude and longitude (in degrees)
    
    Args:
        lat_tuple (tuple): start and end latitudes (degrees) of the grid square (latitude 1, latitude 2)
        long_tuple (tuple): start and end longitude (degrees) of the grid square (longitude 1, longitude 2)
        R (float, radius): radius of earth, defaults to the average meridional radius (6,367,449m)

    Returns:
        area of the rectangle on a sphere, in meters squared
    """
    return (np.pi/180) * R**2 * (long_tuple[0]-long_tuple[1])*(np.sin(np.deg2rad(lat_tuple[0]))-np.sin(np.deg2rad(lat_tuple[1])))


def lat_long_grid_square(lat, long, extent=0.25):
    """Determine the extent a grid square of ME4OH data
    
    Args:
        lat (float): midpoint latitude (degrees) of the grid square
        long (float): midpoint longitude (degrees) of the grid square
        extent (float): range of the grid square (i.e. ME4OH resolution)

    Returns:
        tuple (lats, longs) where
            lats (tuple): start and end latitudes (degrees) of the grid square (latitude 1, latitude 2)
            longs (tuple): start and end longitude (degrees) of the grid square (longitude 1, longitude 2)
    """
    lats = (lat-extent/2, lat+extent/2)
    longs = (long-extent/2, long+extent/2)

    return lats, longs


def area_weighted_mean_eqn(data, lats, longs):
    """Weighted average of data, weighted by area on a spherical globe
    
    Calculated using equation in area_rectangle_sphere()

    Args:
        data (array): data of shape (timesteps, len(lats)) which stores a particular value
            (e.g. SST or OHC) for each grid square of the ME4OH full-field OFAM3 dataset
        lats (numpy array): array of latitudes corresponding to the center of each grid square for which data is stored
        longs (numpy array): array of longitudes corresponding to data values for each timestep for which data is stored

    Return:
        mean (array): weighted average of data, weighted by the area of each grid square using its latitude and longitude coordinates
    """
    areas = []
    for i in range(len(lats)):
        lat_tuple, long_tuple = lat_long_grid_square(lats[i], longs[i])
        areas.append(area_rectangle_sphere(lat_tuple, long_tuple))
    areas = np.array(areas)

    mean = [np.sum(d * areas)/np.sum(areas) for d in data]

    return mean, areas


def grid_square_coords(lat, long, extent=0.25):
    """Determine the coordinates of a grid square of ME4OH data
    
    Args:
        lat (float): midpoint latitude (degrees) of the grid square
        long (float): midpoint longitude (degrees) of the grid square
        extent (float): range of the grid square (i.e. ME4OH resolution)

    Returns:
        coords (list): coordinates in lattude and longitude of the four corners of the grid square
    """
    lat_min = lat-extent/2
    lat_max = lat+extent/2
    long_min = long-extent/2
    long_max = long+extent/2
    coords = [(long_min, lat_min), (long_max, lat_min), (long_max, lat_max), (long_min, lat_max)]

    return coords


def area_weighted_mean(data, lats, longs):
    """Average of data, weighted by area of each grid square on the globe

    Notes: https://pyproj4.github.io/pyproj/stable/api/geod.html
    - only works with areas up to half the size of the globe ;
    - certain large polygons may return negative values.
    - lats should be in the range [-90 deg, 90 deg]

    Args:
        data (array): data of shape (timesteps, len(lats)) which stores a particular value
            (e.g. SST or OHC) for each grid square of the ME4OH full-field OFAM3 dataset
        lats (numpy array): array of latitudes corresponding to the center of each grid square for which data is stored
        longs (numpy array): array of longitudes corresponding to data values for each timestep for which data is stored

    Return:
        mean (array): weighted average of data, weighted by the area of each grid square using its latitude and longitude coordinates
    """
    areas = []
    for lat, long in zip(lats, longs):
        coords = grid_square_coords(lat, long)
        poly = Polygon(coords)
        # plot_polygon(poly)
        # plt.show()
        geod = Geod(ellps="WGS84")
        area_m2, _ = geod.geometry_area_perimeter(poly)
        areas.append(area_m2)
    
    areas = np.array(areas)
    mean = [np.sum(d * areas)/np.sum(areas) for d in data]

    return mean, areas
