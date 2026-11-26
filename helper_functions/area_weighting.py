"""area_weighting.py"""

import numpy as np
import pandas as pd
import torch
from pyproj import Geod
from shapely.geometry import Polygon


def infer_resolution(lats):
    """Infers resolution of data
    
    Infer grid spacing (degrees) from the latitudes of a regular grid
        (0.25 for OFAM3, 1.0 for NOAA).
    
    Args:
        lats (numpy array): array of latitudes corresponding to the
            center of each grid square for which data is stored       
    """
    unique_lats = np.unique(np.asarray(lats, dtype=float))
    if len(unique_lats) < 2:
        raise ValueError("Need at least two distinct latitudes to infer resolution; pass `resolution` explicitly.")
    return float(np.min(np.diff(unique_lats)))
 

def grid_square_coords(lat, long, extent):
    """Determine the coordinates of a grid square centered on lat & long
    
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


def grid_square_area(lat, long, resolution):
    """Calculate area of each grid square on the globe
    
    Notes: https://pyproj4.github.io/pyproj/stable/api/geod.html
    - only works with areas up to half the size of the globe ;
    - certain large polygons may return negative values.
    - lats should be in the range [-90 deg, 90 deg]

    Args:
        lat (float): Latitude corresponding to the center of a grid square
        long (float): Longitude corresponding to the center of a grid square
        resolution (float): extent of a grid square

    Returns:
        area_m2 (float): area of grid cell in m^2
    """
    corner_coords = grid_square_coords(lat, long, extent=resolution)
    poly = Polygon(corner_coords)
    geod = Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(poly)

    return area_m2


def area_weighted_mean(data, area_weights):
    """Average of data, weighted by area of each grid square on the globe

    Args:
        data (numpy array): data of shape (timesteps, full-field) which stores a particular value
            (e.g. SST or OHC) for each grid square of the ME4OH full-field OFAM3 dataset
        area_weights (numpy array): area weights corresponding to the area of each
            grid cell (in m^2) of the full-field grid, shape (full-field,)

    Returns:
        mean (numpy array): weighted average of data, weighted by the area of each grid square
    """
    mean = data @ area_weights/np.sum(area_weights)
    # equivalent to: [np.sum(d * area_weights)/np.sum(area_weights) for d in data]

    return mean


def area_weighted_std(sigma, area_weights):
    """Standard deviation of data weighted by area of each grid square on the globe

    Args:
        sigma (numpy array): standard deviations corresponding to data in area_weighted_mean
            of shape (timesteps, full-field); where data is a mean of multiple locations.
        area_weights (numpy array): area weights corresponding to the area of each
            grid cell (in m^2) of the full-field grid, shape (full-field,)

    Returns:
        std (numpy array): propagated standard deviations, shape (timesteps,)
    """  # noqa: D401
    std = np.sqrt(sigma**2 @ area_weights**2) / np.sum(area_weights)
    # equivalent to: np.sqrt(np.sum(area_weights**2 * y_pred_std[name]**2, axis=1)) / area_weights.sum()
    return std


class LatLongGrid():
    """Lat-long grid and associated cell midpoints, coordinates and areas"""
    def __init__(self, region, resolution):
        """Init

        Args:
            region (str): Area of ocean to use, choices: "world", "NA", "GS"
            resolution (float): grid spacing (degrees) of data
        """
        self.resolution = resolution
        self.__get_grid(region)
        self.__get_area_map()


    def __get_grid(self, region):
        """Get grid cell midpoints in Latitude and Longitude

        Args:
            region (str): Area of ocean to use, choices: "world", "NA", "GS"
        """
        if region == "GS":
            min_lat, max_lat = 30.0, 50.0
            min_long, max_long = -80.0, -30.0
        elif region == "NA":
            min_lat, max_lat = 0.0, 60.0
            min_long, max_long = -80.0, 0.0
        elif region == "world":
            min_lat, max_lat = -90.0, 90.0
            min_long, max_long = -180.0, 180.0
        else:
            ValueError("Area specified is not currently configured.")

        # Produce midpoint grid across area
        lat_range = np.arange(min_lat+self.resolution/2, max_lat, self.resolution)
        long_range = np.arange(min_long+self.resolution/2, max_long, self.resolution)

        # 2D meshgrid if for lat/lon paired at every grid cell
        lon2d, lat2d = np.meshgrid(long_range, lat_range)  # shape (180, 360) each
        long_midpoints = lon2d.reshape(-1)
        lat_midpoints = lat2d.reshape(-1)

        # Store grid as a dataframe
        self.grid = pd.DataFrame({
            "Latitude (degree)": lat_midpoints,
            "Longitude (degree)": long_midpoints
        })


    def __get_area_map(self):
        """Get grid cell areas, and make map as pivot table of Latitudes and Longitudes"""
        areas = [grid_square_area(row["Latitude (degree)"], row["Latitude (degree)"], self.resolution) for _, row in self.grid.iterrows()]
        self.grid["Area (m2)"] = areas
        self.map = self.grid.pivot_table(index="Latitude (degree)", columns="Longitude (degree)", values="Area (m2)")


    def get_area_weights(self, lats, longs):
        """Return area of each grid square specified by lats/longs provided
    
        Args:
            lats (numpy array): array of latitudes corresponding to the center of each grid square for which data is stored
            longs (numpy array): array of longitudes corresponding to data values for each timestep for which data is stored
    
        Returns:
            areas (numpy array): area of each grid cell in m^2, shape (n_cells,)
        """
        areas = np.array([self.map.at[lat, long] for lat, long in zip(lats, longs)])
        return areas


class AreaWeightedMSELoss(torch.nn.Module):
    """Area weighted version of torch.nn.MSELoss that weights each grid cell by its area.
 
    With weights of all ones this is identical to torch.nn.MSELoss().

     Notes:
        Weights should represent relative importance or physical areas
        The weights will be applied to individual grid cells in forward()
    """
    def __init__(self, weights):
        """Init
        
        Args:
            weights (numpy array): Weights as the area for each grid cell
        """
        super().__init__()
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        """Forward pass

        Args:
            inputs (tensor): input data
            targets (tensor): target data (ground truth)

        Return:
            Area Weighted MSE Loss
        """
        return (((inputs - targets)**2 ) * self.weights).mean()
