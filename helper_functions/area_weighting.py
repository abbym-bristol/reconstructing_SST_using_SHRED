"""area_weighting.py"""

import numpy as np
import torch
from pyproj import Geod
from shapely.geometry import Polygon


def infer_resolution(lats):
    # CLAUDE FUNCTION
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


def grid_square_areas(lats, longs):
    """Calculate area of each grid square on the globe

    Notes: https://pyproj4.github.io/pyproj/stable/api/geod.html
    - only works with areas up to half the size of the globe ;
    - certain large polygons may return negative values.
    - lats should be in the range [-90 deg, 90 deg]

    Args:
        lats (numpy array): array of latitudes corresponding to the center of each grid square for which data is stored
        longs (numpy array): array of longitudes corresponding to data values for each timestep for which data is stored

    Returns:
        areas (numpy array): area of each grid cell in m^2, shape (n_cells,)
    """
    resolution = infer_resolution(lats)
    areas = []
    for lat, long in zip(lats, longs):
        coords = grid_square_coords(lat, long, extent=resolution)
        poly = Polygon(coords)
        geod = Geod(ellps="WGS84")
        area_m2, _ = geod.geometry_area_perimeter(poly)
        areas.append(area_m2)
    
    areas = np.array(areas)

    return areas


def area_weighted_mean(data, lats, longs):
    """Average of data, weighted by area of each grid square on the globe

    Args:
        data (array): data of shape (timesteps, len(lats)) which stores a particular value
            (e.g. SST or OHC) for each grid square of the ME4OH full-field OFAM3 dataset
        lats (numpy array): array of latitudes corresponding to the center of each grid square for which data is stored
        longs (numpy array): array of longitudes corresponding to data values for each timestep for which data is stored

    Returns:
        mean (numpy array): weighted average of data, weighted by the area of each grid square using its latitude and longitude coordinates
    """
    areas = grid_square_areas(lats, longs)
    mean = [np.sum(d * areas)/np.sum(areas) for d in data]

    return mean


def area_weights(lats, longs):
    """Normalised area weights
 
    Mean-1 normalisation keeps an area-weighted MSE on the same scale as the unweighted one,
    so learning rates / early-stopping behaviour stay comparable.

    Args:
        lats (numpy array): array of latitudes corresponding to the center of each grid square for which data is stored
        longs (numpy array): array of longitudes corresponding to data values for each timestep for which data is stored

    Returns:
        (numpy array): Area weights normalised to have mean 1 over the cells
    """
    areas = grid_square_areas(lats, longs)
    return areas / areas.mean()


class AreaWeightedMSELoss(torch.nn.Module):
    """Area weighted version of torch.nn.MSELoss that weights each grid cell by its area.
 
    With weights of all ones this is identical to torch.nn.MSELoss().
    """
    def __init__(self, weights):
        """Init
        
        Args:
            weights (numpy array): Area weights normalised to have mean 1 over the cells
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


# TODO:
# - Area weighting for metrics
# - Update area-weighting code for evaluation plots to use this scripts functions
