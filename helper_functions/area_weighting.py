"""area_weighting.py"""

import numpy as np
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

# TODO:
# - Area weighting for loss fucntion
# - Area weighting for metrics
# - Update area-weighting code for evaluation plots to use this scripts functions
