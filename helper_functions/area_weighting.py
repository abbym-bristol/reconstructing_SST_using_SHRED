"""area_weighting.py"""

import numpy as np
import torch
from pyproj import Geod
from shapely.geometry import Polygon

from helper_functions.noaa_data import crop_to_area


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


class LatLongGrid():
    """Lat-long grid and associated cell midpoints, coordinates and areas"""
    def __init__(self, data_lats, area):
        """Init

        Args:
            data_lats (numpy array): array of latitudes corresponding to the center of each grid square for which data is stored
        """
        self.resolution = infer_resolution(data_lats)
        self.__get_midpoints_grid(area)
        self.__get_grid_corner_coords()
        self.__get_grid_cell_areas()


    def __get_midpoints_grid(self, area):
        """Get grid cell midpoints in Latitude and Longitude"""
        lat_range = np.arange(-90+self.resolution/2, 90.0, self.resolution)
        long_range = np.arange(0+self.resolution/2, 360.0, self.resolution)

        # 2D meshgrid if for lat/lon paired at every grid cell
        lon2d, lat2d = np.meshgrid(long_range, lat_range)  # shape (180, 360) each
        long_midpoints = lon2d.reshape(-1)
        long_midpoints = np.where(long_midpoints > 180, long_midpoints - 360, long_midpoints)
        lat_midpoints = lat2d.reshape(-1)

        if area != "world":
            lat_midpoints, long_midpoints = crop_to_area(lats=lat_midpoints, longs=long_midpoints, area=area)

        self.long_midpoints = long_midpoints
        self.lat_midpoints = lat_midpoints

        print(lat_range)
        print(long_range)

        
    def __get_grid_corner_coords(self):
        """Get coordinates of grid cell corners in Latitude and Longitude"""
        self.cell_coords = [grid_square_coords(lat=lat, long=long, extent=self.resolution)
                        for lat, long in zip(self.lat_midpoints, self.long_midpoints)]

    def __get_grid_cell_areas(self):
        """Get grid cell areas"""
        areas = []
        for c in self.cell_coords:
            poly = Polygon(c)
            geod = Geod(ellps="WGS84")
            area_m2, _ = geod.geometry_area_perimeter(poly)
            areas.append(area_m2)

        self.cell_areas = np.array(areas)


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


# TODO: Update area-weighting code for evaluation plots to use this scripts functions
