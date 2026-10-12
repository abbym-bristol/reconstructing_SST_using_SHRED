"""plotting_functions.py"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# TODO: Combine into unified plotting function?

COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
WORLD = gpd.read_file(COUNTRIES_URL).rename({"ADMIN": "name"}, axis="columns")
regions = ['Southern Europe', 'Northern Africa', 'Western Africa', 'Western Europe', 'Northern Europe', 'Central America', 'South America', 'Northern America', 'Caribbean']
NA = WORLD[WORLD['SUBREGION'].isin(regions)]


def set_area(ax, area):
    """For a specified area of the globe, configure the plotting axes

    Args:
        ax (Axes): axis to configure
        area (str): area to use -> currently configured options are:
            "NA" - North Atlantic basin
            "GS" - gulf stream region of NA basin
            "world" - the whole global ocean
    """
     # Get world countries, and plot
    if area is not None:
        if area == "NA":
            ax.set_xlim([-80, 0])
            ax.set_ylim([0, 60])
            NA.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.75)  # Add countries
        elif area == "GS":
            ax.set_xlim([-80, -30])
            ax.set_ylim([30, 50])
            NA.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.75)  # Add countries
        elif area == "world":
            ax.set_xlim([-180, 180])
            ax.set_ylim([-90, 90])
            WORLD.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.75)  # Add countries
        else:
            print(f"Area specified {area} has not yet been configured")
            plt.close()
            return
    else:
        print("No area specified")
        plt.close()
        return


def plot_ohc_map(data, title, min_ohc, max_ohc, area="world", background=None):
    """For a given subset of data, plot the OHC map over the world or filtered to a specific ocean basin
    
    Args:
        data (df or df.groupby object): dataframe (or df.groupby object of a dataframe) containing columns for
            latitude, longitude and observed OHC.
        title (str): title for the plot and name for saved figure.
        min_ohc (float): minimum OHC value across entire dataset (ensures uniform colourbar)
        min_ohc (float): maximum OHC value across entire dataset (ensures uniform colourbar)
        area (str, optional): area to plot for. Current options are "NA" (which will filter to the NA basin),
            "GS" (Gulf stream region of NA basin), or "world" (whole globe, default).
        background (str, optional): if not None, will fill the plot with a background of the colour specified.
            Valid options: "lightblue", "black", etc. Defaults to None.
    Returns:
        Plot of OHC map, saved figure.
    """
    fig, ax = plt.subplots(figsize=(18, 9))

    # Create a shared normalization object
    norm = Normalize(vmin=min_ohc, vmax=max_ohc)
    cmap = plt.cm.plasma   # Define the colormap

    # Plot OHC map
    s = ax.scatter(data.Longitude, data.Latitude, c=data.OHC, s=1, cmap='plasma', norm=norm)  # Plot OHC observations
    
    cbar = fig.colorbar(s, ax=ax, cmap=cmap, norm=norm)
    cbar.set_label('OHC $(Jm^{−2})$')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if background is not None:
        ax.set_facecolor(background)

    set_area(ax, area)

    plt.title(title)
    # plt.savefig(f"images/ohc_{area}_map_{title}.png")
    plt.show()
    plt.close()


def plot_sst_map(data, annotate=False, area="world", background=None, cmap="plasma", min_sst=None, marker_style='s', marker_size=1, max_sst=None, sensor_coords=None, title=None):
    """For a given subset of data, plot the SST map over the world or filtered to a specific area of the ocean
    
    Args:
        data (df or df.groupby object): dataframe (or df.groupby object of a dataframe) containing columns for
            latitude, longitude and observed SST.
        annotate (Bool, optional): TBC
        area (str, optional): area to plot for. Current options are "NA" (which will filter to the NA basin),
            "GS" (Gulf stream region of NA basin), or "world" (whole globe, default).
        background (str, optional): if not None, will fill the plot with a background of the colour specified.
            Valid options: "lightblue", "black", etc. Defaults to None.
        cmap (str or plt colormap, optional): colourmap to use for figure, defaults to "plasma".
        min_sst (float, optional): minimum value across entire dataset (ensures uniform colourbar).
        marker_size (float, optional): size of marker to use in scatter plots.
        max_sst (float, optional): maximum value across entire dataset (ensures uniform colourbar).
        sensor_coords (tuple): tuple of arrays (longs[sensor_locations], lats[sensor_locations]).
            i.e. longitudes and latitudes at the sensor locations used for the reconstruction. If None, not plotted.
        title (str, optional): title for the plot and name for saved figure.

    Returns:
        Plot of OHC map, saved figure.
    """
    plt.rcParams.update({'font.size': 16})
    if area=="world":
        fig, ax = plt.subplots(figsize=(18, 9))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Create a shared normalization object
    if min_sst is None or max_sst is None:
        min_sst = min(data.SST)
        max_sst = max(data.SST)
    norm = Normalize(vmin=min_sst, vmax=max_sst)
    # cmap = plt.cm.plasma   # Define the colormap

    # Plot SST map
    s = ax.scatter(data.Longitude, data.Latitude, c=data.SST, s=marker_size, cmap=cmap, norm=norm, marker=marker_style)
    if type(annotate) is list:
        for i in annotate:
            plt.annotate(i, (data.Longitude[i-1]+0.5, data.Latitude[i-1]), c='k')
    elif annotate:
            for i in range(len(data.Longitude)):
                plt.annotate(i+1, (data.Longitude[i]+0.5, data.Latitude[i]), c='k')

    cbar = fig.colorbar(s, ax=ax, cmap=cmap, norm=norm)
    if cmap == "bwr":  # for difference plots
        cbar.set_label('Error ($\\degree$C)')
    else:
        cbar.set_label('SST ($\\degree$C)')

    if sensor_coords is not None:
        ax.scatter(sensor_coords[0], sensor_coords[1], marker='x', c='k', s=marker_size*10)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if background is not None:
        ax.set_facecolor(background)

    set_area(ax, area)

    if title is not None:
        plt.title(title)
        # plt.savefig(f"images/sst_{area}_map_{title}.png")
    plt.show()
    plt.close()


def plot_compare_sst_recon(recon_data, truth_data, area="world", cmap="plasma", diff_data=None, marker_style='s', marker_size=1, sensor_coords=None, title=None):
    """For a given subset of data, plot the SST map over a set area of the ocean for either reconstruction and ground truth,
    or reconstruction, ground truth and difference between the two.

    Args:
        recon_data (df or df.groupby object): dataframe (or df.groupby object of a dataframe) containing columns for
            latitude, longitude and SST (reconstructed).
        truth_data (df or df.groupby object): dataframe (or df.groupby object of a dataframe) containing columns for
            latitude, longitude and SST (ground truth).
        area (str, optional): area to plot for. Current options are "NA" (which will filter to the NA basin),
            "GS" (Gulf stream region of NA basin), or "world" (whole globe, default).
        cmap (str or plt colormap, optional): colourmap to use for figure, defaults to "plasma".
        diff_data (df or df.groupby object, optional): dataframe (or df.groupby object of a dataframe) containing columns for
            latitude, longitude and SST (difference between recon_data and truth_data values). If None, difference is not plotted.
        marker_size (float, optional): size of marker to use in scatter plots.
        sensor_coords (tuple, optional): tuple of arrays (longs[sensor_locations], lats[sensor_locations])
            i.e. longitudes and latitudes at the sensor locations used for the reconstruction. If None (default), not plotted.
        title (str, optional): title for the plot and name for saved figure. If None, no title used.

    Returns:
        Plot of OHC map, saved figure.
    """  # noqa: D205
    if diff_data is not None:
        fig, ax = plt.subplots(1, 3, figsize=(22, 9))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(22, 9))

    # Create a shared normalization object for SST plots
    min_sst = min([min(recon_data.SST), min(truth_data.SST)])
    max_sst = max([max(recon_data.SST), max(truth_data.SST)])
    norm = Normalize(vmin=min_sst, vmax=max_sst)

    # Reconstruction subfigure
    s = ax[0].scatter(recon_data.Longitude, recon_data.Latitude, c=recon_data.SST, s=marker_size, cmap=cmap, norm=norm)
    ax[0].set_title('Reconstructed SST ($\\degree$C)')
    if sensor_coords is not None:
        ax[0].scatter(sensor_coords[0], sensor_coords[1], marker='.', c='k', s=marker_size)
    if diff_data is not None:
        cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(s, cax=cax, cmap=cmap, norm=norm)

    # Truth subfigure
    s = ax[1].scatter(truth_data.Longitude, truth_data.Latitude, c=truth_data.SST, s=marker_size, marker=marker_style, cmap=cmap, norm=norm)
    ax[1].set_title('True SST ($\\degree$C)')

    if diff_data is not None:
        cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(s, cax=cax, cmap=cmap, norm=norm)
    else:
        cbar = fig.colorbar(s, ax=ax, cmap=cmap, norm=norm)
        cbar.set_label('SST ($\\degree$C)')
    
    # Difference subfigure
    if diff_data is not None:
        # Set colour bar so white always at 0 degrees difference
        extent=max(np.abs(min(diff_data.SST)), max(diff_data.SST))
        norm_diff = Normalize(vmin=-extent, vmax=extent)
        cmap_diff = 'bwr'
        s = ax[2].scatter(diff_data.Longitude, diff_data.Latitude, c=diff_data.SST, s=marker_size, marker=marker_style, cmap=cmap_diff, norm=norm_diff)
        ax[2].set_title('Error ($\\degree$C)')
        cax = make_axes_locatable(ax[2]).append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(s, cax=cax, cmap=cmap, norm=norm_diff)

    # Style
    for a in ax:
        a.set_xlabel('Longitude')
        a.set_ylabel('Latitude')
        set_area(a, area)

    if title is not None:
        fig.suptitle(title)
    # plt.savefig(f"images/sst_comparison_{area}_map_{title}.png")
    plt.show()
    plt.close()


def plot_compare_ff_insitu(ff_data, insitu_data, area="world", title=None):
    """For a given subset of data, plot the SST map over a set area of the ocean for either reconstruction and ground truth,
    or reconstruction, ground truth and difference between the two.

    Args:
        ff_data (df or df.groupby object): dataframe (or df.groupby object of a dataframe) containing columns for
            latitude, longitude and SST (full-field simulation data).
        insitu_data (df or df.groupby object): dataframe (or df.groupby object of a dataframe) containing columns for
            latitude, longitude and SST (in-situ simulation data).
        area (str, optional): area to plot for. Current options are "NA" (which will filter to the NA basin),
            "GS" (Gulf stream region of NA basin), or "world" (whole globe, default).
        title (str, optional): title for the plot and name for saved figure. If None, no title used.

    Returns:
        Plot of OHC map, saved figure.
    """  # noqa: D205
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))

    # Create a shared normalization object for SST plots
    min_sst = min([min(ff_data.SST), min(insitu_data.SST)])
    max_sst = max([max(ff_data.SST), max(insitu_data.SST)])
    norm = Normalize(vmin=min_sst, vmax=max_sst)
    cmap = 'plasma'

    # Full-Field subfigure
    s = ax[0].scatter(ff_data.Longitude, ff_data.Latitude, c=ff_data.SST, s=1, cmap=cmap, norm=norm)
    ax[0].set_title('Full-field SST ($\\degree$C)')

    cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(s, cax=cax, cmap=cmap, norm=norm)
    cbar.set_label('SST ($\\degree$C)')

    # In-situ subfigure
    s = ax[1].scatter(insitu_data.Longitude, insitu_data.Latitude, c=insitu_data.SST, s=1, cmap=cmap, norm=norm)
    ax[1].set_title('In-situ SST ($\\degree$C)')
    ax[1].set_facecolor("lightblue")

    cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(s, cax=cax, cmap=cmap, norm=norm)
    cbar.set_label('SST ($\\degree$C)')

    # Style
    for a in ax:
        a.set_xlabel('Longitude')
        a.set_ylabel('Latitude')
        set_area(a, area)

    if title is not None:
        fig.suptitle(title)
    # plt.savefig(f"images/sst_comparison_{area}_map_{title}.png")
    plt.show()
    plt.close()


def plot_sensors(sensor_coords, area="NA", background=True, title=None):
    """Plot sensor locations on map

    Args:
        sensor_coords (tuple): tuple of arrays (longs[sensor_locations], lats[sensor_locations])
            i.e. longitudes and latitudes at the sensor locations used for reconstructions.
        area (str, optional): area to plot for. Current options are "NA" (which will filter to the NA basin, default),
            "GS" (Gulf stream region of NA basin), or "world" (whole globe).
        background (bool, optional): whether to fill the plot with a blue background, defaults to True.
        title (str, optional): title for the plot and name for saved figure. If None, no title used.

    Returns:
        Plot of sensor locations
    """
    if area=="world":
        fig, ax = plt.subplots(figsize=(18, 9))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(sensor_coords[0], sensor_coords[1], marker='x', c='k', s=4)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if background:
        ax.set_facecolor("lightblue")

    set_area(ax, area)

    plt.title(title)
    # plt.savefig(f"images/sst_{area}_map_{title}.png")
    plt.show()
    plt.close()
