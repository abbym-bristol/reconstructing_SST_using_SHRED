"""load_sst_en411_data.py

Load and preprocess SST ME4OH data for experimentation.

"""

import datetime
import os
from glob import glob

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = 'Data/ME4OH_EN411_OFAM3/data/en4.1.1/1993-2014/'

def load_sst_data(filepath):
    """Load data file, extract variables, basic variable unpacking"""
    file_data = nc.Dataset(filepath,'r')

    # Extract variables
    lat = file_data["ts_lat"][:].data  # Latitude
    long = file_data["ts_lon"][:].data  # Longitude
    time = file_data["en4_ymd"][:].data  # Year , month , day
    sst = file_data["sst"][:].data # Sea surface temperature
    wmo_inst_type = file_data["en4_wmo_inst_type"][:].data
    proj_name = file_data["en4_project_name"][:].data
    plat_num = file_data["en4_platform_number"][:].data

    # Close the file!
    file_data.close()

    # Convert dates and platform info
    processed_date = np.array([datetime.date(int(time[i, 0]), int(time[i, 1]), int(time[i, 2])) for i in range(len(time))])
    processed_wmo = np.array([''.join(list(map(lambda x: x.decode('utf-8'), wmo_type))).strip() for wmo_type in  wmo_inst_type])
    processed_proj_name = np.array([''.join(list(map(lambda x: x.decode('utf-8'), name))).strip() for name in proj_name])
    processed_plat_num = np.array([''.join(list(map(lambda x: x.decode('utf-8'), num))).strip() for num in plat_num])

    # Make df
    sst_df = pd.DataFrame({
        "Date": processed_date,
        "Latitude": lat,
        "Longitude": long,
        "SST": sst,
        "WMO Instrument Type": processed_wmo,
        "Project name": processed_proj_name,
        "Platform number": processed_plat_num
    })
    sst_df = sst_df.dropna()
    
    return sst_df


def crop_df_to_area(df, area, sort_by):
    """Crop a dataframe with Latitude, Longitude and Data (any name) to an area set coordinates

    Args:
        df (pandas Dataframe): containing columns for "Latitude", "Longitude", and other data in other columns
        area (str): area to crop to. Current options are "NA" (which will filter to the NA basin), or
            "GS" (Gulf stream region of NA basin).
        sort_by (str): column to sort df by
    """
    if area == "GS":
        min_lat, max_lat = 30.0, 50.0
        min_long, max_long = -80.0, -30.0
    elif area == "NA":
        min_lat, max_lat = 0.0, 60.0
        min_long, max_long = -80.0, 0.0

    df_mask = ((df['Longitude'] >= min_long) & (df['Longitude'] <= max_long) &
               (df['Latitude'] >= min_lat) & (df['Latitude'] <= max_lat))

    return df.loc[df_mask].sort_values(sort_by).reset_index(drop=True)


if __name__ == "__main__":
    files = [f for f in glob(os.path.join(DATA_PATH, "*.nc"))]
    print(f"Found {len(files)} files in {DATA_PATH} directory")

    # Read all files
    print("Loading files...")
    df = load_sst_data(files[0])
    for path in tqdm(files[1:]):
        df = pd.concat([df, load_sst_data(path)])
        df["Date"] = pd.to_datetime(df["Date"])

    # Convert longitude to -180 to 180 format, allows plotting against geopandas world maps
    print("Converting longitudes...")
    tqdm.pandas()
    df["Longitude"] = df["Longitude"].progress_apply(lambda x: -(360-x) if x > 180 else x)
    world_file_name = "Data/ME4OH_EN411_OFAM3/SST_WORLD_1993_2014.csv"
    df.to_csv(world_file_name, index=False)
    print(f"Whole dataset saved as {world_file_name}")

    # Filter to NA basin
    print("Filtering NA basin...")
    na_df = na_df = crop_df_to_area(df, "NA", "Date")
    na_file_name = "Data/ME4OH_EN411_OFAM3/SST_NA_1993_2014.csv"
    na_df.to_csv(na_file_name, index=False)
    print(f"NA basin dataset saved as {na_file_name}")
