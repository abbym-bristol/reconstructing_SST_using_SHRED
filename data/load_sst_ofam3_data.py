"""load_sst_ofam3_data.py

Load and preprocess ME4OH data SST OFAM3 data for data visualisation.
Puts in the same format as the ME4OH sampled dataset (but stored in separate files by timestamp)

Basic preprocessing done:
- Remove filler values (represent lat/long coordinates on land)
- Convert longitude to -180 to 180 format, allows plotting against geopandas world maps
- Filter to NA basin

"""
import os
from datetime import date, timedelta

import netCDF4 as nc
import pandas as pd
from tqdm import tqdm

from Data.load_sst_en411_data import crop_df_to_area

FILE_PATH = "Data/OFAM3/temp_ofam3_7d_197901-201412.0p25x0p25.nc"
NA_DATA_PATH = "Data/OFAM3/NA/"

def load_data():
    """Load the data from the NetCDF file

    Args:
        None

    Returns:
        temps (array): array with shape (len(dates), 1, len(lats), len(longs)) containing all the SST measurements
        dates (array): array of length 1879 containing corresponding Datetimes for the data
        lats (array): array of length 600 containing the range of latitude values for the data
        longs (array): array of length 1440 containing the range of longitude values for the data
    """
    file_data = nc.Dataset(FILE_PATH,'r')

    # Extract variables
    start_time = date(1979,1,1)
    dates = [start_time + timedelta(t) for t in file_data["Time"][:].data]  # Time stored as days since 1979-01-01 00:00:00
    lats = file_data["yt_ocean"][:].data  # in degrees N
    longs = file_data["xt_ocean"][:].data  # in degrees E
    # depth = file_data["st_ocean"][:].data  # in metres, Z axis
    temps = file_data["temp"][:].data  # in degrees C, sea_water_potential_temperature
    # temp.shape is (Time, st_ocean, yt_ocean, xt_ocean)

    file_data.close()

    return temps, dates, lats, longs


def process_sst_data(temps, dates, lats, longs, data_type="OFAM3", world_data=True):
    """Iterate through temps array to build DataFrames for each timestamp, of the lat, long and SST values.
    Preprocesses this to change the Longitude coordinates ready for geopandas plotting and to filter to the NA basin.

    Args:
        temps (array): array with shape (len(dates), 1, len(lats), len(longs)) containing all the SST measurements
        dates (array): array of length 1879 containing corresponding Datetimes for the data
        lats (array): array of length 600 containing the range of latitude values for the data
        longs (array): array of length 1440 containing the range of longitude values for the data
        data_type (str, optional): string of data type, current choices: OFAM3, NOAA. Defaults to OFAM3.
        world_data (bool, optional): whether to save world data. Defaults to True, saving world data.

    Returns:
        None
    """  # noqa: D205
    for k in tqdm(range(len(dates))):
        sst_array = []
        lat_array = []
        long_array = []
        for i in range(len(lats)):
            for j in range(len(longs)):
                sst_array.append(temps[k][0][i][j])
                lat_array.append(lats[i])
                long_array.append(longs[j])
        df = pd.DataFrame({
            "Latitude": lat_array,
            "Longitude": long_array,
            "SST": sst_array
        })

        # Remove land values - these have filler value of -32768 used
        land_indexes = df.index[df['SST'] == -32768].tolist()  
        df = df.loc[~df.index.isin(land_indexes)]

        # Convert longitude to -180 to 180 format, allows plotting against geopandas world maps
        df['Longitude'] = df['Longitude'].apply(lambda x: -(360-x) if x > 180 else x)

        # Save world file
        if world_data:
            directory = f"Data/{data_type}/world/"
            os.makedirs(directory, exist_ok=True)
            df.to_csv(f"{directory}/{dates[k]}.csv", index=False)

        # Filter to NA basin
        directory = f"Data/{data_type}/NA/"
        os.makedirs(directory, exist_ok=True)
        na_df = crop_df_to_area(df, "NA", "Latitude")  # 12/08/26: sort by changed from Date (which doesn't exist); but haven't tested this code
        na_file_name = f"{directory}/{dates[k]}.csv"
        na_df.to_csv(na_file_name, index=False)


if __name__ == "__main__":
    # Read file
    print("Loading file...")
    temps, dates, lats, longs = load_data()

    print(temps.shape)

    # Preprocess data
    print("Preprocessing data...")
    # process_sst_data(temps, dates, lats, longs)
