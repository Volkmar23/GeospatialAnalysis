import xarray as xr
import numpy as np
import os
import pandas as pd

from typing import Union, List

obj_xr_class = xr.core.dataset.Dataset


# Function to convert cftime.Datetime360Day to pd.Timestamp
def convert_to_timestamp(date):
    try:
        return pd.Timestamp(date.strftime('%Y-%m-%d'))
    except ValueError as e:
        print(f"Error converting date {date}: {e}")
        return None

def has_all_days(date_series:np.array):

    """
    Function to verify that a numpy vectro as all the days...
    was created to see if the chirps files has all the days is saying.
    """
    time_type = str(date_series.dtype)

    if time_type == 'object':

        cftime_date_max = date_series.max()
        cftime_date_min = date_series.min()
        
        # Convert cftime.Datetime360Day to pd.Timestamp
        pd_date_min = pd.Timestamp(cftime_date_min.strftime('%Y-%m-%d'))
        pd_date_max = pd.Timestamp(cftime_date_max.strftime('%Y-%m-%d'))

        ### date_series = pd.to_datetime([pd.Timestamp(date.strftime('%Y-%m-%d')) for date in date_series])
        # Convert the date series to pd.Timestamp and handle errors
        date_series = [convert_to_timestamp(date) for date in date_series]
        
        # Create a date range from the minimum to the maximum date in the series
        full_range = pd.date_range(start = pd_date_min, end = pd_date_max, freq='D')
        
        # Convert to pandas Datetime and handle leap days
        # I made this so i can make a proper ensamble within Nasa mixdataset.
        leap_day_mask = ~((full_range.month == 2) & (full_range.day == 29))
        full_range = full_range[leap_day_mask]

    else:

        # Ensure the series is of datetime type
        date_series = pd.to_datetime(date_series)
        full_range = pd.date_range(start=date_series.min(), end=date_series.max(), freq='D')
        
    is_in_date = full_range.isin(date_series)
    results = is_in_date.all()

    if not results:

        print("This days are outside or Climate data\n")

        full_range_lack = full_range[~is_in_date]
        for day_not in full_range_lack:
            print(day_not)
    else:
        print("Dates complete")
        
    # Check if all dates in the full range are in the date series
    return results

def verify_files(ds:  xr.core.dataset.Dataset,
                 var_name: str,
                 time_name :str):

    
    dim_time = ds.sizes.get(time_name)
    vector_time = ds.get(time_name).values
    unique_values = np.unique(ds.get(var_name).isnull().sum(dim =time_name).values)

    cond1 = (np.isin([0,dim_time],unique_values).all())
    cond2 = (unique_values.size == 2)

    is_complete = (cond1 and cond2 )
    
    if not is_complete:

        
        print(f"Problem, incomplete.\n")
        print(f"Unique values in ds: {unique_values}")
        
        max_value = unique_values[np.argmax(unique_values)]
        print(f"\tComplete : {dim_time} ,current amount {max_value}")

        
    else:
        print(f"\tComplete\n")

        ### if not skip_range_verify:
        ###     print("Lacks some days.")
    
    return is_complete

def verify_completness(txt_names:Union[str, xr.core.dataset.Dataset],
                       var_name: str,
                       time_name = 'time'):

    """
    For temperature the name of the xarray files are : 'tasmax'
    For chirps , var_name = 'precip'
    """

    good_ones = 0

    if isinstance(txt_names, obj_xr_class):

        ## This function was inspired by the fact that nasa now comes in more chunks...
        ds = txt_names
        is_complete = verify_files(ds = ds,var_name =var_name ,time_name = time_name)
        return is_complete

    elif isinstance(txt_names, list):
    
        for idx,txt_name in enumerate(txt_names,start = 1):
            print(f"{txt_name}")
            
            if isinstance(txt_name, str):
                ds = xr.open_dataset(txt_name)  # Open with Dask chunks
                is_complete = verify_files(ds = ds,var_name =var_name ,time_name = time_name)
                good_ones+=int(is_complete)
                if good_ones % 5 == 0:
                    print(f"Going good with {good_ones}")

    print(f"\nGood ones {good_ones}\nbadones {len(txt_names) - good_ones}")

    
