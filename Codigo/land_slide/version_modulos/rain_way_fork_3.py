import xarray as xr
import os
import pandas as pd
import numpy as np
import re
import time
import sys
import cftime
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
path_tools = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\src\utils"

if path_tools not in sys.path:
    sys.path.append(path_tools)

# Import your modules
import tools
from other_variables import nasa_suceptibility
from terrain import general_assigment


def longest_streak_above_threshold(data, threshold):
    """
    ## 1. CDW
    """
    
    above_threshold = data > threshold
    indices = np.where(above_threshold)[0]
    
    if len(indices) == 0:
        return 0
    
    diff = np.diff(indices)
    streaks = np.split(indices, np.where(diff > 1)[0] + 1)
    streak_lengths = [len(streak) for streak in streaks]
    
    return max(streak_lengths)

def rolling_window_max_precipitation(data, window_size):
    """
    ## 2. RX1day

    Ejemplo :
    
    window_size = 5

    result = rolling_window_max_precipitation(numba_file, window_size)
    print("Maximum cumulative precipitation in a window of 5 days:", result)
    """
    
    data = np.array(data)
    rolling_sum = np.convolve(data, np.ones(window_size, dtype=int), 'valid')
    return rolling_sum.max()

def count_days_above_percentile(data, percentile):
    """
    ## 3. R95p
    """
    
    # Convert the data to a NumPy array
    # Calculate the threshold value for the given percentile
    threshold = np.percentile(data, percentile)
    # Count the number of days with precipitation above the threshold
    count = np.sum(data[ data >= threshold ])
    
    return count


def has_all_days(date_series: np.array,
                 non_leap_year=True):
    """
    The purpose of this function is to verify that after concatenating the historical and future data,
    we have all the values except for those of leap years.
    """

    # Convert dates to pandas DatetimeIndex
    date_series_pd = pd.to_datetime([date.strftime('%Y-%m-%d') if isinstance(date, cftime.datetime) else date for date in date_series])
    
    # Create a full range of dates
    full_range = pd.date_range(start=date_series_pd.min(), end=date_series_pd.max(), freq='D')

    if non_leap_year:
        # Handle leap days
        leap_day_mask = ~((full_range.month == 2) & (full_range.day == 29))
        full_range = full_range[leap_day_mask]

    # Check if all dates in the full range are in the date series
    is_equal = full_range.isin(date_series_pd).all()

    if is_equal:
        print("Dates complete")
    else:
        print("Dates Incomplete")

    return is_equal

def common_clip(file_name :str, bbox=None):
    
    current_xr = xr.open_dataset(file_name, chunks={'time':'auto'})    
    current_xr = current_xr.assign_coords(**{'lon':tools.convert_longitude_to_minus180_to_180(current_xr['lon'])})
    current_xr = tools.regular_slice(xr_file=current_xr,
                                     country_bounds=bbox,
                                     lon_name='lon',
                                     lat_name='lat'
                                    )

    return current_xr


def create_df_modelos(url_nasa: str,
                      assets: pd.DataFrame,
                      bbox=None):
    
        
    """
    This functions returns a dataframe with 3 columns of 
    [Modelos , Escenarios , time_object]

    El proposito general de la funcion es creear el dataframe base el cual sera usado para agrupar modelos y escenarios de forma automatizada.
    """

    coords_raw = {
                'vector_lon':None,
                 'vector_lat':None
                }

    re_nasa_pattern = r"SAM_pr_(?P<Modelos>.+?)_(?P<Escenarios>historical|ssp\d+)_(?P<Fold>\d+).nc$"
    modelos_regex = re.compile(re_nasa_pattern)
    np_arr = np.array([var for var in os.listdir(url_nasa) if modelos_regex.match(var)])
    modelos = pd.Series(np_arr).str.extract(modelos_regex).copy()                     ## Dropna removes the .txt doc i have in folder.
    mod_esc = modelos.groupby('Modelos')

    extracted_coords = False
    first_coord = True
    bad_ones = False

    for modelo, df in mod_esc:

        raw_idx = df.index
        files_txt = np_arr[raw_idx]

        path_name = os.path.join(url_nasa, files_txt[0])
        xarray_file = xr.open_dataset(path_name, chunks={'time':'auto'})

        if not extracted_coords:
            current_xr = common_clip(path_name, bbox)
            (slice_tuple, cols_dict_db, dataset_assigment) = tools.extract_slice_tools(nc_file=current_xr,
                                                                                        dataset=assets,
                                                                                        database_name='lon_lat',
                                                                                        column_isa=['Longitud','Latitud'], 
                                                                                        variable_name='pr',
                                                                                        dim_bool=False)
            
            extracted_coords = True

        vector_lon = xarray_file.lon.values
        vector_lat = xarray_file.lat.values
        my_string = str(xarray_file.time.dtype)
            
        modelos.loc[raw_idx, 'time_type'] = my_string

        if first_coord:
            coords_raw['vector_lon'] = vector_lon
            coords_raw['vector_lat'] = vector_lat
            first_coord = False
            
        else:
            equal_lon = (vector_lon == coords_raw['vector_lon']).all()
            equal_lat = (vector_lat == coords_raw['vector_lat']).all()

            are_equal = equal_lon and equal_lat

            if are_equal:
                continue
            else:
                print("Corrds misalign")
                bad_ones = True
                return None

    np_arr = np.array([os.path.join(url_nasa, name_file) for name_file in np_arr])

    coords_dict = {'slice_tuple':slice_tuple,
                   'cols_dict_db': cols_dict_db,
                   'dataset_assigment': dataset_assigment
                }
    
    if not bad_ones:
        print("All share the same coords")
        return (modelos, np_arr, coords_dict)
    else:
        return None
    

def ensamble_folds(bbox=None,
                   list_names=None,
                   non_leap_year=True):
    """
    bbox -> UNION(str,tuple)

    str available : 
        - 'Colombia'
        - 'Peru'
        - 'Brasil'
        - 'Bolivia'
        - 'Chile'
    """
    
    cast_xarray = []

    for idx, file_name in enumerate(list_names, start=1):
        current_xr = common_clip(file_name, bbox)        
        cast_xarray.append(current_xr)
        
    coords_time = [raw_mod.time.values for raw_mod in cast_xarray]
    time_vector = np.concatenate(coords_time)
    time_complete = has_all_days(time_vector, non_leap_year)
    
    if time_complete:
        to_ensamble = xr.concat(cast_xarray, dim='time')
        return to_ensamble
    else:
        return False


def rain_test(url_nsa:str, assets:pd.DataFrame, path_storage:str):
    
    print(f"Amount of files {len(os.listdir(path_storage))}\n")
    
    path_colombia_terrain = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\02_Alos_Palsar\dem_COL\terrain_WGS84"
    terrain_properties = [('slope_Colombia.tif', 'slp'), ('dem_Colombia.tif','dem')]

    (modelos, np_arr, coords_dict) = create_df_modelos(url_nasa=url_nsa,
                                                      assets=assets,
                                                      bbox='Colombia')

    slice_tuple = coords_dict['slice_tuple']
    slice_tuple_2d = slice_tuple[1:]
    cols_dict_db = coords_dict['cols_dict_db']
    dataset_assigment = coords_dict['dataset_assigment']

    assets = nasa_suceptibility(latlon_df=assets)

    if False:
        for var, short_name in terrain_properties:
            path_tif = os.path.join(path_colombia_terrain, var)
            assets = general_assigment(latlon_df=assets,
                                      file_tif=path_tif,
                                      neat_name=short_name,
                                     )
    
    global_output = []
    
    # Define el rango histórico una vez
    historical_name = 'historical'
    historical_range = "1985-2005"
    historical_start_year = 1985
    historical_end_year = 2004
    
    future_name = 'future'
    maper_escenarios = {
                        'historical': 'Historical', 
                        'ssp245': 'SSP2 4.5', 
                        'ssp370': 'SSP3 7.0',
                        'ssp585': 'SSP5 8.5'
                       }
    
    dtype_ssp_grouper = modelos.groupby(['Escenarios',
                                         'Modelos',
                                         'time_type'])
    
    dict_stat_sample = {}
    
    # Define the time ranges (20 years each)
    time_ranges = [(2020, 2039), (2040, 2059), (2060, 2079), (2080, 2100)]
    time_ranges_str = ['2020-2040','2040-2060', '2060-2080', '2080-2100']

    dict_escenarios = {}
    
    for (escenario, modelo, dtype), df_type_escenario in dtype_ssp_grouper:
        if dtype == 'datetime64[ns]':
            non_leap_year = False
        else:
            non_leap_year = True

        escenario_format_name = maper_escenarios[escenario]
        folder_escenario = dict_escenarios.setdefault(escenario_format_name, {})

        longest_streak_dict = folder_escenario.setdefault('longest_streak_result', {})
        rolling_window_dict = folder_escenario.setdefault('rolling_window_result', {})
        count_days_dict = folder_escenario.setdefault('count_days_result', {})
        
        print(f"Working on {escenario_format_name}\n")
        print(f"\tModelo : {modelo}\n")
        index_ssp = df_type_escenario.sort_values(by='Fold', ascending=True).index
        txt_name_ssp = np_arr[index_ssp]

        ds_ssp = ensamble_folds('Colombia', txt_name_ssp, non_leap_year)
        
        # AQUÍ ESTÁ LA MODIFICACIÓN CLAVE: Bifurcar el procesamiento según el escenario
        if escenario == 'historical':
            # PROCESAMIENTO PARA ESCENARIO HISTÓRICO
            print(f'\t\t{historical_start_year}-01-01 {historical_end_year}-12-31')
            longest_streak_list = longest_streak_dict.setdefault(historical_range, [])
            rolling_window_list = rolling_window_dict.setdefault(historical_range, [])
            count_days_list = count_days_dict.setdefault(historical_range, [])
            
            # Select the data for the historical time range
            data_range = ds_ssp.sel(time=slice(f'{historical_start_year}-01-01', f'{historical_end_year}-12-31'))
            chunk_future = data_range.compute()
            
            # El resto del procesamiento es idéntico
            group = chunk_future.pr.resample(time='YE')
            stat_temp = {
                        'longest_streak_result': [],
                        'rolling_window_result': [],
                        'count_days_result': []
                    }
            
            for idx, (rango, slice_numpy) in enumerate(group.groups.items(), start=1):
                numba_raw = chunk_future.pr.values[slice_numpy] * 86_400
                
                # Apply the custom functions along axis 0
                longest_streak_computation = np.apply_along_axis(longest_streak_above_threshold, axis=0, arr=numba_raw, threshold=1)
                rolling_window_computation = np.apply_along_axis(rolling_window_max_precipitation, axis=0, arr=numba_raw, window_size=5)
                count_days_computation = np.apply_along_axis(count_days_above_percentile, axis=0, arr=numba_raw, percentile=95)
                
                stat_temp['longest_streak_result'].append(longest_streak_computation)
                stat_temp['rolling_window_result'].append(rolling_window_computation)
                stat_temp['count_days_result'].append(count_days_computation)
            
            # Procesamiento final para el período histórico
            for variable, list_numba in stat_temp.items():
                stack_array = np.stack(list_numba, axis=0)
                mean_var = np.mean(a=stack_array, q=0.9, axis=0)
                mean_var = mean_var[slice_tuple_2d]
                
                if variable == 'longest_streak_result':
                    longest_streak_list.append(mean_var)
                elif variable == 'rolling_window_result':
                    rolling_window_list.append(mean_var)
                elif variable == 'count_days_result':
                    count_days_list.append(mean_var)
            
            print("Done Historical")
            
        else:
            # PROCESAMIENTO PARA ESCENARIOS FUTUROS
            for (idx, ((start_year, end_year), horizon_code)) in enumerate(zip(time_ranges, time_ranges_str), start=1):
                print(f'\t\t{start_year}-01-01 {end_year}-12-31')
                longest_streak_list = longest_streak_dict.setdefault(horizon_code, [])
                rolling_window_list = rolling_window_dict.setdefault(horizon_code, [])
                count_days_list = count_days_dict.setdefault(horizon_code, [])
                
                # Select the data for the given time range
                data_range = ds_ssp.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
                chunk_future = data_range.compute()
                
                ### Recomensable no agrupar por año.
                group = chunk_future.pr.resample(time='YE')

                ## Diccionario temporal para realizar el promedio de las capas
                stat_temp = {
                            'longest_streak_result': [],
                            'rolling_window_result': [],
                            'count_days_result': []
                        }
                    
                for idx, (rango, slice_numpy) in enumerate(group.groups.items(), start=1):
                    numba_raw = chunk_future.pr.values[slice_numpy] * 86_400
              
                    # Apply the custom functions along axis 0
                    longest_streak_computation = np.apply_along_axis(longest_streak_above_threshold, axis=0, arr=numba_raw, threshold=1)
                    rolling_window_computation = np.apply_along_axis(rolling_window_max_precipitation, axis=0, arr=numba_raw, window_size=5)
                    count_days_computation = np.apply_along_axis(count_days_above_percentile, axis=0, arr=numba_raw, percentile=95)

                    stat_temp['longest_streak_result'].append(longest_streak_computation)
                    stat_temp['rolling_window_result'].append(rolling_window_computation)
                    stat_temp['count_days_result'].append(count_days_computation)
                
                for variable, list_numba in stat_temp.items():
                    stack_array = np.stack(list_numba, axis=0)
                    mean_var = np.mean(a=stack_array, q=0.9, axis=0)
                    mean_var = mean_var[slice_tuple_2d]
                    
                    if variable == 'longest_streak_result':
                        longest_streak_list.append(mean_var)
                    elif variable == 'rolling_window_result':
                        rolling_window_list.append(mean_var)
                    elif variable == 'count_days_result':
                        count_days_list.append(mean_var)

                print("Done")

    # Procesamiento final para todos los escenarios
    for escenario, stat_dict in dict_escenarios.items():
        tortugaso = {}
    
        for stat_way, dict_horizon in stat_dict.items():
            for horizon, list_to_stack in dict_horizon.items():
                ## Ensamble entre modelos.
                compute_var = np.mean(np.stack(list_to_stack, axis=1), axis=1)   
                list_vars = tortugaso.setdefault(horizon, [])
                list_vars.append(compute_var)

        for temp_horizon, list_array in tortugaso.items():
            current_col = f'{escenario}_{temp_horizon}_index'
            mean_rain = np.mean(np.stack(list_array, axis=1), axis=1)  ### (RP95 + CWD + RXX) / 3
            right_df = pd.DataFrame({
                'lon_lat': cols_dict_db,
                current_col: mean_rain
            })
            dataset_assigment = dataset_assigment.merge(right=right_df, on='lon_lat', how='left')

    print("Storing ...")
    dataset_assigment.to_csv(os.path.join(path_storage, 'dataset_assigment.csv'), index=False)
 
    print("Done")

    return None
