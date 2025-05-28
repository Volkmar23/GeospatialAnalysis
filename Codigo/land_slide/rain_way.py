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
from datetime import datetime, timedelta


# Define paths
path_tools = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\utils"


if path_tools not in sys.path:
    sys.path.append(path_tools)

# Import your modules
import tools
from terrain_assigment import calcular_terreno_puntos



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
            (slice_tuple, cols_dict_db, dataset_assigment  ,dim_mapping ) = tools.extract_slice_tools(nc_file=current_xr,
                                                                                                        dataset=assets,
                                                                                                        database_name='lon_lat',
                                                                                                        column_isa=['Longitud','Latitud'], 
                                                                                                        variable_name='pr',
                                                                                                        dim_bool=True)
            
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
                   'dataset_assigment': dataset_assigment,
                   'dim_mapping':dim_mapping
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


## def rain_test(url_nsa: str, assets: pd.DataFrame, path_storage: str) -> None:
def main(url_nsa: str, assets: pd.DataFrame, path_storage: str) -> None:
    """
    Calcula índices de precipitación para escenarios históricos y futuros.
    
    Parameters
    ----------
    url_nsa : str
        Ruta al directorio con archivos de datos climáticos
    assets : pd.DataFrame
        DataFrame con coordenadas de activos/puntos a analizar
    path_storage : str
        Ruta donde se guardarán los resultados
        
    Returns
    -------
    None
        Los resultados se guardan en un archivo CSV
    """
    import logging
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Iniciando procesamiento. Archivos en directorio destino: {len(os.listdir(path_storage))}")
    
    # Constantes
    PRECIPITATION_CONVERSION_FACTOR = 86_400  # Convertir de kg/m2/s a mm/día
    THRESHOLD_WET_DAY = 1.0  # mm/día para días con precipitación
    WINDOW_SIZE = 5  # Tamaño de ventana para precipitación acumulada
    PERCENTILE_THRESHOLD = 95  # Percentil para definir precipitación extrema
    QUANTILE_VALUE = 0.9  # Percentil para el análisis estadístico final
    
    # Rutas de datos de terreno
    path_colombia_terrain = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\02_Alos_Palsar\dem_COL\terrain_WGS84\slope_Colombia.tif"
    # Ejemplo de uso
    ruta_dem = r'C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\01_opentopography_SRTMGL1\01_DEM\Colombia\dem_Colombia_SRTM.tif'


    # Crear dataframe base con modelos y escenarios
    logger.info("Creando dataframe de modelos y escenarios")
    modelos_result = create_df_modelos(url_nasa=url_nsa, 
                                     assets=assets,
                                     bbox='Colombia')
    
    if modelos_result is None:
        logger.error("Error al crear dataframe de modelos. Coordenadas no alineadas.")
        return None
        
    modelos, np_arr, coords_dict = modelos_result

    # Extraer información de cortes espaciales
    slice_tuple = coords_dict['slice_tuple']
    slice_tuple_2d = slice_tuple[1:]
    cols_dict_db = coords_dict['cols_dict_db']

    
    dataset_assigment = coords_dict['dataset_assigment']
    dim_mapping = coords_dict['dim_mapping']


    # Enriquecer dataset con susceptibilidad NASA
    logger.info("Aplicando susceptibilidad NASA a los activos")

    # Define periodos de análisis
    # Periodo histórico
    HISTORICAL_ID = 'historical'
    historical_range = "1985-2005"
    historical_start_year = 1985
    historical_end_year = 2004
    
    # Mapeo de escenarios
    maper_escenarios = {
        'historical': 'Historical', 
        'ssp245': 'SSP2 4.5', 
        'ssp370': 'SSP3 7.0',
        'ssp585': 'SSP5 8.5'
    }
    
    # Agrupar por escenarios y modelos
    dtype_ssp_grouper = modelos.groupby(['Escenarios', 'Modelos', 'time_type'])
    
    # Periodos futuros de 20 años
    time_ranges = [(2020, 2039), (2040, 2059), (2060, 2079), (2080, 2100)]
    time_ranges_str = ['2020-2040', '2040-2060', '2060-2080', '2080-2100']

    # Diccionario para almacenar resultados por escenario
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
                mean_var = np.quantile(a=stack_array,q = 0.9, axis=0)        
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
                    mean_var = np.quantile(a=stack_array,q = 0.9, axis=0)
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

            index_matrix = np.stack(list_array, axis=1)

            if index_matrix.shape[1] != 3:
                print(index_matrix.shape[1])
                print("Wrong")
                return None
                
            mean_rain = np.mean(index_matrix, axis=1)  ### (RP95 + CWD + RXX) / 3
            right_df = pd.DataFrame({
                                    'lon_lat': cols_dict_db,
                                    current_col: mean_rain
                                })
            dataset_assigment = dataset_assigment.merge(right=right_df, on='lon_lat', how='left')

    print("Storing ...")
    dataset_assigment.to_csv(os.path.join(path_storage, 'Resultados_CIGRE.csv'), index=False)
 
    print("Done")

    return None

## Part 2 ## 
### CHIRPS V.2.0 ######  

def chirps_assigment(dataset:pd.DataFrame ):

    slice_tuple = None                
    cols_dict_db = None
    assigment_nowcast = None
    
    path_database = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\4_Rain\03_ChirpsRainFall_SouthAmericaV2"
    print("Working in CHIRPS nowcast assigment [...]\n")

    chirps_txt_paths = os.listdir(path_database)

    total_files = len(chirps_txt_paths) 
    chunk_progress = total_files // 6

    coords_raw = {
                'vector_lon':None,
                 'vector_lat':None
                }
    arrays = [ ]
    numba_vector_time = [ ]
    coords_align = True
    
    for idx, nc in enumerate(chirps_txt_paths ,start = 1):

        complete_path = os.path.join(path_database , nc)
        chunk_nc_now = xr.open_dataset(complete_path)

        if idx == 1:

            (slice_tuple, cols_dict_db, dataset_assigment  ,dim_mapping  ) = tools.extract_slice_tools( nc_file = chunk_nc_now,
                                                                                                         dataset = dataset,
                                                                                                         database_name = 'chirps',
                                                                                                         column_isa = ['Longitud','Latitud'], 
                                                                                                         variable_name = 'precip',
                                                                                                         dim_bool = True)


        
        if idx % chunk_progress == 0:
            print(f"{(idx / total_files) * 100:.3f} [%] progress")

        numba_slice = chunk_nc_now.precip.to_numpy()
        raw_vector_time  = chunk_nc_now.time.values

        numba_vector_time.append(raw_vector_time)
        slice_i = numba_slice[slice_tuple].astype(np.float32)

        arrays.append(slice_i)

        ### Verificar si las cordenadas estan alineadas.

        vector_lon = chunk_nc_now.longitude.values
        vector_lat = chunk_nc_now.latitude.values
  
        if idx == 1:
            coords_raw['vector_lon'] = vector_lon
            coords_raw['vector_lat'] = vector_lat
            
        else:
            equal_lon = (vector_lon == coords_raw['vector_lon']).all()
            equal_lat = (vector_lat == coords_raw['vector_lat']).all()

            are_equal = equal_lon and equal_lat

            if are_equal:
                continue
            else:
                print("Corrds misalign")
                return None

    # Stack the vectors vertically
    stacked_array = np.vstack(arrays)
    dict_matrix_nowcast = pd.DataFrame(stacked_array , index = np.concat(numba_vector_time), columns = cols_dict_db)   
    complete_dates = has_all_days(date_series= dict_matrix_nowcast.index, non_leap_year=False)

    if coords_align:
        print("Coords are in aligment.")
    
    if complete_dates:
        
        print("Done\n")
        return (dataset_assigment, dict_matrix_nowcast)
   
    else:
        return None



def add_precipitation_features(landslide_df, precipitation_df, window_sizes=[3, 7, 15]):
    """
    Pipeline to add precipitation features to landslide dataframe
    Computes static summaries of precipitation patterns for each grid cell
    
    Parameters:
    landslide_df: DataFrame with columns ['ID', ...] where each row is a landslide
    precipitation_df: DataFrame with dates as index and grid cell IDs as columns
    window_sizes: List of rolling window sizes in days
    
    Returns:
    DataFrame with added precipitation features:
    - precip_rollsum_mean_Xd: Mean of rolling sums over entire time series
    - precip_rollmean_mean_Xd: Mean of rolling means over entire time series  
    - precip_rollmax_mean_Xd: Mean of rolling maximums over entire time series
    """
    
    # Create a copy to avoid modifying original dataframe
    result_df = landslide_df.copy()
    
    # Process window by window to minimize memory usage
    print("Processing windows one by one to minimize memory usage...")
    
    for window in window_sizes:
        print(f"  Processing {window}-day rolling statistics...")
        
        # Initialize new columns for this window
        result_df[f'precip_rollsum_mean_{window}d'] = np.nan
        result_df[f'precip_rollmean_mean_{window}d'] = np.nan  
        result_df[f'precip_rollmax_mean_{window}d'] = np.nan
        
        # Calculate rolling statistics for this window only
        print(f"    Calculating rolling statistics...")
        rolling_sum = precipitation_df.rolling(window=window).sum()
        rolling_mean = precipitation_df.rolling(window=window).mean()
        rolling_max = precipitation_df.rolling(window=window).max()
        
        # Compute static summaries (mean over entire time series)
        print(f"    Computing static summaries...")
        rollsum_mean = rolling_sum.mean()    # Mean of rolling sums for each grid cell
        rollmean_mean = rolling_mean.mean()  # Mean of rolling means for each grid cell
        rollmax_mean = rolling_max.mean()    # Mean of rolling maxs for each grid cell
        
        # Map summaries to landslides using groupby for efficiency
        print(f"    Mapping to {len(result_df)} landslides across {result_df['chirps'].nunique()} unique grid cells...")
        
        for grid_cell, group_indices in result_df.groupby('chirps').groups.items():
            
            # Assign same static summaries to all landslides in this grid cell
            result_df.loc[group_indices, f'precip_rollsum_mean_{window}d'] = rollsum_mean[grid_cell]
            result_df.loc[group_indices, f'precip_rollmean_mean_{window}d'] = rollmean_mean[grid_cell]
            result_df.loc[group_indices, f'precip_rollmax_mean_{window}d'] = rollmax_mean[grid_cell]
        
        # Clear memory after processing each window
        del rolling_sum, rolling_mean, rolling_max, rollsum_mean, rollmean_mean, rollmax_mean
        print(f"    Completed {window}-day window and cleared memory")
    
    print(f"Added {len(window_sizes) * 3} precipitation features!")
    return result_df




## def longest_streak_above_threshold(data, threshold=1.0):
##     """
##     CDW - Consecutive Wet Days
##     Calculate the longest streak of days above threshold
##     """
##     above_threshold = data > threshold
##     indices = np.where(above_threshold)[0]
##     
##     if len(indices) == 0:
##         return 0
##     
##     diff = np.diff(indices)
##     streaks = np.split(indices, np.where(diff > 1)[0] + 1)
##     streak_lengths = [len(streak) for streak in streaks]
##     
##     return max(streak_lengths)
## 
## def rolling_window_max_precipitation(data, window_size=5):
##     """
##     RX1day - Maximum cumulative precipitation in rolling window
##     """
##     data = np.array(data)
##     rolling_sum = np.convolve(data, np.ones(window_size, dtype=int), 'valid')
##     return rolling_sum.max()
## 
## def count_days_above_percentile(data, percentile=95):
##     """
##     R95p - Total precipitation from days above 95th percentile
##     """
##     # Calculate the threshold value for the given percentile
##     threshold = np.percentile(data, percentile)
##     # Sum the precipitation on days above the threshold
##     count = np.sum(data[data >= threshold])
##     
##     return count

def compute_precipitation_indices(landslide_df, precipitation_df, 
                                threshold=1.0, window_size=5, percentile=95, 
                                quantile_value=0.9):
    """
    Compute precipitation indices for landslide locations based on CHIRPS data
    
    Parameters:
    -----------
    landslide_df : pd.DataFrame
        DataFrame with landslide events, must have 'id' column with grid cell IDs
    precipitation_df : pd.DataFrame  
        DataFrame with dates as index and grid cell IDs as columns (CHIRPS data)
    threshold : float
        Threshold for wet days (mm/day), default 1.0
    window_size : int
        Rolling window size for RX1day, default 5
    percentile : float
        Percentile threshold for R95p, default 95
    quantile_value : float
        Quantile to use when aggregating across years, default 0.9
        
    Returns:
    --------
    pd.DataFrame
        Landslide dataframe with added precipitation index column
    """
    
    print("Computing precipitation indices for landslide locations...")
    
    # Create copy to avoid modifying original
    result_df = landslide_df.copy()
    
    # Get unique grid cells with landslides
    unique_grid_cells = result_df['chirps'].unique()
    print(f"Processing {len(unique_grid_cells)} unique grid cells...")
    
    # Dictionary to store computed indices for each grid cell
    grid_cell_indices = {}
    
    for i, grid_cell in enumerate(unique_grid_cells, 1):
        # Progress indicator every 10% of cells
        if i % max(1, len(unique_grid_cells) // 10) == 0:
            print(f"  Progress: {i}/{len(unique_grid_cells)} cells processed ({100*i/len(unique_grid_cells):.0f}%)")
                    
        # Extract precipitation time series for this grid cell
        precip_series = precipitation_df[grid_cell]
        
        # Group by year to compute indices annually
        precip_by_year = precip_series.groupby(precip_series.index.year)
        
        # Store yearly indices
        yearly_cdw = []
        yearly_rx1day = []
        yearly_r95p = []
        
        for year, year_data in precip_by_year:
            # Convert to numpy array for processing
            year_precip = year_data.values
            
            # Skip years with insufficient data
            if len(year_precip) < 300:  # At least ~10 months of data
                continue
                
            # Compute the 3 indices for this year
            cdw = longest_streak_above_threshold(year_precip, threshold)
            rx1day = rolling_window_max_precipitation(year_precip, window_size)
            r95p = count_days_above_percentile(year_precip, percentile)
            
            yearly_cdw.append(cdw)
            yearly_rx1day.append(rx1day)
            yearly_r95p.append(r95p)
        
        if len(yearly_cdw) == 0:
            continue
            
        # Aggregate across years using quantile (following original code logic)
        cdw_agg = np.quantile(yearly_cdw, quantile_value)
        rx1day_agg = np.quantile(yearly_rx1day, quantile_value)
        r95p_agg = np.quantile(yearly_r95p, quantile_value)
        
        # Compute final aggregated index: (R95p + CDW + RX1day) / 3
        aggregated_index = (r95p_agg + cdw_agg + rx1day_agg) / 3
        
        # Store results
        grid_cell_indices[grid_cell] = {
            'cdw': cdw_agg,
            'rx1day': rx1day_agg, 
            'r95p': r95p_agg,
            'aggregated_index': aggregated_index
        }
    
    # Add indices to landslide dataframe
    print("\nMapping indices to landslide dataframe...")
    
    # Initialize new columns
    result_df['precip_cdw_index'] = np.nan
    result_df['precip_rx1day_index'] = np.nan
    result_df['precip_r95p_index'] = np.nan
    result_df['precip_aggregated_index'] = np.nan
    
    # Map indices using groupby for efficiency
    for grid_cell, group_indices in result_df.groupby('chirps').groups.items():
        if grid_cell in grid_cell_indices:
            indices = grid_cell_indices[grid_cell]
            
            result_df.loc[group_indices, 'precip_cdw_index'] = indices['cdw']
            result_df.loc[group_indices, 'precip_rx1day_index'] = indices['rx1day']
            result_df.loc[group_indices, 'precip_r95p_index'] = indices['r95p']
            result_df.loc[group_indices, 'precip_aggregated_index'] = indices['aggregated_index']
    
    print(f"Added precipitation indices to {len(result_df)} landslides")
    
    return result_df


def rain_pipe_chirps(dataset:pd.DataFrame):
    
    (dataset_assigment, dict_matrix_nowcast) = chirps_assigment(dataset)
    dataset_assigment = add_precipitation_features(dataset_assigment, dict_matrix_nowcast, window_sizes=[3, 7, 15])
    dataset_assigment =  compute_precipitation_indices(dataset_assigment,
                                                       dict_matrix_nowcast, 
                                                       threshold=1.0, 
                                                       window_size=5, 
                                                       percentile=95, 
                                                       quantile_value=0.9)


    return dataset_assigment


    




