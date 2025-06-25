import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import numpy as np
import os
from collections import OrderedDict
import json
import xarray as xr
import pandas as pd
import re
import time
import sys
import cftime
from datetime import datetime, timedelta
import rioxarray as rio
import rasterio
import logging

# Define paths
path_tools = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\utils"

if path_tools not in sys.path:
    sys.path.append(path_tools)

# Import your modules
import tools
from terrain_assigment import calcular_terreno_puntos

def longest_streak_above_threshold(data, threshold):
    """CDW - Consecutive Wet Days"""
    above_threshold = data > threshold
    indices = np.where(above_threshold)[0]
    
    if len(indices) == 0:
        return 0
    
    diff = np.diff(indices)
    streaks = np.split(indices, np.where(diff > 1)[0] + 1)
    streak_lengths = [len(streak) for streak in streaks]
    
    return max(streak_lengths)

def rolling_window_max_precipitation(data, window_size):
    """RX1day - Maximum precipitation in rolling window"""
    data = np.array(data)
    rolling_sum = np.convolve(data, np.ones(window_size, dtype=int), 'valid')
    return rolling_sum.max()

def count_days_above_percentile(data, percentile):
    """R95p - Count days above percentile"""
    threshold = np.percentile(data, percentile)
    count = np.sum(data[data >= threshold])
    return count

def compute_precipitation_statistics(stack_array):
    """
    Compute multiple statistics for precipitation data
    
    Parameters:
    -----------
    stack_array : np.array
        3D array with shape (years, lat, lon)
        
    Returns:
    --------
    dict : Dictionary with different statistics
    """
    stats = {}
    
    # Mean (existing)
    stats['mean'] = np.mean(stack_array, axis=0)
    
    # Median
    stats['median'] = np.median(stack_array, axis=0)
    
    # Maximum
    stats['max'] = np.max(stack_array, axis=0)
    
    # 95th percentile
    stats['p95'] = np.percentile(stack_array, 95, axis=0)
    
    # Minimum
    stats['min'] = np.min(stack_array, axis=0)
    
    return stats

def create_climate_geotiffs_from_dict(mean_var_raster, 
                                     lon_coords, 
                                     lat_coords, 
                                     affine_transform,
                                     output_directory,
                                     create_single_file=False):
    """
    Convert nested climate dictionary to GeoTIFF files with proper projection.
    Enhanced to handle multiple statistics for precipitation.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Enhanced index names mapping to include precipitation statistics
    index_mapping = {
        'total_precipitation_yearly_mean': 'TP_mean',
        'total_precipitation_yearly_median': 'TP_median', 
        'total_precipitation_yearly_max': 'TP_max',
        'total_precipitation_yearly_p95': 'TP_p95',
        'total_precipitation_yearly_min': 'TP_min',
        'longest_streak_result': 'CWD',
        'rolling_window_result': 'RX5day',
        'count_days_result': 'R95p'
    }
    
    # Set up basic rasterio profile
    base_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999,
        'width': len(lon_coords),
        'height': len(lat_coords),
        'crs': CRS.from_epsg(4326),
        'transform': affine_transform,
        'compress': 'lzw',
        'tiled': True
    }
    
    created_files = {}
    
    if create_single_file:
        # Create one large multi-band file
        all_bands_data, band_descriptions = _prepare_all_bands_data(mean_var_raster, index_mapping)
        
        profile = base_profile.copy()
        profile['count'] = len(all_bands_data)
        
        output_path = os.path.join(output_directory, 'climate_indices_all_scenarios_enhanced.tif')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, (band_data, description) in enumerate(zip(all_bands_data, band_descriptions), 1):
                dst.write(band_data, i)
                dst.set_band_description(i, description)
        
        # Save enhanced metadata as JSON
        metadata_path = os.path.join(output_directory, 'climate_indices_metadata_enhanced.json')
        metadata = {
            'band_descriptions': band_descriptions,
            'coordinate_info': {
                'lon_range': [float(lon_coords.min()), float(lon_coords.max())],
                'lat_range': [float(lat_coords.min()), float(lat_coords.max())],
                'shape': [len(lat_coords), len(lon_coords)]
            },
            'statistics_included': {
                'precipitation': ['mean', 'median', 'max', 'p95', 'min'],
                'indices': ['CWD', 'RX5day', 'R95p']
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        created_files['all_scenarios_enhanced'] = {
            'file_path': output_path,
            'metadata_path': metadata_path,
            'band_count': len(all_bands_data)
        }
        
    else:
        # Create separate files for each scenario and index
        for scenario, scenario_data in mean_var_raster.items():
            scenario_clean = scenario.replace(' ', '_').replace('.', '_')
            
            for index_key, index_data in scenario_data.items():
                index_name = index_mapping.get(index_key, index_key)
                
                # Collect all time periods for this scenario-index combination
                time_periods = list(index_data.keys())
                bands_data = []
                band_descriptions = []
                
                for time_period in time_periods:
                    model_arrays = index_data[time_period]
                    # Ensemble mean across models
                    ensemble_mean = np.mean(np.stack(model_arrays, axis=0), axis=0)
                    bands_data.append(ensemble_mean.astype('float32'))
                    band_descriptions.append(f"{scenario}_{index_name}_{time_period}")
                
                # Create multi-band GeoTIFF for this scenario-index
                profile = base_profile.copy()
                profile['count'] = len(bands_data)
                
                filename = f"{scenario_clean}_{index_name}.tif"
                output_path = os.path.join(output_directory, filename)
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for i, (band_data, description) in enumerate(zip(bands_data, band_descriptions), 1):
                        dst.write(band_data, i)
                        dst.set_band_description(i, description)
                    
                    # Add custom metadata
                    dst.update_tags(
                        scenario=scenario,
                        climate_index=index_name,
                        time_periods=','.join(time_periods),
                        coordinate_system='EPSG:4326',
                        data_type='climate_ensemble_mean'
                    )
                
                created_files[f"{scenario_clean}_{index_name}"] = {
                    'file_path': output_path,
                    'scenario': scenario,
                    'index': index_name,
                    'time_periods': time_periods,
                    'band_count': len(bands_data)
                }
    
    return created_files

def _prepare_all_bands_data(mean_var_raster, index_mapping):
    """Helper function to prepare all bands data for single file creation."""
    all_bands_data = []
    band_descriptions = []
    
    for scenario, scenario_data in mean_var_raster.items():
        for index_key, index_data in scenario_data.items():
            index_name = index_mapping.get(index_key, index_key)

            for time_period, model_arrays in index_data.items():
                ensemble_mean = np.mean(np.stack(model_arrays, axis=0), axis=0)
                all_bands_data.append(ensemble_mean.astype('float32'))
                band_descriptions.append(f"{scenario}_{index_name}_{time_period}")

    return all_bands_data, band_descriptions

def has_all_days(date_series: np.array, non_leap_year=True):
    """
    Verify that after concatenating historical and future data,
    we have all values except for those of leap years.
    """
    date_series_pd = pd.to_datetime([date.strftime('%Y-%m-%d') if isinstance(date, cftime.datetime) else date for date in date_series])
    
    full_range = pd.date_range(start=date_series_pd.min(), end=date_series_pd.max(), freq='D')

    if non_leap_year:
        leap_day_mask = ~((full_range.month == 2) & (full_range.day == 29))
        full_range = full_range[leap_day_mask]

    is_equal = full_range.isin(date_series_pd).all()

    if is_equal:
        print("Dates complete")
    else:
        print("Dates Incomplete")

    return is_equal

def common_clip(path_name, bbox, lon_name, lat_name):
    """Clip dataset to bounding box"""
    if isinstance(path_name, str):
        xarray_file = xr.open_dataset(path_name, chunks={'time':'auto'})
    else:
        xarray_file = path_name
        
    xarray_file = xarray_file.sortby(variables=['lat', 'lon'])
    slice_array = xarray_file.sel({
        lon_name: slice(bbox['lon_min'], bbox['lon_max']),
        lat_name: slice(bbox['lat_min'], bbox['lat_max'])
    })

    return slice_array

def create_df_modelos(url_nasa: str, assets: pd.DataFrame, bbox=None):
    """
    Create dataframe with models, scenarios, and time objects
    """
    coords_raw = {
        'vector_lon': None,
        'vector_lat': None
    }

    dataset_assigment = assets.copy()
    
    re_nasa_pattern = r"SAM_pr_(?P<Modelos>.+?)_(?P<Escenarios>historical|ssp\d+)_(?P<Fold>\d+).nc$"
    modelos_regex = re.compile(re_nasa_pattern)
    np_arr = np.array([var for var in os.listdir(url_nasa) if modelos_regex.match(var)])
    modelos = pd.Series(np_arr).str.extract(modelos_regex).copy()
    mod_esc = modelos.groupby('Modelos')

    extracted_coords = False
    first_coord = True
    bad_ones = False
    database_name = 'idx_affine'

    for modelo, df in mod_esc:
        raw_idx = df.index
        files_txt = np_arr[raw_idx]       
        path_name = os.path.join(url_nasa, files_txt[0])
        xarray_file = xr.open_dataset(path_name, chunks={'time':'auto'})

        if not extracted_coords:
            lon_name = tools.get_coord_name(xarray_file, 'lon')
            lat_name = tools.get_coord_name(xarray_file, 'lat')

            catche_affine = common_clip(path_name=xarray_file,
                                       bbox=bbox,
                                       lon_name=lon_name,
                                       lat_name=lat_name)

            slice_lon = catche_affine.lon.values
            slice_lat = catche_affine.lat.values
    
            in_memory = catche_affine.compute()
            affine = in_memory.rio.transform()
            
            rows, cols = rasterio.transform.rowcol(affine, dataset_assigment.get('Longitud_360').values, dataset_assigment.Latitud.values)
            idx_affine = [f"{row}_{col}" for row, col in zip(rows, cols)]
            
            dataset_assigment.loc[:, database_name] = idx_affine
            dataset_assigment.loc[:, 'row'] = rows
            dataset_assigment.loc[:, 'col'] = cols

            no_duplicates = dataset_assigment.drop_duplicates(subset=database_name).copy()
            no_duplicates.loc[:, ['lon', 'lat']] = no_duplicates.get(database_name).str.extract(r"(?P<lon>\d+)_(?P<lat>\d+)").astype(int)

            cols_dict_db = list(no_duplicates.get(database_name).values)
            rows_unique, cols_unique = no_duplicates.row.values, no_duplicates.col.values
    
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
                print("Coords misalign")
                bad_ones = True
                return None

    np_arr = np.array([os.path.join(url_nasa, name_file) for name_file in np_arr])

    coords_dict = {
        'row': rows_unique,
        'col': cols_unique,
        'cols_dict_db': cols_dict_db,
        'dataset_assigment': dataset_assigment,
        'affine': affine,
        'lon': slice_lon,
        'lat': slice_lat,
        'lon_name': lon_name,
        'lat_name': lat_name
    }
    
    if not bad_ones:
        print("All share the same coords")
        return (modelos, np_arr, coords_dict)
    else:
        return None

def ensamble_folds(bbox=None, list_names=None, non_leap_year=True, lon_name=str, lat_name=str):
    """Ensemble multiple model folds"""
    cast_xarray = []

    for idx, file_name in enumerate(list_names, start=1):
        current_xr = common_clip(file_name, bbox, lon_name, lat_name)        
        cast_xarray.append(current_xr)
        
    coords_time = [raw_mod.time.values for raw_mod in cast_xarray]
    time_vector = np.concatenate(coords_time)
    time_complete = has_all_days(time_vector, non_leap_year)
    
    if time_complete:
        to_ensamble = xr.concat(cast_xarray, dim='time')
        return to_ensamble
    else:
        return False

def main(url_nsa: str, assets: pd.DataFrame, output_directory: str) -> None:
    """
    Calculate precipitation indices for historical and future scenarios.
    Enhanced version with multiple precipitation statistics.
    
    Parameters
    ----------
    url_nsa : str
        Path to directory with climate data files
    assets : pd.DataFrame
        DataFrame with coordinates of assets/points to analyze
    output_directory : str
        Path where results will be saved
        
    Returns
    -------
    tuple
        Created files info and dataset assignment
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    holgura = 1
    
    lon_min = tools.map_to_0_to_360(tools.boundaries['Colombia']['lon_min'] - holgura) 
    lon_max = tools.map_to_0_to_360(tools.boundaries['Colombia']['lon_max'] + holgura)
    lat_min = tools.boundaries['Colombia']['lat_min'] - holgura
    lat_max = tools.boundaries['Colombia']['lat_max'] + holgura

    assets['Longitud_360'] = tools.map_to_0_to_360(assets['Longitud'])

    slice_crops = {
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_min': lat_min,
        'lat_max': lat_max
    }

    logger = logging.getLogger(__name__)
    
    # Constants
    PRECIPITATION_CONVERSION_FACTOR = 86_400
    THRESHOLD_WET_DAY = 1.0
    WINDOW_SIZE = 5
    PERCENTILE_THRESHOLD = 95
    QUANTILE_VALUE = 0.9

    # Create base dataframe with models and scenarios
    logger.info("Creating models and scenarios dataframe")

    modelos_result = create_df_modelos(url_nasa=url_nsa, 
                                      assets=assets,
                                      bbox=slice_crops)
        
    modelos, np_arr, coords_dict = modelos_result

    # Extract spatial information
    col = coords_dict['col']
    row = coords_dict['row']
    lon_name = coords_dict['lon_name']
    lat_name = coords_dict['lat_name']
    lon_coords = coords_dict['lon']
    lat_coords = coords_dict['lat']
    affine = coords_dict['affine']
    cols_dict_db = coords_dict['cols_dict_db']    
    dataset_assigment = coords_dict['dataset_assigment']

    logger.info("Applying NASA susceptibility to assets")

    # Define analysis periods
    HISTORICAL_ID = 'historical'
    historical_range = "1985-2005"
    historical_start_year = 1985
    historical_end_year = 2004
    
    # Scenario mapping
    maper_escenarios = {
        'historical': 'Historical', 
        'ssp245': 'SSP2 4.5', 
        'ssp370': 'SSP3 7.0',
        'ssp585': 'SSP5 8.5'
    }
    
    # Group by scenarios and models
    dtype_ssp_grouper = modelos.groupby(['Escenarios', 'Modelos', 'time_type'])
    
    # Future periods of 20 years
    time_ranges = [(2020, 2039), (2040, 2059), (2060, 2079), (2080, 2100)]
    time_ranges_str = ['2020-2040', '2040-2060', '2060-2080', '2080-2100']

    # Dictionary to store results by scenario
    dict_escenarios = {}
    
    # Enhanced reference template to include all precipitation statistics
    reference_temp_dict = {
                            'total_precipitation_yearly_mean': [],
                            'total_precipitation_yearly_median': [],
                            'total_precipitation_yearly_max': [],
                            'total_precipitation_yearly_p95': [],
                            'total_precipitation_yearly_min': [],
                            'longest_streak_result': [],
                            'rolling_window_result': [],
                            'count_days_result': []
                        }
    
    for (escenario, modelo, dtype), df_type_escenario in dtype_ssp_grouper:
        
        if dtype == 'datetime64[ns]':
            non_leap_year = False
        else:
            non_leap_year = True

        escenario_format_name = maper_escenarios[escenario]
        folder_escenario = dict_escenarios.setdefault(escenario_format_name, {})

        # Create dictionaries for all precipitation statistics
        precip_stats_dicts = {}
        for stat_name in ['mean', 'median', 'max', 'p95', 'min']:
            precip_stats_dicts[f'total_precipitation_yearly_{stat_name}'] = folder_escenario.setdefault(f'total_precipitation_yearly_{stat_name}', {})
        
        # Existing indices dictionaries
        longest_streak_dict = folder_escenario.setdefault('longest_streak_result', {})
        rolling_window_dict = folder_escenario.setdefault('rolling_window_result', {})
        count_days_dict = folder_escenario.setdefault('count_days_result', {})
        
        print(f"Working on {escenario_format_name}")
        print(f"\tModelo : {modelo}")
        
        index_ssp = df_type_escenario.sort_values(by='Fold', ascending=True).index
        txt_name_ssp = np_arr[index_ssp]

        for enumerador, name_fold in enumerate(txt_name_ssp, start=1):
            print(f"\t{enumerador} - {name_fold}")

        ds_ssp = ensamble_folds(slice_crops, txt_name_ssp, non_leap_year, lon_name, lat_name)

        if ds_ssp is False:
            print("Failed to ensemble folds")
            continue
        
        # Process based on scenario type
        if escenario == 'historical':
            # HISTORICAL PROCESSING
            print(f'\t\t{historical_start_year}-01-01 {historical_end_year}-12-31')

            # Get lists for all statistics
            precip_stats_lists = {}
            for stat_name in ['mean', 'median', 'max', 'p95', 'min']:
                precip_stats_lists[f'total_precipitation_yearly_{stat_name}'] = precip_stats_dicts[f'total_precipitation_yearly_{stat_name}'].setdefault(historical_range, [])
            
            longest_streak_list = longest_streak_dict.setdefault(historical_range, [])
            rolling_window_list = rolling_window_dict.setdefault(historical_range, [])
            count_days_list = count_days_dict.setdefault(historical_range, [])
            
            # Select data for historical time range
            data_range = ds_ssp.sel(time=slice(f'{historical_start_year}-01-01', f'{historical_end_year}-12-31'))
            chunk_future = data_range.compute()
            
            group = chunk_future.pr.resample(time='YE')

            # Temporary statistics storage - only for basic calculations
            stat_temp = {
                'total_precipitation_yearly': [],
                'longest_streak_result': [],
                'rolling_window_result': [],
                'count_days_result': []
            }
            
            for idx, (rango, slice_numpy) in enumerate(group.groups.items(), start=1):
                numba_raw = chunk_future.pr.values[slice_numpy] * PRECIPITATION_CONVERSION_FACTOR

                # Apply custom functions
                sum_total_year = np.sum(numba_raw, axis=0)
                longest_streak_computation = np.apply_along_axis(longest_streak_above_threshold, axis=0, arr=numba_raw, threshold=THRESHOLD_WET_DAY)
                rolling_window_computation = np.apply_along_axis(rolling_window_max_precipitation, axis=0, arr=numba_raw, window_size=WINDOW_SIZE)
                count_days_computation = np.apply_along_axis(count_days_above_percentile, axis=0, arr=numba_raw, percentile=PERCENTILE_THRESHOLD)
                
                stat_temp['total_precipitation_yearly'].append(sum_total_year)
                stat_temp['longest_streak_result'].append(longest_streak_computation)
                stat_temp['rolling_window_result'].append(rolling_window_computation)
                stat_temp['count_days_result'].append(count_days_computation)

            # Final processing for historical period
            for variable, list_numba in stat_temp.items():
                stack_array = np.stack(list_numba, axis=0)
                
                if variable == 'total_precipitation_yearly':
                    # Compute all precipitation statistics
                    precip_stats = compute_precipitation_statistics(stack_array)
                    for stat_name, stat_data in precip_stats.items():
                        precip_stats_lists[f'total_precipitation_yearly_{stat_name}'].append(stat_data)
                        
                elif variable == 'longest_streak_result':
                    mean_var_raster = np.mean(stack_array, axis=0)
                    longest_streak_list.append(mean_var_raster)
                elif variable == 'rolling_window_result':
                    mean_var_raster = np.mean(stack_array, axis=0)
                    rolling_window_list.append(mean_var_raster)
                elif variable == 'count_days_result':
                    mean_var_raster = np.mean(stack_array, axis=0)
                    count_days_list.append(mean_var_raster)
                    
            print("Done Historical")
            
        else:
            # FUTURE SCENARIOS PROCESSING
            for (idx, ((start_year, end_year), horizon_code)) in enumerate(zip(time_ranges, time_ranges_str), start=1):
                
                print(f'\t\t{start_year}-01-01 {end_year}-12-31')

                # Get lists for all statistics
                precip_stats_lists = {}
                for stat_name in ['mean', 'median', 'max', 'p95', 'min']:
                    precip_stats_lists[f'total_precipitation_yearly_{stat_name}'] = precip_stats_dicts[f'total_precipitation_yearly_{stat_name}'].setdefault(horizon_code, [])
                
                longest_streak_list = longest_streak_dict.setdefault(horizon_code, [])
                rolling_window_list = rolling_window_dict.setdefault(horizon_code, [])
                count_days_list = count_days_dict.setdefault(horizon_code, [])
                
                # Select data for time range
                data_range = ds_ssp.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
                chunk_future = data_range.compute()

                group = chunk_future.pr.resample(time='YE')

                # Temporary statistics storage
                stat_temp = {
                    'total_precipitation_yearly': [],
                    'longest_streak_result': [],
                    'rolling_window_result': [],
                    'count_days_result': []
                }

                for idx, (rango, slice_numpy) in enumerate(group.groups.items(), start=1):
                    numba_raw = chunk_future.pr.values[slice_numpy] * PRECIPITATION_CONVERSION_FACTOR
                    
                    # Apply custom functions
                    sum_total_year = np.sum(numba_raw, axis=0)
                    longest_streak_computation = np.apply_along_axis(longest_streak_above_threshold, axis=0, arr=numba_raw, threshold=THRESHOLD_WET_DAY)
                    rolling_window_computation = np.apply_along_axis(rolling_window_max_precipitation, axis=0, arr=numba_raw, window_size=WINDOW_SIZE)
                    count_days_computation = np.apply_along_axis(count_days_above_percentile, axis=0, arr=numba_raw, percentile=PERCENTILE_THRESHOLD)

                    stat_temp['total_precipitation_yearly'].append(sum_total_year)
                    stat_temp['longest_streak_result'].append(longest_streak_computation)
                    stat_temp['rolling_window_result'].append(rolling_window_computation)
                    stat_temp['count_days_result'].append(count_days_computation)
                
                for variable, list_numba in stat_temp.items():
                    stack_array = np.stack(list_numba, axis=0)
                    
                    if variable == 'total_precipitation_yearly':
                        # Compute all precipitation statistics
                        precip_stats = compute_precipitation_statistics(stack_array)
                        for stat_name, stat_data in precip_stats.items():
                            precip_stats_lists[f'total_precipitation_yearly_{stat_name}'].append(stat_data)
                            
                    elif variable == 'longest_streak_result':
                        mean_var_raster = np.mean(stack_array, axis=0)
                        longest_streak_list.append(mean_var_raster)
                    elif variable == 'rolling_window_result':
                        mean_var_raster = np.mean(stack_array, axis=0)
                        rolling_window_list.append(mean_var_raster)
                    elif variable == 'count_days_result':
                        mean_var_raster = np.mean(stack_array, axis=0)
                        count_days_list.append(mean_var_raster)

                print("Done")

    # Create enhanced GeoTIFFs with all statistics
    out = create_climate_geotiffs_from_dict(dict_escenarios, 
                                           lon_coords, 
                                           lat_coords, 
                                           affine,
                                           output_directory,
                                           create_single_file=True)
 
    print("Processing complete - Enhanced GeoTIFFs created with multiple precipitation statistics")
    print(f"Total bands created: {out['all_scenarios_enhanced']['band_count']}")

    return out, dataset_assigment