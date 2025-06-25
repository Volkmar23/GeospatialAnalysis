import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import numpy as np
import os
from collections import OrderedDict
import json

def create_climate_geotiffs_from_dict(mean_var_raster, 
                                     lon_coords, 
                                     lat_coords, 
                                     affine_transform,
                                     output_directory,
                                     create_single_file=False):
    """
    Convert nested climate dictionary to GeoTIFF files with proper projection.
    
    Parameters:
    -----------
    mean_var_raster : dict
        Nested dictionary with structure: scenario -> index -> time_range -> list of arrays
    lon_coords : np.array
        Longitude coordinate vector from original NetCDF
    lat_coords : np.array  
        Latitude coordinate vector from original NetCDF
    affine_transform : rasterio.Affine
        Affine transformation from the NetCDF file
    output_directory : str
        Directory to save GeoTIFF files
    create_single_file : bool
        If True, creates one multi-band file. If False, creates separate files.
    
    Returns:
    --------
    dict : Dictionary with created file paths and band information
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Define index names mapping
    index_mapping = {
                    'total_precipitaton_yearly' : 'TP',
                    'longest_streak_result': 'CWD',  # Consecutive Wet Days
                    'rolling_window_result': 'RX5day',  # Max 5-day precipitation  
                    'count_days_result': 'R95p'  # Extreme precipitation days
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
        
        output_path = os.path.join(output_directory, 'climate_indices_all_scenarios.tif')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, (band_data, description) in enumerate(zip(all_bands_data, band_descriptions), 1):
                dst.write(band_data, i)
                dst.set_band_description(i, description)
        
        # Save metadata as JSON
        metadata_path = os.path.join(output_directory, 'climate_indices_metadata.json')
        metadata = {
                    'band_descriptions': band_descriptions,
                    'coordinate_info': {
                        'lon_range': [float(lon_coords.min()), float(lon_coords.max())],
                        'lat_range': [float(lat_coords.min()), float(lat_coords.max())],
                        'shape': [len(lat_coords), len(lon_coords)]
                    }
                }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        created_files['all_scenarios'] = {
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


# Integration with your existing pipeline
def integrate_with_your_pipeline(mean_var_raster, current_xr, output_directory):
    """
    Drop-in function to integrate with your existing code.
    
    Add this at the end of your main() function:
    """
    
    # Extract spatial information
    lon_coords = current_xr.lon.values
    lat_coords = current_xr.lat.values
    affine = current_xr.rio.transform()
    
    # Create GeoTIFFs
    created_files = create_climate_geotiffs_from_dict(
                                        mean_var_raster=mean_var_raster,
                                        lon_coords=lon_coords,
                                        lat_coords=lat_coords,
                                        affine_transform=affine,
                                        output_directory=output_directory,
                                        create_single_file=False  # Creates separate files per scenario-index
                                    )
    
    print(f"Created {len(created_files)} GeoTIFF files:")
    for name, info in created_files.items():
        print(f"  {name}: {info['band_count']} bands")
    
    return created_files