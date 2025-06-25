import rasterio
import rasterio.mask
import xarray as xr
import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import mapping
import warnings

def aggregate_tiff_to_watersheds(tiff_path, watersheds_gdf, stats=['mean', 'min', 'max'], 
                                column_prefix='feature', nodata_threshold=0.5):
    """
    Aggregate TIFF raster values to watershed polygons.
    
    Parameters:
    -----------
    tiff_path : str
        Path to the TIFF file
    watersheds_gdf : GeoDataFrame
        Watershed polygons (must have geometry column)
    stats : list
        Statistics to calculate ['mean', 'min', 'max', 'std', 'median', 'count']
    column_prefix : str
        Prefix for the output columns (e.g., 'elevation', 'slope')
    nodata_threshold : float
        Minimum fraction of valid pixels required (0.0 to 1.0)
    
    Returns:
    --------
    GeoDataFrame
        Original watersheds with new aggregated columns
    """
    
    print(f"Processing TIFF: {tiff_path}")
    
    # Open the raster
    with rasterio.open(tiff_path) as src:
        print(f"  - Raster CRS: {src.crs}")
        print(f"  - Raster shape: {src.shape}")
        print(f"  - Raster bounds: {src.bounds}")
        
        # Check if CRS match, if not reproject watersheds
        if watersheds_gdf.crs != src.crs:
            print(f"  - Reprojecting watersheds from {watersheds_gdf.crs} to {src.crs}")
            watersheds_proj = watersheds_gdf.to_crs(src.crs)
        else:
            watersheds_proj = watersheds_gdf.copy()
        
        # Initialize result columns
        results = {}
        for stat in stats:
            results[f"{column_prefix}_{stat}"] = np.full(len(watersheds_proj), np.nan)
        results[f"{column_prefix}_valid_pixels"] = np.full(len(watersheds_proj), 0)
        
        # Process each watershed
        for idx, row in watersheds_proj.iterrows():
            try:
                # Get geometry for masking
                geom = [mapping(row.geometry)]
                
                # Mask raster with watershed boundary
                masked_array, masked_transform = rasterio.mask.mask(
                                        src, geom, crop=True, nodata=src.nodata, filled=False
                                    )


                return (geom,masked_array,masked_transform)
                
                # Get valid (non-masked) pixels
                valid_data = masked_array[0][~masked_array[0].mask]
                
                # Remove nodata values if they exist
                if src.nodata is not None:
                    valid_data = valid_data[valid_data != src.nodata]
                
                # Check if enough valid pixels
                total_pixels = masked_array[0].size
                valid_pixels = len(valid_data)
                valid_fraction = valid_pixels / total_pixels if total_pixels > 0 else 0
                
                results[f"{column_prefix}_valid_pixels"][idx] = valid_pixels
                
                if valid_fraction >= nodata_threshold and len(valid_data) > 0:
                    # Calculate statistics
                    if 'mean' in stats:
                        results[f"{column_prefix}_mean"][idx] = np.mean(valid_data)
                    if 'min' in stats:
                        results[f"{column_prefix}_min"][idx] = np.min(valid_data)
                    if 'max' in stats:
                        results[f"{column_prefix}_max"][idx] = np.max(valid_data)
                    if 'std' in stats:
                        results[f"{column_prefix}_std"][idx] = np.std(valid_data)
                    if 'median' in stats:
                        results[f"{column_prefix}_median"][idx] = np.median(valid_data)
                    if 'count' in stats:
                        results[f"{column_prefix}_count"][idx] = len(valid_data)
                        
            except Exception as e:
                # print(f"  - Warning: Error processing watershed {idx}: {e}")
                continue
    
    # Add results to the original GeoDataFrame
    result_gdf = watersheds_gdf.copy()
    for col, values in results.items():
        result_gdf[col] = values
    
    # Print summary
    print(f"  - Processed {len(watersheds_proj)} watersheds")
    for stat in stats:
        col_name = f"{column_prefix}_{stat}"
        valid_count = (~np.isnan(result_gdf[col_name])).sum()
        print(f"  - {col_name}: {valid_count}/{len(result_gdf)} valid values")
    
    return result_gdf


def aggregate_netcdf_to_watersheds(netcdf_path, watersheds_gdf, variable_name, 
                                 stats=['mean', 'min', 'max'], column_prefix='climate',
                                 time_slice=None, level_slice=None, 
                                 lat_name='lat', lon_name='lon', nodata_threshold=0.5):
    """
    Aggregate NetCDF data to watershed polygons.
    
    Parameters:
    -----------
    netcdf_path : str
        Path to the NetCDF file
    watersheds_gdf : GeoDataFrame
        Watershed polygons
    variable_name : str
        Name of the variable in NetCDF (e.g., 'precipitation', 'temperature')
    stats : list
        Statistics to calculate
    column_prefix : str
        Prefix for output columns
    time_slice : slice or int
        Time slice to extract (e.g., slice(0, 12) for first year monthly data)
    level_slice : slice or int  
        Vertical level slice if 3D data
    lat_name, lon_name : str
        Names of latitude/longitude coordinates in NetCDF
    nodata_threshold : float
        Minimum fraction of valid pixels required
    
    Returns:
    --------
    GeoDataFrame
        Watersheds with aggregated climate data
    """
    
    print(f"Processing NetCDF: {netcdf_path}")
    
    # Open NetCDF file
    with xr.open_dataset(netcdf_path) as ds:
        print(f"  - Available variables: {list(ds.data_vars.keys())}")
        print(f"  - Dimensions: {dict(ds.dims)}")
        
        # Select variable
        if variable_name not in ds.data_vars:
            raise ValueError(f"Variable '{variable_name}' not found. Available: {list(ds.data_vars.keys())}")
        
        data_array = ds[variable_name]
        
        # Apply time slice if specified
        if time_slice is not None:
            if 'time' in data_array.dims:
                data_array = data_array.isel(time=time_slice)
                print(f"  - Applied time slice: {time_slice}")
        
        # Apply level slice if specified  
        if level_slice is not None and len(data_array.dims) > 2:
            # Try common level dimension names
            level_dims = [dim for dim in data_array.dims if dim in ['level', 'lev', 'pressure', 'plev']]
            if level_dims:
                data_array = data_array.isel({level_dims[0]: level_slice})
                print(f"  - Applied level slice: {level_slice}")
        
        # Aggregate over time if multiple time steps remain
        if 'time' in data_array.dims and len(data_array.time) > 1:
            print(f"  - Aggregating {len(data_array.time)} time steps")
            data_array = data_array.mean(dim='time')
        
        print(f"  - Final data shape: {data_array.shape}")
        print(f"  - Data CRS: Geographic (assumed)")
        
        # Ensure watersheds are in geographic coordinates (EPSG:4326)
        if watersheds_gdf.crs != CRS.from_epsg(4326):
            print(f"  - Reprojecting watersheds from {watersheds_gdf.crs} to EPSG:4326")
            watersheds_geo = watersheds_gdf.to_crs('EPSG:4326')
        else:
            watersheds_geo = watersheds_gdf.copy()
        
        # Get coordinate arrays
        lats = data_array[lat_name].values
        lons = data_array[lon_name].values
        
        # Ensure longitude is in -180 to 180 range
        if lons.max() > 180:
            lons = ((lons + 180) % 360) - 180
            data_array = data_array.assign_coords({lon_name: lons})
            data_array = data_array.sortby(lon_name)
        
        print(f"  - Lat range: {lats.min():.2f} to {lats.max():.2f}")
        print(f"  - Lon range: {lons.min():.2f} to {lons.max():.2f}")
        
        # Initialize results
        results = {}
        for stat in stats:
            results[f"{column_prefix}_{stat}"] = np.full(len(watersheds_geo), np.nan)
        results[f"{column_prefix}_valid_pixels"] = np.full(len(watersheds_geo), 0)
        
        # Process each watershed
        for idx, row in watersheds_geo.iterrows():
            try:
                # Get watershed bounds
                bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
                
                # Find overlapping grid cells
                lat_mask = (lats >= bounds[1]) & (lats <= bounds[3])
                lon_mask = (lons >= bounds[0]) & (lons <= bounds[2])
                
                if not (lat_mask.any() and lon_mask.any()):
                    print(f"  - Warning: No data overlap for watershed {idx}")
                    continue
                
                # Subset data to watershed region
                subset_data = data_array.where(lat_mask & lon_mask[:, np.newaxis].T, drop=True)
                
                # For more precise intersection, we could implement point-in-polygon
                # but for climate data, bounding box is usually sufficient
                
                # Get valid data values
                valid_data = subset_data.values.flatten()
                valid_data = valid_data[~np.isnan(valid_data)]
                
                total_pixels = subset_data.size
                valid_pixels = len(valid_data)
                valid_fraction = valid_pixels / total_pixels if total_pixels > 0 else 0
                
                results[f"{column_prefix}_valid_pixels"][idx] = valid_pixels
                
                if valid_fraction >= nodata_threshold and len(valid_data) > 0:
                    # Calculate statistics
                    if 'mean' in stats:
                        results[f"{column_prefix}_mean"][idx] = np.mean(valid_data)
                    if 'min' in stats:
                        results[f"{column_prefix}_min"][idx] = np.min(valid_data)
                    if 'max' in stats:
                        results[f"{column_prefix}_max"][idx] = np.max(valid_data)
                    if 'std' in stats:
                        results[f"{column_prefix}_std"][idx] = np.std(valid_data)
                    if 'median' in stats:
                        results[f"{column_prefix}_median"][idx] = np.median(valid_data)
                    if 'count' in stats:
                        results[f"{column_prefix}_count"][idx] = len(valid_data)
                        
            except Exception as e:
                print(f"  - Warning: Error processing watershed {idx}: {e}")
                continue
    
    # Add results to original GeoDataFrame
    result_gdf = watersheds_gdf.copy()
    for col, values in results.items():
        result_gdf[col] = values
    
    # Print summary
    print(f"  - Processed {len(watersheds_geo)} watersheds")
    for stat in stats:
        col_name = f"{column_prefix}_{stat}"
        valid_count = (~np.isnan(result_gdf[col_name])).sum()
        print(f"  - {col_name}: {valid_count}/{len(result_gdf)} valid values")
    
    return result_gdf


# Example usage functions
def process_elevation_data(tiff_path, watersheds_gdf):
    """Example: Process elevation/DEM data"""
    return aggregate_tiff_to_watersheds(
        tiff_path=tiff_path,
        watersheds_gdf=watersheds_gdf,
        stats=['mean', 'min', 'max', 'std'],
        column_prefix='elevation'
    )

def process_slope_data(tiff_path, watersheds_gdf):
    """Example: Process slope data"""
    return aggregate_tiff_to_watersheds(
        tiff_path=tiff_path,
        watersheds_gdf=watersheds_gdf,
        stats=['mean', 'max'],  # Min slope usually not meaningful
        column_prefix='slope'
    )

def process_precipitation_data(netcdf_path, watersheds_gdf, scenario='ssp245'):
    """Example: Process CMIP6 precipitation data"""
    return aggregate_netcdf_to_watersheds(
        netcdf_path=netcdf_path,
        watersheds_gdf=watersheds_gdf,
        variable_name='pr',  # Common CMIP6 precipitation variable
        stats=['mean', 'max'],
        column_prefix=f'precip_{scenario}',
        time_slice=slice(0, 12)  # First year of data
    )

# Batch processing function
def batch_process_features(watersheds_gdf, feature_configs):
    """
    Process multiple features at once.
    
    Parameters:
    -----------
    watersheds_gdf : GeoDataFrame
        Base watersheds
    feature_configs : list of dict
        Each dict contains parameters for aggregate_tiff_to_watersheds or aggregate_netcdf_to_watersheds
        
    Example:
    --------
    configs = [
        {
            'type': 'tiff',
            'path': 'elevation.tif',
            'stats': ['mean', 'max'],
            'prefix': 'elevation'
        },
        {
            'type': 'netcdf', 
            'path': 'precipitation_ssp245.nc',
            'variable': 'pr',
            'stats': ['mean'],
            'prefix': 'precip_ssp245'
        }
    ]
    """
    
    result_gdf = watersheds_gdf.copy()
    
    for config in feature_configs:
        print(f"\n--- Processing {config['prefix']} ---")
        
        if config['type'] == 'tiff':
            result_gdf = aggregate_tiff_to_watersheds(
                tiff_path=config['path'],
                watersheds_gdf=result_gdf,
                stats=config.get('stats', ['mean']),
                column_prefix=config['prefix']
            )
        elif config['type'] == 'netcdf':
            result_gdf = aggregate_netcdf_to_watersheds(
                netcdf_path=config['path'],
                watersheds_gdf=result_gdf,
                variable_name=config['variable'],
                stats=config.get('stats', ['mean']),
                column_prefix=config['prefix'],
                time_slice=config.get('time_slice'),
                level_slice=config.get('level_slice')
            )
        else:
            print(f"Unknown type: {config['type']}")
    
    return result_gdf
