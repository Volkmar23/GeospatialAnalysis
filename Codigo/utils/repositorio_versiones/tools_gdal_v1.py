"""
We modify the function named :

-> slice_for_country 

And updatedd from one given by out frind Claude.

"""
from typing import Union, List
import pandas as pd
import rasterio as rio
import os
import numpy as np
from rasterio.windows import Window
import re
from pyproj import Transformer 
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import time


from tools import boundaries


############### 03_AssigningWorldClim. ###############

def get_index_pix(df = None,
                      tiff = None,
                      row_window = 0, 
                      col_window = 0,
                     **kwargs):

    """
    17/02/2025

    This function was deleted.It was replace for slice_for_country in the current module.

    """
    
    return None
        

def slice_for_country(path = str,
                      latlon_df = pd.DataFrame,
                      var_name = str,
                      pais = 'Colombia',
                      read_all = False):
    
    """
    Esta funcion lo que realiza es poder leer del tiff solo lo relevante , lo cual es la region que abarca solo colombia para que asi los
    calculos sean mas rapidos.
    
    area_slice = {
                    'lat_max': 12.6,
                    'lon_min': -79.29,
                    'lat_min': -4.4,
                    'lon_max': -66.29
                }
        
    """

    area_slice = boundaries[pais]
        
    latlon_df_copy = latlon_df.copy()

    lat = latlon_df_copy['Latitud'].values
    lon = latlon_df_copy['Longitud'].values


    raw = rio.open(path, masked=True) 
    dem_transform = raw.transform
    rows_window_i, cols_window_i = rio.transform.rowcol(dem_transform, lon, lat)
    
    rows_outside = (rows_window_i < 0)
    cols_outside = (cols_window_i < 0)
 
    bool_bad_ones = rows_outside | cols_outside
    is_points_outside = bool_bad_ones.any()

    if is_points_outside:
        amount_outside = bool_bad_ones.sum()
        print(f"Theres {amount_outside} outside the current tif file.")


    # Convert the geographic coordinates to pixel coordinates
    lon_min, lat_max = area_slice['lon_min'], area_slice['lat_max']
    lon_max, lat_min = area_slice['lon_max'], area_slice['lat_min']

    row_bottom, col_left = raw.index(lon_min, lat_min)
    row_upper, col_right = raw.index(lon_max, lat_max)
    
    # Window.from_slices((row_start, row_stop), (col_start, col_stop)) <-- Segun documentacion.
    # Define the window or subset of data you want to read
    window_slice = Window.from_slices((row_upper, row_bottom), (col_left, col_right))

    (row_start,row_end),(col_start,col_end) =  window_slice.toranges()


    row_update = rows_window_i - row_start
    col_update = cols_window_i - col_start

    chunk_country = raw.read(1, window = window_slice, masked=True )
    property_assigment = chunk_country[row_update,col_update]

    latlon_df_copy.loc[:,var_name] = property_assigment
    
    
    return latlon_df_copy



### Read DOC string to see how this function was called after i renamed it after 
## I renamed this function from **Assigning_distance** into generated_buffer
def generated_buffer(latlon_df = pd.DataFrame,
                     buffer_distance = Union[int, List[int]],
                     epsg_proj = 4326,
                     new_name_column = str,
                     
                     **kwargs
                      ):
    
    """
    Assigning_distance
    
    Default.
    buffer_distance = 3_000 ; Buffer radius in meters.

    More info from where the code was derived:
    https://gis.stackexchange.com/questions/455354/creating-buffers-in-meters-for-locations-around-the-world-using-python


    This changes were made 15/12/2024

    The root for the implementation to be able to receibe lista is an optimization. Causein that sense we can alter only the radius of the buffer cause all other things are eaqual.

    The fisrt implementaion is located at 03_Escalamiento/Articulador/05_LandSlides/01_Unify_Training.ipynb -> 2.1 Corrida con Assets y LandSlide Catalog
    
    kwargs

    for the case of radius being a list:
        new_name_column = ['id_torre', 'id_ls','id_random']
        path_storage = url.
        name_pattern = "name_sample_{buffer}m.shape"

    for the case of radius being a int:
    
        name_storage = Specify the str for the name of the file
        
    """  
    
    latlon_df_copy = latlon_df.copy()
    path_storage = kwargs.get('path_storage',None)
    # Create geometry column using Shapely Point
    geometry = [Point(xy) for xy in zip(latlon_df_copy['Longitud'], latlon_df_copy['Latitud'])]
    # Convert DataFrame to GeoDataFrame
    gdf_activos = gpd.GeoDataFrame(latlon_df_copy, geometry = geometry, crs = epsg_proj)
    counter = 0
    
    if isinstance(buffer_distance,int):
        print(f"Generating buffer {buffer_distance/1000} km, for {latlon_df_copy.shape[0]}\n")
        
        buffer_geometries = [] #A list to hold buffered points
        name_storage = kwargs.get('name_storage',None)

        for index,(idnum, subframe) in enumerate(gdf_activos.groupby(new_name_column) , start = 1): #For each row
            
            estimated_utm = subframe.estimate_utm_crs() #Estimate a utm crs
            subframe = subframe.to_crs(estimated_utm) #Reproject the point to this
            subframe = subframe.buffer(distance=buffer_distance) #Buffer the point
            buffer_geometries.append(subframe.to_crs(epsg_proj)) #Reproject the resulting series back to 4326 and append to list
    
            counter += 1
    
            if index % 10_000 == 0:
                print(f"Working on {index} tower")
            
        new_geoms = gpd.pd.concat(buffer_geometries) #From a list of series to a data frame
    
        gpd_towers_geomtries = gdf_activos.copy()
        gpd_towers_geomtries["geometry"] = new_geoms
        new_name = f"{name_storage}_buffer_{buffer_distance}_meters.shp"

        print(f"\tStoring {new_name}")

        if path_storage:

            complete_path = os.path.join(path_storage, new_name)
            # Assuming `gdf` is your GeoDataFrame
            gpd_towers_geomtries.to_file(complete_path,driver='ESRI Shapefile')

        tab = "---------" * 4
        print("\tDone!")
        print(f"{tab}\n")

        return gpd_towers_geomtries

    elif isinstance(buffer_distance,list):
        
        print(f"Lista = {[buff/1000 for buff in  buffer_distance]}\n")

        name_pattern = kwargs.get('name_pattern',None)
        
        # Iterate in chunks of 3
        for i in range(0, len(buffer_distance), 4):
            chunk_buffer = buffer_distance[i:i+4]
            print(f"chunk {i+1}")

            dict_geometries = { }
        
            for index,(idnum, subframe_raw) in enumerate(gdf_activos.groupby(new_name_column) , start = 1): #For each row
                
                estimated_utm = subframe_raw.estimate_utm_crs() #Estimate a utm crs
                subframe_raw = subframe_raw.to_crs(estimated_utm) #Reproject the point to this
    
                for buffer_i in chunk_buffer:

                    ### print(f"Working on {buffer_i} tower")

                    buffer_geometries = dict_geometries.setdefault(buffer_i, [ ])
                    subframe_i = subframe_raw.buffer(distance=buffer_i) #Buffer the point
                    buffer_geometries.append(subframe_i.to_crs(epsg_proj)) #Reproject the resulting series back to 4326 and append to list          

            for buffer_key,list_gpd in dict_geometries.items():

                print(f"\tStoring {buffer_key}")
                new_geoms = gpd.pd.concat(list_gpd) #From a list of series to a data frame
                gpd_towers_geomtries = gdf_activos.copy()
                gpd_towers_geomtries["geometry"] = new_geoms
                new_name = name_pattern.format(buffer = buffer_key)

                print(f"\tStoring {new_name}")

                complete_path = os.path.join(path_storage, new_name)
                # Assuming `gdf` is your GeoDataFrame
                ## gpd_towers_geomtries.to_file(new_name)
                gpd_towers_geomtries.to_file(complete_path)


            tab = "---------" * 4
            print(f"\tDone chunk {i+1}!")
            print(f"{tab}\n")

        return None
 

class Project:
    def __init__(self,case = 1, epsg_code = str):
        """
        ## From regular to weird one
        case = 1
        ## From Weird to regular one
        case == 0

        ## From the documentation.
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        transformer.transform(12, 12)
        """
        
        # 'crs': CRS.from_epsg(32618)
        # Define source and destination coordinate systems
        crs_4326 = 'EPSG:4326'  # WGS84 (lat/lon)   #'crs': CRS.from_epsg(4326)
        crs_alos = f'EPSG:{epsg_code}'  # UTM zone 18N (the projection from the TIFF file)  

        print(f"Establish code {crs_alos}")

        ## From regular to weird one
        if case == 1:
            # Initialize the Transformer object
            self._transformer = Transformer.from_crs(crs_4326, crs_alos, always_xy = True)
        elif case == 0:
            # Initialize the Transformer object
            self._transformer = Transformer.from_crs(crs_alos,  crs_4326, always_xy = True)
            
    def reproject(self, lon = float,lat = float):
        # Transform geographic coordinates to projected coordinates
        x_new, y_new = self._transformer.transform(lon, lat)

        return (x_new, y_new)        
              
              
              
              
              
              
              
        