import os
import xarray as xr
import numpy as np
import pandas as pd
import time
import pickle
import re
import rasterio
import sys
from rasterio.windows import Window
from rain_module import has_all_days

path_gloabales = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Python Code\03_Escalamiento\Articulador\01_FuncionesGlobales"
if path_gloabales not in sys.path:
    sys.path.append(path_gloabales)

## from tools import extractUniqueLocation, boundaries
import tools

#### Functions intended to create dict_matrix for Nasa model version 1.

def crop_raster(area_slice:dict,
                src: rasterio.io.DatasetReader 
               ):


    # Convert the geographic coordinates to pixel coordinates
    lon_min, lat_max = area_slice['lon_min'], area_slice['lat_max']
    lon_max, lat_min = area_slice['lon_max'], area_slice['lat_min']

    row_bottom, col_left = src.index(lon_min, lat_min)
    row_upper, col_right = src.index(lon_max, lat_max)
    
    # Window.from_slices((row_start, row_stop), (col_start, col_stop)) <-- Segun documentacion.
    # Define the window or subset of data you want to read
    window_slice = Window.from_slices((row_upper, row_bottom), (col_left, col_right))


    return window_slice

def ids_assigment(
                      raw_tif : rasterio.io.DatasetReader,
                      latlon_df : pd.DataFrame,
                      lon_name = 'Longitud',
                      lat_name = 'Latitud'
                     ):

    """



    """
                      
    dem_transform = raw_tif.transform


    unique_index = latlon_df.index
    lat_ = latlon_df[lat_name].values
    lon_ = latlon_df[lon_name].values


    ### This equation has been verify with the output given by
    rows, cols = rasterio.transform.rowcol(dem_transform, lon_, lat_)

    rows_outside = (rows < 0)
    cols_outside = (cols < 0)
 
    bool_bad_ones = rows_outside | cols_outside
    is_points_outside = bool_bad_ones.any()

    output = None

    if is_points_outside:

        amount_outside = bool_bad_ones.sum()
        print(f"Theres {amount_outside} outside the current tif file.")
        output = latlon_df.loc[bool_bad_ones]

        return output

    latlon_df['rows'] = rows
    latlon_df['cols'] = cols

    rows_unique = latlon_df.rows.values
    cols_unique = latlon_df.cols.values

    ids_cols = [f"{row}_{col}" for row,col in zip(rows_unique,cols_unique)]
    latlon_df['id_nowcast'] = ids_cols

    no_duplicates = latlon_df.drop_duplicates(subset = 'id_nowcast').copy()

    dict_nasa = {'original':latlon_df,
                 'unique': no_duplicates
                    }

    return dict_nasa


def now_cast_v1(latlon_df = pd.DataFrame):


    """
    Metadata NASA:


    {'driver': 'GTiff',
     'dtype': 'uint8',
     'nodata': 255.0,
     'width': 43200,
     'height': 14400,
     'count': 1,
     'crs': CRS.from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'),
     'transform': Affine(0.008333333333325626, 0.0, -180.0,
            0.0, -0.008333333333312291, 60.0000600030303)}


    BoundingBox(left=-180.0, bottom=-59.9999399966667, right=179.999999999667, top=60.0000600030303)
    """
    
    global boundaries

    copy_latlon_df = latlon_df.copy()

    folder_path =  r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\08_LandSlideCatalog\07_GlobalDailyNowCast"
    regex_pattern = r"Global_Landslide_Nowcast_v1.1_(?P<Year>\d{4})(?P<Month>\d{2})(?P<Day>\d{2}).tif"
    pais = 'Colombia'
    date_pattern = re.compile(regex_pattern)

    area_slice = tools.boundaries[pais] 

    # List all tif files in the folder
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]


    amount_tif_images = len(tif_files)
    chunk_progress = amount_tif_images // 6

    ## Array that holds tuple like ('Year','Month','Day')
    tuplas_vector_time = [ ]
    
    # Initialize an empty list to store the arrays
    arrays = []
    
    dict_nasa = None
    window_slice = None
    row_update = None
    col_update = None
    
    # Read each tif file and append the array to the list
    for idx, tif_file in enumerate(tif_files, start = 1):

        ## Extract the time of the .tif images.
        nasa_match = date_pattern.match(tif_file)
    
        if bool(nasa_match):
            tupla_date = ( nasa_match.group('Year'),nasa_match.group('Month'),nasa_match.group('Day') )
            path_complete = os.path.join(folder_path, tif_file)

            with rasterio.open(path_complete) as src:
                
                if idx == 1:
    
                    dict_nasa = ids_assigment(raw_tif = src, latlon_df = copy_latlon_df)
    
                    ## Establish the window:  window_slice = Window.from_slices((row_upper, row_bottom), (col_left, col_right))
                    window_slice = crop_raster(area_slice = area_slice ,src =src )
    
                    if isinstance( dict_nasa, dict):
    
                        rows_unique = dict_nasa['unique'].rows.values
                        cols_unique = dict_nasa['unique'].cols.values

                        id_cols =  dict_nasa['unique']['id_nowcast'].values
    
                        print(f"Establish window for {pais}\n{window_slice}\n")

                        (row_start,row_end), (col_start,col_end) =  window_slice.toranges()
                        row_update = rows_unique - row_start
                        col_update = cols_unique - col_start
                       
                    elif isinstance( dict_nasa, pd.DataFrame):
        
                        print("A mistake happen. ")
                        return dict_nasa
    
                country_slice = src.read(1, window = window_slice, masked=True )
                slice_i = country_slice[row_update,col_update]
                vector_probabilty = slice_i.data
                is_nan = slice_i.mask
    
                if is_nan.any():

                    year, month, day  =  tupla_date
                    nan_amount = is_nan.sum()
                    print(f"Skipping the following date {year}-{month}-{day}.\nTheres {nan_amount} nan in some tif file.")

                else:
                    
                    arrays.append(vector_probabilty)  # Read the first band
                    tuplas_vector_time.append(tupla_date)
                    
        else:
            print(f"Stop\nName {tif_file} no match with the current pattern.")
            return tif_file,date_pattern

        if idx % chunk_progress == 0:
            print(f"\tProgress {idx / amount_tif_images:.2f}")

    # Stack the vectors vertically
    stacked_array = np.vstack(arrays)
    # Convertir la lista de tuplas a un vector de rango de datetime de pandas
    date_range = pd.to_datetime([f"{year}-{month}-{day}" for year, month, day in tuplas_vector_time])
    dff = pd.DataFrame(stacked_array , index = date_range,columns = id_cols)
    output = ( dict_nasa['original'],dff )
    
    return output

def assigment_now_cast():

    """
    ### IN MAINTANCE ###
    ### Nowcast V.1 ####
    In progress.
    This is the backbone to make the realtionship between the rain and the risk now cast.
    I just brought this form the notebooks so have it now in prodcution , but it still in maintance.For now we will just focus on the already created **NowCastV2**
    """


    parent_path = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\2_Processed_Data\05_LandSlide\04_Training_cache\Metodologia 2\cache_trash\training_set_1"
    
    
    final_table = [ ]
    current_amount = 0 
    whole_size = group_block.size().size
    chunk = whole_size // 8 
    
    
    basket = [ ] 
    
    counter_basket = 0
    
    for counter , ((cell_chirps, id_nowcast), df_block) in enumerate(group_block,start = 1):
            
        vector_1 = df_matrix_nowcast[df_matrix_nowcast[id_nowcast] == 1].index
        vector_2 = df_matrix_nowcast[df_matrix_nowcast[id_nowcast] == 2].index
        vector_0 = whole_range - (set(vector_1) | set(vector_2))
    
    
        slices = [ ]
    
        len_1 = len(vector_1)
        len_2 = len(vector_2)

        
        if len_1 == 0 and len_2 == 0:
            sample_size = 30
        else:
    
            sample_size = 0
    
            if len_1 > 0:
                series_slice_1 = rolling_df_sum.loc[vector_1, cell_chirps]
                series_slice_1 = transform_series(series_slice_1, 1)
                sample_size += len_1
                slices.append(series_slice_1)
    
            if len_2 > 0:
                series_slice_2 = rolling_df_sum.loc[vector_2, cell_chirps]
                series_slice_2 = transform_series(series_slice_2, 2)
                sample_size += len_2
                slices.append(series_slice_2)
            
        sampled_vector_0 = random.sample(list(vector_0), sample_size)
        series_slice_0 = rolling_df_sum.loc[sampled_vector_0, cell_chirps]
        series_slice_0 = transform_series(series_slice_0, 0)
        slices.append(series_slice_0)
        
        # Combine all slices
        series_slice = pd.concat(slices)
        
        # Get IDs
        id_identifier = df_block.id.values
        
        # Replicate series_slice for each ID
        df_with_ids = pd.concat([series_slice.assign(**{'id' : id_}) for id_ in id_identifier], ignore_index=True)
    
        df_with_ids = df_with_ids.astype({'risk':'category',
                                          'pr':np.float32,
                                          'id':'category'})
    
        basket.append(df_with_ids)
    
        if counter % chunk == 0:
            
            print(f"Progress {(counter / whole_size) * 100 :.2f}")
            flush_dataframe = pd.concat(basket, axis = 0 , ignore_index = True)
            counter_basket += 1
            save_name = os.path.join(parent_path , f'{counter_basket}_basket.csv')
            flush_dataframe.to_csv(save_name , index = False)
            basket = [ ]

    return None



### Nowcast V.2.0 ######   From here start part 2.

def now_cast_assigment(dataset:pd.DataFrame ):

    slice_tuple = None
    cols_dict_db = None
    assigment_nowcast = None
    path_database = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\08_LandSlideCatalog\08_GlobalDailyNowCast_2.0\Colombia\02_Slices"
    ### print("Working in NASA nowcast assigment [...]\n")
    nc_now_v2 = os.listdir(path_database)

    total_files = len(nc_now_v2) 
    chunk_progress = total_files // 6

    date_pattern = r".+?_(?P<Year>\d{4})(?P<Month>\d{2})(?P<Day>\d{2})"
    date_pattern = re.compile(date_pattern)
    
    tuplas_list = [ ]
    arrays = [ ]
    
    for idx, nc in enumerate(nc_now_v2,start = 1):

        
        nasa_match = date_pattern.match(nc)
        tupla_date = ( nasa_match.group('Year'),nasa_match.group('Month'),nasa_match.group('Day') )
        tuplas_list.append(tupla_date)

        complete_path = os.path.join(path_database , nc)
        chunk_nc_now = xr.open_dataset(complete_path)

        if idx == 1:

            (slice_tuple, cols_dict_db, assigment_nowcast) = tools.extract_slice_tools( nc_file = chunk_nc_now,
                                                                                         dataset = dataset,
                                                                                         database_name = 'id_nowcast',
                                                                                         column_name_pro = ['lon', 'lat' ], 
                                                                                         column_isa = ['Longitud','Latitud'], 
                                                                                         variable_name = 'p_landslide')
            

        if idx % chunk_progress == 0:
            print(f"{(idx / total_files) * 100:.3f} [%] progress")

        numba_slice = chunk_nc_now.p_landslide.to_numpy()
        slice_i = numba_slice[slice_tuple].astype(np.float32)

        arrays.append(slice_i)
        
    # Stack the vectors vertically
    stacked_array = np.vstack(arrays)
    
    # Convertir la lista de tuplas a un vector de rango de datetime de pandas
    date_range = pd.to_datetime([f"{year}-{month}-{day}" for year, month, day in tuplas_list])
    dict_matrix_nowcast = pd.DataFrame(stacked_array , index = date_range, columns = cols_dict_db)

    complete_dates = has_all_days(date_series = dict_matrix_nowcast.index, non_leap_year=False)

    if complete_dates:
        print("Done\n")
        return (assigment_nowcast, dict_matrix_nowcast)

    print("Done\n")



