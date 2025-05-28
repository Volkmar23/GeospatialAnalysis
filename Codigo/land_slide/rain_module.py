import xarray as xr
import os
import pandas as pd
import numpy as np
import re
import time
import sys


# Define paths
path_tools = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Python Code\03_Escalamiento\Articulador\01_FuncionesGlobales"

# Add paths to sys.path if not already present
if path_tools not in sys.path:
    sys.path.append(path_tools)

# Import your modules
import tools



def weighted_average(matrix1, matrix2, weight1, weight2):

    if matrix1.shape == matrix2.shape:
        
        # Ensure the matrices have the same shape
    
        matrix1_time = matrix1.index
        matrix2_time = matrix2.index
    
        matrix1_cols = matrix1.columns
        matrix2_cols = matrix2.columns
        
        cols_verify = (matrix1_cols  == matrix2_cols).all()
        time_verify  = (matrix1_time  == matrix2_time).all()
    
        is_okey = cols_verify and time_verify
            
        if is_okey:
            # Calculate the weighted average
            weighted_avg = (weight1 * matrix1 + weight2 * matrix2) / (weight1 + weight2)
            return weighted_avg

        else:
            print("Matrices are not equal")
    else:
        raise ValueError("Matrices must have the same shape")


def create_df_modelos( url_nasa):
    
        
    """
    This functions returns a dataframe with 3 columns of 
    [Modelos , Escenarios , time_object]

    El proposito general de la funcion es creear el dataframe base el cual sera usado para agrupar modelos y escenarios de forma automatizada.
    """

    coords_raw = {
                'vector_lon':None,
                 'vector_lat':None
                }


    re_nasa_pattern = f"pr_(?P<Modelos>.+?)_(?P<Escenarios>historical|ssp\d+)_(?P<Fold>\d+).nc$"
    modelos_regex = re.compile(re_nasa_pattern)
    
    np_arr = np.array([var for var in os.listdir(url_nasa) if modelos_regex.match(var)]   )
    modelos  = pd.Series(np_arr).str.extract(modelos_regex).dropna().copy()   ## Dropna removes the .txt doc i have in folder.

    mod_esc = modelos.groupby('Modelos')

    first = True
    bad_ones = False

    for modelo, df in mod_esc:

        ### df_copy = df.copy()

        raw_idx = df.index
        files_txt  = np_arr[raw_idx]

        path_name = os.path.join(url_nasa,files_txt[0])
        xarray_file = xr.open_dataset(path_name,chunks = {'time':'auto'})

        vector_lon = xarray_file.lon.values
        vector_lat = xarray_file.lat.values
        my_string = str(xarray_file.time.dtype)
            
        modelos.loc[raw_idx, 'time_type'] = my_string

        if first:

            coords_raw['vector_lon'] = vector_lon
            coords_raw['vector_lat'] = vector_lat
            ## succed_equal += 1
            first = False
            
        else:

            equal_lon = (vector_lon == coords_raw['vector_lon']).all()
            equal_lat = (vector_lat == coords_raw['vector_lat']).all()

            are_equal = equal_lon and equal_lat

            if are_equal:
                ## succed_equal += 1
                continue
            else:
                print("Corrds misalign")
                bad_ones = True
                return None

    np_arr = np.array([os.path.join(url_nasa,name_file) for name_file in np_arr ])
    
    if not bad_ones:
        print("All share the same coords")
    
    return (modelos,np_arr)


def has_all_days(date_series:np.array,non_leap_year = True):

    """
    The purpose of this function is to verify that after concating the historical and future we have all the values save for those of leap year.
    """

    full_range = pd.date_range(start = date_series.min(), end = date_series.max(), freq='D')

    if non_leap_year:
        
        
        # Convert to pandas Datetime and handle leap days
        # I made this so i can make a proper ensamble within Nasa mixdataset.
        leap_day_mask = ~((full_range.month == 2) & (full_range.day == 29))
        full_range = full_range[leap_day_mask]
        
    is_equal = (full_range == date_series).all()

    if is_equal:
        print("Dates complete")
    else:
        print("Dates Incomplete")
        
    # Check if all dates in the full range are in the date series
    return is_equal
    

def verify_aligment(coords:list):
 
     first_coords = coords[0]
     holder_bool = [ ] 
     for coord in coords[1:]:

         if coord.size == first_coords.size:
             is_equal = (coord == first_coords).all()
             holder_bool.append(is_equal)
         else:
             print("Coords unequal")
             return None
     if np.all(holder_bool):
         return True
     else: 
         False

def compute_quantile(numba:np.array  ,
                     choosen_quantile:float):


    if isinstance(choosen_quantile,float):
        raw_quantile = np.quantile(numba,q = choosen_quantile,axis = 0)
      
    elif choosen_quantile == 'max':
        raw_quantile = np.max(numba,axis = 0)
  
    elif choosen_quantile == 'mean':
        raw_quantile = np.mean(numba,axis = 0)

    return raw_quantile


def ensamble_folds(bbox = None,
                    list_names = str,
                    fold_number_int = int
                ):
    

    new_dimention = f'fold_temp_{fold_number_int}'
    cast_xarray = [ ] 

    for idx, file_name in enumerate(list_names,start = 1):
        
        current_xr = xr.open_dataset(file_name,chunks = {'time':'auto'})
        current_xr = current_xr.assign_coords(**{'lon':tools.convert_longitude_to_minus180_to_180(current_xr['lon'])})
        current_xr = tools.regular_slice(xr_file = current_xr,
                                           country_bounds = bbox,
                                           lon_name = 'lon',
                                           lat_name = 'lat'
                                          )
        cast_xarray.append(current_xr)
        

    coords_time = [raw_mod.time.values for raw_mod in cast_xarray]
    coords_lat = [raw_mod.lat.values for raw_mod in cast_xarray]
    coords_lon = [raw_mod.lon.values for raw_mod in cast_xarray]

    coords_holder = {
                     'coords_time': coords_time ,
                     'coords_lat':coords_lat,
                     'coords_lon':coords_lon
                    }
    
    
    aprobado = True

    for name_coords, coord in coords_holder.items():
        is_equal = verify_aligment(coord)

        if  is_equal:
            continue
        else:
            print("Misaligment.")
            aprobado = False
            break
    
    if aprobado:
        to_ensamble = xr.concat( cast_xarray , dim = new_dimention)
        return to_ensamble

    else:
        return False


def lazy_read( 
              url_nasa =str ,
              chosen_escenario = str,
              bbox = None):

    """
    
    This functon is in charge of computing a dictionary with keys:

    {
       historical_name: { 'tables': {time_type_1:[xrarray file read as chunks. ],
                                     time_type_2:[xrarray file read as chunks. ]},
                                     
                          'weights': {time_type_1: w1,time_type_2: w2  }  
                          }

    self._ensamble_functions() will then receive this dictionary as input and compute the respective quantile comptation.
    """

    (modelos , np_arr)  = create_df_modelos( url_nasa)
    
    print("Ensmabling xarray files [...]\n")

    global_output = [ ] 
    historical_name = 'historical'
    future_name = 'future'
    maper_escenarios  = {
                                  'historical' : 'Historical' , 
                                  'ssp245': 'SSP2 4.5', 
                                  'ssp370': 'SSP3 7.0',
                                  'ssp585': 'SSP5 8.5'
                                 }

    
    hist_models = modelos.loc[modelos.Escenarios == 'historical']
    ssp_models = modelos.loc[modelos.Escenarios != 'historical']
    dtype_ssp_grouper = ssp_models.groupby('Escenarios')     ## Primero agrupamos por el tipos de vector de Escenarios que tenemos.

    dict_dtype = {
                       historical_name: { 'tables': { },
                                          'weights': {}  },
             
                       future_name:  {'tables': {},
                                      'weights': {} }
                      }
    
    for  escenario , df_type_escenario in dtype_ssp_grouper: 

        if chosen_escenario != escenario:
            continue
            
        escenario_format_name = maper_escenarios[escenario]
        print(f"Woring on {escenario_format_name}")
        kind_type = df_type_escenario.groupby('time_type')

        for time_type,df_kind in kind_type:

            list_historicals = dict_dtype[historical_name]['tables'].setdefault(time_type, [ ] )
            list_ssp = dict_dtype[future_name]['tables'].setdefault(time_type, [ ] )
    
            models_ssp_unique  = df_kind.Modelos.unique()
            weight_ssp = models_ssp_unique.size
            
            hist_filter = hist_models.loc[hist_models.Modelos.isin(models_ssp_unique)]
            weight_hist  = hist_filter.Modelos.unique().size
    
            dict_dtype[historical_name]['weights'].setdefault(time_type, weight_hist  )
            dict_dtype[future_name]['weights'].setdefault(time_type, weight_ssp  )
            
            mod_historical = hist_filter.groupby('Fold')
            mod_esc = df_kind.groupby('Fold')     ### 
             
            iter_time = {'historical' : mod_historical, escenario : mod_esc}
            
            for time_line , grouper in iter_time.items():
            
                for fold ,df_fold in grouper:
     
                     txt_name = np_arr[df_fold.index]
                     to_ensamble = ensamble_folds(bbox,
                                                 txt_name,
                                                 fold)

                     if time_line == 'historical':
                         list_historicals.append(to_ensamble)

                     elif time_line == escenario:
                         list_ssp.append(to_ensamble)

    return dict_dtype



def main_gpdx(url_nasa = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\4_Rain\02_NasaGPDX_SouthAmericaV2" ,
             dataset = pd.DataFrame ,
             pais = 'Colombia',
             chosen_escenario = str,
             choosen_quantile = 0.5
                ):

    common_format_time  = '%Y-%m-%d'
    start_year = 1985
    scale = 86_400
    dict_dtype = lazy_read(url_nasa,  chosen_escenario ,bbox = pais)

    ensamble_historical =  dict_dtype['historical'] 
    ensamble_ssp =  dict_dtype['future'] 

    create_cache = True
    slice_tuple = None
    cols_dict_db = None
    dataset_assigment = None


    results = { 'historical':  ensamble_historical, 'future':  ensamble_ssp}

    for horizon, diccionario in results.items():

        temp_dict = { }
        
        weights_dictionary = diccionario['weights']
        count_types = len(weights_dictionary.keys())
        
        for kind_calendar ,list_xr_to_concat in diccionario['tables'].items():
    
            df_to_concat = [ ]
    
            for mod_ensamble in list_xr_to_concat:
    
                if create_cache:
                    
                    (slice_tuple, cols_dict_db, dataset_assigment) = tools.extract_slice_tools( nc_file = mod_ensamble,
                                                                                                 dataset = dataset,
                                                                                                 database_name = 'nasa',
                                                                                                 column_name_pro = ['lon', 'lat' ], 
                                                                                                 column_isa = ['Longitud','Latitud'], 
                                                                                                 variable_name = 'pr',
                                                                                               dim_mapping = False)
                    
                    create_cache = False
    
                numba_to_slice = mod_ensamble.pr.values
                raw_vector_time = mod_ensamble.time.values
    
                if numba_to_slice.shape[0] == weights_dictionary[kind_calendar]:     ### Making sure we are reducing the dimension holding the models.
                    
                    rescale_rain = numba_to_slice[slice_tuple] * scale
                    quantile_numba = compute_quantile(rescale_rain,choosen_quantile)
                    reduced_shape = quantile_numba.shape
    
                    if kind_calendar == 'datetime64[ns]':
                        time_index_leap_year = pd.to_datetime(raw_vector_time ).normalize()  
                        leap_day_mask = ~((time_index_leap_year.month == 2) & (time_index_leap_year.day == 29))
                        time_index = time_index_leap_year[leap_day_mask]
    
                        if len(reduced_shape) == 2 and reduced_shape[0] == time_index_leap_year.size:
                            quantile_numba = quantile_numba[leap_day_mask,:]
                        else:
                            raise ValueError("Time not aligns.")
    
                        normalized = pd.to_datetime(time_index.strftime(common_format_time))
                        
                    elif kind_calendar == 'object':
                        normalized = pd.to_datetime([pd.Timestamp(date.strftime(common_format_time)) for date in raw_vector_time])
    
                    df_chunk = pd.DataFrame(quantile_numba , index = normalized,columns = cols_dict_db)
                    
                    df_to_concat.append(df_chunk)
                else:
                    raise ValueError("Wrong slice time")
    
            concat_all_dfs =  pd.concat(df_to_concat,axis = 0)
            temp_dict.setdefault(kind_calendar , concat_all_dfs)
    
    
        if count_types == 2:
             
             ensamble = weighted_average(matrix1 = temp_dict['datetime64[ns]'],
                                          matrix2 = temp_dict['object'],
                                          weight1 = weights_dictionary['datetime64[ns]'],
                                          weight2 = weights_dictionary['object'])
            
        elif count_types == 1:
            
             dtypes_available = temp_dict.keys()
             ensamble =  temp_dict[dtypes_available[0]]

        results[ horizon]  = ensamble

    dict_matrix_db = pd.concat([results['historical'] , results['future']] , axis = 0)
    dict_matrix_db = dict_matrix_db.loc[dict_matrix_db.index.year >= start_year ,:]

    complete_dates = has_all_days(date_series = dict_matrix_db.index,non_leap_year = True)

    if complete_dates:
        print("Complete")

    return dataset_assigment,dict_matrix_db 


class NasaGPDX:


    def __init__(self,database
                     ,window,
                      escenario ):


        self.database = database
        self.window = window
        path_nasa_gpdx = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\4_Rain\02_NasaGPDX_SouthAmericaV2" 

        (self.dict_matrix_db ,self.dataset_assigment) = main_gpdx(url_nasa = path_nasa_gpdx ,
                                                                 dataset = database ,
                                                                 pais = 'Colombia',
                                                                 chosen_escenario = str,
                                                                 choosen_quantile = 0.5
                                                                    )
        
        self.dict_matrix_cum_window = self.dict_matrix_db.rolling(window = window).sum().astype(np.float32).copy()


        if self.database != 'nasa':
            
            self._sample_cum_name =  f'cum_sum_{self.database}_{self.window}'
        else:
            self._sample_cum_name = None


    def recompute_window(self,window):

        """
        kwargs = [escenario , window]
        """

        if self._dict_matrix_db is None:
            
            print("Need to run assigment_cell() first")
            
            return None

        if isinstance(window , int):

            if window > 1:
                
                if window != self.window:

                    print(f"\tRecomputing the matrixes , replacing a window = {self.window} for {window}.")
                    self._dict_matrix_cum_window = self._dict_matrix_db.rolling(window = window).sum().copy()
                    self.window = window
                else:
                    pass
            else:
                print("Window must be greater than 1")       
        return None

### CHIRPS V.2.0 ######   From here start part 2.

## Already move to Codigo/land_slide/rain_way.py

## def chirps_assigment(dataset:pd.DataFrame ):
## 
##     slice_tuple = None                
##     cols_dict_db = None
##     assigment_nowcast = None
##     
##     path_database = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\4_Rain\03_ChirpsRainFall_SouthAmericaV2"
## 
##     print("Working in CHIRPS nowcast assigment [...]\n")
## 
##     nc_now_v2 = os.listdir(path_database)
## 
##     total_files = len(nc_now_v2) 
##     chunk_progress = total_files // 6
## 
##     ### date_pattern = r"CHIRPS_daily_(?P<Year>\d{4}).nc"    is not being used !!!!
##     ### date_pattern = re.compile(date_pattern)      is not being used !!!!
##     
##     arrays = [ ]
##     numba_vector_time = [ ]
##     
##     for idx, nc in enumerate(nc_now_v2,start = 1):
## 
##         #### nasa_match = date_pattern.match(nc)    is not being used !!!!
##         complete_path = os.path.join(path_database , nc)
##         chunk_nc_now = xr.open_dataset(complete_path)
## 
##         if idx == 1:
## 
##             (slice_tuple, cols_dict_db, assigment_nowcast) = tools.extract_slice_tools( nc_file = chunk_nc_now,
##                                                                                          dataset = dataset,
##                                                                                          database_name = 'chirps',
##                                                                                          column_name_pro = ['longitude', 'latitude' ], 
##                                                                                          column_isa = ['Longitud','Latitud'], 
##                                                                                          variable_name = 'precip')
##         if idx % chunk_progress == 0:
##             print(f"{(idx / total_files) * 100:.3f} [%] progress")
## 
##         numba_slice = chunk_nc_now.precip.to_numpy()
##         raw_vector_time  = chunk_nc_now.time.values
## 
##         numba_vector_time.append(raw_vector_time)
##         slice_i = numba_slice[slice_tuple].astype(np.float32)
## 
##         arrays.append(slice_i)
##         
##     # Stack the vectors vertically
##     stacked_array = np.vstack(arrays)
##     dict_matrix_nowcast = pd.DataFrame(stacked_array , index = np.concat(numba_vector_time), columns = cols_dict_db)
##     complete_dates = has_all_days(date_series= dict_matrix_nowcast.index, non_leap_year=False)
## 
##     if complete_dates:
##         
##         print("Done\n")
##         return (assigment_nowcast, dict_matrix_nowcast)
## 
##     else:
##         return None

