import pickle
import pandas as pd
import os
import numpy as np


## from create_cache import nasa_name_db,chirps_name_db,meta_database

from .metadata import meta_database

## Own computer
common_path = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\2_Processed_Data\05_LandSlide\01_CumRainFall"


path_pickle = os.path.join(common_path,r"01_Pickles")
path_df = os.path.join(common_path,r"02_DataFrames")



def weighted_average(matrix1, matrix2, weight1, weight2):
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
        
        raise ValueError("Matrices must have the same shape")

# Function to create a mask for closed intervals around cross dates
def create_intervals_mask(cross_dates, available_time, buffer_days = 30 ):
    mask = np.ones(len(available_time), dtype=bool)
    for cross_date in cross_dates:
        start_buffer = cross_date - pd.Timedelta(days=buffer_days)
        end_buffer = cross_date        
        mask &= (available_time < start_buffer) | (available_time > end_buffer)
    return mask

# Function to get random dates outside the buffer period
def get_random_dates(data_dict, available_time, num_dates = 100):

    dict_random = {}
    
    for cell, info in data_dict.items():
        cross_dates, codes = info['cross_date']

        mask = create_intervals_mask(cross_dates, available_time)
        valid_dates = available_time[mask]
    
        # Select num_dates from valid_dates without replacement
        selected_dates = np.random.choice(valid_dates, num_dates, replace=False)
        
        # Select num_dates from three_values with replacement
        selected_values = np.random.choice(codes, num_dates, replace=True)

        # Combine into a tuple of vectors
        result = (selected_dates, selected_values)

        dict_random[cell] =  result

    return dict_random


class Rain:


    def __init__(self,database = 'chirps',pais = 'Colombia',id_code = 'id_ls',**kwargs ):

        """
        Available databases = ['chirps', 'era5', 'nasa']
        """

        self._path_pickle = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\2_Processed_Data\05_LandSlide\01_CumRainFall\01_Pickles"
        self._path_df = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\2_Processed_Data\05_LandSlide\01_CumRainFall\02_DataFrames"

        
        os.chdir(self._path_df)
        csv_txt = f"{pais}_{database}.csv"

        self.pais = pais
        self.database = database
        self._df_merged = pd.read_csv(csv_txt)
        self._df_assigment = None
        self.id_code = id_code

    # Function to find the nearest key from self._df_merged for each row in df2
    def _find_nearest_key(self,row):
        # Calculate the distance between the current row and all rows in self._df_merged
        distances = np.sqrt((self._df_merged['Longitud'] - row['Longitud'])**2 + (self._df_merged['Latitud'] - row['Latitud'])**2)
        # Find the index of the minimum distance
        min_index = distances.idxmin()
        # Return the corresponding key from self._df_merged
        return self._df_merged.loc[min_index, self.database]


    def _ensamble_nsa(self,dict_matrix_db = dict,
                              mingle = True):

        """
        mingle : If True then we take the weight average between ns and obj types.,else only work with ns.    
        """

        model_weights = {'ns': 4, 'obj':2 }
        dict_escenarios_avg = {self.database: {'ns': dict() }}

        tipe_objs = dict_matrix_db[self.database].keys()
        dict_type_holder = { }

        for type_obj in tipe_objs:
            escenarios = dict_matrix_db[self.database][type_obj].keys()
            for escenario in escenarios:

                df_escenario = dict_matrix_db[self.database][type_obj][escenario]
                dict_obj = dict_type_holder.setdefault(escenario, { } )
                dict_obj.setdefault(type_obj,df_escenario )

        for escenario, dict_types in dict_type_holder.items():   
            if mingle:
                ensemble_escenario = weighted_average(dict_types['ns'], 
                                                          dict_types['obj'],
                                                          model_weights['ns'],
                                                          model_weights['obj'])
            else:
                ensemble_escenario = dict_types['ns']
            dict_foloup = dict_escenarios_avg[self.database]['ns'].setdefault(escenario , ensemble_escenario)
            
        return dict_escenarios_avg


    
    def _matrix(self,):
        
        pattern_df = r'(?P<set>\w+)_(?P<db>\w+).csv'
        pattern_pickle = r'(?P<set>\w+)_(?P<db>\w+)_(?P<Escenario>historical|ssp\d{3})_(?P<fold>\d+)_(?P<time_type>\w+).pickle'
        df_merged_catalog = pd.Series(os.listdir(self._path_df)).str.extract(pattern_df)
        df_merged_pickle = pd.Series(os.listdir(path_pickle)).str.extract(pattern_pickle)
        
        slice_table = df_merged_pickle.loc[(df_merged_pickle.set == self.pais) & (df_merged_pickle.db == self.database) ]
        global_df = { } 
        
        group_db = slice_table.groupby(['time_type','db','Escenario'])
        
        for (tipe_obj,db,escenario),df_time in  group_db:
        
            dict_db = global_df.setdefault(db, { } )
            dict_escenarios = dict_db.setdefault(tipe_obj,{ })
        
            df_cells = self._df_assigment.get(self.database).unique()
            set_df_cells  = set(df_cells)
            
            cell_index = None
            dfs_holder = [ ]
            cols_final = [ ]
            counter = 0
            progress = 0
            for counter,(idx,row) in enumerate(df_time.iterrows(),start = 1):
        
                os.chdir(path_pickle)
                filename = f"{self.pais}_{db}_{escenario}_{row['fold']}_{tipe_obj}.pickle"
                
                # Load the pickle file
                with open( filename, 'rb') as file:
                    data = pickle.load(file)
                    set_pickle = set(data['values'].keys())
                    if counter ==  1:
                        cell_index = data['vector_time']
                    intersection =  set_df_cells.intersection(set_pickle)
                    if not intersection:
                        continue
                    else:
                        matrix_vectors = [ ]
                        columns_aligment = [ ]
                        for cell in intersection:
                            vector_cell = data['values'][cell]
                            matrix_vectors.append(vector_cell)
                            columns_aligment.append(cell)
                        matrix_vectors = np.stack(matrix_vectors,axis = 1)
                        dfs_holder.append(matrix_vectors)
                        cols_final += columns_aligment
                        progress += len(intersection)
                        if progress == len(set_df_cells):
                            break
            matrix_series = pd.DataFrame(np.concatenate(dfs_holder,axis =1),columns=cols_final,index = cell_index ) 

            if self.database == 'era5':
                matrix_series = matrix_series.resample("D").sum()

            dict_escenarios.setdefault(escenario ,matrix_series) 

        if self.database == 'nasa':
            global_df = self._ensamble_nsa(dict_matrix_db = global_df,  mingle = True)

        return global_df
 
       
    # Function to verify if any cells in the list are beside each other using vectorized operations in numpy
    def verify_adjacent_cells_numpy_vectorized(self, cell_list):
        cell_coords = np.array([list(map(int, cell.split('_'))) for cell in cell_list])
        
        # Calculate the differences between each pair of cells
        diff = np.abs(cell_coords[:, np.newaxis] - cell_coords)
        
        # Check if the differences are within the 3x3 matrix bounds
        adjacent_mask = np.all(diff <= 1, axis=2)
        
        # Get the indices of adjacent cells
        adjacent_indices = np.argwhere(adjacent_mask)
        
        # Create a dictionary to hold the results
        result_dict = {cell: {'neighbors': [], 'cross_date': None}  for cell in cell_list}
        
        # Populate the dictionary with adjacent cells
        for i, j in adjacent_indices:
            if i != j:
                result_dict[cell_list[i]]['neighbors'].append(cell_list[j])

        
        for cell ,dict_neighbors in result_dict.items():
        
            list_neighbors = dict_neighbors['neighbors'] + [cell ]
            df_dates = self._df_assigment.loc[self._df_assigment.get(self.database).isin(list_neighbors), ['Fecha',self.id_code,self.database] ]
            df_id_codes = df_dates.loc[df_dates.get(self.database) == cell,self.id_code].values

            # Create a tuple of tuples from the DataFrame columns
            tuple_of_tuples = (df_dates['Fecha'].values, df_id_codes)

            dict_neighbors["cross_date"] = tuple_of_tuples
    
     
        return result_dict


    def assigment_cell(self,data:pd.DataFrame):

        self._df_assigment = data.copy()

        # Apply the function to each row in self._df_merged2
        self._df_assigment[self.database] = self._df_assigment.apply(lambda row: self._find_nearest_key(row), axis=1)
        self._dict_matrix_db = self._matrix()
    

    def fit(self):

        ### dict_matrix_db = self._matrix()
        dict_neighbors = self.verify_adjacent_cells_numpy_vectorized(self._df_assigment.get(self.database).unique())

        output_dict = {'df_assigment':self._df_assigment,
                       'dict_matrix_db':self._dict_matrix_db,
                       'dict_neighbors':dict_neighbors,
                      } 
         
        print("Done Assigment.")

        return output_dict



class Load:
    
    def __init__(self,df_coords =  pd.DataFrame,
                     pais = 'Colombia',
                     database = 'chirps'):
        
        """
        Available databases = ['chirps', 'era5', 'nasa']
        """

        object_rain = Rain(database = database, pais=pais)
        object_rain.assigment_cell(data = df_coords)
        rain_assigment = object_rain.fit()

        self.df_assigment = rain_assigment['df_assigment']
        self._dict_matrix_db = rain_assigment['dict_matrix_db']
        self.dict_neighbors = rain_assigment['dict_neighbors']
        self._random_dates = None

        self._available_time = pd.date_range(start='1981-01-01', end='2023-12-31', freq='D')
        self.database = database
        
        self._code = 'id_ls'
  

    # Function to turn a set of Timestamps into a tuple with time deltas, ordered with the oldest date first
    # and account for the case where going back in time runs out of the available range of time
    def _timestamps_to_timedelta(self,timestamps):
        result = []
        for timestamp in timestamps:
            new_timestamp = timestamp - pd.Timedelta(days=self._buffer_date)
            # Ensure the new timestamp does not go out of the available range
            if new_timestamp < self._available_time.min():
                result.append((None, timestamp))
            else:
                result.append((new_timestamp, timestamp))
        return tuple(result)
        
    
    def _cum_sum_ensamble(self,
                          matrix_df = pd.DataFrame,
                          escenario = str,
                          **kwargs):


        # Get random dates outside the buffer period for each cell
        self._random_dates = get_random_dates(data_dict = self.dict_neighbors,
                                                available_time = self._available_time[self._window_size + 1:],
                                                num_dates = self._num_dates)
                                              

        
        df_merged_copy = self.df_assigment.copy()

        rolling_df_sum = matrix_df.rolling(window = self._window_size).sum()  
        rolling_df_max = matrix_df.rolling(window = self._window_size).max()

        not_date = 0
        
        list_tuple = [ ]
        random_dates = [ ]
        dates_slides = [ ]


        if self.database != 'nasa':

            cum_name =  f'cum_sum_{self.database}_{self._window_size}'
            max_name = f'max_{self.database}_{self._window_size}'
        else:
            cum_name =  f'cum_sum_{self.database}_{self._window_size}_{escenario}'
            max_name = f'max_{self.database}_{self._window_size}_{escenario}'

    
        for cell_group,df_group in df_merged_copy.groupby(self.database):

            random_numpy_dates,code = self._random_dates[cell_group]
            cum_sum_random = rolling_df_sum.loc[random_numpy_dates, cell_group]
            max_sum_random = rolling_df_max.loc[random_numpy_dates, cell_group]
            cum_sum_random.name = cum_name
            max_sum_random.name = max_name
            random_df_dates = pd.concat([cum_sum_random,max_sum_random],axis = 1)
            random_df_dates = random_df_dates.reset_index(names = 'Fecha')
            random_df_dates = random_df_dates.assign(**{'Label' : 0,self._code :code})
            random_dates.append(random_df_dates)

            timestamps_tuple = self.dict_neighbors[cell_group]['cross_date']
            cross_dates, codes = timestamps_tuple
            # Convert the set of Timestamps into a tuple with time deltas, ordered with the oldest date first
            result_tuple = self._timestamps_to_timedelta(cross_dates)
            
            # Iterate over both tuples
            for (start_date ,end_date), code_date in zip(result_tuple, codes):
                
                try:
                    cum_sum_slide = rolling_df_sum.loc[start_date:end_date, cell_group].max()
                    max_sum_slide = rolling_df_max.loc[start_date:end_date, cell_group].max()
                    
                except KeyError:

                    not_date += 1
                    ## print(f"Date Not Found {date}")
                    cum_sum_slide = np.nan
                    max_sum_slide = np.nan


                tuple_out = ( end_date, cum_sum_slide, max_sum_slide, code_date,1)
                list_tuple.append(tuple_out)

            else: 
                continue

        random_dates  = pd.concat(random_dates)
        ensamble_result = pd.DataFrame(list_tuple, columns = ['Fecha', cum_name, max_name ,self._code,'Label'])
        
        if self._random_sample == '2:1':
            print("Ratio 2:1")
            random_dates = random_dates.sample(ensamble_result.shape[0] * 2)
        elif isinstance(self._random_sample,int):
            print(f"Extracting {self._random_sample}")
            random_dates = random_dates.sample(self._random_sample  )

        out_df_training = pd.concat([random_dates,ensamble_result],ignore_index = True)

        print(f"Amount of nan values {not_date}")
    
        return out_df_training


    def compute_cum_max(self, window = int ,
                            buffer_date = int,
                            num_dates = int,
                            random_sample = str,
                       **kwargs):


        """
        window = Is the windows size. The window for which we are taking the cumsum and and maximun cum sum.
        buffer_date = Is the buffer generated around each incident. We can go back 10 ,20 ,30 days. Recall that we are not only using incidentes (1's) but also randomly sampling after we cross the complete range with this intervals.

        window != buffer_date , in the sense taht the first one is the one we use to compute the cumulative water while the second is to make the global buffer around the vector time to sample later.The buffer date is constrols how close we want our landslide date be from our non- landslide dates.
        
        """

        self._window_size = window           
        self._buffer_date = buffer_date        # days.
        self._num_dates = num_dates        # Number of days.
        self._random_sample = random_sample



        df_merged_copy = self.df_assigment.copy()

        df_merged_copy.drop(columns = ['Fecha',self.database] ,inplace = True)
        db_bases = self._dict_matrix_db.keys()
        global_output = { }

        
        for db in db_bases:
        
            tipe_objs = self._dict_matrix_db[db].keys()

            if db == meta_database['nasa']['name']:
                escenario = kwargs.get('escenario')
            else:
                escenario = 'historical'

            matrix_df = self._dict_matrix_db[db]['ns'][ escenario]
            df_escenario = self._cum_sum_ensamble( matrix_ensamble,escenario = escenario)
            df_merged_copy = pd.merge(left = df_merged_copy,right = df_escenario,on = self._code)
                           
        return df_merged_copy
    
        
    def __repr__(self):
    
        return f"{self.database}({self.database},id_code = {self._code})"

