"""

This was a copy from the original at Python_CC/03_Escalamiento/Articulador/01_FuncionesGlobales/tools.py


Fecha de este fork : 20/05/2025

Cambios a realizar:

Eliminar 

-> class Country
-> col_boundaries_v1
-> raw_itch
-> getTimeSeries

-> load_assets:
    Esta funcion se remplazara por

    load_activos

"""


import pandas as pd
import os
import numpy as np
import re
import time
import xarray as xr  

from datetime import datetime




"""
date : 26/10/2024

I no loger required to import all this libraries no more.
Now all the function related to GDAL transfomration are located in tools_gdal in the current package.

03_Escalamiento/Articulador/01_FuncionesGlobales/tools_gdal.py

## import rasterio
## from rasterio.windows import Window
## import geopandas as gpd
## from pyproj import Geod
## from shapely.geometry import Point, LineString
### from shapely.ops import nearest_points

Despite this i will preser the function for referenc ein case something goes wrong.



date: 5/02/2025


i will tranfer the functions 



"""

parent = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\11_dataset_assets\02_DemasFiliales\KMZs_SouthAmerica\SouthAmerica_Datasets"


 ## This Colombian Boundary is an old one but still works with the chirps dataset
col_boundaries_v1 = {
                    'lat_max': 12.6,
                    'lon_min': -79.29,    
                    'lat_min': -4.4,
                    'lon_max': -66.29
                        }

## This is the new boudnaries.
col_boundaries_v2 =  {
                    'lat_max': 12.590276718139648,
                    'lon_min': -81.72014617919922,    
                    'lat_min': -4.236873626708984,
                    'lon_max':  -66.87045288085938
                        }

boundaries = {
              'Colombia': col_boundaries_v2, 
              'Peru':  {
                        'lat_max': 0.8683535942075974,
                        'lon_min': -82.22535820037143,
                        'lat_min': -18.889094645160505,
                        'lon_max': -66.9948695889273
                        }  ,
              'Brasil':   {
                            'lat_max': 5.659364931405092,
                            'lon_min':  -74.97876784988931,
                            'lat_min': -34.24820136342015,
                            'lon_max':  -32.94565154858247
                            } ,
              
              'Bolivia': {  
                            'lat_max': -9.139067067621765,
                            'lon_min': -70.21589811156173,
                            'lat_min': -23.416587810916724,
                            'lon_max':  -56.84845140180418
                                } ,
              
              'Chile': {  
                        'lat_max': -16.818341017253367,
                        'lon_min': -77.472441757313,
                        'lat_min': -50.94650666450796,
                        'lon_max':  -66.67902689406914
                        } ,


    
            'south_america':   {  
                                'lat_max': 13.592638486971083,
                                'lon_min': -84.0833129206595,
                                'lat_min': -53.43306775676489,
                                'lon_max':   -34.73304413246237
                                }
                     }


def get_current_time():

    print("\n")
    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y %H:%M:%S")
    print("Current Time =", current_time)
    print("\n")
    return None


class Country:
    
    def __init__(self,name = str,
                     database   = pd.DataFrame,
                     se_country = pd.DataFrame,
                     index_shape_geometry = None,
                     boundaries = None,
                     quantile_choosen = None,
                     windgust_threshold = None,
                     model_factor = list
                ):

        self.wind_threshold = windgust_threshold
        self.q_choosen = quantile_choosen
        self._model_factor = model_factor
        
        
        self.name = name
        self.database = database
        self.se = se_country
        self.index_shape_geometry = index_shape_geometry
        self.boundaries = boundaries


        root_folder = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Escalamiento"
       
        self.url_location = {
                            'windgust' : os.path.join(root_folder, r'01_WindGust\02_DataComplemento\01_CORDEX'),
                            'windgust_ERA5': os.path.join(root_folder, r'01_WindGust\02_DataComplemento\02_ERA'),
                            'tasmax' :  os.path.join(root_folder, r'02_Tasmax_CMIP6\02_DataComplemento'),
                            'wri' :  os.path.join(root_folder, r'03_WRI')
                            }

        self._recent_change_w = True
        self._recent_change_q = True
    
    def root_databases(self, variable = str,folder_name = str):
        
        """
        variable = [
                    "tasmax",
                    "windgust",
                    "wri",
                    "windgust_ERA5"
                    ]
        """
        
        dir_name = os.path.join(self.url_location[variable], folder_name)
        dir_country = os.path.join(dir_name, self.name)
        
        return dir_country
  
    def set_quantile(self, q = float):
        """
        q = New Quantile.
        """
        if self.q_choosen != q:
            self.q_choosen = q
            self._recent_change_q = True            
        else:
            self._recent_change = False

    def windgust_threshold(self, w = float):
        """
        W = Threshold .
        """
        
        if self.wind_threshold != w:
            self.wind_threshold = w
            self._recent_change_w = True
            
        else:
            self._recent_change = False
            
    @property
    def displayCountryProperties(self):
        
        num_towers = self.database.shape[0]
        print(f"Number of towers:\t{num_towers}\nCountry:\t{self.name}")
        
    def __repr__(self):
        return f'{self.name}'

############### Preparing assets. ###############

def raw_assets(version  = 2):
    
    """
    The purpose of this function is to reduce the amount of towers duplicated in our ISA assets database.
    The only thing we did was to group lines that comes with 2 circuits into one , and the drop the duplicates in the columns ["lat","lon","name"]

    Esta funcion devulve la base de datos con lso activos para Colombia.
    """

    
    path_assets = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\11_dataset_assets\01_Colombia\02_PowerLine\Dataset_v2"
    path_tran = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\11_dataset_assets\01_Colombia\02_PowerLine\Dataset_v1\Activos_Transelca"

    
    if version == 1:
            
        assets_complete = pd.read_csv(os.path.join(path_assets, 'Activos ITCO & Transelca.csv')  ,low_memory=False)
        patron_tran = r"(?P<name_linea>.+)?\s+\d\s+(34.5|\d{1,4}).*"
        patron_itco = r"(?P<name_linea>.+)\s+C?c?ircuito.+"
        
        match_name = re.compile(patron_itco)
        
        # Function to clean names based on regex pattern
        def clean_name(name,regex):
            is_match = regex.match(name)
            if is_match:
                return is_match.group('name_linea').strip()
            else:
                return name
        
        replacement_dict = {
                               'Bacatá Noroeste a  230 kV':'Bacatá Noroeste a 230 kV',
                                'Comuneros Primavera  a 230 kV':'Comuneros Primavera a 230 kV',
                                "Guatapé Jaguas a 230 kV": "Guatapé - Jaguas a 230 kV",
                                'Reforma Tunal  a 230 kV':'Reforma Tunal a 230 kV',
                                'Playas - Primavera a  230 kV' : 'Playas - Primavera a 230 kV'
                               }
        
        assets_complete.loc[: , 'linea_clean'] = assets_complete.get('linea_clean').fillna(assets_complete['Linea'],)
        assets_complete.loc[:, 'Linea'] =   assets_complete.loc[:, 'linea_clean'].apply(lambda x: clean_name( x,match_name)).replace(replacement_dict)
        assets_complete.drop(columns = 'linea_clean',inplace = True)
        
        assets_complete = assets_complete.drop_duplicates(subset = ["Latitud","Longitud","Linea"]).reset_index(drop = True)
        
        pick_cols = ['Torre',	'Latitud'	,'Longitud',	'Linea',	'CTE']
    
        filter_df = assets_complete.loc[:, pick_cols]
        filter_df = filter_df.reset_index(names = 'ID')

    elif version == 2:

        assets_itco = pd.read_csv(os.path.join(path_assets, 'Activos ITCO.csv') )
        assets_trans = pd.read_csv(os.path.join(path_tran, 'Activos Transelca.csv') )

        assets_itco = assets_itco.assign(**{"Empresa":'ITCO'})
        assets_trans = assets_trans.assign(**{"Empresa":'TCA'})
        
        filter_df = pd.concat([ assets_itco , assets_trans] , ignore_index = True)
        
    return filter_df


## def raw_itch():
## 
##     global parent
##     
##     ## Chile
##     url_chile = os.path.join(parent, '04_Chile')
##     chile_data = pd.read_csv(os.path.join( url_chile , "LT_Chile.csv"))
##     chile_se = pd.read_csv(os.path.join( url_chile , "SE_Chile.csv"))
## 
##     out_dict = {'LT' : chile_data ,
##                 'SE': chile_se
##                } 
## 
##     return out_dict


def load_activos(pais = str ,se = False):
    
    
    """
    Cargar todas las bases de datos de todos los paises.
    variable  = ['load_in_wri', 'nasa_gpdx' , 'cordex_wind']
    load_in_wri = bool
    """
    
    global boundaries,parent

    
    ## Colombia
    if pais == 'Colombia':
        
        data_towers = raw_assets()
        ### path_se_colombia = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Mision Volkmar\03_AqueductFloods\01_DataSE"
        ## path_se_colombia = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\4_Datasets_Location\01_Colombia\01_DataSE"
        path_se_colombia = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\11_dataset_assets\01_Colombia\01_DataSE"
        
        ### os.chdir(path_se_colombia)
        subestaciones = pd.read_csv(os.path.join(path_se_colombia, r"01_Subestaciones ITCO & ITCO.csv"))
        subestaciones = subestaciones[subestaciones['Longitud']   <= -66.29 ].reset_index(drop = True)

    elif pais == 'Bolivia':

        ## Bolivia.
        url_bolivia =  os.path.join(parent, '01_Bolivia')
        data_towers = pd.read_csv( os.path.join( url_bolivia , "LT_Bolivia.csv"))
    
        subestaciones = pd.read_csv("SE_Bolivia.csv")
        subestaciones = subestaciones.rename(columns= {'Longitud':'Latitud' , 
                                                        'Latitud':'Longitud'}
                                     )
    elif pais == 'Brasil':

        ## Brasil
        url_brasil =  os.path.join(parent, '02_Brazil')
        data_towers = pd.read_csv( os.path.join(url_brasil,   "LT_Brasil.csv"))
        data_towers = data_towers.sort_values(by = ['Linea']).drop_duplicates(subset = ['Longitud','Latitud'],keep = 'first').reset_index(drop = True)
        subestaciones = pd.read_csv( os.path.join(url_brasil, "SE_Brasil.csv"))

    elif pais == 'Peru':
        
        ## Peru
        url_peru =  os.path.join(parent, '03_Peru')
        os.chdir(url_peru)
        data_towers = pd.read_csv("LT_Peru.csv")
        data_towers = peru_data[peru_data['Longitud'] < -66.9948695889273].reset_index(drop = True)
        
        ## Drop outlier in the sea : 13580
        data_towers = peru_data.drop(13580).reset_index(drop = True)
        subestaciones = pd.read_csv('SE_REP.csv')

    elif pais == 'Chile':
        ## Chile
        url_chile = os.path.join(parent, '04_Chile')
        os.chdir(url_chile)
        data_towers = pd.read_csv("LT_Chile.csv")
        subestaciones = pd.read_csv("SE_Chile.csv")

    if se:
        out = {'data': data_towers,
                'se': subestaciones}
        return out
    else:
        return data_towers
    
    return subset

    
def load_assets(variable = str ):
    
    
    """
    Cargar todas las bases de datos de todos los paises.
    variable  = ['load_in_wri', 'nasa_gpdx' , 'cordex_wind']
    load_in_wri = bool
    """
    
    global boundaries,parent
    
    ### parent = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Escalamiento\KMZs\SouthAmerica_Datasets"
    ## parent = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\4_Datasets_Location\02_DemasFiliales\KMZs_SouthAmerica\SouthAmerica_Datasets"
    ### parent = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\11_dataset_assets\02_DemasFiliales\KMZs_SouthAmerica\SouthAmerica_Datasets"
    
    ## Colombia
    colombia_data = raw_assets()
    ### path_se_colombia = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Mision Volkmar\03_AqueductFloods\01_DataSE"
    ## path_se_colombia = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\4_Datasets_Location\01_Colombia\01_DataSE"
    path_se_colombia = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\11_dataset_assets\01_Colombia\01_DataSE"
    
    
    ### os.chdir(path_se_colombia)
    colombia_se = pd.read_csv(os.path.join(path_se_colombia, r"01_Subestaciones ITCO & ITCO.csv"))
    colombia_se = colombia_se[colombia_se['Longitud']   <= -66.29 ].reset_index(drop = True)
    
    ## Bolivia.
    url_bolivia =  os.path.join(parent, '01_Bolivia')
    os.chdir(url_bolivia)
    bolivia_data = pd.read_csv("LT_Bolivia.csv")

    bolivia_se = pd.read_csv("SE_Bolivia.csv")
    bolivia_se = bolivia_se.rename(columns= {'Longitud':'Latitud' , 
                                            'Latitud':'Longitud'}
                                 )
    
    
    ## Brasil
    url_brasil =  os.path.join(parent, '02_Brazil')
    os.chdir(url_brasil)
    brasil_data = pd.read_csv("LT_Brasil.csv")
    brasil_data = brasil_data.sort_values(by = ['Linea']).drop_duplicates(subset = ['Longitud','Latitud'],keep = 'first').reset_index(drop = True)
    brasil_se = pd.read_csv("SE_Brasil.csv")
    
    ## Peru
    url_peru =  os.path.join(parent, '03_Peru')
    os.chdir(url_peru)
    peru_data = pd.read_csv("LT_Peru.csv")
    peru_data = peru_data[peru_data['Longitud'] < -66.9948695889273].reset_index(drop = True)
    
    ## Drop outlier in the sea : 13580
    peru_data = peru_data.drop(13580).reset_index(drop = True)
    peru_se = pd.read_csv('SE_REP.csv')
    
    ## Chile
    url_chile = os.path.join(parent, '04_Chile')
    os.chdir(url_chile)
    chile_data = pd.read_csv("LT_Chile.csv")
    chile_se = pd.read_csv("SE_Chile.csv")


    datasets = {'Colombia': {'Boundaries':  boundaries['Colombia']  , 
                           'data': colombia_data,
                           'se': colombia_se,
                           'index_shape_geometries': 4
                                    },
                
              'Peru': {'Boundaries':  boundaries['Peru']  ,
                           'data': peru_data,
                           'se': peru_se ,
                           'index_shape_geometries': 11
                                      },
                
              
              'Brasil': {'Boundaries':  boundaries['Brasil'],
                          'data': brasil_data,
                          'se': brasil_se,
                          'index_shape_geometries': 2
                        },
              
              'Bolivia': {'Boundaries':  boundaries['Bolivia'] ,
                          'data': bolivia_data,
                          'se': bolivia_se,
                          'index_shape_geometries': 1
                         },
              
              
              'Chile': {'Boundaries':  boundaries['Chile'] ,
                        'data': chile_data,
                        'se': chile_se  ,
                        'index_shape_geometries': 3
                       }
                     }

    
    if variable == 'load_in_wri':
        keys_to_delete = ['index_shape_geometries']
        subset = datasets.copy()
        
        for pais , dict_variables in subset.items():
            for parameter in keys_to_delete:
                del dict_variables[parameter]
        
    elif variable == 'nasa_gpdx':
        
        keys_to_delete = ['se']
        subset = datasets.copy()
        
        for pais , dict_variables in subset.items():

            for parameter in keys_to_delete:
                del dict_variables[parameter]

    elif variable == 'cordex_wind':

        subset_main = datasets.copy()
        subset = { }
        
        for pais , dict_variables in subset_main.items():
            subset.setdefault(pais, dict_variables['data'])

    return subset



def define_boundaries(all_south_america  = False,**kwargs):
    
    global boundaries
    
    
    dataset = kwargs.get('dataset',False)
    
    holgura = 0.1/4
    
    if  isinstance(dataset,pd.DataFrame):
        
        min_boundaries = dataset.get(['Latitud','Longitud']).min().to_dict()
        max_boundaries = dataset.get(['Latitud','Longitud']).max().to_dict()
        min_lon,max_lon = min_boundaries.get('Longitud'), max_boundaries.get('Longitud')
        min_lat,max_lat = min_boundaries.get('Latitud'), max_boundaries.get('Latitud')

        
        paises_boundaries = {
                            'lat_max': max_lat + holgura,
                            'lon_min': min_lon - holgura,    
                            'lat_min': min_lat - holgura,
                            'lon_max': max_lon + holgura
                                }

    
    elif all_south_america:
        
        paises_boundaries = {"south_america": [boundaries['south_america']['lat_max'],
                                              boundaries['south_america']['lon_min'],
                                              boundaries['south_america']['lat_min'],
                                              boundaries['south_america']['lon_max']]}
                             
        print(f"Working on all South America {paises_boundaries['south_america']}")
        
    else:
        
        activos = load_assets(variable = 'nasa_gpdx')

        name_paises = activos.keys()

        paises_boundaries = { }

        for pais_name in name_paises:

            min_boundaries = activos[pais_name]['data'].get(['Latitud','Longitud']).min().to_dict()
            max_boundaries = activos[pais_name]['data'].get(['Latitud','Longitud']).max().to_dict()
            min_lon,max_lon = min_boundaries.get('Longitud'), max_boundaries.get('Longitud')
            min_lat,max_lat = min_boundaries.get('Latitud'), max_boundaries.get('Latitud')

            boundaries_copernicus = [max_lat + holgura,
                                     min_lon - holgura,
                                     min_lat - holgura,
                                     max_lon + holgura]

            paises_boundaries.setdefault(pais_name, boundaries_copernicus)

        for pais , limits in paises_boundaries.items():

            print(f"{pais}:\t{limits}")
        

    return paises_boundaries
                     

##### To The assigment of ID's ###################################

def getTimeSeries(xarray = None,
                  lon_axes = int,
                  lat_axes = int,
                  data_base = str
                 ):
    
    """
    wind_gust is being return in [km/h]
    Available variables = ['nasa_pr', 'nasa_tasmax','wind_gust','chirps']
    Returns a Series with the values based on the variables . 
    The index will be the time.
    """
    
        
    meta_data = {'nasa_pr': {'short_name':'pr','factor':86400},
                 'nasa_tasmax':{'short_name':'tasmax','factor': 1},
                 'wind_gust':{'short_name':'wsgsmax','factor':3.6},    
                 'chirps':{'short_name':'precip','factor': 86400}
                }


    short_name = meta_data[data_base]['short_name']
    factor = meta_data[data_base]['factor']     

    if data_base == 'wind_gust':
        raw_values = xarray.get(short_name).isel( rlon = lon_axes ,rlat = lat_axes ).values * factor
        timeSerie = xarray.get(short_name).isel(rlon = lon_axes ,rlat = lat_axes ).time.values
        
    else:
        raw_values = xarray.get(short_name).isel( lon = lon_axes ,lat = lat_axes ).values * factor
        timeSerie = xarray.get(short_name).isel(lon = lon_axes ,lat = lat_axes ).time.values


    rain_time_series = pd.Series(timeSerie).astype(str).str.strip().str.extract(r"(.+)\s+(\d+.+)").rename(columns = {0:'date',1:'hour'})
    time_series = pd.to_datetime(rain_time_series['date'].values , errors='coerce')
    pd_series = pd.Series(data = raw_values, index = time_series)
    final_pd_series = pd_series[pd_series.index.notnull()]

    return final_pd_series
        
def map_to_0_to_360(regular_longitude):
    
    """
    A good reference for this function 
    https://gis.stackexchange.com/questions/201789/verifying-formula-that-will-convert-longitude-0-360-to-180-to-180
    Sample Usage:
    
    map_to_0_to_360(regular_longitude= -75.4976)
    >> 284.50239999999997
    """
    
    converted_longitude = (regular_longitude + 360) % 360
    return converted_longitude

def convert_longitude_to_minus180_to_180(lon):
    
    """
    convert_longitude_to_minus180_to_180(lon =284.50239999999997)
    >>> -75.49760000000003
    """

    return np.mod(lon + 180, 360) - 180

def extractUniqueLocation(proyeccion_bd = None,
                           infrastructure = None,
                           isPRO = ['lon','lat'],
                           isISA = list,
                           name = None):

    """
    This function have been approved.
    date : 12/08/2024
    This function was onece again aprovved 18/01/2025 . Please do not review it again.
    """


    # Cordenadas con la infraestructura de ISA
    lon_array_isa = infrastructure[isISA[0]].values
    lat_array_isa = infrastructure[isISA[1]].values

    # Se les da una nueva forma para asi ser restadas ams adelante.
    lon_reshaped_isa = lon_array_isa[:, np.newaxis]
    lat_reshaped_isa = lat_array_isa[:, np.newaxis]

    # Se acomoda luego el vector de temperatura creando un vector de la forma array([[*,*,..* ]])
    lon_reshaped_cordex = proyeccion_bd[isPRO[0]].variable.values.reshape(1, -1)
    lat_reshaped_cordex = proyeccion_bd[isPRO[1]].variable.values.reshape(1, -1)

    # Con este lo que se hace es replicar el cetor con la latitud y longitud [[*],
    #                                                                         [*],
    #                                                                         [*]]

    lon_reshaped_cordex = np.tile(lon_reshaped_cordex  , (lon_reshaped_isa.shape[0], 1))
    lat_reshaped_cordex = np.tile(lat_reshaped_cordex  , (lat_reshaped_isa.shape[0], 1))

    # Con las formas adecuadas , estos ya se pueden sumar.
    lat_min_idx = np.argmin(np.abs(lat_reshaped_cordex - lat_reshaped_isa),axis = 1)
    lon_min_idx = np.argmin(np.abs(lon_reshaped_cordex - lon_reshaped_isa),axis = 1)


    ## Change made 9/01/2025. Its more explict the other way . More safe. This is one of the most old codes and a change will make great impact.
    ## ids_stations = ["{}_{}".format(x,y) for x, y in zip(lon_min_idx ,lat_min_idx)]
    ids_stations = [f"{x}_{y}" for x, y in zip(lon_min_idx ,lat_min_idx)]
    
    infrastructure[name] = ids_stations

    return infrastructure

def is_coordinate_within_bounds(lat = None, lon = None, bounds = None):
    
    """
    Check if a latitude and longitude coordinate falls within the bounds.

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        bounds (tuple): Bounding box coordinates (left, bottom, right, top).

    Returns:
        bool: True if the coordinate is within the bounds, False otherwise.
    """
    left, bottom, right, top = bounds
    if left <= lon <= right and bottom <= lat <= top:
        return True
    else:
        return np.nan

### Functions in charge

def regular_slice(xr_file:None,
                  country_bounds: None,
                  **kwargs
                 ):
        

    """
    5/02/2025


    
    this functions comes from 03_Escalamiento/Articulador/05_LandSlides/02_funciones/cumrain_sum/main_logic.py
    this was the one used at a moment to make the slices and create the cache for Colombia with respect to the rain.

    The old implementation was this way:
    regular_slice(xr_file:None,
                  bbox:tuple,
                  lon_name = str,
                  lat_name = str):

    7/02/2025 This functions has been verify !

    This is now outdated 

    ## lon_name = str,
    ## lat_name = str):
    """

    global boundaries

    columns_dict = kwargs.get('column_name_pro', False)

    if  columns_dict:

        lon_name = columns_dict['lon']
        lat_name = columns_dict['lat']
        
    else:

        lon_name = get_coord_name(xr_file, 'lon')
        lat_name = get_coord_name(xr_file, 'lat')
    
        
    column_name_pro = [lon_name,lat_name ]    ## isPRO = ['lon','lat']

    if isinstance(country_bounds,str):

        if country_bounds in boundaries.keys():

            bbox = boundaries[country_bounds]

            lon_min = bbox['lon_min']
            lon_max = bbox['lon_max']
            lat_min = bbox['lat_min']
            lat_max = bbox['lat_max']

        else:
            print("Not a country")
            return None

    elif isinstance(country_bounds,tuple):

        (lon_min,lon_max,lat_min,lat_max) = country_bounds


##     print("Sliced made at bounds:\n\n")
##     print(f"Min lon :{lon_min:.2f}")
##     print(f"Max lon :{lon_max:.2f}")
##     print(f"Min lon :{lat_min:.2f}")
##     print(f"Max lon :{lat_max:.2f}\n")

    xr_file = xr_file.sortby(column_name_pro ,ascending  = True)  # Sort in ascending order
    xr_file = xr_file.sel({lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)})

    return xr_file

## coord_options

# Helper function to get the correct coordinate name
def get_coord_name(xr_file:None, 
                   dim:str ):    


    """
    This function recibe a xarray file and we specify the dimension we want to get name of.
    This is a way to retrive the code of the xarray dinamically cause in some istances the code names come like 'longitud', 'longitude' ...
    
    
    lon_name = get_coord_name(ds, 'lon')
    lat_name = get_coord_name(ds, 'lat')  
    time_name = get_coord_name(ds, 'time')

    [...] lon_name -> 'longitude'
    
    """


    # Define possible names for coordinates
    coord_options = {
                        'lon': ['lon', 'longitude', 'rlon'],
                        'lat': ['lat', 'latitude', 'rlat'],
                        'time':['time','valid_time']
                    }

    for name in coord_options[dim]:
        if name in xr_file.coords:
            return name
    raise ValueError(f"No matching coordinate found in {options}")


"""
Function in charge of extracting the rows,cols form a xarray file.

Name of the functions :

1 . 

"""

# Function to map dictionary dimensions to numpy array shape
def map_coords_to_shape(shape, coords):
    dim_mapping = {}
    for key, size in coords.items():
        for i, dim_size in enumerate(shape):
            if size == dim_size:
                dim_mapping[key] = i
                break
    return dim_mapping



def extract_slice_tools( nc_file = xr.core.dataset.Dataset,
                        dataset = pd.DataFrame,
                        database_name = 'lon_lat',
                        column_isa = ['Longitud','Latitud'], 
                        variable_name = str,
                       dim_bool = bool):

    """
    database_name = 'id_nowcast'
    column_name_pro = ['lon', 'lat' ]    ## isPRO = ['lon','lat'],
    column_isa = ['Longitud','Latitud']
    variable_name = ['p_landslide' , 'pr']
    """
    
    dataset_copy = dataset.copy()

    lon_name = get_coord_name(nc_file, 'lon')
    lat_name = get_coord_name(nc_file, 'lat')
    time_name = get_coord_name(nc_file, 'time')

    column_name_pro = [lon_name,lat_name] 

    dataset_assigment = extractUniqueLocation(
                                            proyeccion_bd = nc_file,
                                            infrastructure = dataset_copy,
                                            name = database_name,
                                            isISA = column_isa,
                                            isPRO = column_name_pro
                                        )
    
    no_duplicates = dataset_assigment.drop_duplicates(subset=database_name).copy()
    no_duplicates.loc[:, ['lon', 'lat']] = no_duplicates.get(database_name).str.extract(r"(?P<lon>\d+)_(?P<lat>\d+)").astype(int)
    
    lon_slice = no_duplicates.lon.values
    lat_slice = no_duplicates.lat.values

    dict_coords = nc_file.coords.xindexes.dims.mapping
    numba_slice = getattr(nc_file, variable_name).to_numpy()

    
    multiple_models = False
    ## (# Modelos , #vector time, rows,cols)
    if len(numba_slice.shape) == 4:
        multiple_models = True


    if multiple_models:
        shape = numba_slice.shape[1:] 
    else:
        shape = numba_slice.shape

    
    dim_mapping = map_coords_to_shape(shape, dict_coords)

    slice_arrays = {
                    time_name: slice(None),
                    lat_name: lat_slice,
                    lon_name: lon_slice
                    }
    
    slc = [slice(None)] * len(shape)
    for key, array in slice_arrays.items():
        index_position = dim_mapping[key]
        slc[index_position] = array

    if multiple_models:
        slc = [slice(None)] + slc     
        
    slice_tuple = tuple(slc)
    cols_dict_db = list(no_duplicates.get(database_name).values)

    if  dim_bool:
        output = (slice_tuple, cols_dict_db, dataset_assigment  ,dim_mapping  )
    else:
        output = (slice_tuple, cols_dict_db, dataset_assigment )
        
    return output



              
        