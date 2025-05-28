"""
Modulo original de riesgos para inundaciones
"""
import sys
path_tools = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\utils"

if path_tools not in sys.path:
    sys.path.append(path_tools)

import os
import numpy as np
import pandas as pd
import geopandas as gpd 
import re
from .rain_way import rain_pipe_chirps
from .wranggling import load_inventario
from .openstreet import assigment_road_river
from .other_variables import crop_regular_map
from tools_gdal import generated_buffer
from terrain_assigment import calcular_terreno_puntos,general_assigment
from tools import get_current_time 



def compute_distances(path_buffer_storage:str):

    ## path_buffer_storage = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\cache_results"
    regex_pattern = r"inventario_(?P<radius>\d+)m.shape"
    unit_file = r".+\.shp$"
    
    catch_name = re.compile(unit_file)
    match_shp = re.compile(regex_pattern)
    
    shapes_radius = [txt for txt in os.listdir(path_buffer_storage) if match_shp.match(txt) ]
    
    for radius in shapes_radius:

        radio_meters = match_shp.match(radius).group('radius')
    
        path_radius = os.path.join(path_buffer_storage, radius)
        files_shp = os.listdir(path_radius)
    
        for file in files_shp:
    
            if catch_name.match(file):
    
                final_path = os.path.join(path_radius ,file)
                buffer_gdp = gpd.read_file(final_path)

                river_assigment = assigment_road_river(gpd_cache = buffer_gdp,  buffer_radius = path_radius,id_column = 'ID')
                river_assigment.to_csv( os.path.join( path_buffer_storage, f"Invenatario_{radio_meters}_COMPUTED.csv") ,index = False)
                
            else:
                continue

    return None


def create_dataset(path_buffer_storage:str ):


    """
    ['ID', 'Year', 'Month',
    'Day', 'Country', 'Region',
    'Department','Municipality', 'Place',
    'Site', 'Latitud', 'Longitud', 'Type',
    'uncertainty', 'Cause', 'Fatalities',
    'losses', 'Source 1', 'Source 2',
     'Source 3', 'Source 4', 'add', 'triggering_description',
     'EFFECT OBSERVATIONS', 'Magnitud', 'subregion']
    """



    ## path_buffer_storage = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\cache_results"
    print(f"Amount of files {len(os.listdir(path_buffer_storage))}\n{os.listdir(path_buffer_storage)}")

    path_colombia_dem = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\02_Alos_Palsar\dem_COL\terrain_WGS84\dem_Colombia.tif"
    # Rutas de datos de terreno
    path_colombia_slope = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\02_Alos_Palsar\dem_COL\terrain_WGS84\slope_Colombia.tif"
    # Ejemplo de uso
    ruta_dem = r'C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\01_opentopography_SRTMGL1\01_DEM\Colombia\V2\dem_Colombia_SRTM.tif'

    get_current_time()
    resultado = load_inventario( verbose=  False  )
    
    safe_points = pd.read_csv(r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\cache_results_v2\SafePoints.csv")
    ## All the computations are being made with the data with incidents.
    inventario = resultado['incidentes_withdates']
    
    slice_inventario = inventario.loc[: , [ 'ID', 'Latitud', 'Longitud']]
    duplicated = slice_inventario.ID.duplicated().any()
    
    if duplicated:
        return None
        
    start_range_non = slice_inventario.ID.max() +1 
    safe_points['ID'] = range(start_range_non, start_range_non + len(safe_points))
    slice_inventario['label'] = 1
    safe_points['label'] = 0
    
    dataset_assigment = pd.concat([slice_inventario, safe_points], axis=0, ignore_index=True)

    print(f"Amount points {dataset_assigment.shape}\n")

    dataset_assigment = rain_pipe_chirps(dataset_assigment)
    print("=="*10 +" End with rain " +"=="*10) 

    
    id_identifiers = 'ID'
    name_file = "inventario_{buffer}m.shape"
    regex_pattern = r"inventario_(?P<radius>\d+)m.shape"

    match_compile = re.compile(regex_pattern)
    
    dataset_assigment = crop_regular_map(latlon_df=dataset_assigment)
    dataset_assigment = general_assigment(latlon_df = dataset_assigment,
                                          file_tif = path_colombia_dem,
                                          neat_name = 'elevacion_alos')
    
    dataset_assigment = general_assigment(latlon_df = dataset_assigment,
                                          file_tif = path_colombia_slope,
                                          neat_name = 'slp_alos')

    print("\n\n")
    # Calcular pendientes con método simplificado
    dataset_assigment = calcular_terreno_puntos(
                                                dataset_assigment,
                                                ruta_dem,
                                                nombre_pendiente='slp_srtm',
                                                tam_ventana=5000,
                                                solapamiento=2,
                                                resolucion_constante=None
                                                )
                                    
    dataset_assigment['Pendiente_12_5m'] = dataset_assigment['slp_alos'].fillna(dataset_assigment['slp_srtm'])
    dataset_assigment['Elevacion_12_5m'] = dataset_assigment['elevacion_alos'].fillna(dataset_assigment['elevacion_srtm'])

    dataset_assigment.to_csv(os.path.join( path_buffer_storage, "Inventario_raw.csv") ,index = False)

    RAW_COORDS = dataset_assigment.loc[: ,['ID',  'Latitud', 'Longitud']]
    folder_storage = os.listdir(path_buffer_storage)
    amount_gpd = len([match_compile.match(var) for var in folder_storage])

    list_radius = [ 1000 ]
        
    generated_buffer(latlon_df = RAW_COORDS,
                     buffer_distance = list_radius,   ##  [metros]
                     epsg_proj = 4326,
                     new_name_column = id_identifiers,
                     name_pattern = name_file,
                     path_storage = path_buffer_storage
                    )

    compute_distances(path_buffer_storage)

    print("Done with the assigment !! ")
    return None

