
"""
In construction ...MAITANCE.



"""






import os
import sys


path_tools = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\src\utils"
func_landslide = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\src\landslide\ml"

path_to_add = [path_tools , func_landslide]

for path in path_to_add:
    # Add paths to sys.path if not already present
    if path not in sys.path:
        sys.path.append(path)
        
import pandas as pd
import re
import geopandas as gpd
import tools
import matplotlib.pyplot as plt
from rain_way import rain_test

## Comes from tools
from tools_gdal import generated_buffer

## Comes from the functions used in Landslides.
## from open_street import assigment_road_river
## from terrain import general_assigment
from terrain_assigment import general_assigment



name_id = 'ID'

def name_radius_tuple(path:str,regex_pattern :None):

    lista_return = [ ]
    for variable in os.listdir(path):
        is_match = regex_pattern.match(variable)
        if is_match:
            radius = is_match.group('radius')
            lista_return.append( (variable , radius) )
        else:
            continue
    return lista_return

def get_max_index(path_sampling):
    # Define the regex pattern
    re_idx_pattern = re.compile(r'raw_(?P<number>\d+)')
    
    # Get the list of files in the directory
    current_amount = os.listdir(path_sampling)
    
    # Initialize max_value to None
    max_value = None
    
    # Iterate over the list of files
    for file_name in current_amount:
        match = re_idx_pattern.search(file_name)
        if match:
            number = int(match.group('number'))
            if max_value is None or number > max_value:
                max_value = number
    
    # If no files matched the pattern, set idx to 1
    if max_value is None:
        idx = 1
    else:
        idx = max_value + 1
    
    return idx

def main_pipe_line(df = pd.DataFrame,
                   path_sampling = str):

    global name_id

    
    final_sample_copy = df.copy()  ## bug.
    unit_name = 'raw_{idx}'
    road_river_distance = 'river_road_distance_{radius}{idx}.csv'
    terrain_name  = 'terrain_{idx}.csv'

    print(f"Amount of files {len(os.listdir(path_sampling))}\n")
    idx_new = get_max_index(path_sampling)

    print(f"Creating Cache number {idx_new}\n")

    buffer_distances = [500, 1500] ## units [meters]
    choosen_projection = 4326
    

    tab = "########" * 5

    buffer_name = unit_name.format(idx = idx_new) + '_{buffer}m.shp'    

    final_sample_copy.to_csv(os.path.join(path_sampling, unit_name.format(idx = idx_new) + '.csv'),index = False)

    print(f"{tab} 1. Generando buffers {tab} [...]")
    generated_buffer(latlon_df = final_sample_copy,
                    buffer_distance = buffer_distances,
                    epsg_proj = choosen_projection,
                    name_pattern =  buffer_name,
                    new_name_column = name_id,
                    path_storage = path_sampling)

    matcher_shape = re.compile(r"raw_{idx}_(?P<radius>\d+)m.shp".format(idx = idx_new) )
    name_radius = name_radius_tuple(path_sampling , matcher_shape)

    print(f"{tab} 2. Generando distancias {tab} [...]")
    for partial_name, radius in name_radius:

        print(f"Radius \t: {radius} km")
        
        buffer_gpd_file = gpd.read_file(os.path.join(path_sampling, partial_name))
        computed_river_road = assigment_road_river(buffer_gpd_file, radius, name_id)
        computed_river_road.drop(columns = 'geometry',inplace = True)
        complete_name_road = os.path.join(path_sampling , road_river_distance.format(radius = radius , idx = idx_new))
        computed_river_road.to_csv(complete_name_road,index = False)
        print("\n")## 

    ## Clear memory .
    computed_river_road = None
    
    print(f"{tab} 3. Starting terrain Assigment. {tab} [...]")
    complete_terrain = os.path.join(path_sampling, terrain_name.format(idx = idx_new ))
    activos_dem = general_assigment( pais =  'Colombia',latlon_df = final_sample_copy)

    # Forward fill NaN values within each category
    activos_dem['slope'] = activos_dem.groupby('Linea')['slope'].ffill()
    activos_dem['curvature'] = activos_dem.groupby('Linea')['curvature'].ffill()
    activos_dem['dem'] = activos_dem.groupby('Linea')['dem'].ffill()

    # Normalize using Min-Max Scaling
    activos_dem['slope_normalized'] = (activos_dem['slope'] - activos_dem['slope'].min()) / (activos_dem['slope'].max() - activos_dem['slope'].min())

    
    activos_dem.to_csv(complete_terrain,index = False)
    
    return None 

def create_training_set(idx = int, 
                        radius = 500,
                       path_sampling = str):

    global name_id
    print("Generando la data principal:\n")
    
    df_unit = pd.read_csv(os.path.join(path_sampling,f'raw_{idx}.csv'))
    terrain_df  = pd.read_csv( os.path.join(path_sampling, f'terrain_{idx}.csv' ))
    road_river_distance = pd.read_csv( os.path.join(path_sampling,f'river_road_distance_{radius}{idx}.csv'))
    
    dfs_to_concat = [terrain_df,road_river_distance ]
    merged_df = None

    for idx, df in enumerate(dfs_to_concat,start = 1):

        if idx == 1:
            set_df_unit = set(df_unit.columns)
            new_cols = set(df.columns)
            filter_cols = list(new_cols.difference(set_df_unit)) + [name_id  ]
            merged_df = pd.merge(left = df_unit , right = df[filter_cols] , on = name_id )

        else:
            set_df_unit = set(merged_df.columns)
            new_cols = set(df.columns)
            filter_cols = list(new_cols.difference(set_df_unit)) + [name_id  ]

            merged_df = pd.merge(left = merged_df , right = df[filter_cols] , on = name_id   )


    return merged_df









