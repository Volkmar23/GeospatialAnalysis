

"""
This is a copy from the original one at 
-> src/landslide/ml/open_street.py

Referencia [1]

Inspirados del siguiente link
https://gis.stackexchange.com/questions/436938/calculate-length-using-geopandas

Vimos que de acuerdo a la ubicacion espacial los calculos de longitud pueden variar asi que aca muestran como calculan para una LineString la ubicacion de su centroid y depseus estiman su utc para
asi luego calcual la distacia que es muy similiar a la que arroja arcgis.

Aca tambien hacemos lo mismo pero con la funcion interna de .estimate_utm_crs.
Aunque se asume que el area no cambia , con respecto a la longitud si tomamos la ubicacion estimada de cada toore por que en funcion de su ubicacion el calculo de la longitud cambia .

"""

import os
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import geopandas as gpd
import sys
from pyproj import Geod
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import time
import re

from terrain import general_assigment

## This one works !!
def area_buffer(df_match:None):


    # Reproject to a projected CRS (e.g., EPSG:3857 or a local projection)
    gdf_projected = df_match.to_crs(epsg = 3857)

    # gdf_projected = df_match.to_crs(epsg = estimated_utm)

    # Pick a specific polygon (e.g., the first one)
    polygon = gdf_projected.iloc[0].geometry

    # Compute the area in square kilometers
    area_km2 = polygon.area / 1e6  # Convert from square meters to square kilometers
    
    return area_km2

# Create a Geod object using the WGS84 ellipsoid

geod = Geod(ellps="WGS84")      # Con esta luego se calcula las distancias entre la torre y el punto mas cercano al shape.


def assigment_road_river(gpd_cache_copy ):

    """
    <Geographic 2D CRS: EPSG:4326>
    Name: WGS 84
    Axis Info [ellipsoidal]:
    - Lat[north]: Geodetic latitude (degree)
    - Lon[east]: Geodetic longitude (degree)
    Area of Use:
    - name: World.
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984 ensemble
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich
    """

    file_tif_dem = r'C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\01_opentopography_SRTMGL1\01_DEM\Colombia\dem_Colombia_SRTM.tif'     
    computed_area = False

    #### 'gis_osm_waterways_free_1.shp', 'gis_osm_water_a_free_1.shp'
    riverway_path = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\09_Roads\Colombia\colombia-latest-free.shp\gis_osm_waterways_free_1.shp"
    open_street = gpd.read_file(riverway_path)

    open_street = open_street.loc[open_street.fclass == 'river']
    
    id_column = 'ID'

    ## Relaziamos la interseccion 
    join_variable = gpd.sjoin(left_df = open_street,
                               right_df= gpd_cache_copy,
                               how="right",                              ## Me muestra todo lo del lado derecho , por enede me dice que torre no esta dentro del radio.
                               predicate="intersects")

    join_variable = (
                    join_variable.dropna(subset = 'index_left')               ## Como el join fue de intersection y ademas right , botamos aquellos rios o vias que no cruzaron a alguna torre.
                                  .sort_values(by = id_column)               ## Ordenamos la tabla de menor a mayor para eventualmente agrupar.
                                    .astype({id_column:'category',           ## Trasfromamos por facilidad la id_torre (identificador unico de una torre a categoria)
                                            'index_left':int})                ## Vimos que cuando cruzabamos la tabla indice de los shape venia en un formato float ,asi que la pasamos a entremso por que eventualment usaremos .loc[ ] con ellos . Solo para aseguranos que estamos fiultrando con un entero y no un flaat.
                    )



    ## Aca esta lo calve y es que una vez ordenada la tabal, tendiramos agrupados cada torre con su conjunto de rios que cruzo su radio.Cada torre tiene al menos 1 rio o via que lo cruza. Esto esta asegurado desde el join.
    group_merged = join_variable.groupby(id_column,observed=True)

    list_that_cross = [ ]

    ## Basta con iterar sobre el groupo. 
    for id_tower , df_match in group_merged:

        ## Donde almacenae la interseccion de los shapes que cruzan el buffe de la torre.
        list_geoms_sections = [ ]
        list_ids_sections = []  # <--- Create a parallel list for IDs
        min_distance = np.inf
        punto_rio_mas_cercano = None

        ## Se asume que con calcular el area una sola vez para las dos variables es suficinete. Cada torre tendra el mismo valor de area.
        if not computed_area:
            
            area_buff = area_buffer(df_match)
            print(f"Area {area_buff:.2f}")
            computed_area = True

        ## Extrameos del df_match la geometria del buffer.
        buffer_geom = df_match.iloc[0].geometry

        ## Esto es para luego tomar las distancias del punto mas cercano del rio a la torre. Aca despojo el buffer y solo queda el punto desnudo.
        centroid_buff = buffer_geom.centroid  
        
        # Referencia [1]
        estimated_utm = df_match.estimate_utm_crs() #Estimate a utm crs   , 
        index_left = df_match.get('index_left').values
        df_open_street = open_street.loc[index_left]

        for idx,  row in df_open_street.iterrows():

            river_geometry = row['geometry']
            unique_id = row['osm_id']
            
            # Compute the intersection
            intersection = river_geometry.intersection(buffer_geom)

            nearest_point_river, _ = nearest_points(river_geometry,centroid_buff)
            distance = geod.geometry_length(LineString([(nearest_point_river.x,nearest_point_river.y),centroid_buff]))

            if distance < min_distance:

                punto_rio_mas_cercano = nearest_point_river
                min_distance = distance

            list_geoms_sections.append(intersection)
            list_ids_sections.append(unique_id)  # <--- Save the id!

        resultado = (id_tower  , punto_rio_mas_cercano.x , punto_rio_mas_cercano.y,min_distance )
        list_that_cross.append(resultado)

    merge_along = [ id_column ,'Longitud_rio',	'Latitud_rio' ,  'min_distance' ]
    open_stret_df = pd.DataFrame(list_that_cross, columns = merge_along  )

    open_stret_df = general_assigment(latlon_df = open_stret_df,
                                      file_tif = file_tif_dem,
                                      neat_name = 'dem_river',
                                      lon_name = 'Longitud_rio',
                                      lat_name = 'Latitud_rio')
    
    gpd_cache_copy = pd.merge(left = gpd_cache_copy , right = open_stret_df ,on = id_column, how = 'left')
    gpd_cache_copy.loc[: , merge_along] = gpd_cache_copy.loc[:, merge_along ].astype(float).fillna({'min_distance': 1_000 })
    gpd_cache_copy.drop(columns = 'geometry',inplace = True)

    return gpd_cache_copy




















