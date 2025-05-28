import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
from tools_gdal import slice_for_country


def UnidadGeologica(latlon_df = None):
    
    """
    
    """
 
    # mapa geologico colombiano.
    ## mgc_2023 = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\06_InputGeologico\05_SGC2023\mgc2023.gdb"
    mgc_2023 = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\05_SGC2023\mgc2023.gdb"
    
    # Create geometry column using Shapely Point
    geometry = [Point(xy) for xy in zip(latlon_df['Longitud'], latlon_df['Latitud'])]

    # Convert DataFrame to GeoDataFrame
    gdf_activos = gpd.GeoDataFrame(latlon_df, geometry=geometry)
    gdf_activos = gdf_activos.drop(columns = ['Latitud','Longitud'])
    mapa_geologico = gpd.read_file(os.path.join( mgc_2023, 'a0000018f.gdbtable'))
    gdf_activos = gdf_activos.set_crs(epsg=mapa_geologico.crs.to_epsg())
    joined_df = gdf_activos.sjoin(mapa_geologico, how='left')
       
    latlon_df['Litologia'] = joined_df['SimboloUC'].str.extract(r"(?P<crono>.+)-(?P<Litologia>.+)").get("Litologia")
     
    return latlon_df

def PGA(latlon_df = None):
    
    """
    Peak Ground Acceleration: ....
    
    'crs': CRS.from_epsg(4326)
    BoundingBox(left=-180.0, bottom=-60.00000000000101, right=180.0, top=89.99)
    
        {
         'driver': 'GTiff',
         'dtype': 'float64',
         'nodata': 1.7976931348623157e+308,
         'width': 7200,
         'height': 3000,
         'count': 1,
         'crs': CRS.from_epsg(4326),
         'transform': Affine(0.05, 0.0, -180.0,0.0, -0.049996666666667, 89.99)
        }
        
    """
    send_copy = latlon_df.copy()
    url_ground_accelaration = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\06_InputGeologico\04_PGA"

    tiff_name = "v2023_1_pga_475_rock_3min.tif"
    name_column = "PGA"    
    complete_path = os.path.join(url_ground_accelaration, tiff_name)
    
    send_copy = slice_for_country(path = complete_path,
                                  latlon_df = send_copy,
                                  var_name = name_column,
                                  pais = 'Colombia',
                              )
    
    return send_copy

def nasa_suceptibility(latlon_df = None):
    
    """
    Suceptibility

    category of susptibility = [0, 1, 2, 3, 4, 5]
    
    {'driver': 'GTiff',
     'dtype': 'int8',
     'nodata': 127.0,
     'width': 43200,
     'height': 15841,
     'count': 1,
     'crs': CRS.from_epsg(4326),
     'transform': Affine(0.00833333333333333, 0.0, -180.0,
            0.0, -0.00833333333333333, 72.00006000333326)}
        
    """
    send_copy = latlon_df.copy()
    url_suseptibility = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\08_LandSlideCatalog\09_GlobalLandSlideSuseptibility"

    tiff_name = "global-landslide-susceptibility-map-2-27-23.tif"
    name_column = "suceptilibty"
    complete_path = os.path.join(url_suseptibility, tiff_name)
    send_copy = slice_for_country(path = complete_path,
                                  latlon_df = send_copy,
                                  var_name = name_column,
                                  pais = 'Colombia',
                              )
    
    return send_copy


def crop_regular_map(latlon_df:pd.DataFrame):

    latlon_df = nasa_suceptibility(latlon_df)
    latlon_df = PGA(latlon_df)
    latlon_df = UnidadGeologica( latlon_df)


    return latlon_df
