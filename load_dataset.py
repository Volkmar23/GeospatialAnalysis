import geopandas as gpd
from pathlib import Path
import json

def restore_original_columns(gdf, mapping_path):
    """
    Restore original column names using the mapping dictionary.
    """
    # Load mapping
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Create reverse mapping (encoded_name -> original_name)
    gdf_restored = gdf.copy()
    
    # Rename columns back to original names
    for encoded_name, original_name in mapping.items():
        if encoded_name in gdf_restored.columns:
            gdf_restored = gdf_restored.rename(columns={encoded_name: original_name})
    
    return gdf_restored

def load_shapefile_with_original_names(shapefile_path, mapping_path=None):
    """
    Load shapefile and restore original column names.
    """
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Determine mapping path if not provided
    if mapping_path is None:
        mapping_path = str(Path(shapefile_path).with_suffix('.json'))
    
    # Restore original column names
    gdf_restored = restore_original_columns(gdf, mapping_path)
    
    return gdf_restored

# Later, load back with original names
gdf_restored = load_shapefile_with_original_names(r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\GEE_Landslide_Pipelines\main_training_set.shp")