import pandas as pd
import geopandas as gpd
# Fix for KMeans memory leak warning on Windows - MUST be before imports
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Using the suggested value from the warning
import seaborn as sns
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point




def plots_cloroplets(areas = gpd.GeoDataFrame,
                    name_output = 'Municipio'):

    path_storage = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\Graficas\PatternPoints"
    
    # Ensure your GeoDataFrame is in the right CRS for web mapping
    # If your data is in a different CRS, reproject to EPSG:3857 (Web Mercator)
    areas_proj = areas.to_crs(epsg=3857)
    
    # Set up figure and axis
    f, ax = plt.subplots(1, figsize=(12, 10))
    
    # Plot the choropleth map
    areas_proj.plot(column='cuenta_incidentes', 
                    scheme='quantiles',  # Better contrast - divides data into equal-sized groups
                    ax=ax,
                    legend=True,
                    cmap='OrRd',  # Good for showing intensity - you could also try 'Reds', 'YlOrRd', 'plasma'
                    edgecolor='white',  # White edges often look cleaner
                    linewidth=0.5,
                    k=5,  # Number of classes - adjust as needed
                    legend_kwds={"loc": "upper left", 
                               "bbox_to_anchor": (1, 1),
                               "title": "Incidentes de\nDeslizamiento"}
                   )
    
    
    # Add OpenStreetMap basemap
    ctx.add_basemap(ax, 
                    crs=areas_proj.crs,  # Use the projected CRS
                    source=ctx.providers.OpenStreetMap.Mapnik,  # Standard OSM
                    alpha=0.7)  # Make basemap slightly transparent
    
    # Customize the plot
    ax.set_title("Mapa de Incidentes de Deslizamientos en Colombia", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Optional: Remove axes for cleaner look
    # ax.set_axis_off()
    
    # Or keep axes but make them cleaner
    ax.set_xlabel("Longitud", fontsize=12)
    ax.set_ylabel("Latitud", fontsize=12)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Show the map
    plt.show()
    
    # Optional: Save the map
    plt.savefig(os.path.join(path_storage , f'mapa_deslizamientos_{name_output}.png'), dpi=300, bbox_inches='tight')




def create_cloropltes(inventario_gdf: gpd.GeoDataFrame):
    

    path_municipio_colombia = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\databases\Municipios\Municipios.shp"
    path_departamentos_colombia = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\databases\Servicio-609\Departamentos_Abril_2025_shp\Departamento.shp"
    
    municipios = gpd.read_file(path_municipio_colombia ,)
    departamentos = gpd.read_file(path_departamentos_colombia ,)
    
    # Asignar un sistema de referencia espacial (CRS), por ejemplo WGS84
    departamentos.to_crs(epsg=4326, inplace=True)
    # Asignar un sistema de referencia espacial (CRS), por ejemplo WGS84
    municipios.to_crs(epsg=4326, inplace=True)
    
    # Leer el GeoDataFrame de puntos (deslizamientos)
    # Asegúrate de que gdf esté definido y tenga geometría tipo Point
    # gdf = gpd.read_file("ruta_a_tus_puntos.shp").to_crs(epsg=4326)
    
    shapes = [('municipio', municipios ),('departamento', departamentos)  ]
    
    output = { }
    
    for tipo,shp in shapes:
    
        if tipo  == 'municipio':
            identifier = 'OBJECTID'
        else:
            identifier = 'DeCodigo'
            
        join_areas = gpd.sjoin(inventario_gdf, shp[[identifier, 'geometry']], how="left", predicate='intersects')
     
        empety_join =  join_areas.index_right.isna()   
        if empety_join.any():
            remaining_rows = empety_join.sum()
        
            print(f"{tipo} - No hubo con {remaining_rows} incidentes del inventario de movimento en masas.")
        
        conteo = join_areas.groupby(identifier).size().reset_index(name='cuenta_incidentes')
        shp = shp.merge(conteo , on = identifier)
        output.setdefault(tipo , shp)


    for tipo,shp in output.items():

        plots_cloroplets(areas = shp, name_output = tipo)





def create_joint_plot(inventario_gdf:gpd.GeoDataFrame):


    
    # Convert to Web Mercator for basemap compatibility
    inventario_proj = inventario_gdf.to_crs(epsg=3857)
    
    # Set better style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Generate improved scatter plot with map background
    g = sns.jointplot(
        x=inventario_proj.geometry.x,  # Use projected coordinates
        y=inventario_proj.geometry.y, 
        data=None,  # We're using the coordinates directly
        kind='scatter',           
        s=12,                     # Slightly larger for visibility over map
        alpha=0.8,               # More opaque over map background
        color='red',             # Strong color that stands out on map
        height=12,               # Larger to accommodate map details
        ratio=4,                 # Adjust ratio for better proportions
        marginal_kws=dict(       
            bins=40, 
            fill=True, 
            alpha=0.7,
            color='red'
        ),
        joint_kws=dict(          
            edgecolors='darkred',    # Dark edge for better contrast
            linewidth=0.5,
            rasterized=True      
        )
    )
    
    # Add basemap to the main scatter plot
    ctx.add_basemap(g.ax_joint, 
                    crs='EPSG:3857',
                    
                source=    ctx.providers.CartoDB.Voyager,
                #    source=ctx.providers.OpenStreetMap.Mapnik,  # You can change this
                    
                    alpha=0.7,  # Slightly transparent so points stand out
                    zoom='auto')
    
    # Improve labels and title  
    g.set_axis_labels('Longitud', 'Latitud', fontsize=14)
    g.fig.suptitle('Distribución Espacial de Deslizamientos en Colombia', 
                   fontsize=18, fontweight='bold', y=0.98)
    
    # Add subtle grid
    g.ax_joint.grid(True, alpha=0.2, linestyle='--', color='white', linewidth=0.5)
    
    # Customize ticks - format large numbers nicely
    g.ax_joint.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    g.ax_joint.tick_params(labelsize=11)
    
    # Add a legend/info box
    info_text = f'Total: {len(inventario_proj):,} deslizamientos'
    g.ax_joint.text(0.02, 0.98, info_text, 
                   transform=g.ax_joint.transAxes, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                   fontsize=12, verticalalignment='top')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Alternative basemap options you can try:
    # ctx.providers.CartoDB.Positron      # Clean, minimal
    # ctx.providers.CartoDB.Voyager       # Good contrast
    # ctx.providers.Esri.WorldImagery     # Satellite imagery
    # ctx.providers.OpenStreetMap.Mapnik  # Standard OSM

    

def kdensit_points(df):


    # === Crear GeoDataFrame ===
    
    # Asegúrate de tener columnas llamadas 'longitude' y 'latitude'
    df = df.dropna(subset=['Longitud', 'Latitud'])  # Eliminar filas con NaN
    
    # Crear geometría y transformar a Web Mercator
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['Longitud'], df['Latitud']),
        crs='EPSG:4326'  # Sistema de coordenadas geográficas
    ).to_crs(epsg=3857)  # Proyección compatible con contextily
    
    # Extraer coordenadas proyectadas
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    
    # === Submuestreo ===
    
    sample_size = 5000
    if len(gdf) > sample_size:
        gdf_sample = gdf.sample(n=sample_size, random_state=42)
    else:
        gdf_sample = gdf
    
    # === KDE Plot ===
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Gráfico KDE
    sns.kdeplot(
                x=gdf_sample['x'],
                y=gdf_sample['y'],
                levels=25,
                fill=True,
                alpha=0.6,
                cmap='viridis_r',
                ax=ax
            )
    
    # Añadir mapa base
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        crs=gdf.crs.to_string()
    )
    
    # Ajustar aspecto para evitar distorsión
    ax.set_aspect('equal')
    
    # Mejorar visual
    ax.set_title("Densidad de deslizamientos (KDE)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()




    



