"""
This are the functions to assign properties.

"""
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import rasterio as rio
from shapely.geometry import Point
from rasterio.windows import Window
import re
import matplotlib.pyplot as plt
import time


def extract_statistic(chunk_country, 
                      row_update,
                      col_update, 
                      n=3,
                      statistic='nanmax'):
    """
    Extracts a specified statistic from an nxn slice around each (row, col) pair in the arrays row_update and col_update.
    
    Parameters:
    - chunk_country: 2D numpy array containing the terrain data.
    - row_update: 1D numpy array of row indices.
    - col_update: 1D numpy array of column indices.
    - n: Size of the nxn window (default is 3).
    - variable: Name of the terrain variable (default is 'default').
    - statistic: Statistic to compute ('nanmax', 'nanmean', etc.).
    
    Returns:
    - List of computed statistics for each (row, col) pair.
    """


    amount_nans = 0
    
    if statistic:
        
        terrain_property = []

        for row, col in zip(row_update, col_update):
            
            # Define the boundaries for the nxn slice
            row_start = max(row - n//2, 0)
            row_end = min(row + n//2 + 1, chunk_country.shape[0])
            col_start = max(col - n//2, 0)
            col_end = min(col + n//2 + 1, chunk_country.shape[1])
            
            # Extract the 3x3 slice
            slice_nxn = chunk_country[row_start:row_end, col_start:col_end]
        
            raw_data = slice_nxn.data
            mask_data = slice_nxn.mask
    
            if mask_data.all():
         
                stat_value = np.nan
                amount_nans += 1
            
            elif mask_data.any():
                
                raw_data = raw_data.astype('float')
                raw_data[mask_data] = np.nan
                
                # Compute the specified statistic
                if statistic == 'max':
                    stat_value = np.nanmax(raw_data)
                elif statistic == 'mean':
                    stat_value = np.nanmean(raw_data)
                else:
                    raise ValueError(f"Unsupported statistic: {statistic}")
                    
            else:
                # Compute the specified statistic
                if statistic == 'max':
                    stat_value = np.max(raw_data)
                elif statistic == 'mean':
                    stat_value = np.mean(raw_data)
    
            terrain_property.append(stat_value)

    else:
       
        raw_data = chunk_country.data
        mask_data = chunk_country.mask

        single_mask = mask_data[row_update,col_update]
        terrain_property = raw_data[row_update,col_update]

        if single_mask.any():
            amount_nans += single_mask.sum()
            terrain_property = terrain_property.astype('float')
            terrain_property[single_mask] = np.nan

    return (terrain_property, amount_nans)
  


def general_assigment( 
                      latlon_df = pd.DataFrame,       
                      file_tif = str,
    
                      neat_name = None,
                      lon_name = 'Longitud',
                      lat_name = 'Latitud'
                       ):
    
    chunk_size= 10_000
    latlon_df_copy = latlon_df.copy()
    
    has_lat_lon = ~ latlon_df.get( [lon_name , lat_name]).isna().any(axis = 1)
    latlon_df_good = latlon_df_copy.loc[has_lat_lon]

    unique_index = latlon_df_good.index
    lat_ = latlon_df_good[lat_name].values
    lon_ = latlon_df_good[lon_name].values

    raw = rio.open(file_tif, masked=True) 
    dem_transform = raw.transform
    rows, cols = rio.transform.rowcol(dem_transform, lon_, lat_)

    min_row, max_row = rows.min() - 2 , rows.max() + 2 
    min_col, max_col = cols.min() - 2 , cols.max() + 2
    
    #List to store the windows
    windows = []
    
    # Iterate over the larger window and create smaller chunks
    for row_start in range(min_row, max_row , chunk_size):
        for col_start in range(min_col, max_col , chunk_size):
            
            row_end = min(row_start + chunk_size, max_row )
            col_end = min(col_start + chunk_size, max_col )
            window = Window.from_slices((row_start, row_end), (col_start, col_end))
             
            windows.append({
                            'window': window,
                            'row_off': window.row_off,
                            'col_off': window.col_off,
                            'height': window.height,
                            'width': window.width,
                            'points': None
                        })

    total_amount = len(windows)
    chunk_window = total_amount // 4

    if chunk_window == 0:
        chunk_window = 1
        
    # Create arrays for window boundaries
    row_offs = np.array([win['row_off'] for win in windows])
    col_offs = np.array([win['col_off'] for win in windows])
    heights = np.array([win['height'] for win in windows])
    widths = np.array([win['width'] for win in windows])

    # Calculate the end boundaries
    row_ends = row_offs + heights
    col_ends = col_offs + widths

    print(f"Size of the window :\t{chunk_size}\n")
    print(f"Amount of windows created {len(windows)}\n")
    
    # Use broadcasting to find the window for each point
    row_in_window = (rows[:, None] >= row_offs) & (rows[:, None] < row_ends)
    col_in_window = (cols[:, None] >= col_offs) & (cols[:, None] < col_ends)

    cross = row_in_window & col_in_window
    
    # Find the indices of the windows for each point
    window_indices = np.argmax(cross, axis=1)

    ## This code is to verify that each point
    verify_uniqueness = ( (cross).sum(axis = 1) == 1 ).all()
    total_amount_nan = 0
    
    if verify_uniqueness:
        print("All points belong to a unqiue window")

        df_to_group = pd.DataFrame({'idx_window': window_indices,
                                    'idx_latlon_df': unique_index,
                                    'rows': rows,
                                    'cols': cols
                                   })
        
        group_complete_window = df_to_group.groupby('idx_window')

        for idx_window ,df_window in group_complete_window:
            
            real_index = df_window.idx_latlon_df.values
            global_rows = df_window.rows.values
            global_cols = df_window.cols.values

            tuple_points = (real_index, global_rows, global_cols)
            windows[idx_window]['points'] = tuple_points
            
        amount_blanck_windows = 0

        for id_window,window in enumerate(windows,start = 1):
            
            crop_window = window['window']
            points = window['points']

            if id_window % chunk_window == 0:
                pct = (id_window / total_amount)*100
                print(f"\tProgress {pct :.2f} ...")

            if points is None:
                amount_blanck_windows += 1
                continue

            else:

                indices_window_i,rows_window_i,cols_window_i = points
                (row_start,row_end),(col_start,col_end) =  crop_window.toranges()
     
                row_update = rows_window_i - row_start
                col_update = cols_window_i - col_start
     
                chunk_country = raw.read(1, window = crop_window, masked=True )
   
                (terrain_property, amount_nans) = extract_statistic(chunk_country, 
                                                      row_update,
                                                      col_update, 
                                                      n =  3,
                                                      statistic= False)

                total_amount_nan += amount_nans
                latlon_df_copy.loc[indices_window_i,neat_name] = terrain_property

        delta_ = total_amount - amount_blanck_windows
        print(f"From the total amount of windows {total_amount} , the points are located in  {delta_}.\npct = {(delta_/total_amount) * 100:.2f}\n")
        print(f"Amount of nans : { total_amount_nan} ")  
 
    else:
        print("Theres points that belongs to more than 1 window")
        return None

    return latlon_df_copy


def calcular_terreno_puntos(
                                df,
                                ruta_dem,
                                nombre_pendiente='pendiente',
                                nombre_elevacion='elevacion_srtm',  # Nuevo parámetro para la columna de elevación
                                nombre_lon='Longitud',
                                nombre_lat='Latitud',
                                tam_ventana=5000,
                                solapamiento=2,
                                resolucion_constante=None
                            ):
    """
    Calcula elevación y pendientes en un DEM con enfoque de dos fases.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con coordenadas de longitud y latitud
    ruta_dem : str
        Ruta al archivo DEM en formato GeoTIFF
    nombre_pendiente : str
        Nombre de la columna para almacenar los valores de pendiente
    nombre_elevacion : str
        Nombre de la columna para almacenar los valores de elevación
    nombre_lon : str
        Nombre de la columna con longitudes
    nombre_lat : str
        Nombre de la columna con latitudes
    tam_ventana : int
        Tamaño de las ventanas en píxeles
    solapamiento : int
        Número de píxeles de solapamiento entre ventanas
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame original con las columnas de elevación y pendiente añadidas
    """
    print("Iniciando cálculo de elevación y pendientes con método de dos fases...")

    if resolucion_constante is not None:
        print(f"Usando resolución constante de {resolucion_constante} metros para todos los cálculos")
    else:
        print("Calculando resolución variable basada en la latitud de cada ventana")

    # Copiar el DataFrame
    df_copia = df.copy()
    
    points_with_no_slope = False
    amount_no_valid = 0
    
    tiempo_inicio = time.time()
    dem = rio.open(ruta_dem) 

    # Filtrar filas con coordenadas válidas
    filas_validas = ~df[[nombre_lon, nombre_lat]].isna().any(axis=1)
    df_valido = df_copia.loc[filas_validas]    
    latitudes = df_valido[nombre_lat].values
    longitudes = df_valido[nombre_lon].values

    # Transformar coordenadas a filas y columnas
    filas, cols = rio.transform.rowcol(dem.transform, longitudes, latitudes)
    no_valid_points_mask = (filas < 0) |  (cols < 0)

    if no_valid_points_mask.any():

        filas = filas[~no_valid_points_mask]
        cols = cols[~no_valid_points_mask]
        amount_no_valid = no_valid_points_mask.sum()
        points_with_no_slope = df_valido.loc[no_valid_points_mask]
        print(f"Theres {amount_no_valid} invalid points ")

        df_valido = df_valido.loc[~no_valid_points_mask]

    # Obtener índices y coordenadas
    indices_unicos = df_valido.index
  
    print(f"Procesando {len(indices_unicos)} puntos válidos...")    
    print(f"DEM cargado: {dem.width}x{dem.height} píxeles, CRS: {dem.crs}")

    # Determinar límites
    min_fila, max_fila = min(filas), max(filas) 
    min_col, max_col = min(cols), max(cols) 
    
    print(f"Rango de filas: {min_fila}-{max_fila}, columnas: {min_col}-{max_col}")
    
    # Obtener resolución en grados
    resx, resy = dem.res
    print(f"Resolución del DEM: {resx} x {resy} grados")

    # FASE 1: Crear ventanas núcleo sin solapamiento
    ventanas_nucleo = []
    
    for fila_inicio in range(min_fila, max_fila, tam_ventana):
        for col_inicio in range(min_col, max_col, tam_ventana):
            fila_fin = min(fila_inicio + tam_ventana, dem.height)
            col_fin = min(col_inicio + tam_ventana, dem.width)
            
            # Ventana sin solapamiento
            ventana = Window.from_slices((fila_inicio, fila_fin), 
                                        (col_inicio, col_fin))

            # Calcular resolución en metros (variable o constante)
            if resolucion_constante is None:
            
                # Calcular coordenadas del centro para conversión
                centro_fila = (fila_inicio + fila_fin) / 2
                centro_col = (col_inicio + col_fin) / 2
                lon_centro, lat_centro = rio.transform.xy(dem.transform, centro_fila, centro_col)
                
                # Calcular factores de conversión de grados a metros para esta latitud
                metros_por_grado_lat = 111320.0  # metros por grado de latitud
                metros_por_grado_lon = 111320.0 * np.cos(np.radians(lat_centro))
                
                # Convertir resolución de grados a metros
                metros_por_pixel_x = resx * metros_por_grado_lon
                metros_por_pixel_y = resy * metros_por_grado_lat

            else:
                # Usar valor constante proporcionado
                metros_por_pixel_x = resolucion_constante
                metros_por_pixel_y = resolucion_constante
                lat_centro = None  # No necesitamos guardar lat_centro con resolución constante
        
            
            ventanas_nucleo.append({
                                    'ventana': ventana,
                                    'fila_inicio': fila_inicio,
                                    'col_inicio': col_inicio,
                                    'altura': ventana.height,
                                    'ancho': ventana.width,
                                    'lat_centro': lat_centro,
                                    'metros_por_pixel_x': metros_por_pixel_x,
                                    'metros_por_pixel_y': metros_por_pixel_y,
                                    'puntos': None
                                })
    
    total_ventanas = len(ventanas_nucleo)
    print(f"Creadas {total_ventanas} ventanas núcleo sin solapamiento")
    
    # Crear arrays para límites de ventanas núcleo
    filas_inicio = np.array([v['fila_inicio'] for v in ventanas_nucleo])
    cols_inicio = np.array([v['col_inicio'] for v in ventanas_nucleo])
    alturas = np.array([v['altura'] for v in ventanas_nucleo])
    anchos = np.array([v['ancho'] for v in ventanas_nucleo])
    
    # Calcular límites finales
    filas_fin = filas_inicio + alturas
    cols_fin = cols_inicio + anchos
    
    # Usar broadcasting para encontrar la ventana para cada punto
    punto_en_fila = (filas[:, None] >= filas_inicio) & (filas[:, None] < filas_fin)
    punto_en_col = (cols[:, None] >= cols_inicio) & (cols[:, None] < cols_fin)
    
    # Puntos contenidos en cada ventana
    cross = punto_en_fila & punto_en_col
    
    # Verificar unicidad (ventanas disjuntas, cada punto debe estar en exactamente una ventana)
    verify_uniqueness = ((cross).sum(axis=1) == 1).all()

    not_window_at_all = np.sum((cross).sum(axis=1) == 0)
    
    if not verify_uniqueness:
        print("¡ADVERTENCIA! Hay puntos que no pertenecen a exactamente una ventana")
        print(f"  - {not_window_at_all} puntos no están en ninguna ventana")
        print(f"  - {np.sum((cross).sum(axis=1) > 1)} puntos están en múltiples ventanas")
        # Si hay puntos en múltiples ventanas con este enfoque, hay un problema de diseño
    else:
        print("Todos los puntos pertenecen a exactamente una ventana núcleo")
    
    # Obtener índice de ventana para cada punto
    window_indices = np.argmax(cross, axis=1)
    
    # Agrupar puntos por ventana
    df_to_group = pd.DataFrame({
                                'idx_window': window_indices,
                                'idx_latlon_df': indices_unicos,
                                'rows': filas,
                                'cols': cols
                            })
    group_complete_window = df_to_group.groupby('idx_window')
    
    # Asignar puntos a cada ventana
    for idx_window, df_window in group_complete_window:
        real_index = df_window.idx_latlon_df.values
        global_rows = df_window.rows.values
        global_cols = df_window.cols.values
        
        tuple_points = (real_index, global_rows, global_cols)
        ventanas_nucleo[idx_window]['puntos'] = tuple_points
    
    # Contar ventanas con puntos
    ventanas_con_puntos = sum(1 for v in ventanas_nucleo if v['puntos'] is not None)
    print(f"De {total_ventanas} ventanas, {ventanas_con_puntos} contienen puntos ({ventanas_con_puntos/total_ventanas*100:.2f}%)")
    
    # FASE 2: Procesar ventanas con puntos, expandiendo con solapamiento para cálculos
    total_nan_elevacion = 0
    total_nan_pendiente = 0
    ventanas_procesadas = 0
    
    # Inicializar columnas con NaN
    df_copia[nombre_pendiente] = np.nan
    df_copia[nombre_elevacion] = np.nan  # Nueva columna para elevación
    
    for i, ventana_nucleo in enumerate(ventanas_nucleo):
        if ventana_nucleo['puntos'] is None:
            continue
        
        ventanas_procesadas += 1
        if ventanas_procesadas % 10 == 0 or ventanas_procesadas == ventanas_con_puntos:
            print(f"Procesando ventana {ventanas_procesadas} de {ventanas_con_puntos} ({ventanas_procesadas/ventanas_con_puntos*100:.1f}%)")
        
        # Extraer información de la ventana núcleo
        indices, filas_ventana, cols_ventana = ventana_nucleo['puntos']
        
        # Expandir ventana con solapamiento para cálculos
        fila_inicio_exp = max(0, ventana_nucleo['fila_inicio'] - solapamiento)
        col_inicio_exp = max(0, ventana_nucleo['col_inicio'] - solapamiento)
        
        fila_fin_exp = min(ventana_nucleo['fila_inicio'] + ventana_nucleo['altura'] + solapamiento, dem.height)
        col_fin_exp = min(ventana_nucleo['col_inicio'] + ventana_nucleo['ancho'] + solapamiento, dem.width)
        
        # Definir ventana expandida para cálculos
        ventana_expandida = Window.from_slices((fila_inicio_exp, fila_fin_exp), 
                                              (col_inicio_exp, col_fin_exp))
        
        # Obtener factores de conversión para esta ventana
        metros_por_pixel_x = abs(ventana_nucleo['metros_por_pixel_x'])
        metros_por_pixel_y = abs(ventana_nucleo['metros_por_pixel_y'])
        
        # Leer datos de la ventana expandida (mantenemos los valores originales)
        datos_ventana = dem.read(1, window=ventana_expandida, masked=True)
        
        # Ajustar coordenadas de los puntos relativas a la ventana expandida
        filas_locales = filas_ventana - fila_inicio_exp
        cols_locales = cols_ventana - col_inicio_exp
        
        # Verificar límites válidos
        validos = (filas_locales >= 0) & (filas_locales < ventana_expandida.height) & \
                  (cols_locales >= 0) & (cols_locales < ventana_expandida.width)
        
        if not validos.all():
            print(f"¡ADVERTENCIA! {np.sum(~validos)} puntos fuera de los límites en la ventana {i}")
            total_nan_elevacion += np.sum(~validos)
            total_nan_pendiente += np.sum(~validos)
        
        # 1. EXTRAER ELEVACIÓN (valores originales)
        try:
            # Mantenemos los datos originales para la elevación
            if isinstance(datos_ventana, np.ma.MaskedArray):
                # Extraer valores solo para puntos válidos
                valores_elevacion = datos_ventana.data[filas_locales[validos], cols_locales[validos]]
                # Marcamos como NaN los puntos enmascarados
                mascara_elevacion = datos_ventana.mask[filas_locales[validos], cols_locales[validos]]
                
                # Para puntos enmascarados, no asignamos valor (queda NaN)
                if np.any(mascara_elevacion):
                    total_nan_elevacion += np.sum(mascara_elevacion)
                    df_copia.loc[indices[validos][~mascara_elevacion], nombre_elevacion] = valores_elevacion[~mascara_elevacion]
                else:
                    df_copia.loc[indices[validos], nombre_elevacion] = valores_elevacion
            else:
                # Si no está enmascarado, simplemente extraemos los valores
                valores_elevacion = datos_ventana[filas_locales[validos], cols_locales[validos]]
                df_copia.loc[indices[validos], nombre_elevacion] = valores_elevacion
        except Exception as e:
            print(f"Error al extraer elevación en ventana {i}: {e}")
            total_nan_elevacion += len(indices[validos])
        
        # 2. CALCULAR PENDIENTE (convertir a float)
        try:
            # Convertir a float para cálculos de pendiente
            if isinstance(datos_ventana, np.ma.MaskedArray):
                datos_array = datos_ventana.astype(np.float32)
                datos_array = np.ma.filled(datos_array, np.nan)
            else:
                datos_array = datos_ventana.astype(np.float32)
            
            # Calcular gradientes y pendiente
            dx, dy = np.gradient(datos_array, metros_por_pixel_x, metros_por_pixel_y)
            slope = np.sqrt(dx**2 + dy**2)
            slope_deg = np.degrees(np.arctan(slope))
            
            # Extraer valores de pendiente para puntos válidos
            valores_pendiente = slope_deg[filas_locales[validos], cols_locales[validos]]
            
            # Verificar NaN en pendiente
            nan_mask = np.isnan(valores_pendiente)
            if np.any(nan_mask):
                total_nan_pendiente += np.sum(nan_mask)
                
            # Asignar valores de pendiente al DataFrame
            df_copia.loc[indices[validos][~nan_mask], nombre_pendiente] = valores_pendiente[~nan_mask]
        except Exception as e:
            print(f"Error al calcular pendiente en ventana {i}: {e}")
            total_nan_pendiente += len(indices[validos])
    
    print(f"\nProcesamiento completado:")
    print(f"- Elevación: {total_nan_elevacion} puntos sin valor (NaN)")
    print(f"- Pendiente: {total_nan_pendiente} puntos sin valor (NaN)")
    print(f"- Total points with out value assign {not_window_at_all + amount_no_valid}")
    print(f"Tiempo total: {(time.time() - tiempo_inicio) / 60:.2f} minutos")
        
    return df_copia


class TerrainPlotter:

    """
    10/01/2025
    
    This class is in charge of plot every one of the 8 terrain properties that there are.
    Till this day only plot the 8 terrain porpoerties in a fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    """

    
    
    def __init__(self, tiff_paths, df):
        self.tiff_paths = tiff_paths
        self.df = df
        self.tiffs = [rio.open(path) for path in tiff_paths]


        self._lat_name = 'Latitud'
        self._lon_name = 'Longitud'

    
    def _process_dataframe(self, power_line_name,src ):
        filtered_df = self.df[self.df['Linea'] == power_line_name].copy()
        lon_min = filtered_df[self._lon_name].min() 
        lon_max = filtered_df[self._lon_name].max() 
        lat_min = filtered_df[self._lat_name].min() 
        lat_max = filtered_df[self._lat_name].max() 

        row_bottom, col_left = src.index(lon_min, lat_min)
        row_upper, col_right = src.index(lon_max, lat_max)

        delta_row = 30 

        # Define the window or subset of data you want to read
        window_slice = Window.from_slices((row_upper - delta_row, row_bottom +delta_row ), (col_left - delta_row, col_right + delta_row ))

        return filtered_df, window_slice

    def _normalize_raster(self, data):
        mean = np.mean(data)
        std = np.std(data)
    
        # Clip to 2 standard deviations
        vmin, vmax = mean - 2 * std, mean + 2 * std
        data_clipped = np.clip(data, vmin, vmax)
    
        # Normalize to 0–255
        data_normalized = ((data_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        return data_normalized

    def plot_terrain(self, power_line_name,var_name,cmapa = str):

        matcher = re.compile(r'.+\\(\w+)_Colombia\.tif$')

        filtered_df = None
        window_slice = None
        row_update = None
        col_update = None
        
        
        fig, axes = plt.subplots( figsize=(20, 10))

        
        for tiff_raw in self.tiffs:
            
            is_match = matcher.match(tiff_raw.name)
    
            catch_name = is_match.group(1)

            if var_name != catch_name:
                continue
            else:
                
                filtered_df, window_slice = self._process_dataframe(power_line_name = power_line_name,src = tiff_raw)
        
                dem_transform = tiff_raw.transform
        
                ## unique_index = filtered_df.index
                lat_ = filtered_df[self._lat_name].values
                lon_ = filtered_df[self._lon_name].values
            
                rows, cols = rio.transform.rowcol(dem_transform, lon_, lat_)
                (row_start,row_end),(col_start,col_end) =  window_slice.toranges()
                row_update = rows - row_start
                col_update = cols - col_start
                
                
                numba_slice = tiff_raw.read(1, window=window_slice)

                if var_name not in ['dem', 'slope']:
                    numba_slice = self._normalize_raster(numba_slice)
                    
                im = axes.imshow(numba_slice, cmap=cmapa,  vmin=np.min(numba_slice), vmax=np.max(numba_slice))
                axes.axis('off')
                axes.set_title(catch_name)

                # Add color bar
                cbar = fig.colorbar(im, ax=axes, orientation='vertical')
                cbar.set_label('Color Bar Label')  # You can customize the label
        
                # Plot points
                for row_, col_ in zip(row_update,col_update):
                    axes.plot(col_, row_, 'ro', markersize= 1)  # 'ro' for red dots
                    ### ax.annotate(power_line_name, (row_, col_), color='white', fontsize=8, ha='right')
        
        plt.show()

        return None

