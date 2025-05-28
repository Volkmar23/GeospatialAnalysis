import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Límites precisos de Colombia definidos por el usuario
col_boundaries_v2 = {
    'lat_max': 12.590276718139648,
    'lon_min': -81.72014617919922,    
    'lat_min': -4.236873626708984,
    'lon_max': -66.87045288085938
}

def validar_incidentes_deslizamiento_colombia(incidentes_df, verbose=True):
    """
    Valida las coordenadas de incidentes de deslizamiento usando los límites precisos de Colombia
    
    Parameters:
    -----------
    incidentes_df : pandas.DataFrame
        DataFrame con columnas 'Longitud' y 'Latitud' de incidentes de deslizamiento
    verbose : bool, default=True
        Si True, imprime información detallada del proceso
    
    Returns:
    --------
    tuple: (mask_dentro_colombia, incidentes_colombia, incidentes_fuera, estadisticas)
    """
    
    if verbose:
        print("COLOMBIA " + "="*60 + " COLOMBIA")
        print("  VALIDACION DE INCIDENTES DE DESLIZAMIENTO EN COLOMBIA")
        print("COLOMBIA " + "="*60 + " COLOMBIA")
        
        print(f"\nLIMITES GEOGRAFICOS USADOS:")
        print(f"• Longitud: {col_boundaries_v2['lon_min']:.6f}° a {col_boundaries_v2['lon_max']:.6f}°")
        print(f"• Latitud:  {col_boundaries_v2['lat_min']:.6f}° a {col_boundaries_v2['lat_max']:.6f}°")
    
    # Crear máscara para coordenadas dentro de Colombia
    dentro_colombia = (
        (incidentes_df['Longitud'] >= col_boundaries_v2['lon_min']) & 
        (incidentes_df['Longitud'] <= col_boundaries_v2['lon_max']) &
        (incidentes_df['Latitud'] >= col_boundaries_v2['lat_min']) & 
        (incidentes_df['Latitud'] <= col_boundaries_v2['lat_max'])
    )
    
    # Separar incidentes dentro y fuera de Colombia
    incidentes_colombia = incidentes_df[dentro_colombia].copy()
    incidentes_fuera = incidentes_df[~dentro_colombia].copy()
    
    if verbose:
        print(f"\nRESULTADOS DE VALIDACION:")
        print(f"• Total de incidentes analizados: {len(incidentes_df):,}")
        print(f"• Incidentes dentro de Colombia: {len(incidentes_colombia):,} ({len(incidentes_colombia)/len(incidentes_df)*100:.2f}%)")
        print(f"• Incidentes fuera de limites: {len(incidentes_fuera):,} ({len(incidentes_fuera)/len(incidentes_df)*100:.2f}%)")
    
    # Análisis detallado de los incidentes fuera de límites
    if len(incidentes_fuera) > 0 and verbose:
        print(f"\nANALISIS DE INCIDENTES FUERA DE LIMITES:")
        
        # Verificar qué límite se viola
        fuera_lon_min = incidentes_fuera['Longitud'] < col_boundaries_v2['lon_min']
        fuera_lon_max = incidentes_fuera['Longitud'] > col_boundaries_v2['lon_max']
        fuera_lat_min = incidentes_fuera['Latitud'] < col_boundaries_v2['lat_min']
        fuera_lat_max = incidentes_fuera['Latitud'] > col_boundaries_v2['lat_max']
        
        print(f"• Longitud muy occidental (< {col_boundaries_v2['lon_min']:.2f}°): {fuera_lon_min.sum()}")
        print(f"• Longitud muy oriental (> {col_boundaries_v2['lon_max']:.2f}°): {fuera_lon_max.sum()}")
        print(f"• Latitud muy sur (< {col_boundaries_v2['lat_min']:.2f}°): {fuera_lat_min.sum()}")
        print(f"• Latitud muy norte (> {col_boundaries_v2['lat_max']:.2f}°): {fuera_lat_max.sum()}")
        
        print(f"\nEjemplos de coordenadas fuera de limites:")
        print(incidentes_fuera[['Longitud', 'Latitud']].head(10))
        
        # Posibles ubicaciones de incidentes fuera de Colombia
        print(f"\nPOSIBLES UBICACIONES DE INCIDENTES FUERA DE COLOMBIA:")
        if fuera_lon_min.sum() > 0:
            print(f"• {fuera_lon_min.sum()} incidentes al oeste (posiblemente Oceano Pacifico/Ecuador)")
        if fuera_lon_max.sum() > 0:
            print(f"• {fuera_lon_max.sum()} incidentes al este (posiblemente Venezuela/Brasil)")
        if fuera_lat_min.sum() > 0:
            print(f"• {fuera_lat_min.sum()} incidentes al sur (posiblemente Peru/Brasil/Ecuador)")
        if fuera_lat_max.sum() > 0:
            print(f"• {fuera_lat_max.sum()} incidentes al norte (posiblemente Mar Caribe)")
    
    # Estadísticas de incidentes válidos en Colombia
    if len(incidentes_colombia) > 0 and verbose:
        print(f"\nESTADISTICAS DE INCIDENTES EN COLOMBIA:")
        print(f"• Rango longitud: {incidentes_colombia['Longitud'].min():.4f}° a {incidentes_colombia['Longitud'].max():.4f}°")
        print(f"• Rango latitud: {incidentes_colombia['Latitud'].min():.4f}° a {incidentes_colombia['Latitud'].max():.4f}°")
        print(f"• Centro geografico: ({incidentes_colombia['Longitud'].mean():.4f}°, {incidentes_colombia['Latitud'].mean():.4f}°)")
        
        # Cobertura del territorio colombiano donde ocurren deslizamientos
        cobertura_lon = (incidentes_colombia['Longitud'].max() - incidentes_colombia['Longitud'].min()) / (col_boundaries_v2['lon_max'] - col_boundaries_v2['lon_min']) * 100
        cobertura_lat = (incidentes_colombia['Latitud'].max() - incidentes_colombia['Latitud'].min()) / (col_boundaries_v2['lat_max'] - col_boundaries_v2['lat_min']) * 100
        
        print(f"• Cobertura territorial afectada: {cobertura_lon:.1f}% (longitud), {cobertura_lat:.1f}% (latitud)")
        
        # Densidad de incidentes por grado cuadrado
        area_afectada = (incidentes_colombia['Longitud'].max() - incidentes_colombia['Longitud'].min()) * \
                       (incidentes_colombia['Latitud'].max() - incidentes_colombia['Latitud'].min())
        densidad = len(incidentes_colombia) / area_afectada if area_afectada > 0 else 0
        print(f"• Densidad de incidentes: {densidad:.1f} incidentes por grado cuadrado")
    
    # Calcular densidad aunque no se imprima
    densidad = 0
    if len(incidentes_colombia) > 0:
        area_afectada = (incidentes_colombia['Longitud'].max() - incidentes_colombia['Longitud'].min()) * \
                       (incidentes_colombia['Latitud'].max() - incidentes_colombia['Latitud'].min())
        densidad = len(incidentes_colombia) / area_afectada if area_afectada > 0 else 0
    
    # Crear diccionario con estadísticas
    estadisticas = {
        'total_incidentes': len(incidentes_df),
        'incidentes_colombia': len(incidentes_colombia),
        'incidentes_fuera': len(incidentes_fuera),
        'porcentaje_validos': len(incidentes_colombia)/len(incidentes_df)*100,
        'densidad_incidentes': densidad,
        'limites_usados': col_boundaries_v2
    }
    
    return dentro_colombia, incidentes_colombia, incidentes_fuera, estadisticas

def filtrar_incidentes_colombia(incidentes_df, verbose=True):
    """
    Filtra el DataFrame para mantener solo los incidentes dentro de Colombia
    
    Parameters:
    -----------
    incidentes_df : pandas.DataFrame
        DataFrame con coordenadas de incidentes limpias
    verbose : bool, default=True
        Si True, imprime información del proceso
    
    Returns:
    --------
    pandas.DataFrame: DataFrame filtrado solo con incidentes en Colombia
    """
    
    # Aplicar filtro usando el diccionario de límites
    dentro_colombia = (
        (incidentes_df['Longitud'] >= col_boundaries_v2['lon_min']) & 
        (incidentes_df['Longitud'] <= col_boundaries_v2['lon_max']) &
        (incidentes_df['Latitud'] >= col_boundaries_v2['lat_min']) & 
        (incidentes_df['Latitud'] <= col_boundaries_v2['lat_max'])
    )
    
    incidentes_colombia_final = incidentes_df[dentro_colombia].copy()
    
    if verbose:
        print(f"FILTRADO FINAL PARA INCIDENTES EN COLOMBIA:")
        print(f"• Incidentes antes del filtro: {len(incidentes_df):,}")
        print(f"• Incidentes en Colombia: {len(incidentes_colombia_final):,}")
        print(f"• Incidentes removidos: {len(incidentes_df) - len(incidentes_colombia_final):,}")
        print(f"• Porcentaje conservado: {len(incidentes_colombia_final)/len(incidentes_df)*100:.2f}%")
    
    return incidentes_colombia_final

def crear_mapa_incidentes_deslizamiento(incidentes_df, dentro_colombia_mask, verbose=True):
    """
    Crea visualizaciones de la validación geográfica para incidentes de deslizamiento
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Validacion Geografica - Incidentes de Deslizamiento en Colombia', fontsize=16, fontweight='bold')
    
    # Mapa 1: Todas las coordenadas con límites
    incidentes_dentro = incidentes_df[dentro_colombia_mask]
    incidentes_fuera = incidentes_df[~dentro_colombia_mask]
    
    if len(incidentes_dentro) > 0:
        axes[0].scatter(incidentes_dentro['Longitud'], incidentes_dentro['Latitud'], 
                       c='red', s=4, alpha=0.7, label=f'Incidentes en Colombia ({len(incidentes_dentro):,})')
    
    if len(incidentes_fuera) > 0:
        axes[0].scatter(incidentes_fuera['Longitud'], incidentes_fuera['Latitud'], 
                       c='gray', s=10, alpha=0.8, label=f'Incidentes fuera Colombia ({len(incidentes_fuera):,})')
    
    # Dibujar límites precisos de Colombia
    axes[0].axvline(col_boundaries_v2['lon_min'], color='blue', linestyle='-', linewidth=2, alpha=0.8, label='Limites Colombia')
    axes[0].axvline(col_boundaries_v2['lon_max'], color='blue', linestyle='-', linewidth=2, alpha=0.8)
    axes[0].axhline(col_boundaries_v2['lat_min'], color='blue', linestyle='-', linewidth=2, alpha=0.8)
    axes[0].axhline(col_boundaries_v2['lat_max'], color='blue', linestyle='-', linewidth=2, alpha=0.8)
    
    # Sombrear el área de Colombia
    from matplotlib.patches import Rectangle
    colombia_rect = Rectangle((col_boundaries_v2['lon_min'], col_boundaries_v2['lat_min']), 
                             col_boundaries_v2['lon_max'] - col_boundaries_v2['lon_min'],
                             col_boundaries_v2['lat_max'] - col_boundaries_v2['lat_min'],
                             linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.2)
    axes[0].add_patch(colombia_rect)
    
    axes[0].set_title('Validacion Geografica de Incidentes')
    axes[0].set_xlabel('Longitud (grados)')
    axes[0].set_ylabel('Latitud (grados)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mapa 2: Solo incidentes válidos en Colombia (zoom)
    if len(incidentes_dentro) > 0:
        axes[1].scatter(incidentes_dentro['Longitud'], incidentes_dentro['Latitud'], 
                       c='darkred', s=3, alpha=0.6)
        
        axes[1].set_title(f'Incidentes de Deslizamiento en Colombia ({len(incidentes_dentro):,})')
        axes[1].set_xlabel('Longitud (grados)')
        axes[1].set_ylabel('Latitud (grados)')
        axes[1].grid(True, alpha=0.3)
        
        # Ajustar límites del zoom
        margin = 0.5
        axes[1].set_xlim(incidentes_dentro['Longitud'].min() - margin, 
                        incidentes_dentro['Longitud'].max() + margin)
        axes[1].set_ylim(incidentes_dentro['Latitud'].min() - margin, 
                        incidentes_dentro['Latitud'].max() + margin)
        
        # Añadir información de densidad
        axes[1].text(0.02, 0.98, 
                    f'Densidad: {len(incidentes_dentro):,} incidentes\n'
                    f'Area: {(incidentes_dentro["Longitud"].max()-incidentes_dentro["Longitud"].min()):.1f}° x '
                    f'{(incidentes_dentro["Latitud"].max()-incidentes_dentro["Latitud"].min()):.1f}°',
                    transform=axes[1].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('validacion_incidentes_deslizamiento_colombia.png', dpi=300, bbox_inches='tight')
    if verbose:
        print("Mapa guardado como: validacion_incidentes_deslizamiento_colombia.png")
    plt.show()

def analizar_patrones_deslizamiento(incidentes_colombia, verbose=True):
    """
    Análisis específico para patrones de deslizamiento en Colombia
    """
    
    if verbose:
        print(f"\nANALISIS DE PATRONES DE DESLIZAMIENTO EN COLOMBIA:")
        print("=" * 60)
    
    if len(incidentes_colombia) == 0:
        if verbose:
            print("No hay incidentes válidos para analizar.")
        return None
    
    # Análisis por regiones geográficas de Colombia
    def clasificar_region_colombia(row):
        lon, lat = row['Longitud'], row['Latitud']
        
        # Clasificación aproximada por regiones de Colombia
        if lat >= 8:  # Norte
            if lon <= -75:
                return "Caribe Occidental"
            else:
                return "Caribe Oriental"
        elif lat >= 4:  # Centro
            if lon <= -75:
                return "Andina Central"
            else:
                return "Orinoquia"
        else:  # Sur
            if lon <= -75:
                return "Andina Sur/Pacifico"
            else:
                return "Amazonia"
    
    # Aplicar clasificación (solo para una muestra si hay muchos datos)
    if len(incidentes_colombia) > 5000:
        muestra = incidentes_colombia.sample(5000, random_state=42)
        if verbose:
            print(f"Usando muestra de 5,000 incidentes para análisis regional")
    else:
        muestra = incidentes_colombia
    
    muestra_regiones = muestra.apply(clasificar_region_colombia, axis=1)
    distribucion_regional = muestra_regiones.value_counts()
    
    if verbose:
        print(f"DISTRIBUCION REGIONAL DE DESLIZAMIENTOS:")
        for region, cantidad in distribucion_regional.items():
            porcentaje = cantidad / len(muestra) * 100
            print(f"• {region}: {cantidad} incidentes ({porcentaje:.1f}%)")
    
    # Análisis de hotspots (zonas de alta concentración)
    if verbose:
        print(f"\nANALISIS DE HOTSPOTS:")
    
    # Dividir Colombia en grid para identificar zonas de mayor actividad
    lon_bins = np.linspace(incidentes_colombia['Longitud'].min(), incidentes_colombia['Longitud'].max(), 10)
    lat_bins = np.linspace(incidentes_colombia['Latitud'].min(), incidentes_colombia['Latitud'].max(), 10)
    
    # Crear grid de conteo
    grid_counts, _, _ = np.histogram2d(incidentes_colombia['Longitud'], incidentes_colombia['Latitud'], 
                                     bins=[lon_bins, lat_bins])
    
    # Encontrar las celdas con más incidentes
    max_count = grid_counts.max()
    max_indices = np.where(grid_counts == max_count)
    
    if max_count > 0 and verbose:
        print(f"• Zona de mayor actividad: {max_count:.0f} incidentes en una celda de ~0.1° x 0.1°")
        print(f"• Coordenadas aproximadas del hotspot principal:")
        for i in range(len(max_indices[0])):
            lon_idx, lat_idx = max_indices[0][i], max_indices[1][i]
            lon_center = (lon_bins[lon_idx] + lon_bins[lon_idx+1]) / 2
            lat_center = (lat_bins[lat_idx] + lat_bins[lat_idx+1]) / 2
            print(f"  - ({lon_center:.3f}°, {lat_center:.3f}°)")
    
    return distribucion_regional

def proceso_completo_limpieza_y_validacion(inventario_raw, verbose=True):
    """
    Proceso completo: limpieza + validación con límites de Colombia
    
    Parameters:
    -----------
    inventario_raw : pandas.DataFrame
        DataFrame original con coordenadas sin limpiar
    verbose : bool, default=True
        Si True, imprime información detallada del proceso
        
    Returns:
    --------
    dict: Diccionario con todos los resultados del proceso
    """
    
    if verbose:
        print("PROCESAMIENTO " + "="*50 + " PROCESAMIENTO")
        print("  PROCESO COMPLETO: LIMPIEZA + VALIDACION DE INCIDENTES")
        print("PROCESAMIENTO " + "="*50 + " PROCESAMIENTO")
    
    # === FASE 1: LIMPIEZA ===
    if verbose:
        print("\nFASE 1: LIMPIEZA DE COORDENADAS")
        print("-" * 40)
    
    # Copiar para no modificar el original
    inventario = inventario_raw.copy()
    registros_inicial = len(inventario)
    
    if verbose:
        print(f"Registros iniciales: {registros_inicial:,}")
    
    # Eliminar nulos
    inventario.dropna(subset=['Longitud', 'Latitud'], inplace=True)
    if verbose:
        print(f"Después de eliminar nulos: {len(inventario):,} registros")
    
    # Identificar valores no convertibles
    longitud_numeric = pd.to_numeric(inventario['Longitud'], errors='coerce')
    valores_problematicos = (inventario['Longitud'].notna()) & (longitud_numeric.isna())
    
    if verbose and valores_problematicos.sum() > 0:
        print(f"Valores problemáticos encontrados: {valores_problematicos.sum():,}")
        # Mostrar algunos ejemplos
        valores_unicos_problematicos = inventario.loc[valores_problematicos, 'Longitud'].unique()[:5]
        print("Ejemplos de valores problemáticos:")
        for i, valor in enumerate(valores_unicos_problematicos):
            print(f"  {i+1}. '{valor}'")
    
    # Obtener solo filas válidas
    incidentes_numericos = inventario[~valores_problematicos].copy()
    
    # Convertir tipos de datos
    incidentes_numericos = incidentes_numericos.astype({
        'Latitud': np.float32,
        'Longitud': np.float32
    })
    
    # Agregar columna ID si no existe
    if 'ID' not in incidentes_numericos.columns:
        incidentes_numericos['ID'] = range(1, len(incidentes_numericos) + 1)
    else:
        incidentes_numericos = incidentes_numericos.astype({'ID': np.int64})
    
    # Eliminar longitudes positivas (incorrectas para Colombia)
    drop_invalid_rows = incidentes_numericos.loc[incidentes_numericos.Longitud >= 0].index
    incidentes_numericos.drop(drop_invalid_rows, inplace=True)
    
    if verbose:
        print(f"Longitudes positivas eliminadas: {len(drop_invalid_rows):,}")
        print(f"Después de limpieza completa: {len(incidentes_numericos):,} registros")
    
    # === FASE 2: VALIDACIÓN CON LÍMITES DE COLOMBIA ===
    if verbose:
        print(f"\nFASE 2: VALIDACION CON LIMITES DE COLOMBIA")
        print("-" * 40)
    
    # Aplicar filtro geográfico
    dentro_colombia = (
        (incidentes_numericos['Longitud'] >= col_boundaries_v2['lon_min']) & 
        (incidentes_numericos['Longitud'] <= col_boundaries_v2['lon_max']) &
        (incidentes_numericos['Latitud'] >= col_boundaries_v2['lat_min']) & 
        (incidentes_numericos['Latitud'] <= col_boundaries_v2['lat_max'])
    )
    
    incidentes_colombia = incidentes_numericos[dentro_colombia].copy()
    incidentes_fuera = incidentes_numericos[~dentro_colombia].copy()
    
    if verbose:
        print(f"Incidentes dentro de Colombia: {len(incidentes_colombia):,}")
        print(f"Incidentes fuera de limites: {len(incidentes_fuera):,}")
    
    # === FASE 3: ESTADÍSTICAS FINALES ===
    if verbose:
        print(f"\nFASE 3: RESULTADOS FINALES")
        print("-" * 40)
        print(f"RESUMEN DEL PROCESO:")
        print(f"• Registros originales: {registros_inicial:,}")
        print(f"• Después de limpieza: {len(incidentes_numericos):,} ({len(incidentes_numericos)/registros_inicial*100:.1f}%)")
        print(f"• Incidentes en Colombia: {len(incidentes_colombia):,} ({len(incidentes_colombia)/registros_inicial*100:.1f}%)")
        print(f"• Tasa de éxito total: {len(incidentes_colombia)/registros_inicial*100:.1f}%")
        
        if len(incidentes_colombia) > 0:
            print(f"\nESTADISTICAS DE COORDENADAS FINALES:")
            print(f"• Rango longitud: {incidentes_colombia['Longitud'].min():.4f}° a {incidentes_colombia['Longitud'].max():.4f}°")
            print(f"• Rango latitud: {incidentes_colombia['Latitud'].min():.4f}° a {incidentes_colombia['Latitud'].max():.4f}°")
            print(f"• Centro geográfico: ({incidentes_colombia['Longitud'].mean():.4f}°, {incidentes_colombia['Latitud'].mean():.4f}°)")
    
    # Calcular estadísticas
    densidad = 0
    if len(incidentes_colombia) > 0:
        area_afectada = (incidentes_colombia['Longitud'].max() - incidentes_colombia['Longitud'].min()) * \
                       (incidentes_colombia['Latitud'].max() - incidentes_colombia['Latitud'].min())
        densidad = len(incidentes_colombia) / area_afectada if area_afectada > 0 else 0
    
    estadisticas = {
        'total_inicial': registros_inicial,
        'total_limpio': len(incidentes_numericos),
        'incidentes_colombia': len(incidentes_colombia),
        'incidentes_fuera': len(incidentes_fuera),
        'tasa_exito': len(incidentes_colombia)/registros_inicial*100,
        'densidad_incidentes': densidad,
        'limites_usados': col_boundaries_v2
    }
    
    # Resultados finales
    resultados = {
        'inventario_original': inventario_raw,
        'torres_limpias': incidentes_numericos,  # Mantengo el nombre por compatibilidad
        'incidentes_limpios': incidentes_numericos,  # Nombre más apropiado
        'torres_colombia': incidentes_colombia,  # Mantengo por compatibilidad
        'incidentes_colombia': incidentes_colombia,  # Nombre más apropiado
        'incidentes_fuera_colombia': incidentes_fuera,
        'mask_colombia': dentro_colombia,
        'estadisticas': estadisticas,
        'limites_colombia': col_boundaries_v2
    }
    
    return resultados

def load_inventario(ruta_archivo=None, verbose=True, aplicar_validacion_geografica=True, 
                   guardar_resultados=False, carpeta_salida=None):
    """
    Carga y procesa completamente el inventario de incidentes de deslizamiento
    
    Parameters:
    -----------
    ruta_archivo : str, optional
        Ruta completa al archivo Excel. Si None, usa la ruta por defecto
    verbose : bool, default=True
        Si True, imprime información detallada del proceso
    aplicar_validacion_geografica : bool, default=True
        Si True, aplica filtros geográficos y crea visualizaciones
    guardar_resultados : bool, default=False
        Si True, guarda los DataFrames resultantes en CSV
    carpeta_salida : str, optional
        Carpeta donde guardar los resultados. Si None, usa la carpeta actual
        
    Returns:
    --------
    dict: Diccionario con todos los resultados del procesamiento
        - 'incidentes_colombia_final': DataFrame listo para análisis
        - 'incidentes_todos_limpios': DataFrame limpio sin filtro geográfico
        - 'estadisticas': Diccionario con métricas del proceso
        - 'distribucion_regional': Análisis por regiones (si verbose=True)
    """
    
    # Configurar ruta por defecto
    if ruta_archivo is None:
        path_inventario_landslide = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\databases"
        ruta_archivo = os.path.join(path_inventario_landslide, 'Colombia_database1900_2021(completa).xlsx')
    
    if verbose:
        print("CARGA DE INVENTARIO " + "="*40 + " INICIO")
        print(f"Archivo: {os.path.basename(ruta_archivo)}")
        print(f"Ruta: {ruta_archivo}")
    
    try:
        # Cargar archivo
        if verbose:
            print(f"\nCargando archivo Excel...")
        
        inventario_raw = pd.read_excel(ruta_archivo)
        
        if verbose:
            print(f"Archivo cargado exitosamente!")
            print(f"Dimensiones: {inventario_raw.shape}")
            print(f"Columnas: {list(inventario_raw.columns)}")
        
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo: {ruta_archivo}")
        return None
    except Exception as e:
        print(f"ERROR al cargar el archivo: {e}")
        return None
    
    # Proceso completo de limpieza y validación
    if verbose:
        print(f"\nIniciando proceso de limpieza y validación...")
    
    resultados = proceso_completo_limpieza_y_validacion(inventario_raw, verbose=verbose)
    
    # Obtener el DataFrame final de incidentes en Colombia
    incidentes_colombia_final = resultados['incidentes_colombia']
    
    # Aplicar validación geográfica adicional si se solicita
    distribucion_regional = None
    if aplicar_validacion_geografica and len(incidentes_colombia_final) > 0:
        
        if verbose:
            print(f"\nAPLICANDO VALIDACION GEOGRAFICA ADICIONAL...")
            
            # Validación detallada
            mask_colombia, incidentes_col, incidentes_fuera, stats = validar_incidentes_deslizamiento_colombia(resultados['incidentes_limpios'], verbose=verbose)
            
            # Crear visualizaciones
            print(f"\nCreando visualizaciones...")
            crear_mapa_incidentes_deslizamiento(resultados['incidentes_limpios'], mask_colombia, verbose=verbose)
            
            # Análisis de patrones
            print(f"\nAnalizando patrones regionales...")
            distribucion_regional = analizar_patrones_deslizamiento(incidentes_colombia_final, verbose=verbose)
        else:
            # Ejecutar validación sin prints
            mask_colombia, incidentes_col, incidentes_fuera, stats = validar_incidentes_deslizamiento_colombia(resultados['incidentes_limpios'], verbose=False)
            distribucion_regional = analizar_patrones_deslizamiento(incidentes_colombia_final, verbose=False)
    
    # Guardar resultados si se solicita
    if guardar_resultados:
        if carpeta_salida is None:
            carpeta_salida = os.path.dirname(ruta_archivo)
        
        if verbose:
            print(f"\nGuardando resultados en: {carpeta_salida}")
        
        # Guardar diferentes versiones
        archivo_colombia = os.path.join(carpeta_salida, 'incidentes_deslizamiento_colombia_final.csv')
        archivo_limpio = os.path.join(carpeta_salida, 'incidentes_deslizamiento_todos_limpios.csv')
        archivo_fuera = os.path.join(carpeta_salida, 'incidentes_fuera_colombia.csv')
        
        incidentes_colombia_final.to_csv(archivo_colombia, index=False)
        resultados['incidentes_limpios'].to_csv(archivo_limpio, index=False)
        
        if len(resultados['incidentes_fuera_colombia']) > 0:
            resultados['incidentes_fuera_colombia'].to_csv(archivo_fuera, index=False)
        
        if verbose:
            print(f"Archivos guardados:")
            print(f"• {os.path.basename(archivo_colombia)}")
            print(f"• {os.path.basename(archivo_limpio)}")
            if len(resultados['incidentes_fuera_colombia']) > 0:
                print(f"• {os.path.basename(archivo_fuera)}")
    
    # Preparar resultados finales
    resultados_finales = {
        'incidentes_colombia_final': incidentes_colombia_final,
        'incidentes_todos_limpios': resultados['incidentes_limpios'],
        'incidentes_fuera_colombia': resultados['incidentes_fuera_colombia'],
        'estadisticas': resultados['estadisticas'],
        'distribucion_regional': distribucion_regional,
        'resultados_completos': resultados  # Para compatibilidad
    }
    
    if verbose:
        print(f"\nPROCESO COMPLETADO EXITOSAMENTE!")
        print(f"Dataset final listo para análisis:")
        print(f"• Incidentes en Colombia: {len(incidentes_colombia_final):,}")
        print(f"• Calidad de datos: {resultados['estadisticas']['tasa_exito']:.1f}% de éxito")
        print("="*70)
    
    return resultados_finales

# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

"""
# USO BÁSICO (con verbose completo)
resultados = load_inventario()
incidentes_finales = resultados['incidentes_colombia_final']

# USO SILENCIOSO (sin prints)
resultados = load_inventario(verbose=False)

# USO CON ARCHIVO ESPECÍFICO
resultados = load_inventario(
    ruta_archivo="C:/mi_carpeta/mi_inventario.xlsx",
    verbose=True,
    guardar_resultados=True,
    carpeta_salida="C:/mi_carpeta/resultados"
)

# USO SOLO PARA LIMPIEZA (sin validación geográfica)
resultados = load_inventario(
    verbose=True,
    aplicar_validacion_geografica=False
)

# ACCESO A RESULTADOS
incidentes_colombia = resultados['incidentes_colombia_final']
todos_los_incidentes = resultados['incidentes_todos_limpios']  
estadisticas = resultados['estadisticas']
distribucion = resultados['distribucion_regional']

print(f"Incidentes procesados: {len(incidentes_colombia):,}")
print(f"Tasa de éxito: {estadisticas['tasa_exito']:.1f}%")
"""