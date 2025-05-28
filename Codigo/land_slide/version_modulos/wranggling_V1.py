

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Límites precisos de Colombia definidos por el usuario
col_boundaries_v2 = {
    'lat_max': 12.590276718139648,
    'lon_min': -81.72014617919922,    
    'lat_min': -4.236873626708984,
    'lon_max': -66.87045288085938
}

def validar_incidentes_deslizamiento_colombia(incidentes_df):
    """
    Valida las coordenadas de incidentes de deslizamiento usando los límites precisos de Colombia
    
    Parameters:
    -----------
    incidentes_df : pandas.DataFrame
        DataFrame con columnas 'Longitud' y 'Latitud' de incidentes de deslizamiento
    
    Returns:
    --------
    tuple: (mask_dentro_colombia, incidentes_colombia, incidentes_fuera, estadisticas)
    """
    
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
    
    print(f"\nRESULTADOS DE VALIDACION:")
    print(f"• Total de incidentes analizados: {len(incidentes_df):,}")
    print(f"• Incidentes dentro de Colombia: {len(incidentes_colombia):,} ({len(incidentes_colombia)/len(incidentes_df)*100:.2f}%)")
    print(f"• Incidentes fuera de limites: {len(incidentes_fuera):,} ({len(incidentes_fuera)/len(incidentes_df)*100:.2f}%)")
    
    # Análisis detallado de los incidentes fuera de límites
    if len(incidentes_fuera) > 0:
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
    if len(incidentes_colombia) > 0:
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
    
    # Crear diccionario con estadísticas
    estadisticas = {
        'total_incidentes': len(incidentes_df),
        'incidentes_colombia': len(incidentes_colombia),
        'incidentes_fuera': len(incidentes_fuera),
        'porcentaje_validos': len(incidentes_colombia)/len(incidentes_df)*100,
        'densidad_incidentes': densidad if len(incidentes_colombia) > 0 else 0,
        'limites_usados': col_boundaries_v2
    }
    
    return dentro_colombia, incidentes_colombia, incidentes_fuera, estadisticas

def filtrar_incidentes_colombia(incidentes_df):
    """
    Filtra el DataFrame para mantener solo los incidentes dentro de Colombia
    
    Parameters:
    -----------
    incidentes_df : pandas.DataFrame
        DataFrame con coordenadas de incidentes limpias
    
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
    
    print(f"FILTRADO FINAL PARA INCIDENTES EN COLOMBIA:")
    print(f"• Incidentes antes del filtro: {len(incidentes_df):,}")
    print(f"• Incidentes en Colombia: {len(incidentes_colombia_final):,}")
    print(f"• Incidentes removidos: {len(incidentes_df) - len(incidentes_colombia_final):,}")
    print(f"• Porcentaje conservado: {len(incidentes_colombia_final)/len(incidentes_df)*100:.2f}%")
    
    return incidentes_colombia_final

def crear_mapa_incidentes_deslizamiento(incidentes_df, dentro_colombia_mask):
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
    plt.show()

def analizar_patrones_deslizamiento(incidentes_colombia):
    """
    Análisis específico para patrones de deslizamiento en Colombia
    """
    
    print(f"\nANALISIS DE PATRONES DE DESLIZAMIENTO EN COLOMBIA:")
    print("=" * 60)
    
    if len(incidentes_colombia) == 0:
        print("No hay incidentes válidos para analizar.")
        return
    
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
        print(f"Usando muestra de 5,000 incidentes para análisis regional")
    else:
        muestra = incidentes_colombia
    
    muestra_regiones = muestra.apply(clasificar_region_colombia, axis=1)
    distribucion_regional = muestra_regiones.value_counts()
    
    print(f"DISTRIBUCION REGIONAL DE DESLIZAMIENTOS:")
    for region, cantidad in distribucion_regional.items():
        porcentaje = cantidad / len(muestra) * 100
        print(f"• {region}: {cantidad} incidentes ({porcentaje:.1f}%)")
    
    # Análisis de hotspots (zonas de alta concentración)
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
    
    if max_count > 0:
        print(f"• Zona de mayor actividad: {max_count:.0f} incidentes en una celda de ~0.1° x 0.1°")
        print(f"• Coordenadas aproximadas del hotspot principal:")
        for i in range(len(max_indices[0])):
            lon_idx, lat_idx = max_indices[0][i], max_indices[1][i]
            lon_center = (lon_bins[lon_idx] + lon_bins[lon_idx+1]) / 2
            lat_center = (lat_bins[lat_idx] + lat_bins[lat_idx+1]) / 2
            print(f"  - ({lon_center:.3f}°, {lat_center:.3f}°)")
    
    return distribucion_regional

def proceso_completo_incidentes_deslizamiento(inventario_raw):
    """
    Proceso completo para incidentes de deslizamiento: limpieza + validación
    """
    
    print("PROCESAMIENTO " + "="*50 + " PROCESAMIENTO")
    print("  PROCESO COMPLETO: INCIDENTES DE DESLIZAMIENTO")
    print("PROCESAMIENTO " + "="*50 + " PROCESAMIENTO")
    
    # Aplicar tu proceso de limpieza existente
    # (Aquí iría tu código de limpieza que ya tienes funcionando)
    
    print(f"\nTu proceso de limpieza ya funciona perfectamente!")
    print(f"Continuemos con la validación geografica...")
    
    return None




def load_inventario():

    # ============================================================================
    # CÓDIGO PARA USAR CON TU INVENTARIO (después de tu proceso de limpieza)
    # ============================================================================

    path_inventario_landslide = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\databases"

## Not use for now
###     wrangling_cols = [ 'ID' , 'Latitud', 'Longitud']
###     complete_cols = ['ID', 'Year', 'Month', 'Day','Region', 'Department', 'Municipality', 'Place', 'Site','Latitud', 'Longitud','Cause','triggering_description']
    
    
    inventario = pd.read_excel(os.path.join(path_inventario_landslide, 'Colombia_database1900_2021(completa).xlsx'))
    inventario = inventario.dropna(subset = ['Latitud' , 'Longitud'])

    # Si quieres hacer el proceso completo desde cero:
    resultados = proceso_completo_limpieza_y_validacion(inventario)

    # ============================================================================
    # CODIGO PARA USAR CON TUS INCIDENTES DE DESLIZAMIENTO
    # ============================================================================
    
    
    # Para usar con tu inventario_valido de incidentes:
    
    inventario_valido = resultados['torres_limpias']


    if verbose == True:
        
        # 1. Validar coordenadas de incidentes con límites precisos
        mask_colombia, incidentes_colombia, incidentes_fuera, stats = validar_incidentes_deslizamiento_colombia(resultados['torres_limpias'])
        
        # 2. Crear visualizaciones específicas para deslizamientos
        crear_mapa_incidentes_deslizamiento(inventario_valido, mask_colombia)
        
        # 3. Filtrar solo incidentes en Colombia
        incidentes_colombia_final = filtrar_incidentes_colombia(inventario_valido)
        
        # 4. Análisis de patrones específicos de deslizamiento
        distribucion_regional = analizar_patrones_deslizamiento(incidentes_colombia_final)

    return resultados









