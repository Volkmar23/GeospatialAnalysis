import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional, Union
import warnings

# Configuración de warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# Diccionario de remapeo de categorías de causas
DICT_REMAPPING = {
    # Desconocida
    'Unknown': 'Desconocida',
    
    # Rainfall (lluvias)
    'Lluvias': 'Rainfall',
    'lluvias': 'Rainfall', 
    'Erosion': 'Rainfall',
    
    # Antropico
    'Anthropic': 'Antropico',
    'Anthropogenic': 'Antropico',
    
    # Earthquake (sismo)
    'Sismo': 'Earthquake',
    'Falla': 'Earthquake',
    
    # Actividad volcánica
    'Volcanic Activity': 'Actividad volcánica',
    
    # Otra causa
    'Other cause': 'Otra causa'
}

# Límites precisos de Colombia
COLOMBIA_BOUNDARIES = {
    'lat_max': 12.590276718139648,
    'lon_min': -81.72014617919922,    
    'lat_min': -4.236873626708984,
    'lon_max': -66.87045288085938
}

# Configuración de impresión
SEPARADOR_PRINCIPAL = "=" * 70
SEPARADOR_SECUNDARIO = "-" * 50
TITULO_COLOMBIA = "COLOMBIA " + "=" * 60 + " COLOMBIA"

# ============================================================================
# FUNCIONES AUXILIARES DE FORMATO
# ============================================================================

def imprimir_titulo_principal(titulo: str) -> None:
    """Imprime un título principal con formato estándar."""
    print(f"\n{SEPARADOR_PRINCIPAL}")
    print(f"  {titulo.upper()}")
    print(f"{SEPARADOR_PRINCIPAL}")

def imprimir_seccion(titulo: str) -> None:
    """Imprime un título de sección con formato estándar."""
    print(f"\n{titulo.upper()}")
    print(f"{SEPARADOR_SECUNDARIO}")

def imprimir_estadisticas(stats_dict: Dict, titulo: str = "ESTADÍSTICAS") -> None:
    """Imprime estadísticas en formato estándar."""
    print(f"\n{titulo}:")
    for key, value in stats_dict.items():
        if isinstance(value, (int, float)):
            if isinstance(value, int):
                print(f"• {key}: {value:,}")
            else:
                print(f"• {key}: {value:.2f}")
        else:
            print(f"• {key}: {value}")

def imprimir_coordenadas_info(df: pd.DataFrame, nombre: str = "Dataset") -> None:
    """Imprime información estándar de coordenadas."""
    if len(df) == 0:
        print(f"• {nombre}: Sin datos")
        return
    
    stats = {
        "Registros": len(df),
        "Rango longitud": f"{df['Longitud'].min():.4f}° a {df['Longitud'].max():.4f}°",
        "Rango latitud": f"{df['Latitud'].min():.4f}° a {df['Latitud'].max():.4f}°",
        "Centro geográfico": f"({df['Longitud'].mean():.4f}°, {df['Latitud'].mean():.4f}°)"
    }
    
    for key, value in stats.items():
        print(f"• {key}: {value}")

# ============================================================================
# FUNCIONES DE CLASIFICACIÓN GEOGRÁFICA
# ============================================================================

def clasificar_region_colombia(row: pd.Series) -> str:
    """
    Clasifica un punto geográfico en regiones de Colombia.
    
    Parameters:
    -----------
    row : pd.Series
        Serie con columnas 'Longitud' y 'Latitud'
        
    Returns:
    --------
    str: Nombre de la región
    """
    lon, lat = row['Longitud'], row['Latitud']
    
    if lat >= 8:  # Norte
        return "Caribe Occidental" if lon <= -75 else "Caribe Oriental"
    elif lat >= 4:  # Centro
        return "Andina Central" if lon <= -75 else "Orinoquia"
    else:  # Sur
        return "Andina Sur/Pacifico" if lon <= -75 else "Amazonia"

def calcular_densidad_incidentes(df: pd.DataFrame) -> float:
    """Calcula la densidad de incidentes por grado cuadrado."""
    if len(df) == 0:
        return 0.0
    
    area_afectada = (
        (df['Longitud'].max() - df['Longitud'].min()) * 
        (df['Latitud'].max() - df['Latitud'].min())
    )
    
    return len(df) / area_afectada if area_afectada > 0 else 0.0

# ============================================================================
# FUNCIONES PRINCIPALES DE VALIDACIÓN
# ============================================================================

def validar_incidentes_deslizamiento_colombia(
                                                incidentes_df: pd.DataFrame, 
                                                verbose: bool = True
                                            ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Valida las coordenadas de incidentes de deslizamiento usando los límites precisos de Colombia.
    
    Parameters:
    -----------
    incidentes_df : pd.DataFrame
        DataFrame con columnas 'Longitud' y 'Latitud' de incidentes de deslizamiento
    verbose : bool, default=True
        Si True, imprime información detallada del proceso
    
    Returns:
    --------
    tuple: (mask_dentro_colombia, incidentes_colombia, incidentes_fuera, estadisticas)
    """
    
    if verbose:
        imprimir_titulo_principal("VALIDACIÓN DE INCIDENTES DE DESLIZAMIENTO EN COLOMBIA")
        
        print(f"\nLÍMITES GEOGRÁFICOS UTILIZADOS:")
        coords_info = {
            "Longitud": f"{COLOMBIA_BOUNDARIES['lon_min']:.6f}° a {COLOMBIA_BOUNDARIES['lon_max']:.6f}°",
            "Latitud": f"{COLOMBIA_BOUNDARIES['lat_min']:.6f}° a {COLOMBIA_BOUNDARIES['lat_max']:.6f}°"
        }
        for key, value in coords_info.items():
            print(f"• {key}: {value}")
    
    # Crear máscara para coordenadas dentro de Colombia
    dentro_colombia = (
        (incidentes_df['Longitud'] >= COLOMBIA_BOUNDARIES['lon_min']) & 
        (incidentes_df['Longitud'] <= COLOMBIA_BOUNDARIES['lon_max']) &
        (incidentes_df['Latitud'] >= COLOMBIA_BOUNDARIES['lat_min']) & 
        (incidentes_df['Latitud'] <= COLOMBIA_BOUNDARIES['lat_max'])
    )
    
    # Separar incidentes dentro y fuera de Colombia
    incidentes_colombia = incidentes_df[dentro_colombia].copy()
    incidentes_fuera = incidentes_df[~dentro_colombia].copy()
    
    if verbose:
        imprimir_seccion("RESULTADOS DE VALIDACIÓN")
        
        porcentaje_validos = (len(incidentes_colombia) / len(incidentes_df) * 100) if len(incidentes_df) > 0 else 0
        
        resultados = {
                "Total de incidentes analizados": len(incidentes_df),
                "Incidentes dentro de Colombia": f"{len(incidentes_colombia):,} ({porcentaje_validos:.2f}%)",
                "Incidentes fuera de límites": f"{len(incidentes_fuera):,} ({100-porcentaje_validos:.2f}%)"
            }
        
        for key, value in resultados.items():
            print(f"• {key}: {value}")
    
    # Análisis detallado de incidentes fuera de límites
    if len(incidentes_fuera) > 0 and verbose:
        _analizar_incidentes_fuera_limites(incidentes_fuera)
    
    # Estadísticas de incidentes válidos en Colombia
    if len(incidentes_colombia) > 0 and verbose:
        _mostrar_estadisticas_colombia(incidentes_colombia)
    
    # Crear estadísticas finales
    densidad = calcular_densidad_incidentes(incidentes_colombia)
    porcentaje_validos = (len(incidentes_colombia) / len(incidentes_df) * 100) if len(incidentes_df) > 0 else 0
    
    estadisticas = {
        'total_incidentes': len(incidentes_df),
        'incidentes_colombia': len(incidentes_colombia),
        'incidentes_fuera': len(incidentes_fuera),
        'porcentaje_validos': porcentaje_validos,
        'densidad_incidentes': densidad,
        'limites_usados': COLOMBIA_BOUNDARIES
    }
    
    return dentro_colombia, incidentes_colombia, incidentes_fuera, estadisticas

def _analizar_incidentes_fuera_limites(incidentes_fuera: pd.DataFrame) -> None:
    """Analiza los incidentes que están fuera de los límites de Colombia."""
    imprimir_seccion("ANÁLISIS DE INCIDENTES FUERA DE LÍMITES")
    
    # Verificar qué límite se viola
    violaciones = {
        f"Longitud muy occidental (< {COLOMBIA_BOUNDARIES['lon_min']:.2f}°)": 
            (incidentes_fuera['Longitud'] < COLOMBIA_BOUNDARIES['lon_min']).sum(),
        f"Longitud muy oriental (> {COLOMBIA_BOUNDARIES['lon_max']:.2f}°)": 
            (incidentes_fuera['Longitud'] > COLOMBIA_BOUNDARIES['lon_max']).sum(),
        f"Latitud muy sur (< {COLOMBIA_BOUNDARIES['lat_min']:.2f}°)": 
            (incidentes_fuera['Latitud'] < COLOMBIA_BOUNDARIES['lat_min']).sum(),
        f"Latitud muy norte (> {COLOMBIA_BOUNDARIES['lat_max']:.2f}°)": 
            (incidentes_fuera['Latitud'] > COLOMBIA_BOUNDARIES['lat_max']).sum()
    }
    
    for descripcion, cantidad in violaciones.items():
        print(f"• {descripcion}: {cantidad}")
    
    # Mostrar ejemplos
    print(f"\nEjemplos de coordenadas fuera de límites:")
    print(incidentes_fuera[['Longitud', 'Latitud']].head(10).to_string(index=False))
    
    # Posibles ubicaciones
    print(f"\nPOSIBLES UBICACIONES DE INCIDENTES FUERA DE COLOMBIA:")
    ubicaciones = [
        (violaciones[f"Longitud muy occidental (< {COLOMBIA_BOUNDARIES['lon_min']:.2f}°)"], 
         "al oeste (posiblemente Océano Pacífico/Ecuador)"),
        (violaciones[f"Longitud muy oriental (> {COLOMBIA_BOUNDARIES['lon_max']:.2f}°)"], 
         "al este (posiblemente Venezuela/Brasil)"),
        (violaciones[f"Latitud muy sur (< {COLOMBIA_BOUNDARIES['lat_min']:.2f}°)"], 
         "al sur (posiblemente Perú/Brasil/Ecuador)"),
        (violaciones[f"Latitud muy norte (> {COLOMBIA_BOUNDARIES['lat_max']:.2f}°)"], 
         "al norte (posiblemente Mar Caribe)")
    ]
    
    for cantidad, descripcion in ubicaciones:
        if cantidad > 0:
            print(f"• {cantidad} incidentes {descripcion}")

def _mostrar_estadisticas_colombia(incidentes_colombia: pd.DataFrame) -> None:
    """Muestra estadísticas detalladas de los incidentes en Colombia."""
    imprimir_seccion("ESTADÍSTICAS DE INCIDENTES EN COLOMBIA")
    
    # Estadísticas básicas
    imprimir_coordenadas_info(incidentes_colombia, "Incidentes en Colombia")
    
    # Cobertura territorial
    cobertura_lon = (
        (incidentes_colombia['Longitud'].max() - incidentes_colombia['Longitud'].min()) / 
        (COLOMBIA_BOUNDARIES['lon_max'] - COLOMBIA_BOUNDARIES['lon_min']) * 100
    )
    cobertura_lat = (
        (incidentes_colombia['Latitud'].max() - incidentes_colombia['Latitud'].min()) / 
        (COLOMBIA_BOUNDARIES['lat_max'] - COLOMBIA_BOUNDARIES['lat_min']) * 100
    )
    
    densidad = calcular_densidad_incidentes(incidentes_colombia)
    
    cobertura_stats = {
        "Cobertura territorial longitud": f"{cobertura_lon:.1f}%",
        "Cobertura territorial latitud": f"{cobertura_lat:.1f}%", 
        "Densidad de incidentes": f"{densidad:.1f} incidentes por grado cuadrado"
    }
    
    for key, value in cobertura_stats.items():
        print(f"• {key}: {value}")

def filtrar_incidentes_colombia(
    incidentes_df: pd.DataFrame, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filtra el DataFrame para mantener solo los incidentes dentro de Colombia.
    
    Parameters:
    -----------
    incidentes_df : pd.DataFrame
        DataFrame con coordenadas de incidentes limpias
    verbose : bool, default=True
        Si True, imprime información del proceso
    
    Returns:
    --------
    pd.DataFrame: DataFrame filtrado solo con incidentes en Colombia
    """
    
    # Aplicar filtro usando el diccionario de límites
    dentro_colombia = (
        (incidentes_df['Longitud'] >= COLOMBIA_BOUNDARIES['lon_min']) & 
        (incidentes_df['Longitud'] <= COLOMBIA_BOUNDARIES['lon_max']) &
        (incidentes_df['Latitud'] >= COLOMBIA_BOUNDARIES['lat_min']) & 
        (incidentes_df['Latitud'] <= COLOMBIA_BOUNDARIES['lat_max'])
    )
    
    incidentes_colombia_final = incidentes_df[dentro_colombia].copy()
    
    if verbose:
        imprimir_seccion("FILTRADO FINAL PARA INCIDENTES EN COLOMBIA")
        
        filtro_stats = {
            "Incidentes antes del filtro": len(incidentes_df),
            "Incidentes en Colombia": len(incidentes_colombia_final),
            "Incidentes removidos": len(incidentes_df) - len(incidentes_colombia_final),
            "Porcentaje conservado": f"{len(incidentes_colombia_final)/len(incidentes_df)*100:.2f}%"
        }
        
        imprimir_estadisticas(filtro_stats, "RESULTADOS DEL FILTRADO")
    
    return incidentes_colombia_final

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def crear_mapa_incidentes_deslizamiento(
    incidentes_df: pd.DataFrame, 
    dentro_colombia_mask: np.ndarray, 
    verbose: bool = True
) -> None:
    """Crea visualizaciones de la validación geográfica para incidentes de deslizamiento."""
    
    if verbose:
        print(f"\nCreando visualizaciones de validación geográfica...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Validación Geográfica - Incidentes de Deslizamiento en Colombia', 
                 fontsize=16, fontweight='bold')
    
    # Separar datos
    incidentes_dentro = incidentes_df[dentro_colombia_mask]
    incidentes_fuera = incidentes_df[~dentro_colombia_mask]
    
    # Mapa 1: Todas las coordenadas con límites
    _crear_mapa_completo(axes[0], incidentes_dentro, incidentes_fuera)
    
    # Mapa 2: Solo incidentes válidos en Colombia (zoom)
    _crear_mapa_colombia_zoom(axes[1], incidentes_dentro)
    
    plt.tight_layout()
    filename = 'validacion_incidentes_deslizamiento_colombia.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"Mapa guardado como: {filename}")
    
    plt.show()

def _crear_mapa_completo(ax, incidentes_dentro: pd.DataFrame, incidentes_fuera: pd.DataFrame) -> None:
    """Crea el mapa completo con todos los incidentes y límites."""
    
    # Plotear incidentes
    if len(incidentes_dentro) > 0:
        ax.scatter(incidentes_dentro['Longitud'], incidentes_dentro['Latitud'], 
                  c='red', s=4, alpha=0.7, 
                  label=f'Incidentes en Colombia ({len(incidentes_dentro):,})')
    
    if len(incidentes_fuera) > 0:
        ax.scatter(incidentes_fuera['Longitud'], incidentes_fuera['Latitud'], 
                  c='gray', s=10, alpha=0.8, 
                  label=f'Incidentes fuera Colombia ({len(incidentes_fuera):,})')
    
    # Dibujar límites de Colombia
    boundaries = COLOMBIA_BOUNDARIES
    ax.axvline(boundaries['lon_min'], color='blue', linestyle='-', linewidth=2, alpha=0.8, label='Límites Colombia')
    ax.axvline(boundaries['lon_max'], color='blue', linestyle='-', linewidth=2, alpha=0.8)
    ax.axhline(boundaries['lat_min'], color='blue', linestyle='-', linewidth=2, alpha=0.8)
    ax.axhline(boundaries['lat_max'], color='blue', linestyle='-', linewidth=2, alpha=0.8)
    
    # Sombrear el área de Colombia
    from matplotlib.patches import Rectangle
    colombia_rect = Rectangle(
        (boundaries['lon_min'], boundaries['lat_min']), 
        boundaries['lon_max'] - boundaries['lon_min'],
        boundaries['lat_max'] - boundaries['lat_min'],
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.2
    )
    ax.add_patch(colombia_rect)
    
    ax.set_title('Validación Geográfica de Incidentes')
    ax.set_xlabel('Longitud (grados)')
    ax.set_ylabel('Latitud (grados)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _crear_mapa_colombia_zoom(ax, incidentes_dentro: pd.DataFrame) -> None:
    """Crea el mapa con zoom a Colombia."""
    
    if len(incidentes_dentro) == 0:
        ax.text(0.5, 0.5, 'No hay incidentes válidos en Colombia', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Sin datos válidos')
        return
    
    ax.scatter(incidentes_dentro['Longitud'], incidentes_dentro['Latitud'], 
              c='darkred', s=3, alpha=0.6)
    
    ax.set_title(f'Incidentes de Deslizamiento en Colombia ({len(incidentes_dentro):,})')
    ax.set_xlabel('Longitud (grados)')
    ax.set_ylabel('Latitud (grados)')
    ax.grid(True, alpha=0.3)
    
    # Ajustar límites del zoom
    margin = 0.5
    ax.set_xlim(incidentes_dentro['Longitud'].min() - margin, 
                incidentes_dentro['Longitud'].max() + margin)
    ax.set_ylim(incidentes_dentro['Latitud'].min() - margin, 
                incidentes_dentro['Latitud'].max() + margin)
    
    # Añadir información de densidad
    area_lon = incidentes_dentro['Longitud'].max() - incidentes_dentro['Longitud'].min()
    area_lat = incidentes_dentro['Latitud'].max() - incidentes_dentro['Latitud'].min()
    
    info_text = (f'Densidad: {len(incidentes_dentro):,} incidentes\n'
                f'Área: {area_lon:.1f}° x {area_lat:.1f}°')
    
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=10)

# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def analizar_patrones_deslizamiento(incidentes_colombia: pd.DataFrame, verbose: bool = True) -> Optional[pd.Series]:
    """Análisis específico para patrones de deslizamiento en Colombia."""
    
    if verbose:
        imprimir_titulo_principal("ANÁLISIS DE PATRONES DE DESLIZAMIENTO EN COLOMBIA")
    
    if len(incidentes_colombia) == 0:
        if verbose:
            print("No hay incidentes válidos para analizar.")
        return None
    
    # Análisis por regiones (usando muestra si hay muchos datos)
    muestra = _obtener_muestra_analisis(incidentes_colombia, verbose)
    distribucion_regional = _analizar_distribucion_regional(muestra, verbose)
    
    # Análisis de hotspots
    if verbose:
        _analizar_hotspots(incidentes_colombia)
    
    return distribucion_regional

def _obtener_muestra_analisis(incidentes_colombia: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Obtiene una muestra para análisis si el dataset es muy grande."""
    if len(incidentes_colombia) > 5000:
        muestra = incidentes_colombia.sample(5000, random_state=42)
        if verbose:
            print(f"Usando muestra de 5,000 incidentes para análisis regional")
        return muestra
    return incidentes_colombia

def _analizar_distribucion_regional(muestra: pd.DataFrame, verbose: bool) -> pd.Series:
    """Analiza la distribución regional de deslizamientos."""
    muestra_regiones = muestra.apply(clasificar_region_colombia, axis=1)
    distribucion_regional = muestra_regiones.value_counts()
    
    if verbose:
        imprimir_seccion("DISTRIBUCIÓN REGIONAL DE DESLIZAMIENTOS")
        
        for region, cantidad in distribucion_regional.items():
            porcentaje = cantidad / len(muestra) * 100
            print(f"• {region}: {cantidad} incidentes ({porcentaje:.1f}%)")
    
    return distribucion_regional

def _analizar_hotspots(incidentes_colombia: pd.DataFrame) -> None:
    """Analiza zonas de alta concentración de deslizamientos."""
    imprimir_seccion("ANÁLISIS DE HOTSPOTS")
    
    # Dividir Colombia en grid para identificar zonas de mayor actividad
    lon_bins = np.linspace(incidentes_colombia['Longitud'].min(), 
                          incidentes_colombia['Longitud'].max(), 10)
    lat_bins = np.linspace(incidentes_colombia['Latitud'].min(), 
                          incidentes_colombia['Latitud'].max(), 10)
    
    # Crear grid de conteo
    grid_counts, _, _ = np.histogram2d(incidentes_colombia['Longitud'], 
                                     incidentes_colombia['Latitud'], 
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

# ============================================================================
# FUNCIONES DE PROCESAMIENTO PRINCIPAL
# ============================================================================

def limpiar_coordenadas(inventario_raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Limpia y valida las coordenadas del inventario.
    
    Parameters:
    -----------
    inventario_raw : pd.DataFrame
        DataFrame original con coordenadas sin limpiar
    verbose : bool, default=True
        Si True, imprime información del proceso
        
    Returns:
    --------
    pd.DataFrame: DataFrame con coordenadas limpias
    """
    
    if verbose:
        imprimir_seccion("LIMPIEZA DE COORDENADAS")
    
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
        for i, valor in enumerate(valores_unicos_problematicos, 1):
            print(f"  {i}. '{valor}'")
    
    # Obtener solo filas válidas
    incidentes_numericos = inventario[~valores_problematicos].copy()
    
    # Convertir tipos de datos
    incidentes_numericos = incidentes_numericos.astype({
        'ID': np.int64,
        'Latitud': np.float32,
        'Longitud': np.float32
    })
    
    # Eliminar longitudes positivas (incorrectas para Colombia)
    longitudes_invalidas = incidentes_numericos['Longitud'] >= 0
    incidentes_numericos = incidentes_numericos[~longitudes_invalidas]
    
    if verbose:
        print(f"Longitudes positivas eliminadas: {longitudes_invalidas.sum():,}")
        print(f"Después de limpieza completa: {len(incidentes_numericos):,} registros")
    
    return incidentes_numericos

def procesar_fechas(incidentes_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Procesa y valida las fechas del inventario.
    
    Parameters:
    -----------
    incidentes_df : pd.DataFrame
        DataFrame con columnas Year, Month, Day
    verbose : bool, default=True
        Si True, imprime información del proceso
        
    Returns:
    --------
    pd.DataFrame: DataFrame con fechas válidas
    """
    
    if verbose:
        print(f"\nProcesando fechas...")
    
    # Convertir columnas a numéricas
    fecha_cols = ['Year', 'Month', 'Day']
    for col in fecha_cols:
        incidentes_df[col] = pd.to_numeric(incidentes_df[col], errors='coerce')
    
    # Filtrar solo filas con valores válidos
    incidentes_with_dates = incidentes_df.dropna(subset=fecha_cols).copy()
    
    # Filtrar rangos válidos
    incidentes_with_dates = incidentes_with_dates[
        (incidentes_with_dates['Year'] >= 1900) & (incidentes_with_dates['Year'] <= 2030) &
        (incidentes_with_dates['Month'] >= 1) & (incidentes_with_dates['Month'] <= 12) &
        (incidentes_with_dates['Day'] >= 1) & (incidentes_with_dates['Day'] <= 31)
    ]
    
    # Crear columna de fecha
    incidentes_with_dates.loc[:, 'Fecha'] = pd.to_datetime(
        incidentes_with_dates[fecha_cols], errors='coerce'
    )
    
    # Eliminar fechas imposibles
    incidentes_with_dates = incidentes_with_dates.dropna(subset=['Fecha']).copy()
    
    if verbose:
        print(f"Registros con fechas válidas: {len(incidentes_with_dates):,}")
    
    return incidentes_with_dates

def remapear_causas(incidentes_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aplica el remapeo de categorías de causas.
    
    Parameters:
    -----------
    incidentes_df : pd.DataFrame
        DataFrame con columna 'Cause'
    verbose : bool, default=True
        Si True, imprime información del proceso
        
    Returns:
    --------
    pd.DataFrame: DataFrame con causas remapeadas
    """
    
    if 'Cause' not in incidentes_df.columns:
        if verbose:
            print("Advertencia: Columna 'Cause' no encontrada")
        return incidentes_df
    
    if verbose:
        valores_antes = incidentes_df['Cause'].value_counts()
        imprimir_seccion("REMAPEO DE CATEGORÍAS DE CAUSAS")
        print(f"Valores únicos antes del remapeo: {len(valores_antes)}")
        print("\nCategorías originales más frecuentes:")
        print(valores_antes.head(10))
    
    # Aplicar remapeo
    incidentes_df.loc[:, 'Cause'] = incidentes_df['Cause'].map(DICT_REMAPPING).fillna(incidentes_df['Cause'])
    
    if verbose:
        valores_despues = incidentes_df['Cause'].value_counts()
        print(f"\nValores únicos después del remapeo: {len(valores_despues)}")
        print("\nCategorías finales:")
        print(valores_despues)
        
        # Mostrar cambios aplicados
        print(f"\nCambios aplicados:")
        for original, nuevo in DICT_REMAPPING.items():
            if original in valores_antes.index:
                print(f"• '{original}' -> '{nuevo}'")
    
    return incidentes_df

def proceso_completo_limpieza_y_validacion(
    inventario_raw: pd.DataFrame, 
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Proceso completo: limpieza + validación con límites de Colombia.
    
    Parameters:
    -----------
    inventario_raw : pd.DataFrame
        DataFrame original con coordenadas sin limpiar
    verbose : bool, default=True
        Si True, imprime información detallada del proceso
        
    Returns:
    --------
    dict: Diccionario con todos los resultados del proceso
    """
    
    if verbose:
        imprimir_titulo_principal("PROCESO COMPLETO: LIMPIEZA + VALIDACIÓN DE INCIDENTES")
    
    registros_inicial = len(inventario_raw)
    
    # FASE 1: Limpieza de coordenadas
    incidentes_numericos = limpiar_coordenadas(inventario_raw, verbose)
    
    # FASE 2: Validación geográfica
    if verbose:
        imprimir_seccion("VALIDACIÓN CON LÍMITES DE COLOMBIA")
    
    # Aplicar filtro geográfico
    dentro_colombia = (
        (incidentes_numericos['Longitud'] >= COLOMBIA_BOUNDARIES['lon_min']) & 
        (incidentes_numericos['Longitud'] <= COLOMBIA_BOUNDARIES['lon_max']) &
        (incidentes_numericos['Latitud'] >= COLOMBIA_BOUNDARIES['lat_min']) & 
        (incidentes_numericos['Latitud'] <= COLOMBIA_BOUNDARIES['lat_max'])
    )
    
    incidentes_colombia = incidentes_numericos[dentro_colombia].copy()
    incidentes_fuera = incidentes_numericos[~dentro_colombia].copy()
    
    # FASE 3: Remapeo de causas
    incidentes_colombia = remapear_causas(incidentes_colombia, verbose)
    
    # FASE 4: Procesamiento de fechas
    incidentes_with_dates = procesar_fechas(incidentes_colombia, verbose)
    
    if verbose:
        imprimir_seccion("RESULTADOS FINALES")
        
        resumen = {
            "Registros originales": registros_inicial,
            "Después de limpieza": f"{len(incidentes_numericos):,} ({len(incidentes_numericos)/registros_inicial*100:.1f}%)",
            "Incidentes en Colombia": f"{len(incidentes_colombia):,} ({len(incidentes_colombia)/registros_inicial*100:.1f}%)",
            "Con fechas válidas": f"{len(incidentes_with_dates):,} ({len(incidentes_with_dates)/registros_inicial*100:.1f}%)",
            "Tasa de éxito total": f"{len(incidentes_colombia)/registros_inicial*100:.1f}%"
        }
        
        imprimir_estadisticas(resumen, "RESUMEN DEL PROCESO")
        
        if len(incidentes_colombia) > 0:
            print(f"\nESTADÍSTICAS DE COORDENADAS FINALES:")
            imprimir_coordenadas_info(incidentes_colombia)
    
    # Resultados finales
    resultados = {
        'incidentes_colombia': incidentes_colombia,
        'incidentes_withdates': incidentes_with_dates,
        'incidentes_fuera_colombia': incidentes_fuera,
    }
    
    return resultados

# ============================================================================
# FUNCIÓN PRINCIPAL DE CARGA
# ============================================================================

def load_inventario(
                    ruta_archivo: Optional[str] = None, 
                    verbose: bool = True, 
                    aplicar_validacion_geografica: bool = True, 
                    guardar_resultados: bool = False, 
                    carpeta_salida: Optional[str] = None
                ) -> Dict[str, pd.DataFrame]:
    """
    Carga y procesa completamente el inventario de incidentes de deslizamiento.
    
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
    dict: Diccionario con elementos principales:
        - 'data_cruda': DataFrame original sin modificaciones
        - 'data_transformada': DataFrame final procesado y listo para análisis
        - 'incidentes_withdates': DataFrame con fechas válidas
    
    Examples:
    ---------
    # Uso básico
    >>> resultado = load_inventario()
    >>> data_cruda = resultado['data_cruda']
    >>> data_transformada = resultado['data_transformada']
    
    # Uso silencioso
    >>> resultado = load_inventario(verbose=False)
    
    # Con archivo específico y guardado
    >>> resultado = load_inventario(
    ...     ruta_archivo="C:/mi_carpeta/mi_inventario.xlsx",
    ...     guardar_resultados=True,
    ...     carpeta_salida="C:/mi_carpeta/resultados"
    ... )
    """
    
    # Configurar ruta por defecto
    if ruta_archivo is None:
        path_inventario_landslide = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\databases"
        ruta_archivo = os.path.join(path_inventario_landslide, 'Colombia_database1900_2021(completa).xlsx')
    
    if verbose:
        imprimir_titulo_principal("CARGA DE INVENTARIO DE DESLIZAMIENTOS")
        print(f"Archivo: {os.path.basename(ruta_archivo)}")
        print(f"Ruta: {ruta_archivo}")
    
    try:
        # Cargar archivo
        if verbose:
            print(f"\nCargando archivo Excel...")
        
        data_cruda = pd.read_excel(ruta_archivo)
        
        if verbose:
            archivo_info = {
                "Estado": "Cargado exitosamente",
                "Dimensiones": f"{data_cruda.shape[0]:,} filas x {data_cruda.shape[1]} columnas",
                "Columnas": f"{len(data_cruda.columns)} columnas disponibles"
            }
            
            for key, value in archivo_info.items():
                print(f"• {key}: {value}")
            
            print(f"• Lista de columnas: {list(data_cruda.columns)}")
        
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo: {ruta_archivo}")
        return {}
    except Exception as e:
        print(f"ERROR al cargar el archivo: {e}")
        return {}
    
    # Proceso completo de limpieza y validación
    if verbose:
        print(f"\nIniciando proceso de transformación...")
    
    resultados = proceso_completo_limpieza_y_validacion(data_cruda, verbose=verbose)
    data_transformada = resultados['incidentes_colombia']
    
    # Aplicar validación geográfica adicional si se solicita
    if aplicar_validacion_geografica and len(data_transformada) > 0:
        
        if verbose:
            print(f"\nAPLICANDO VALIDACIÓN GEOGRÁFICA ADICIONAL...")
        
            # Validación detallada
            mask_colombia, incidentes_col, incidentes_fuera, stats = validar_incidentes_deslizamiento_colombia(
                                                                                        data_transformada, verbose=verbose
                                                                                    )
            print(f"\nCreando visualizaciones...")
            crear_mapa_incidentes_deslizamiento(data_transformada, mask_colombia, verbose=verbose)
            
            print(f"\nAnalizando patrones regionales...")
            distribucion_regional = analizar_patrones_deslizamiento(data_transformada, verbose=verbose)
    
    # Guardar resultados si se solicita
    if guardar_resultados:
        _guardar_resultados(data_cruda, data_transformada, ruta_archivo, carpeta_salida, verbose)
    
    # Resultado simplificado
    resultado_simplificado = {
                            'data_cruda': data_cruda,
                            'data_transformada': data_transformada,
                            'incidentes_withdates': resultados['incidentes_withdates']
                        }
    
    if verbose:
        _imprimir_resumen_final(data_cruda, data_transformada)
    
    return resultado_simplificado

def _guardar_resultados(
                        data_cruda: pd.DataFrame, 
                        data_transformada: pd.DataFrame, 
                        ruta_archivo: str, 
                        carpeta_salida: Optional[str], 
                        verbose: bool
                    ) -> None:
    """Guarda los resultados en archivos CSV."""
    
    if carpeta_salida is None:
        carpeta_salida = os.path.dirname(ruta_archivo)
    
    if verbose:
        print(f"\nGuardando resultados en: {carpeta_salida}")
    
    # Guardar ambas versiones
    archivo_crudo = os.path.join(carpeta_salida, 'inventario_deslizamientos_data_cruda.csv')
    archivo_transformado = os.path.join(carpeta_salida, 'inventario_deslizamientos_data_transformada.csv')
    
    data_cruda.to_csv(archivo_crudo, index=False)
    data_transformada.to_csv(archivo_transformado, index=False)
    
    if verbose:
        print(f"Archivos guardados:")
        print(f"• {os.path.basename(archivo_crudo)}")
        print(f"• {os.path.basename(archivo_transformado)}")

def _imprimir_resumen_final(data_cruda: pd.DataFrame, data_transformada: pd.DataFrame) -> None:
    """Imprime el resumen final del proceso."""
    
    imprimir_titulo_principal("PROCESO COMPLETADO EXITOSAMENTE")
    
    tasa_exito = (len(data_transformada) / len(data_cruda) * 100) if len(data_cruda) > 0 else 0
    
    resumen_final = {
        "Data cruda": f"{len(data_cruda):,} registros originales",
        "Data transformada": f"{len(data_transformada):,} registros procesados",
        "Tasa de éxito": f"{tasa_exito:.1f}%"
    }
    
    print("Resultado simplificado:")
    for key, value in resumen_final.items():
        print(f"• {key}: {value}")
    
    if 'Cause' in data_transformada.columns:
        causas_finales = data_transformada['Cause'].value_counts()
        print(f"• Categorías de causas finales: {len(causas_finales)}")
    
    print(f"{SEPARADOR_PRINCIPAL}")

# ============================================================================
# EJEMPLO DE USO
# ============================================================================

# if __name__ == "__main__":
#     """
#     Ejemplo de uso del sistema de análisis de deslizamientos.
#     """
#     
#     # Ejemplo básico
#     print("Ejecutando ejemplo de análisis de deslizamientos en Colombia...")
#     
#     # resultado = load_inventario()
#     # 
#     # if resultado:
#     #     data_cruda = resultado['data_cruda']
#     #     data_transformada = resultado['data_transformada']
#     #     
#     #     print(f"\nAnálisis completado:")
#     #     print(f"- Datos originales: {len(data_cruda):,} registros")
#     #     print(f"- Datos procesados: {len(data_transformada):,} registros")
#     #     
#     #     if 'Cause' in data_transformada.columns:
#     #         print(f"\nCausas principales:")
#     #         causas = data_transformada['Cause'].value_counts().head()
#     #         for causa, cantidad in causas.items():
#     #             print(f"  • {causa}: {cantidad:,} incidentes")
#     
#     print("Para ejecutar el análisis, descomenta las líneas en el bloque __main__")