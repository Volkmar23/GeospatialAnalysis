import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import os

# Definir orden de las categorías de riesgo para que aparezcan consistentemente
RISK_ORDER = ['Muy baja', 'Baja', 'Intermedia', 'Alta', 'Muy Alta']

# Definir paleta de colores corporativa para categorías de riesgo según los colores especificados
# Desde verde (riesgo muy bajo) hasta morado (riesgo muy alto)
RISK_COLORS = {
                'Muy baja': '#008000',  # Verde
                'Baja': '#FFFF00',      # Amarillo
                'Intermedia': '#FFA500', # Naranja
                'Alta': '#FF0000',      # Rojo
                'Muy Alta': '#800080'   # Morado
            }

# Definir paleta de colores para escenarios climáticos
SCENARIO_COLORS = {
                    'SSP2 4.5': '#4575b4',  # Azul (escenario más optimista)
                    'SSP3 7.0': '#FFA500',  # Naranja (escenario intermedio)
                    'SSP5 8.5': '#FF0000'   # Rojo (escenario más pesimista)
                }

# Función para establecer la ruta de almacenamiento
def get_storage_path(filename):
    """
    Devuelve la ruta completa para guardar archivos en la carpeta especificada
    """
    url_storage = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Notebooks_dowjones\Landslide\PLOTS\Landslide"
    return os.path.join(url_storage, filename)

# Función para preparar el dataframe para análisis
def prepare_data(df):
    """
    Prepara los datos para el análisis, asegurando los tipos correctos
    y calculando algunas métricas útiles.
    """
    # Asegurar que las columnas categóricas son de tipo categoría con el orden correcto
    df['landslide_label'] = pd.Categorical(df['landslide_label'], categories=RISK_ORDER, ordered=True)
    df['Escenario'] = pd.Categorical(df['Escenario'], categories=['SSP2 4.5', 'SSP3 7.0', 'SSP5 8.5'], ordered=True)
    df['horizonte'] = pd.Categorical(df['horizonte'], categories=['2020-2040', '2040-2060', '2060-2080', '2080-2100'], ordered=True)
    
    # Verificar si hay valores NaN solo en las columnas relevantes para nuestras visualizaciones
    relevant_columns = ['Escenario', 'horizonte', 'landslide_label', 'Pendiente_12_5m_30m']
    missing_data = df[relevant_columns].isnull().sum()
    
    if missing_data.sum() > 0:
        print("Advertencia: Se encontraron valores nulos en columnas importantes:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"  - {col}: {count} valores nulos")
        
    return df

# Función para crear gráfico de líneas que muestre tendencias temporales
def plot_risk_trend_lines(df, title="Variación del riesgo de deslizamiento para circuitos en Colombia", 
                         filename="tendencias_riesgo_alto.png"):
    """
    Crea un gráfico de líneas que muestra cómo evoluciona el porcentaje de torres
    con riesgo 'Alto' o 'Muy Alto' a lo largo del tiempo para cada escenario.
    """
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Definir categorías de alto riesgo
    high_risk_categories = ['Alta', 'Muy Alta']
    
    # Mapeo de horizontes a años individuales para el eje X
    horizon_to_year = {
        '2020-2040': '2025',
        '2040-2060': '2040',
        '2060-2080': '2060',
        '2080-2100': '2080'
    }
    
    # Para cada escenario
    for scenario in df['Escenario'].unique():
        scenario_data = df[df['Escenario'] == scenario]
        
        # Calcular porcentaje de torres con alto riesgo por horizonte
        high_risk_percentages = []
        horizons = []
        years_for_plot = []
        
        for horizon in scenario_data['horizonte'].unique():
            horizon_data = scenario_data[scenario_data['horizonte'] == horizon]
            
            # Calcular el porcentaje de torres con riesgo alto o muy alto
            high_risk_count = horizon_data[horizon_data['landslide_label'].isin(high_risk_categories)].shape[0]
            total_count = horizon_data.shape[0]
            percentage = (high_risk_count / total_count) * 100 if total_count > 0 else 0
            
            high_risk_percentages.append(percentage)
            horizons.append(horizon)
            years_for_plot.append(horizon_to_year[horizon])
        
        # Plotear línea para este escenario
        ax.plot(years_for_plot, high_risk_percentages, marker='o', linewidth=3, 
               label=scenario, color=SCENARIO_COLORS[scenario])
    
    # Configurar ejes y etiquetas
    ax.set_xlabel('Horizonte Temporal', fontsize=14, fontweight='bold')
    ax.set_ylabel('Porcentaje de Torres con Riesgo Alto o Muy Alto (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Mejoras estéticas
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)
    
    # Añadir leyenda
    ax.legend(title="Escenario", fontsize=12, title_fontsize=13)
    
    # Añadir anotaciones para cada punto
    for scenario in df['Escenario'].unique():
        scenario_data = df[df['Escenario'] == scenario]
        
        # Contar riesgos altos por horizonte
        for i, horizon in enumerate(scenario_data['horizonte'].unique()):
            horizon_data = scenario_data[scenario_data['horizonte'] == horizon]
            
            high_risk_count = horizon_data[horizon_data['landslide_label'].isin(high_risk_categories)].shape[0]
            total_count = horizon_data.shape[0]
            percentage = (high_risk_count / total_count) * 100 if total_count > 0 else 0
            
            ax.annotate(f'{percentage:.1f}%', 
                      (horizon_to_year[horizon], percentage),
                      textcoords="offset points", 
                      xytext=(0, 10), 
                      ha='center', 
                      fontsize=10, 
                      fontweight='bold')
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura en la ruta especificada
    plt.savefig(get_storage_path(filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

# Función principal simplificada para generar solo el gráfico de tendencias
def generate_visualization(df):
    """
    Genera únicamente la visualización de tendencias de riesgo alto.
    """
    # Preparar datos
    df = prepare_data(df)
    
    # Generar visualización de tendencias
    plot_risk_trend_lines(df)


    ### las imagenes estan ahora en Notebooks_dowjones/Resumen
    print("Visualización generada exitosamente en la ruta:", 
          r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Notebooks_dowjones\Landslide\PLOTS\Landslide")

