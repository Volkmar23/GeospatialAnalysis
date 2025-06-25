import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import contextily as ctx
import warnings
warnings.filterwarnings('ignore')

# Load your GeoDataFrame (update path as needed)
# gdf = gpd.read_file('path/to/colombia_watersheds_elevation_slope_shapefile.shp')



# ===================================================================
# 1. BASIC DATA EXPLORATION
# ===================================================================

def explore_slope_data(gdf):
    """Explore the slope statistics in your GeoDataFrame"""
    
    print("📊 SLOPE DATA SUMMARY:")
    print("-" * 30)
    
    slope_columns = ['slope_min', 'slope_mean', 'slope_max', 'slope_p90', 'slope_p95']
    
    for col in slope_columns:
        if col in gdf.columns:
            print(f"{col:12}: {gdf[col].min():.1f}° to {gdf[col].max():.1f}° (mean: {gdf[col].mean():.1f}°)")
    
    print(f"\nTotal watersheds: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    
    # Check for missing values
    missing_slopes = gdf[slope_columns].isnull().sum()
    if missing_slopes.any():
        print(f"\n⚠️ Missing values:")
        print(missing_slopes[missing_slopes > 0])
    else:
        print("✅ No missing slope values")
    
    return gdf

# ===================================================================
# 2. CHOROPLETH MAP FUNCTIONS
# ===================================================================

def plot_slope_choropleth(gdf, slope_column='slope_p95', figsize=(15, 12)):
    """
    Create a choropleth map highlighting high slope areas
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Your watershed data
    slope_column : str
        Which slope column to use ('slope_mean', 'slope_max', 'slope_p95', etc.)


    # Quick single plot example
    plot_slope_choropleth(gdf, slope_column='slope_p95', figsize=(12, 10))
    """
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Color schemes for different maps
    color_schemes = ['Reds', 'YlOrRd', 'plasma', 'viridis']
    classification_schemes = ['quantiles', 'equal_interval', 'natural_breaks', 'percentiles']
    
    for i, (scheme, cmap) in enumerate(zip(classification_schemes, color_schemes)):
        ax = axes[i]
        
        # Create the choropleth map
        if scheme == 'percentiles':
            # Custom percentile-based classification
            vmin, vmax = gdf[slope_column].quantile([0.1, 0.9])
        else:
            vmin, vmax = gdf[slope_column].min(), gdf[slope_column].max()
        
        gdf.plot(
            column=slope_column,
            ax=ax,
            cmap=cmap,
            legend=True,
            scheme=scheme if scheme != 'percentiles' else None,
            k=5,  # Number of classes
            edgecolor='white',
            linewidth=0.1
        )
        
        ax.set_title(f'{slope_column.title()} - {scheme.replace("_", " ").title()}', 
                     fontsize=12, fontweight='bold')
        ax.set_axis_off()
    
    plt.suptitle(f'Slope Analysis: {slope_column.replace("_", " ").title()} (degrees)', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Graficas\Entrega 2\FixedVariables\slope_coefficient.png', dpi=300, bbox_inches='tight')  # LINE 1: Save figure
    plt.close()

def plot_high_slope_focus(gdf, threshold_percentile=90, slope_column='slope_p95'):
    """
    Focus specifically on the highest slope areas
    """
    
    # Calculate threshold
    threshold = gdf[slope_column].quantile(threshold_percentile/100)
    
    # Create high slope mask
    high_slope_mask = gdf[slope_column] >= threshold
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: All watersheds with high slopes highlighted
    ax1 = axes[0]
    
    # Plot all watersheds in light gray
    gdf.plot(ax=ax1, color='lightgray', edgecolor='white', linewidth=0.1, alpha=0.7)
    
    # Highlight high slope watersheds
    gdf[high_slope_mask].plot(
        ax=ax1, 
        column=slope_column,
        cmap='Reds',
        legend=True,
        edgecolor='darkred',
        linewidth=0.5
    )
    
    ax1.set_title(f'High Slope Watersheds\n(Top {100-threshold_percentile}% - {slope_column} ≥ {threshold:.1f}°)', 
                  fontsize=14, fontweight='bold')
    ax1.set_axis_off()
    
    # Right plot: Only high slope watersheds with detailed classification
    ax2 = axes[1]
    
    high_slope_gdf = gdf[high_slope_mask].copy()
    
    if len(high_slope_gdf) > 0:
        high_slope_gdf.plot(
            column=slope_column,
            ax=ax2,
            cmap='plasma',
            legend=True,
            scheme='quantiles',
            k=5,
            edgecolor='black',
            linewidth=0.3
        )
        
        ax2.set_title(f'Detailed View: High Slope Areas Only\n({len(high_slope_gdf)} watersheds)', 
                      fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No high slope areas found', 
                 transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('No High Slope Areas', fontsize=14)
    
    ax2.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
    print(f"📈 High Slope Analysis (≥{threshold:.1f}°):")
    print(f"   • {len(high_slope_gdf)} watersheds ({len(high_slope_gdf)/len(gdf)*100:.1f}%)")
    print(f"   • Slope range: {high_slope_gdf[slope_column].min():.1f}° to {high_slope_gdf[slope_column].max():.1f}°")
    
    return high_slope_gdf

def plot_slope_comparison(gdf):
    """
    Compare different slope measures side by side
    """
    
    slope_columns = ['slope_mean', 'slope_max', 'slope_p90', 'slope_p95']
    available_columns = [col for col in slope_columns if col in gdf.columns]
    
    n_cols = len(available_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 8))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(available_columns):
        ax = axes[i]
        
        gdf.plot(
            column=col,
            ax=ax,
            cmap='YlOrRd',
            legend=True,
            scheme='quantiles',
            k=5,
            edgecolor='white',
            linewidth=0.1
        )
        
        ax.set_title(f'{col.replace("_", " ").title()}\n(Mean: {gdf[col].mean():.1f}°)', 
                     fontsize=12, fontweight='bold')
        ax.set_axis_off()
    
    plt.suptitle('Comparison of Different Slope Measures', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ===================================================================
# 3. STATISTICAL ANALYSIS
# ===================================================================

def analyze_slope_distribution(gdf):
    """
    Analyze the statistical distribution of slope values
    """
    
    slope_columns = ['slope_mean', 'slope_max', 'slope_p90', 'slope_p95']
    available_columns = [col for col in slope_columns if col in gdf.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(available_columns[:4]):
        ax = axes[i]
        
        # Histogram with KDE
        gdf[col].hist(bins=30, alpha=0.7, ax=ax, color='skyblue', edgecolor='black')
        
        # Add vertical lines for key statistics
        mean_val = gdf[col].mean()
        median_val = gdf[col].median()
        p90_val = gdf[col].quantile(0.9)
        
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}°')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}°')
        ax.axvline(p90_val, color='orange', linestyle='--', label=f'90th percentile: {p90_val:.1f}°')
        
        ax.set_title(f'Distribution: {col.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Slope (degrees)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_slope_categories(gdf, slope_column='slope_p95'):
    """
    Create categorical classifications for slope
    """
    
    # Define slope categories (adjust thresholds as needed)
    def categorize_slope(slope_value):
        if slope_value < 5:
            return 'Flat (< 5°)'
        elif slope_value < 10:
            return 'Gentle (5-10°)'
        elif slope_value < 15:
            return 'Moderate (10-15°)'
        elif slope_value < 25:
            return 'Steep (15-25°)'
        else:
            return 'Very Steep (> 25°)'
    
    # Apply categorization
    gdf['slope_category'] = gdf[slope_column].apply(categorize_slope)
    
    # Plot categorical map
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors for categories
    colors = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494']
    categories = ['Flat (< 5°)', 'Gentle (5-10°)', 'Moderate (10-15°)', 'Steep (15-25°)', 'Very Steep (> 25°)']
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors[:len(gdf['slope_category'].unique())])
    
    gdf.plot(
        column='slope_category',
        ax=ax,
        cmap=cmap,
        legend=True,
        edgecolor='white',
        linewidth=0.1
    )
    
    ax.set_title(f'Slope Categories Based on {slope_column.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Graficas\Entrega 2\FixedVariables\slope_distributions.png', dpi=300, bbox_inches='tight')  # LINE 1: Save figure
    
    # Print category statistics
    category_stats = gdf['slope_category'].value_counts().sort_index()
    print("📊 Slope Category Distribution:")
    print("-" * 40)
    for category, count in category_stats.items():
        percentage = (count / len(gdf)) * 100
        print(f"{category:20}: {count:4d} watersheds ({percentage:5.1f}%)")
    
    return gdf

# ===================================================================
# 4. MAIN EXECUTION FUNCTION
# ===================================================================

def main_slope_analysis(gdf):
    """
    Run the complete slope analysis


    # ===================================================================
    # 5. USAGE EXAMPLE
    # ===================================================================
    
    # Example usage (uncomment and modify path):
    
    # Load your data
    #gdf = gpd.read_file('colombia_watersheds_elevation_slope_shapefile.shp')
    
    # Run complete analysis
    gdf_analyzed, high_slope_areas = main_slope_analysis(file)
    
    
    """

    print("🗺️ Choropleth Map Analysis for Slope Data")
    print("=" * 50)
    
    print("🚀 Starting Comprehensive Slope Analysis")
    print("=" * 50)
    
    # 1. Explore the data
    gdf = explore_slope_data(gdf)
    
    print("\n" + "="*50)
    
    # 2. Basic choropleth maps
    print("🗺️ Creating choropleth maps...")
    plot_slope_choropleth(gdf, slope_column='slope_p95')
    
    # 3. Focus on high slope areas
##     print("🔍 Analyzing high slope areas...")
##     high_slope_gdf = plot_high_slope_focus(gdf, threshold_percentile=90)
##     
##     # 4. Compare different slope measures
##     print("📊 Comparing slope measures...")
##     plot_slope_comparison(gdf)
##     
##     # 5. Statistical analysis
##     print("📈 Analyzing slope distributions...")
##     analyze_slope_distribution(gdf)
    
    # 6. Create categorical classification
    print("🏷️ Creating slope categories...")
    gdf = create_slope_categories(gdf, slope_column='slope_mean')
    
    print("\n✅ Analysis complete!")



    print("🎯 Ready to analyze your slope data!")
    print("👆 Load your GeoDataFrame and run: main_slope_analysis(gdf)")
    ## return gdf, high_slope_gdf
    return gdf




