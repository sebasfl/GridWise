# src/analysis/eda_building_quality_fast.py
# Optimized version for large datasets
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(parquet_path: Path, sample_size: int = None) -> pd.DataFrame:
    """Load parquet data and prepare for analysis with optional sampling."""
    df = pd.read_parquet(parquet_path)
    
    # Ensure we have the right columns
    ts_col = "timestamp_local" if "timestamp_local" in df.columns else "timestamp_utc"
    if "meter" in df.columns:
        df = df[df["meter"] == "electricity"].copy()
    
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df[ts_col] = pd.to_datetime(df[ts_col])
    
    # Rename timestamp column for consistency
    if ts_col != "timestamp_local":
        df = df.rename(columns={ts_col: "timestamp_local"})
    
    # Optional sampling for very large datasets
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} records from {len(df):,} total records...")
        # Stratified sampling by building to maintain representation
        df = df.groupby('building_id').apply(
            lambda x: x.sample(min(len(x), sample_size // df['building_id'].nunique()), 
                              random_state=42)
        ).reset_index(drop=True)
    
    return df.sort_values(["timestamp_local", "building_id"]).reset_index(drop=True)

def analyze_missing_data_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Fast missing data analysis using groupby operations."""
    ts_col = "timestamp_local"
    
    # Get date range
    min_date = df[ts_col].min()
    max_date = df[ts_col].max()
    expected_hours = int((max_date - min_date).total_seconds() / 3600) + 1
    
    # Aggregate statistics per building
    building_stats = df.groupby('building_id').agg({
        'value': ['count', lambda x: x.isna().sum(), 'mean', 'std', 'min', 'max'],
        'timestamp_local': ['min', 'max', 'nunique']
    }).round(3)
    
    # Flatten column names
    building_stats.columns = ['total_readings', 'missing_values', 'mean_consumption', 
                             'std_consumption', 'min_consumption', 'max_consumption',
                             'start_date', 'end_date', 'unique_timestamps']
    
    # Calculate derived metrics
    building_stats['zero_values'] = df.groupby('building_id')['value'].apply(lambda x: (x == 0).sum())
    building_stats['negative_values'] = df.groupby('building_id')['value'].apply(lambda x: (x < 0).sum())
    
    # Completeness calculations
    building_stats['expected_readings'] = expected_hours
    building_stats['missing_timestamps'] = expected_hours - building_stats['unique_timestamps']
    building_stats['completeness_pct'] = ((building_stats['total_readings'] - building_stats['missing_values']) / expected_hours * 100).round(2)
    
    # Outlier detection (IQR method)
    def calculate_outliers(group):
        values = group['value'].dropna()
        if len(values) < 4:
            return pd.Series({'outliers': 0, 'outlier_pct': 0})
        
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((values < lower) | (values > upper)).sum()
        
        return pd.Series({
            'outliers': outliers,
            'outlier_pct': round((outliers / len(values)) * 100, 2)
        })
    
    outlier_stats = df.groupby('building_id').apply(calculate_outliers)
    building_stats = building_stats.join(outlier_stats)
    
    # Quality score calculation
    building_stats['coefficient_variation'] = (building_stats['std_consumption'] / 
                                              building_stats['mean_consumption']).fillna(np.inf)
    
    building_stats['completeness_score'] = building_stats['completeness_pct'].clip(0, 100)
    building_stats['stability_score'] = 100 / (1 + building_stats['coefficient_variation'].replace([np.inf, -np.inf], 10))
    building_stats['consistency_score'] = (100 - building_stats['outlier_pct']).clip(0, 100)
    
    building_stats['quality_score'] = (
        0.4 * building_stats['completeness_score'] +
        0.3 * building_stats['stability_score'] +
        0.3 * building_stats['consistency_score']
    ).round(2)
    
    return building_stats.reset_index().sort_values('quality_score', ascending=False)

def create_fast_visualizations(quality_df: pd.DataFrame, output_dir: Path):
    """Create essential visualizations efficiently."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    
    # 1. Quality overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Completeness distribution
    axes[0,0].hist(quality_df['completeness_pct'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Completeness %')
    axes[0,0].set_ylabel('Number of Buildings')
    axes[0,0].set_title('Data Completeness Distribution')
    axes[0,0].axvline(quality_df['completeness_pct'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {quality_df["completeness_pct"].mean():.1f}%')
    axes[0,0].legend()
    
    # Quality score distribution
    axes[0,1].hist(quality_df['quality_score'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_xlabel('Quality Score')
    axes[0,1].set_ylabel('Number of Buildings')
    axes[0,1].set_title('Quality Score Distribution')
    axes[0,1].axvline(quality_df['quality_score'].mean(), color='red', linestyle='--',
                     label=f'Mean: {quality_df["quality_score"].mean():.1f}')
    axes[0,1].legend()
    
    # Outlier percentage distribution
    axes[1,0].hist(quality_df['outlier_pct'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1,0].set_xlabel('Outlier %')
    axes[1,0].set_ylabel('Number of Buildings')
    axes[1,0].set_title('Outlier Percentage Distribution')
    
    # Consumption statistics
    axes[1,1].scatter(quality_df['mean_consumption'], quality_df['std_consumption'], 
                     alpha=0.6, s=20, c=quality_df['quality_score'], cmap='RdYlGn')
    axes[1,1].set_xlabel('Mean Consumption')
    axes[1,1].set_ylabel('Std Consumption') 
    axes[1,1].set_title('Consumption Variability')
    cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
    cbar.set_label('Quality Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_overview_fast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top/Bottom buildings
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Worst quality buildings
    worst_10 = quality_df.nsmallest(10, 'quality_score')
    axes[0].barh(range(len(worst_10)), worst_10['quality_score'])
    axes[0].set_yticks(range(len(worst_10)))
    axes[0].set_yticklabels(worst_10['building_id'].astype(str), fontsize=8)
    axes[0].set_xlabel('Quality Score')
    axes[0].set_title('10 Lowest Quality Buildings')
    
    # Best quality buildings  
    best_10 = quality_df.nlargest(10, 'quality_score')
    axes[1].barh(range(len(best_10)), best_10['quality_score'])
    axes[1].set_yticks(range(len(best_10)))
    axes[1].set_yticklabels(best_10['building_id'].astype(str), fontsize=8)
    axes[1].set_xlabel('Quality Score')
    axes[1].set_title('10 Highest Quality Buildings')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_bottom_buildings.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_fast_report(quality_df: pd.DataFrame, output_dir: Path, 
                        original_records: int, processed_records: int):
    """Generate summary report with processing info."""
    with open(output_dir / 'quality_report_fast.txt', 'w') as f:
        f.write("SMART GRID - FAST DATA QUALITY ANALYSIS\n")
        f.write("=" * 45 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original Records: {original_records:,}\n")
        f.write(f"Processed Records: {processed_records:,}\n")
        f.write(f"Total Buildings: {len(quality_df):,}\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Quality Score: {quality_df['quality_score'].mean():.2f}\n")
        f.write(f"Median Quality Score: {quality_df['quality_score'].median():.2f}\n")
        f.write(f"Average Completeness: {quality_df['completeness_pct'].mean():.2f}%\n")
        f.write(f"Average Outlier Rate: {quality_df['outlier_pct'].mean():.2f}%\n\n")
        
        # Quality categories
        high_quality = quality_df[quality_df['quality_score'] >= 80]
        medium_quality = quality_df[(quality_df['quality_score'] >= 60) & (quality_df['quality_score'] < 80)]
        low_quality = quality_df[quality_df['quality_score'] < 60]
        
        f.write("QUALITY CATEGORIES\n")
        f.write("-" * 18 + "\n")
        f.write(f"High Quality (≥80): {len(high_quality):,} buildings ({len(high_quality)/len(quality_df)*100:.1f}%)\n")
        f.write(f"Medium Quality (60-79): {len(medium_quality):,} buildings ({len(medium_quality)/len(quality_df)*100:.1f}%)\n")
        f.write(f"Low Quality (<60): {len(low_quality):,} buildings ({len(low_quality)/len(quality_df)*100:.1f}%)\n\n")
        
        f.write("RECOMMENDED THRESHOLDS FOR CLEANING\n")
        f.write("-" * 35 + "\n")
        f.write("Conservative (keep most data):\n")
        f.write("  --min-quality-score 40 --min-completeness 60 --max-outlier-pct 20\n")
        f.write("Balanced (recommended):\n")
        f.write("  --min-quality-score 60 --min-completeness 75 --max-outlier-pct 15\n")
        f.write("Strict (best quality):\n")
        f.write("  --min-quality-score 75 --min-completeness 85 --max-outlier-pct 10\n\n")
        
        f.write("TOP 5 BEST BUILDINGS\n")
        f.write("-" * 20 + "\n")
        for _, row in quality_df.head(5).iterrows():
            f.write(f"{row['building_id']}: Quality={row['quality_score']:.1f}, "
                   f"Completeness={row['completeness_pct']:.1f}%, "
                   f"Outliers={row['outlier_pct']:.1f}%\n")
        
        f.write("\nTOP 5 WORST BUILDINGS\n")
        f.write("-" * 21 + "\n")
        for _, row in quality_df.tail(5).iterrows():
            f.write(f"{row['building_id']}: Quality={row['quality_score']:.1f}, "
                   f"Completeness={row['completeness_pct']:.1f}%, "
                   f"Outliers={row['outlier_pct']:.1f}%\n")

def main():
    parser = argparse.ArgumentParser(description='Fast building energy data quality analysis')
    parser.add_argument('--parquet', type=Path, required=True, help='Path to parquet file')
    parser.add_argument('--output', type=Path, default='data/analysis', help='Output directory')
    parser.add_argument('--sample-size', type=int, help='Limit analysis to N records (for very large datasets)')
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_and_prepare_data(args.parquet, args.sample_size)
    original_size = pd.read_parquet(args.parquet).shape[0] if not args.sample_size else len(df)
    
    print(f"Analyzing {len(df):,} records for {df['building_id'].nunique():,} buildings...")
    
    print("Running fast quality analysis...")
    quality_df = analyze_missing_data_fast(df)
    
    print("Creating visualizations...")
    create_fast_visualizations(quality_df, args.output)
    
    print("Generating report...")
    generate_fast_report(quality_df, args.output, original_size, len(df))
    
    # Save results
    quality_df.to_csv(args.output / 'building_quality_scores.csv', index=False)
    
    print(f"\n✅ Fast analysis complete! Results in {args.output}")
    print(f"📊 Quality score range: {quality_df['quality_score'].min():.1f} - {quality_df['quality_score'].max():.1f}")
    print(f"🏢 Buildings analyzed: {len(quality_df):,}")
    print(f"⭐ High quality (≥80): {(quality_df['quality_score'] >= 80).sum():,}")
    print(f"⚠️  Low quality (<60): {(quality_df['quality_score'] < 60).sum():,}")

if __name__ == "__main__":
    main()