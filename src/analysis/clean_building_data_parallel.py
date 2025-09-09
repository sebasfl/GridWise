# src/analysis/clean_building_data_parallel.py
# GPU-accelerated parallel version using CuDF and CuPy
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available (CuDF + CuPy)")
    # Import pandas/numpy as fallback aliases
    import pandas as pd
    import numpy as np
except ImportError:
    # Fallback to pandas/numpy if RAPIDS not available
    import pandas as pd
    import numpy as np
    GPU_AVAILABLE = False
    print("⚠️  GPU libraries not available, using CPU fallback")
    # Create dummy cudf/cupy aliases
    cudf = pd
    cp = np

from pathlib import Path
from typing import Dict, List, Tuple

class ParallelBuildingDataCleaner:
    """GPU-accelerated parallel data cleaning for massive datasets."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cleaning_stats = {}
        self.gpu_available = GPU_AVAILABLE
        
        if not self.gpu_available and self.verbose:
            print("📝 Note: Install cudf-cu11 and cupy-cuda11x for GPU acceleration")
    
    def load_and_prepare_data(self, parquet_path: Path):
        """Load parquet data with GPU acceleration if available."""
        if self.verbose:
            print(f"Loading data {'on GPU' if self.gpu_available else 'on CPU'}...")
        
        if self.gpu_available:
            df = cudf.read_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)
        
        # Ensure we have the right columns
        ts_col = "timestamp_local" if "timestamp_local" in df.columns else "timestamp_utc"
        if "meter" in df.columns:
            df = df[df["meter"] == "electricity"]
        
        if self.gpu_available:
            df["value"] = cudf.to_numeric(df["value"], errors="coerce")
            df[ts_col] = cudf.to_datetime(df[ts_col])
        else:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df[ts_col] = pd.to_datetime(df[ts_col])
        
        # Rename timestamp column for consistency
        if ts_col != "timestamp_local":
            df = df.rename(columns={ts_col: "timestamp_local"})
        
        return df.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)
    
    def parallel_outlier_capping(self, df):
        """GPU-accelerated outlier capping using groupby operations."""
        if self.verbose:
            print(f"Capping outliers {'on GPU' if self.gpu_available else 'vectorized on CPU'}...")
        
        original_outliers = 0
        
        def cap_outliers_group(group):
            nonlocal original_outliers
            values = group['value']
            
            # Skip if insufficient data
            if len(values.dropna()) < 4:
                return group
            
            # Calculate IQR bounds vectorized
            if self.gpu_available:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
            else:
                q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = max(0, q1 - 1.5 * iqr)  # Energy can't be negative
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers before capping
            outliers_mask = (values < lower_bound) | (values > upper_bound)
            original_outliers += outliers_mask.sum()
            
            # Cap outliers
            group['value'] = values.clip(lower=lower_bound, upper=upper_bound)
            
            return group
        
        # Apply capping per building in parallel
        df_clean = df.groupby('building_id', group_keys=False).apply(cap_outliers_group)
        
        self.cleaning_stats['outliers_capped'] = original_outliers
        
        if self.verbose:
            print(f"  Capped {original_outliers:,} outlier values")
        
        return df_clean
    
    def parallel_interpolation(self, df):
        """Vectorized time-aware interpolation."""
        if self.verbose:
            print("Interpolating missing values (vectorized)...")
        
        original_missing = df['value'].isna().sum()
        
        def interpolate_group(group):
            # Sort by timestamp to ensure proper time order
            group = group.sort_values('timestamp_local')
            
            # Linear interpolation (CuDF doesn't support 'time' method)
            if self.gpu_available:
                group['value'] = group['value'].interpolate(method='linear')
                group['value'] = group['value'].fillna(method='ffill')
                group['value'] = group['value'].fillna(method='bfill')
            else:
                group['value'] = group['value'].interpolate(method='time', limit_direction='both')
                group['value'] = group['value'].fillna(method='ffill', limit=48)
                group['value'] = group['value'].fillna(method='bfill', limit=48)
            
            return group
        
        # Apply interpolation per building
        df_clean = df.groupby('building_id', group_keys=False).apply(interpolate_group)
        
        final_missing = df_clean['value'].isna().sum()
        imputed = original_missing - final_missing
        
        self.cleaning_stats['values_imputed'] = imputed
        
        if self.verbose:
            print(f"  Imputed {imputed:,} missing values")
            print(f"  Remaining missing: {final_missing:,}")
        
        return df_clean
    
    def remove_extreme_patterns_vectorized(self, df):
        """Vectorized removal of extreme patterns."""
        if self.verbose:
            print("Removing extreme patterns (vectorized)...")
        
        # Calculate building-level statistics vectorized
        if self.gpu_available:
            # CuDF doesn't support lambda in agg, so calculate everything separately
            building_stats = df.groupby('building_id')['value'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            # Calculate near zero counts using vectorized operations
            df_temp = df.copy()
            df_temp['is_near_zero'] = df_temp['value'] <= 0.01
            near_zero_stats = df_temp.groupby('building_id')['is_near_zero'].sum()
            building_stats['near_zero_count'] = near_zero_stats
            building_stats['near_zero_pct'] = building_stats['near_zero_count'] / building_stats['count']
        else:
            building_stats = df.groupby('building_id')['value'].agg([
                'count', 'mean', 'std', 'min', 'max',
                lambda x: (x <= 0.01).sum(),  # near_zero_count
                lambda x: (x <= 0.01).sum() / len(x)  # near_zero_pct
            ]).rename(columns={
                '<lambda_0>': 'near_zero_count',
                '<lambda_1>': 'near_zero_pct'
            })
        
        # Calculate coefficient of variation
        building_stats['cv'] = building_stats['std'] / building_stats['mean']
        building_stats['cv'] = building_stats['cv'].fillna(0)
        
        # Define extreme patterns
        extreme_mask = (
            (building_stats['near_zero_pct'] > 0.95) |  # >95% near zero
            (building_stats['cv'] > 5) |  # Extremely volatile
            (building_stats['max'] > 1000) |  # Unrealistically high (>1000 kWh/hour)
            (building_stats['count'] < 100)  # Too few observations
        )
        
        if self.gpu_available:
            extreme_buildings = building_stats[extreme_mask].index.to_arrow().to_pylist()
        else:
            extreme_buildings = building_stats[extreme_mask].index.tolist()
        
        # Remove extreme buildings
        df_clean = df[~df['building_id'].isin(extreme_buildings)].copy()
        
        self.cleaning_stats['extreme_buildings_removed'] = len(extreme_buildings)
        self.cleaning_stats['remaining_buildings'] = df_clean['building_id'].nunique()
        
        if self.verbose:
            print(f"  Removed {len(extreme_buildings)} extreme buildings")
            print(f"  Remaining: {df_clean['building_id'].nunique()} buildings")
        
        return df_clean
    
    def add_time_features_vectorized(self, df):
        """Vectorized time feature creation."""
        if self.verbose:
            print("Adding time features (vectorized)...")
        
        if self.gpu_available:
            # CuDF timestamp handling
            if not hasattr(df['timestamp_local'].dtype, 'tz'):
                df['timestamp_local'] = cudf.to_datetime(df['timestamp_local'])
            
            # Extract time features using CuDF
            df['hour'] = df['timestamp_local'].dt.hour.astype('int8')
            df['day_of_week'] = df['timestamp_local'].dt.dayofweek.astype('int8') 
            df['month'] = df['timestamp_local'].dt.month.astype('int8')
            df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
            df['is_working_hours'] = (
                (df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)
            ).astype('int8')
            
            # Seasonal features  
            df['quarter'] = df['timestamp_local'].dt.quarter.astype('int8')
            df['day_of_year'] = df['timestamp_local'].dt.dayofyear.astype('int16')
        else:
            # Ensure timestamp is datetime
            df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
            
            # Extract all time features at once (vectorized)
            df['hour'] = df['timestamp_local'].dt.hour.astype('int8')
            df['day_of_week'] = df['timestamp_local'].dt.dayofweek.astype('int8')
            df['month'] = df['timestamp_local'].dt.month.astype('int8')
            df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
            df['is_working_hours'] = (
                (df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)
            ).astype('int8')
            
            # Seasonal features
            df['quarter'] = df['timestamp_local'].dt.quarter.astype('int8')
            df['day_of_year'] = df['timestamp_local'].dt.dayofyear.astype('int16')
        
        if self.verbose:
            print(f"  Added 7 time features")
        
        return df
    
    def generate_cleaning_report(self, output_dir: Path, original_size: int, final_size: int):
        """Generate comprehensive cleaning report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'parallel_cleaning_report.txt', 'w') as f:
            f.write("SMART GRID - PARALLEL DATA CLEANING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Cleaning Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Records: {original_size:,}\n")
            f.write(f"Final Records: {final_size:,}\n")
            f.write(f"Data Retention: {(final_size/original_size)*100:.2f}%\n\n")
            
            f.write("CLEANING OPERATIONS\n")
            f.write("-" * 20 + "\n")
            for key, value in self.cleaning_stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
            
            f.write(f"\nCLEANING STRATEGY\n")
            f.write("-" * 17 + "\n")
            f.write("• Vectorized operations for maximum speed\n")
            f.write("• Parallel processing per building group\n")
            f.write("• Conservative outlier capping (preserve data)\n")
            f.write("• Time-aware interpolation for missing values\n")
            f.write("• Only remove truly extreme patterns\n")
            f.write("• Rich time feature engineering\n")
    
    def clean_data_parallel(self, df):
        """Main parallel cleaning pipeline."""
        original_size = len(df)
        
        if self.verbose:
            print(f"Starting parallel cleaning of {original_size:,} records...")
        
        # Step 1: Cap outliers (vectorized)
        df = self.parallel_outlier_capping(df)
        
        # Step 2: Interpolate missing values (vectorized)
        df = self.parallel_interpolation(df)
        
        # Step 3: Remove extreme patterns (vectorized)
        df = self.remove_extreme_patterns_vectorized(df)
        
        # Step 4: Add time features (vectorized)
        df = self.add_time_features_vectorized(df)
        
        # Step 5: Final cleanup
        df = df.dropna(subset=['value'])  # Remove any remaining NaNs
        
        final_size = len(df)
        self.cleaning_stats['final_records'] = final_size
        self.cleaning_stats['retention_rate'] = (final_size / original_size) * 100
        
        if self.verbose:
            print(f"\n✅ Parallel cleaning completed!")
            print(f"📊 Retention rate: {(final_size/original_size)*100:.2f}%")
            print(f"🏢 Final buildings: {df['building_id'].nunique():,}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Parallel building energy data cleaning')
    parser.add_argument('--parquet', type=Path, required=True, help='Input parquet file')
    parser.add_argument('--output', type=Path, required=True, help='Output cleaned parquet file')
    parser.add_argument('--analysis-output', type=Path, default='data/analysis', help='Analysis output directory')
    
    args = parser.parse_args()
    
    print("🚀 Initializing parallel data cleaner...")
    cleaner = ParallelBuildingDataCleaner(verbose=True)
    
    # Load data
    df = cleaner.load_and_prepare_data(args.parquet)
    original_size = len(df)
    print(f"📥 Loaded {original_size:,} records for {df['building_id'].nunique():,} buildings")
    
    # Clean data using parallel/vectorized operations
    df_clean = cleaner.clean_data_parallel(df)
    
    # Generate report
    print("📝 Generating cleaning report...")
    cleaner.generate_cleaning_report(args.analysis_output, original_size, len(df_clean))
    
    # Save cleaned dataset
    print("💾 Saving cleaned dataset...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if cleaner.gpu_available:
        # Convert to pandas for saving if using CuDF
        df_clean.to_pandas().to_parquet(args.output, index=False)
    else:
        df_clean.to_parquet(args.output, index=False)
    
    print(f"\n🎉 Success! Clean data saved to: {args.output}")
    print(f"📈 Final dataset: {len(df_clean):,} records, {df_clean['building_id'].nunique():,} buildings")

if __name__ == "__main__":
    main()