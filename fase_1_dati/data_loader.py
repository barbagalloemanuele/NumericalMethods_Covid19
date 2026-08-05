import pandas as pd
import numpy as np

def fetch_and_preprocess_data():
    """Fetches COVID-19 provincial data and filters for the First Wave."""
    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"
    print("Downloading data from Protezione Civile...")
    df = pd.read_csv(url)
    
    # Convert dates and filter (Feb 24, 2020 to May 31, 2020)
    # Ensure UTC timezone handling if present in CSV
    df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None)
    start_date = pd.to_datetime("2020-02-24")
    end_date = pd.to_datetime("2020-05-31")
    mask = (df['data'] >= start_date) & (df['data'] <= end_date)
    df = df.loc[mask].copy()
    
    # Filter out entries without valid geographic coordinates (e.g. "In fase di definizione")
    df = df.dropna(subset=['lat', 'long'])
    df = df[(df['lat'] != 0) & (df['long'] != 0)]
    
    return df

def create_spatial_grid(df, grid_size=(50, 50)):
    """
    Maps lat/long coordinates to a 2D discrete grid.
    This step is essential to prepare data for PDE discretization.
    """
    df = df.copy()
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_long, max_long = df['long'].min(), df['long'].max()
    
    # Map longitudes to X axis [0, grid_size[0]-1]
    df['x_idx'] = ((df['long'] - min_long) / (max_long - min_long) * (grid_size[0] - 1)).astype(int)
    
    # Map latitudes to Y axis [0, grid_size[1]-1]
    df['y_idx'] = ((df['lat'] - min_lat) / (max_lat - min_lat) * (grid_size[1] - 1)).astype(int)
    
    return df

if __name__ == "__main__":
    df_raw = fetch_and_preprocess_data()
    print(f"Data filtered. Shape: {df_raw.shape}")
    
    df_spatial = create_spatial_grid(df_raw)
    
    print("\nSample of Spatial Mapped Data:")
    print(df_spatial[['data', 'denominazione_provincia', 'totale_casi', 'x_idx', 'y_idx']].head(10))
    print("\nUnique grid coordinates generated:", df_spatial[['x_idx', 'y_idx']].drop_duplicates().shape[0])
