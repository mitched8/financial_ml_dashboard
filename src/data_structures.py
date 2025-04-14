"""
Implementation of financial data structures from López de Prado's book
"Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd


def get_time_bars(df, timeframe='1H'):
    """
    Create time bars from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index and OHLCV data
    timeframe : str
        Timeframe for resampling (e.g., '1H', '1D')
        
    Returns:
    --------
    pandas.DataFrame
        Resampled time bars
    """
    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Resample to the specified timeframe
    resampled = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    return resampled.dropna()


def get_tick_bars(df, threshold):
    """
    Create tick bars from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with tick data
    threshold : int
        Number of ticks per bar
        
    Returns:
    --------
    pandas.DataFrame
        Tick bars
    """
    # Initialize variables
    bars = []
    bar_open = df['Close'].iloc[0]
    bar_high = df['Close'].iloc[0]
    bar_low = df['Close'].iloc[0]
    bar_volume = 0
    tick_count = 0
    
    for idx, row in df.iterrows():
        # Update high and low
        bar_high = max(bar_high, row['Close'])
        bar_low = min(bar_low, row['Close'])
        bar_volume += row['Volume'] if 'Volume' in df.columns else 0
        tick_count += 1
        
        # If we've reached the threshold, create a new bar
        if tick_count >= threshold:
            bars.append({
                'Timestamp': idx,
                'Open': bar_open,
                'High': bar_high,
                'Low': bar_low,
                'Close': row['Close'],
                'Volume': bar_volume
            })
            
            # Reset for next bar
            bar_open = row['Close']
            bar_high = row['Close']
            bar_low = row['Close']
            bar_volume = 0
            tick_count = 0
    
    # Create DataFrame from bars
    bars_df = pd.DataFrame(bars)
    if len(bars_df) > 0:
        bars_df.set_index('Timestamp', inplace=True)
    
    return bars_df


def get_volume_bars(df, threshold):
    """
    Create volume bars from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    threshold : float
        Volume threshold per bar
        
    Returns:
    --------
    pandas.DataFrame
        Volume bars
    """
    # Initialize variables
    bars = []
    bar_open = df['Close'].iloc[0]
    bar_high = df['Close'].iloc[0]
    bar_low = df['Close'].iloc[0]
    bar_volume = 0
    
    for idx, row in df.iterrows():
        # Update high and low
        bar_high = max(bar_high, row['Close'])
        bar_low = min(bar_low, row['Close'])
        
        # Add current volume
        current_volume = row['Volume'] if 'Volume' in df.columns else 0
        bar_volume += current_volume
        
        # If we've reached the threshold, create a new bar
        if bar_volume >= threshold:
            bars.append({
                'Timestamp': idx,
                'Open': bar_open,
                'High': bar_high,
                'Low': bar_low,
                'Close': row['Close'],
                'Volume': bar_volume
            })
            
            # Reset for next bar
            bar_open = row['Close']
            bar_high = row['Close']
            bar_low = row['Close']
            bar_volume = 0
    
    # Create DataFrame from bars
    bars_df = pd.DataFrame(bars)
    if len(bars_df) > 0:
        bars_df.set_index('Timestamp', inplace=True)
    
    return bars_df


def get_dollar_bars(df, threshold):
    """
    Create dollar bars from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    threshold : float
        Dollar value threshold per bar
        
    Returns:
    --------
    pandas.DataFrame
        Dollar bars
    """
    # Initialize variables
    bars = []
    bar_open = df['Close'].iloc[0]
    bar_high = df['Close'].iloc[0]
    bar_low = df['Close'].iloc[0]
    bar_volume = 0
    dollar_value = 0
    
    for idx, row in df.iterrows():
        # Update high and low
        bar_high = max(bar_high, row['Close'])
        bar_low = min(bar_low, row['Close'])
        
        # Add current volume and dollar value
        current_volume = row['Volume'] if 'Volume' in df.columns else 0
        bar_volume += current_volume
        dollar_value += current_volume * row['Close']
        
        # If we've reached the threshold, create a new bar
        if dollar_value >= threshold:
            bars.append({
                'Timestamp': idx,
                'Open': bar_open,
                'High': bar_high,
                'Low': bar_low,
                'Close': row['Close'],
                'Volume': bar_volume,
                'DollarValue': dollar_value
            })
            
            # Reset for next bar
            bar_open = row['Close']
            bar_high = row['Close']
            bar_low = row['Close']
            bar_volume = 0
            dollar_value = 0
    
    # Create DataFrame from bars
    bars_df = pd.DataFrame(bars)
    if len(bars_df) > 0:
        bars_df.set_index('Timestamp', inplace=True)
    
    return bars_df


def get_tick_imbalance_bars(df, expected_imbalance=None, window=100):
    """
    Create tick imbalance bars (TIBs) from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with tick data
    expected_imbalance : float, optional
        Expected tick imbalance, if None it will be estimated from initial window
    window : int
        Window size for estimating expected imbalance
        
    Returns:
    --------
    pandas.DataFrame
        Tick imbalance bars
    """
    # Calculate tick signs (bt)
    df = df.copy()
    df['TickSign'] = np.sign(df['Close'].diff().fillna(0))
    
    # Initialize variables
    bars = []
    bar_open = df['Close'].iloc[0]
    bar_high = df['Close'].iloc[0]
    bar_low = df['Close'].iloc[0]
    bar_volume = 0
    
    # Initialize tick imbalance
    tick_imbalance = 0
    
    # Estimate expected imbalance if not provided
    if expected_imbalance is None:
        init_window = min(window, len(df))
        bt_window = df['TickSign'].iloc[:init_window]
        expected_imbalance = abs(bt_window.mean()) * init_window
    
    for idx, row in df.iterrows():
        # Update high and low
        bar_high = max(bar_high, row['Close'])
        bar_low = min(bar_low, row['Close'])
        
        # Add current volume
        current_volume = row['Volume'] if 'Volume' in df.columns else 0
        bar_volume += current_volume
        
        # Update tick imbalance
        tick_imbalance += row['TickSign']
        
        # If absolute tick imbalance exceeds expected imbalance, create a new bar
        if abs(tick_imbalance) >= expected_imbalance:
            bars.append({
                'Timestamp': idx,
                'Open': bar_open,
                'High': bar_high,
                'Low': bar_low,
                'Close': row['Close'],
                'Volume': bar_volume,
                'TickImbalance': tick_imbalance
            })
            
            # Reset for next bar
            bar_open = row['Close']
            bar_high = row['Close']
            bar_low = row['Close']
            bar_volume = 0
            tick_imbalance = 0
    
    # Create DataFrame from bars
    bars_df = pd.DataFrame(bars)
    if len(bars_df) > 0:
        bars_df.set_index('Timestamp', inplace=True)
    
    return bars_df


def get_volume_imbalance_bars(df, expected_imbalance=None, window=100):
    """
    Create volume imbalance bars from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    expected_imbalance : float, optional
        Expected volume imbalance, if None it will be estimated from initial window
    window : int
        Window size for estimating expected imbalance
        
    Returns:
    --------
    pandas.DataFrame
        Volume imbalance bars
    """
    # Calculate tick signs (bt)
    df = df.copy()
    df['TickSign'] = np.sign(df['Close'].diff().fillna(0))
    
    # Calculate signed volume
    df['SignedVolume'] = df['TickSign'] * df['Volume']
    
    # Initialize variables
    bars = []
    bar_open = df['Close'].iloc[0]
    bar_high = df['Close'].iloc[0]
    bar_low = df['Close'].iloc[0]
    bar_volume = 0
    
    # Initialize volume imbalance
    volume_imbalance = 0
    
    # Estimate expected imbalance if not provided
    if expected_imbalance is None:
        init_window = min(window, len(df))
        vt_window = df['SignedVolume'].iloc[:init_window]
        expected_imbalance = abs(vt_window.sum())
    
    for idx, row in df.iterrows():
        # Update high and low
        bar_high = max(bar_high, row['Close'])
        bar_low = min(bar_low, row['Close'])
        
        # Add current volume
        current_volume = row['Volume'] if 'Volume' in df.columns else 0
        bar_volume += current_volume
        
        # Update volume imbalance
        volume_imbalance += row['SignedVolume']
        
        # If absolute volume imbalance exceeds expected imbalance, create a new bar
        if abs(volume_imbalance) >= expected_imbalance:
            bars.append({
                'Timestamp': idx,
                'Open': bar_open,
                'High': bar_high,
                'Low': bar_low,
                'Close': row['Close'],
                'Volume': bar_volume,
                'VolumeImbalance': volume_imbalance
            })
            
            # Reset for next bar
            bar_open = row['Close']
            bar_high = row['Close']
            bar_low = row['Close']
            bar_volume = 0
            volume_imbalance = 0
    
    # Create DataFrame from bars
    bars_df = pd.DataFrame(bars)
    if len(bars_df) > 0:
        bars_df.set_index('Timestamp', inplace=True)
    
    return bars_df
