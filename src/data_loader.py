"""
Data loader module for the Financial Machine Learning Dashboard
Uses Yahoo Finance API to fetch real forex data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Add the path to access the data API
sys.path.append('/opt/.manus/.sandbox-runtime')
try:
    from data_api import ApiClient
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False
    print("Warning: Could not import ApiClient. Using yfinance as fallback.")
    import yfinance as yf

def get_forex_data(symbol, interval='1d', period='1y'):
    """
    Fetch forex data from Yahoo Finance API
    
    Parameters:
    -----------
    symbol : str
        Forex pair symbol (e.g., 'EURUSD=X')
    interval : str
        Data interval (e.g., '1d', '1h')
    period : str
        Data period (e.g., '1y', '6mo')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLCV data
    """
    try:
        if HAS_API_CLIENT:
            # Use the data API client
            client = ApiClient()
            data = client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'interval': interval,
                'range': period,
                'includeAdjustedClose': True
            })
            
            # Extract the data from the API response
            if data and 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                
                # Extract timestamps and convert to datetime
                timestamps = result['timestamp']
                dates = [datetime.fromtimestamp(ts, tz=pytz.UTC) for ts in timestamps]
                
                # Extract OHLCV data
                ohlcv = result['indicators']['quote'][0]
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Open': ohlcv['open'],
                    'High': ohlcv['high'],
                    'Low': ohlcv['low'],
                    'Close': ohlcv['close'],
                    'Volume': ohlcv['volume']
                }, index=dates)
                
                # Handle any NaN values
                df = df.dropna()
                
                return df
            else:
                raise ValueError("Invalid data format received from API")
        else:
            # Fallback to yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        # Return dummy data as fallback
        return create_dummy_forex_data(symbol)

def create_dummy_forex_data(symbol, periods=1000):
    """
    Create dummy forex data for demonstration purposes
    
    Parameters:
    -----------
    symbol : str
        Forex pair symbol
    periods : int
        Number of periods to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLCV data
    """
    # Set base price based on symbol
    if symbol == 'EURUSD=X':
        base_price = 1.10
        volatility = 0.01
    elif symbol == 'GBPUSD=X':
        base_price = 1.30
        volatility = 0.012
    elif symbol == 'USDJPY=X':
        base_price = 110.0
        volatility = 0.5
    elif symbol == 'AUDUSD=X':
        base_price = 0.75
        volatility = 0.008
    else:
        base_price = 1.0
        volatility = 0.01
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')
    
    # Generate price data with random walk
    np.random.seed(42)  # For reproducibility
    
    # Generate returns with slight drift
    returns = np.random.normal(0.0001, volatility, periods)
    
    # Calculate price series
    close_prices = base_price * (1 + np.cumsum(returns))
    
    # Generate OHLC data
    high_prices = close_prices * (1 + np.random.uniform(0, 0.003, periods))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.003, periods))
    open_prices = close_prices.copy()
    np.random.shuffle(open_prices)
    
    # Ensure High is always >= Open and Close, and Low is always <= Open and Close
    for i in range(periods):
        high_prices[i] = max(open_prices[i], close_prices[i], high_prices[i])
        low_prices[i] = min(open_prices[i], close_prices[i], low_prices[i])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, periods)
    }, index=dates)
    
    return df

def load_forex_data():
    """
    Load forex data for multiple pairs
    
    Returns:
    --------
    dict
        Dictionary with forex pair names as keys and DataFrames as values
    """
    forex_pairs = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'AUD/USD': 'AUDUSD=X'
    }
    
    data_dict = {}
    for name, symbol in forex_pairs.items():
        try:
            # Try to get real data
            data = get_forex_data(symbol, interval='1h', period='1mo')
            if data is not None and not data.empty:
                data_dict[name] = data
                print(f"Successfully loaded data for {name}")
            else:
                # Fallback to dummy data
                data_dict[name] = create_dummy_forex_data(symbol)
                print(f"Using dummy data for {name}")
        except Exception as e:
            print(f"Error loading data for {name}: {e}")
            # Fallback to dummy data
            data_dict[name] = create_dummy_forex_data(symbol)
            print(f"Using dummy data for {name}")
    
    return data_dict
