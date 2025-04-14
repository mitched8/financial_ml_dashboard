"""
Implementation of utility functions for data processing and visualization
for the Financial Machine Learning Dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_and_process_forex_data(file_path):
    """
    Load and process forex data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with OHLCV data
    """
    try:
        # Create dummy data as a fallback
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
        dummy_data = {
            'Open': np.random.normal(1.0, 0.01, 1000),
            'High': np.random.normal(1.01, 0.01, 1000),
            'Low': np.random.normal(0.99, 0.01, 1000),
            'Close': np.random.normal(1.0, 0.01, 1000),
            'Volume': np.random.randint(1000, 10000, 1000)
        }
        
        # Ensure High is always >= Open and Close, and Low is always <= Open and Close
        for i in range(1000):
            dummy_data['High'][i] = max(dummy_data['Open'][i], dummy_data['Close'][i], dummy_data['High'][i])
            dummy_data['Low'][i] = min(dummy_data['Open'][i], dummy_data['Close'][i], dummy_data['Low'][i])
        
        # First, check the file format by reading the first few lines
        with open(file_path, 'r') as f:
            first_lines = [f.readline() for _ in range(4)]
        
        # Check if we have the unusual header format with Price, Ticker, Datetime rows
        if 'Price' in first_lines[0] and 'Ticker' in first_lines[1] and 'Datetime' in first_lines[2]:
            # Skip the first 3 rows and use the 4th row data onwards
            df = pd.read_csv(file_path, skiprows=3)
            
            # Set the first column as the index and parse dates
            df.set_index(df.columns[0], inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # The first column after the index is the Price column which should be renamed to Close
            if len(df.columns) >= 1:
                df = df.rename(columns={df.columns[0]: 'Close'})
            
            # Ensure all required columns exist
            if 'Close' not in df.columns:
                print(f"Warning: 'Close' column not found in {file_path}. Using dummy data.")
                return pd.DataFrame(dummy_data, index=dates)
        else:
            # Standard format
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Convert all numeric columns to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if we have all required columns
        if not all(col in df.columns for col in required_columns):
            # If Volume is missing, add a placeholder column
            if 'Volume' not in df.columns:
                df['Volume'] = 0
                
            # If we're missing other columns, try to infer them
            if 'Open' not in df.columns and 'Close' in df.columns:
                df['Open'] = df['Close'].shift(1)
                df.loc[df.index[0], 'Open'] = df['Close'].iloc[0]
                
            if 'High' not in df.columns and 'Close' in df.columns:
                df['High'] = df['Close']
                
            if 'Low' not in df.columns and 'Close' in df.columns:
                df['Low'] = df['Close']
        
        # Sort by date
        df = df.sort_index()
        
        # Remove any NaN values
        df = df.dropna()
        
        # Print column names for debugging
        print(f"Columns in dataframe from {file_path}: {df.columns.tolist()}")
        
        # Final check for Close column
        if 'Close' not in df.columns:
            print(f"Error: 'Close' column still not found in {file_path} after processing. Using dummy data.")
            return pd.DataFrame(dummy_data, index=dates)
        
        return df
    
    except Exception as e:
        # If there's an error, return a simple DataFrame with dummy data
        print(f"Error loading data from {file_path}: {e}")
        print("Returning dummy data for demonstration purposes.")
        
        return pd.DataFrame(dummy_data, index=dates)


def calculate_returns(prices, method='arithmetic'):
    """
    Calculate returns from price series.
    
    Parameters:
    -----------
    prices : pandas.Series
        Price series
    method : str
        'arithmetic' or 'log' returns
        
    Returns:
    --------
    pandas.Series
        Returns series
    """
    # Ensure prices is numeric
    prices = pd.to_numeric(prices, errors='coerce')
    
    if method == 'arithmetic':
        returns = prices.pct_change().dropna()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'arithmetic' or 'log'")
    
    return returns


def calculate_volatility(returns, window=20, min_periods=5):
    """
    Calculate rolling volatility.
    
    Parameters:
    -----------
    returns : pandas.Series
        Returns series
    window : int
        Rolling window size
    min_periods : int
        Minimum number of observations required
        
    Returns:
    --------
    pandas.Series
        Volatility series
    """
    # Ensure returns is numeric
    returns = pd.to_numeric(returns, errors='coerce')
    
    return returns.rolling(window=window, min_periods=min_periods).std()


def plot_price_with_signals(prices, signals, title='Price Chart with Signals'):
    """
    Plot price chart with buy/sell signals.
    
    Parameters:
    -----------
    prices : pandas.Series
        Price series
    signals : pandas.Series
        Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
    title : str
        Chart title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Ensure prices is numeric
    prices = pd.to_numeric(prices, errors='coerce')
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Add buy signals
    buy_signals = signals[signals == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=prices.loc[buy_signals.index],
            mode='markers',
            name='Buy',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))
    
    # Add sell signals
    sell_signals = signals[signals == -1]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=prices.loc[sell_signals.index],
            mode='markers',
            name='Sell',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    return fig


def plot_equity_curve(backtest_results):
    """
    Plot equity curve and drawdowns from backtest results.
    
    Parameters:
    -----------
    backtest_results : pandas.DataFrame
        Backtest results with Equity and DrawdownPct columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05,
                       subplot_titles=('Equity Curve', 'Drawdown'))
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=backtest_results.index,
            y=backtest_results['Equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=backtest_results.index,
            y=backtest_results['DrawdownPct'],
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Strategy Performance',
        template='plotly_white',
        height=600
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text='Equity', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown', row=2, col=1)
    
    return fig


def plot_bar_types_comparison(price_data, bar_types_dict):
    """
    Plot comparison of different bar types.
    
    Parameters:
    -----------
    price_data : pandas.DataFrame
        Original price data with OHLCV columns
    bar_types_dict : dict
        Dictionary with bar type names as keys and bar DataFrames as values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Ensure numeric data
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in price_data.columns:
            price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
    
    # Create subplots
    n_bars = len(bar_types_dict) + 1  # +1 for original data
    fig = make_subplots(rows=n_bars, cols=1, shared_xaxes=True,
                       vertical_spacing=0.02,
                       subplot_titles=['Original Data'] + list(bar_types_dict.keys()))
    
    # Add original data
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='Original Data'
        ),
        row=1, col=1
    )
    
    # Add each bar type
    for i, (name, bars) in enumerate(bar_types_dict.items(), start=2):
        # Ensure numeric data in each bar type
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in bars.columns:
                bars[col] = pd.to_numeric(bars[col], errors='coerce')
        
        fig.add_trace(
            go.Candlestick(
                x=bars.index,
                open=bars['Open'],
                high=bars['High'],
                low=bars['Low'],
                close=bars['Close'],
                name=name
            ),
            row=i, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Comparison of Bar Types',
        template='plotly_white',
        height=200 * n_bars,
        showlegend=False
    )
    
    return fig


def plot_fractional_differentiation(original_series, diff_series_dict):
    """
    Plot original series and fractionally differentiated series.
    
    Parameters:
    -----------
    original_series : pandas.Series
        Original time series
    diff_series_dict : dict
        Dictionary with d values as keys and differentiated series as values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Ensure original_series is numeric
    original_series = pd.to_numeric(original_series, errors='coerce')
    
    # Create figure
    fig = go.Figure()
    
    # Add original series
    fig.add_trace(go.Scatter(
        x=original_series.index,
        y=original_series,
        mode='lines',
        name='Original'
    ))
    
    # Add differentiated series
    for d, series in diff_series_dict.items():
        # Ensure series is numeric
        series = pd.to_numeric(series, errors='coerce')
        
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series,
            mode='lines',
            name=f'd={d}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Fractional Differentiation Comparison',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    
    return fig


def plot_feature_importance(importance_series, title='Feature Importance'):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_series : pandas.Series
        Series with feature names as index and importance values
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Ensure importance_series is numeric
    importance_series = pd.to_numeric(importance_series, errors='coerce')
    
    # Sort by importance
    importance_series = importance_series.sort_values(ascending=True)
    
    # Create figure
    fig = go.Figure()
    
    # Add horizontal bar chart
    fig.add_trace(go.Bar(
        y=importance_series.index,
        x=importance_series.values,
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.7)')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_white',
        height=max(300, len(importance_series) * 25)
    )
    
    return fig
