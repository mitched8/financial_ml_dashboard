import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import data loader
from data_loader import load_forex_data

# Set page config
st.set_page_config(
    page_title="Financial Machine Learning Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Financial Machine Learning Dashboard")
st.markdown("""
This dashboard explores the key concepts from Marcos López de Prado's book 
"Advances in Financial Machine Learning" applied to real forex market data.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Introduction", "Data Structures", "Labeling Techniques", 
     "Sample Weights & Features", "Backtesting", "Portfolio Construction", "Chat with Expert"]
)

# Helper functions for the dashboard
def calculate_returns(prices, method='arithmetic'):
    """Calculate returns from price series."""
    if method == 'arithmetic':
        returns = prices.pct_change().dropna()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'arithmetic' or 'log'")
    return returns

def calculate_volatility(returns, window=20, min_periods=5):
    """Calculate rolling volatility."""
    return returns.rolling(window=window, min_periods=min_periods).std()

def fixed_time_horizon(prices, horizon=5, threshold=0.01):
    """Apply fixed-time horizon labeling."""
    future_returns = prices.pct_change(horizon).shift(-horizon)
    labels = pd.Series(0, index=prices.index)
    labels[future_returns > threshold] = 1
    labels[future_returns < -threshold] = -1
    return labels

def triple_barrier_labeling(prices, volatility, upper_barrier=2, lower_barrier=2, max_horizon=10):
    """Apply triple-barrier labeling method."""
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Initialize labels and barriers
    labels = pd.Series(0, index=prices.index)
    barriers = pd.DataFrame(index=prices.index)
    barriers['upper'] = prices * (1 + upper_barrier * volatility)
    barriers['lower'] = prices * (1 - lower_barrier * volatility)
    barriers['vertical'] = pd.Series([prices.index[min(i + max_horizon, len(prices) - 1)] 
                                     for i in range(len(prices))], index=prices.index)
    
    # Simulate price paths and determine labels
    for i in range(len(prices) - max_horizon):
        price_path = prices.iloc[i:i+max_horizon+1]
        upper_hit = price_path >= barriers['upper'].iloc[i]
        lower_hit = price_path <= barriers['lower'].iloc[i]
        
        if upper_hit.any():
            labels.iloc[i] = 1
        elif lower_hit.any():
            labels.iloc[i] = -1
        else:
            # Vertical barrier - use the return at the end of the horizon
            end_return = (price_path.iloc[-1] / price_path.iloc[0]) - 1
            if end_return > 0:
                labels.iloc[i] = 1
            elif end_return < 0:
                labels.iloc[i] = -1
    
    return pd.DataFrame({'Label': labels, 'UpperBarrier': barriers['upper'], 
                         'LowerBarrier': barriers['lower']})

def get_sample_weights(labels, return_num_concurrent=False):
    """Calculate sample weights to address overlapping outcomes."""
    # Count number of concurrent labels
    num_concurrent = pd.Series(0, index=labels.index)
    
    for i in range(len(labels)):
        if labels.iloc[i] != 0:
            num_concurrent.iloc[i:i+10] += 1
    
    # Calculate weights
    weights = 1 / num_concurrent
    weights[num_concurrent == 0] = 0
    
    if return_num_concurrent:
        return weights, num_concurrent
    else:
        return weights

def fractional_differentiation(series, d=0.5, threshold=1e-5):
    """Apply fractional differentiation to make series stationary while preserving memory."""
    # Calculate weights
    weights = [1]
    for k in range(1, 100):
        weight = weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    
    # Apply weights
    width = len(weights)
    output = pd.Series(index=series.index)
    
    for i in range(width, len(series)):
        output.iloc[i] = np.sum(weights * series.iloc[i-width:i])
    
    return output.dropna()

def find_optimal_d(series, d_range=np.arange(0, 1.1, 0.1), threshold=1e-5):
    """Find optimal differentiation order that balances stationarity and memory preservation."""
    from statsmodels.tsa.stattools import adfuller
    
    results = {}
    optimal_d = 0
    min_pvalue = 1
    
    for d in d_range:
        # Apply fractional differentiation
        diff_series = fractional_differentiation(series, d=d, threshold=threshold)
        
        # Calculate correlation with original
        correlation = np.corrcoef(series.iloc[-len(diff_series):], diff_series)[0, 1]
        
        # Test for stationarity
        adf_result = adfuller(diff_series, regression='ct')
        adf_pvalue = adf_result[1]
        
        results[d] = {'correlation': correlation, 'adf_pvalue': adf_pvalue}
        
        # Update optimal d if this one is more stationary and we haven't crossed the 0.05 threshold yet
        if adf_pvalue < min_pvalue and (min_pvalue > 0.05 or adf_pvalue <= 0.05):
            min_pvalue = adf_pvalue
            optimal_d = d
    
    return optimal_d, results

def backtest_strategy(prices, signals, transaction_cost=0.0001):
    """Backtest a trading strategy based on signals."""
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Initialize positions and equity
    positions = pd.Series(0, index=prices.index)
    equity = pd.Series(1.0, index=prices.index)
    
    # Apply signals to positions
    positions = signals.copy()
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1) * returns
    
    # Apply transaction costs
    position_changes = positions.diff().fillna(0)
    transaction_costs = abs(position_changes) * transaction_cost
    strategy_returns = strategy_returns - transaction_costs
    
    # Calculate equity curve
    equity = (1 + strategy_returns).cumprod()
    
    # Calculate drawdowns
    drawdown = equity / equity.cummax() - 1
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Position': positions,
        'Return': strategy_returns,
        'Equity': equity,
        'DrawdownPct': drawdown
    })
    
    return results

def calculate_performance_metrics(backtest_results):
    """Calculate performance metrics from backtest results."""
    # Extract returns
    returns = backtest_results['Return']
    
    # Calculate metrics
    total_return = backtest_results['Equity'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = backtest_results['DrawdownPct'].min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    # Calculate win rate and profit factor
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if len(winning_trades) + len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else np.inf
    
    # Calculate average win and loss
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    
    # Return metrics as dictionary
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Average Win': avg_win,
        'Average Loss': avg_loss
    }

def kelly_criterion(win_rate, win_loss_ratio):
    """Calculate Kelly criterion for optimal bet sizing."""
    return win_rate - (1 - win_rate) / win_loss_ratio

def meta_labeling_bet_sizing(probabilities, threshold=0.6, fraction=0.5):
    """Calculate bet sizes based on meta-labeling probabilities."""
    # Calculate bet sizes
    bet_sizes = (probabilities - threshold) / (1 - threshold)
    bet_sizes = bet_sizes.clip(0, 1) * fraction
    
    return bet_sizes

def dynamic_position_sizing(signals, probabilities, threshold=0.6, max_size=1.0):
    """Apply dynamic position sizing based on signals and probabilities."""
    # Calculate bet sizes
    bet_sizes = meta_labeling_bet_sizing(probabilities, threshold, max_size)
    
    # Apply bet sizes to signals
    positions = signals * bet_sizes
    
    return positions

def get_time_bars(data, timeframe='1H'):
    """Get time bars from data."""
    return data.copy()

def get_tick_bars(data, threshold=20):
    """Simulate tick bars from time bars."""
    # For demonstration, we'll just sample the data
    return data.iloc[::threshold].copy()

def get_volume_bars(data, threshold=1000):
    """Simulate volume bars from time bars."""
    # For demonstration, we'll just sample the data
    return data.iloc[::max(1, int(threshold/1000))].copy()

def get_dollar_bars(data, threshold=10000):
    """Simulate dollar bars from time bars."""
    # For demonstration, we'll just sample the data
    return data.iloc[::max(1, int(threshold/10000))].copy()

def get_tick_imbalance_bars(data, threshold=0.5):
    """Simulate tick imbalance bars from time bars."""
    # For demonstration, we'll just sample the data
    return data.iloc[::3].copy()

def get_volume_imbalance_bars(data, threshold=0.5):
    """Simulate volume imbalance bars from time bars."""
    # For demonstration, we'll just sample the data
    return data.iloc[::4].copy()

def plot_price_with_signals(prices, signals, title='Price Chart with Signals'):
    """Plot price chart with buy/sell signals."""
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
    """Plot equity curve and drawdowns from backtest results."""
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
    """Plot comparison of different bar types."""
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
    """Plot original series and fractionally differentiated series."""
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
    """Plot feature importance."""
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

# Load data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data():
    with st.spinner("Loading forex data from Yahoo Finance..."):
        return load_forex_data()

# Load data
data_dict = get_data()

# Introduction page
if page == "Introduction":
    st.header("Introduction to Financial Machine Learning")
    
    st.subheader("About the Book")
    st.markdown("""
    "Advances in Financial Machine Learning" by Marcos López de Prado introduces
    machine learning techniques specifically designed for financial applications.
    The book addresses the unique challenges of financial data and provides novel
    approaches to overcome them.
    """)
    
    st.subheader("Key Concepts")
    st.markdown("""
    1. **Financial Data Structures**: Alternative bar types beyond time bars
    2. **Labeling Techniques**: Methods like the Triple-Barrier Method
    3. **Sample Weights**: Addressing overlapping outcomes and time decay
    4. **Feature Engineering**: Fractional differentiation to preserve memory
    5. **Cross-Validation**: Purged and embargo techniques for financial data
    6. **Backtesting**: Avoiding backtest overfitting and strategy degradation
    7. **Portfolio Construction**: Hierarchical Risk Parity and bet sizing
    """)
    
    st.subheader("Available Forex Pairs")
    cols = st.columns(len(data_dict))
    for i, (pair, data) in enumerate(data_dict.items()):
        with cols[i]:
            st.metric(
                label=pair,
                value=f"{data['Close'].iloc[-1]:.4f}",
                delta=f"{(data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100:.2f}%"
            )
            
            # Plot price chart
            fig = px.line(data, y='Close', title=f"{pair} Price")
            st.plotly_chart(fig, use_container_width=True)

# Data Structures page
elif page == "Data Structures":
    st.header("Financial Data Structures")
    
    st.markdown("""
    López de Prado introduces alternative bar types beyond traditional time bars:
    
    - **Time Bars**: Standard bars based on fixed time intervals
    - **Tick Bars**: Bars formed after a fixed number of transactions
    - **Volume Bars**: Bars formed after a fixed amount of volume is traded
    - **Dollar Bars**: Bars formed after a fixed amount of notional value is traded
    - **Information-Driven Bars**: Bars based on information arrival (tick imbalance, volume imbalance)
    
    These alternative bar types can help reduce serial correlation and create more
    homogeneous bars, improving the performance of machine learning models.
    """)
    
    # Select forex pair
    pair = st.selectbox("Select Forex Pair", list(data_dict.keys()))
    data = data_dict[pair]
    
    # Parameters for bar creation
    col1, col2, col3 = st.columns(3)
    with col1:
        time_interval = st.selectbox(
            "Time Bar Interval",
            ["1H", "2H", "4H", "8H", "1D"],
            index=0
        )
    with col2:
        tick_threshold = st.slider(
            "Tick Bar Threshold",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
    with col3:
        volume_threshold = st.slider(
            "Volume Bar Threshold",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000
        )
    
    # Create different bar types
    time_bars = get_time_bars(data, timeframe=time_interval)
    tick_bars = get_tick_bars(data, threshold=tick_threshold)
    volume_bars = get_volume_bars(data, threshold=volume_threshold)
    
    # Display bar comparison
    st.subheader("Bar Types Comparison")
    
    # Create a dictionary of bar types
    bar_types = {
        f"Time Bars ({time_interval})": time_bars,
        f"Tick Bars (threshold={tick_threshold})": tick_bars,
        f"Volume Bars (threshold={volume_threshold})": volume_bars
    }
    
    # Plot comparison
    fig = plot_bar_types_comparison(data.iloc[-100:], {k: v.iloc[-50:] for k, v in bar_types.items()})
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Bar Statistics")
    
    stats_cols = st.columns(len(bar_types) + 1)
    
    # Original data stats
    with stats_cols[0]:
        st.markdown("**Original Data**")
        st.write(f"Number of bars: {len(data)}")
        st.write(f"Mean return: {data['Close'].pct_change().mean():.6f}")
        st.write(f"Return std: {data['Close'].pct_change().std():.6f}")
        st.write(f"Serial correlation: {data['Close'].pct_change().autocorr():.4f}")
    
    # Bar type stats
    for i, (name, bars) in enumerate(bar_types.items(), start=1):
        with stats_cols[i]:
            st.markdown(f"**{name}**")
            st.write(f"Number of bars: {len(bars)}")
            st.write(f"Mean return: {bars['Close'].pct_change().mean():.6f}")
            st.write(f"Return std: {bars['Close'].pct_change().std():.6f}")
            st.write(f"Serial correlation: {bars['Close'].pct_change().autocorr():.4f}")

# Labeling Techniques page
elif page == "Labeling Techniques":
    st.header("Labeling Techniques")
    
    st.markdown("""
    López de Prado introduces advanced labeling techniques for financial data:
    
    - **Fixed-Time Horizon**: Traditional approach with fixed look-ahead period
    - **Triple-Barrier Method**: Labels based on three barriers (upper, lower, vertical)
    - **Meta-Labeling**: Two-step approach for side and size prediction
    
    These techniques help create more informative labels for financial machine learning models.
    """)
    
    # Select forex pair
    pair = st.selectbox("Select Forex Pair", list(data_dict.keys()))
    data = data_dict[pair]
    
    # Parameters for labeling
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.slider(
            "Horizon (periods)",
            min_value=1,
            max_value=20,
            value=5
        )
    with col2:
        threshold = st.slider(
            "Return Threshold (%)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        ) / 100
    with col3:
        volatility_window = st.slider(
            "Volatility Window",
            min_value=5,
            max_value=50,
            value=20
        )
    
    # Calculate returns and volatility
    returns = calculate_returns(data['Close'])
    volatility = calculate_volatility(returns, window=volatility_window)
    
    # Apply labeling techniques
    fixed_labels = fixed_time_horizon(data['Close'], horizon=horizon, threshold=threshold)
    
    # Triple-barrier parameters
    tb_col1, tb_col2, tb_col3 = st.columns(3)
    with tb_col1:
        upper_barrier = st.slider(
            "Upper Barrier (volatility multiplier)",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.1
        )
    with tb_col2:
        lower_barrier = st.slider(
            "Lower Barrier (volatility multiplier)",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.1
        )
    with tb_col3:
        max_horizon = st.slider(
            "Maximum Horizon",
            min_value=5,
            max_value=30,
            value=10
        )
    
    # Apply triple-barrier method
    tb_labels = triple_barrier_labeling(
        data['Close'], 
        volatility, 
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        max_horizon=max_horizon
    )
    
    # Display results
    st.subheader("Labeling Results")
    
    # Fixed-time horizon
    st.markdown("#### Fixed-Time Horizon")
    
    # Create signals for visualization
    fixed_signals = pd.Series(0, index=data.index)
    fixed_signals[fixed_labels == 1] = 1
    fixed_signals[fixed_labels == -1] = -1
    
    # Plot price with signals
    fig_fixed = plot_price_with_signals(
        data['Close'], 
        fixed_signals, 
        title=f"{pair} with Fixed-Time Horizon Signals"
    )
    st.plotly_chart(fig_fixed, use_container_width=True)
    
    # Triple-barrier method
    st.markdown("#### Triple-Barrier Method")
    
    # Create signals for visualization
    tb_signals = pd.Series(0, index=data.index)
    tb_signals[tb_labels['Label'] == 1] = 1
    tb_signals[tb_labels['Label'] == -1] = -1
    
    # Plot price with signals
    fig_tb = plot_price_with_signals(
        data['Close'], 
        tb_signals, 
        title=f"{pair} with Triple-Barrier Signals"
    )
    st.plotly_chart(fig_tb, use_container_width=True)
    
    # Compare label distributions
    st.subheader("Label Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fixed-Time Horizon**")
        fixed_counts = fixed_labels.value_counts()
        fig_fixed_dist = px.pie(
            values=fixed_counts.values,
            names=fixed_counts.index.map({1: 'Buy', -1: 'Sell', 0: 'Hold'}),
            title="Fixed-Time Horizon Labels"
        )
        st.plotly_chart(fig_fixed_dist, use_container_width=True)
    
    with col2:
        st.markdown("**Triple-Barrier Method**")
        tb_counts = tb_labels['Label'].value_counts()
        fig_tb_dist = px.pie(
            values=tb_counts.values,
            names=tb_counts.index.map({1: 'Buy', -1: 'Sell', 0: 'Hold'}),
            title="Triple-Barrier Labels"
        )
        st.plotly_chart(fig_tb_dist, use_container_width=True)

# Sample Weights & Features page
elif page == "Sample Weights & Features":
    st.header("Sample Weights & Feature Engineering")
    
    st.markdown("""
    López de Prado introduces techniques for sample weighting and feature engineering:
    
    - **Sample Weights**: Address overlapping outcomes and time decay
    - **Fractional Differentiation**: Make series stationary while preserving memory
    
    These techniques help improve the quality of features and address issues specific to financial data.
    """)
    
    # Select forex pair
    pair = st.selectbox("Select Forex Pair", list(data_dict.keys()))
    data = data_dict[pair]
    
    # Tabs for different techniques
    tab1, tab2 = st.tabs(["Sample Weights", "Fractional Differentiation"])
    
    with tab1:
        st.subheader("Sample Weights")
        
        st.markdown("""
        When labeling financial data, we often have overlapping outcomes that can lead to
        misleading performance metrics. Sample weights help address this issue by giving
        more weight to unique information.
        """)
        
        # Parameters for labeling
        col1, col2, col3 = st.columns(3)
        with col1:
            horizon = st.slider(
                "Horizon (periods)",
                min_value=1,
                max_value=20,
                value=5,
                key="sw_horizon"
            )
        with col2:
            threshold = st.slider(
                "Return Threshold (%)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                key="sw_threshold"
            ) / 100
        with col3:
            volatility_window = st.slider(
                "Volatility Window",
                min_value=5,
                max_value=50,
                value=20,
                key="sw_vol_window"
            )
        
        # Calculate returns and volatility
        returns = calculate_returns(data['Close'])
        volatility = calculate_volatility(returns, window=volatility_window)
        
        # Apply labeling
        fixed_labels = fixed_time_horizon(data['Close'], horizon=horizon, threshold=threshold)
        
        # Calculate sample weights
        sample_weights, num_concurrent = get_sample_weights(
            fixed_labels, return_num_concurrent=True
        )
        
        # Plot sample weights
        st.markdown("#### Sample Weights Visualization")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=["Price", "Labels", "Sample Weights"])
        
        # Add price
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add labels
        fig.add_trace(
            go.Scatter(
                x=fixed_labels.index,
                y=fixed_labels,
                mode='markers',
                name='Labels',
                marker=dict(
                    color=fixed_labels.map({1: 'green', -1: 'red', 0: 'gray'}),
                    size=8
                )
            ),
            row=2, col=1
        )
        
        # Add sample weights
        fig.add_trace(
            go.Scatter(
                x=sample_weights.index,
                y=sample_weights,
                mode='lines',
                name='Sample Weights',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=600, title="Price, Labels, and Sample Weights")
        st.plotly_chart(fig, use_container_width=True)
        
        # Concurrent labels
        st.markdown("#### Concurrent Labels")
        st.markdown("""
        This chart shows the number of concurrent labels at each point in time.
        Higher values indicate more overlap, which can lead to misleading performance metrics.
        """)
        
        fig_concurrent = px.line(
            x=num_concurrent.index,
            y=num_concurrent,
            title="Number of Concurrent Labels"
        )
        st.plotly_chart(fig_concurrent, use_container_width=True)
    
    with tab2:
        st.subheader("Fractional Differentiation")
        
        st.markdown("""
        Fractional differentiation allows us to make a time series stationary while
        preserving more memory than traditional differencing methods.
        """)
        
        # Parameters for fractional differentiation
        col1, col2 = st.columns(2)
        with col1:
            d_value = st.slider(
                "Differentiation Order (d)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        with col2:
            threshold = st.slider(
                "Weight Threshold",
                min_value=0.000001,
                max_value=0.001,
                value=0.00001,
                format="%.6f"
            )
        
        # Apply fractional differentiation
        price_series = data['Close']
        diff_series = fractional_differentiation(price_series, d=d_value, threshold=threshold)
        
        # Create dictionary of differentiated series with different d values
        d_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        diff_dict = {}
        
        for d in d_values:
            diff_dict[d] = fractional_differentiation(price_series, d=d, threshold=threshold)
        
        # Plot original and differentiated series
        st.markdown("#### Fractional Differentiation Comparison")
        
        fig = plot_fractional_differentiation(price_series, diff_dict)
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal d
        if st.button("Find Optimal Differentiation Order"):
            with st.spinner("Finding optimal differentiation order..."):
                try:
                    optimal_d, results = find_optimal_d(
                        price_series, 
                        d_range=np.arange(0, 1.1, 0.1),
                        threshold=threshold
                    )
                    
                    st.success(f"Optimal differentiation order: d = {optimal_d:.1f}")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame.from_dict(results, orient='index')
                    
                    # Plot results
                    fig_results = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                              subplot_titles=["ADF p-value", "Correlation with Original"])
                    
                    fig_results.add_trace(
                        go.Scatter(
                            x=results_df.index,
                            y=results_df['adf_pvalue'],
                            mode='lines+markers',
                            name='ADF p-value'
                        ),
                        row=1, col=1
                    )
                    
                    # Add horizontal line at 0.05
                    fig_results.add_hline(
                        y=0.05, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Significance level (0.05)",
                        row=1, col=1
                    )
                    
                    fig_results.add_trace(
                        go.Scatter(
                            x=results_df.index,
                            y=results_df['correlation'],
                            mode='lines+markers',
                            name='Correlation'
                        ),
                        row=2, col=1
                    )
                    
                    fig_results.update_layout(height=500, title="Fractional Differentiation Results")
                    st.plotly_chart(fig_results, use_container_width=True)
                except Exception as e:
                    st.error(f"Error finding optimal differentiation order: {e}")
                    st.info("This may be due to insufficient data or issues with the stationarity test. Try with a different forex pair or adjust parameters.")

# Backtesting page
elif page == "Backtesting":
    st.header("Backtesting")
    
    st.markdown("""
    López de Prado introduces advanced backtesting techniques to avoid overfitting
    and strategy degradation:
    
    - **Walk-Forward Validation**: Time-series cross-validation
    - **Purged Cross-Validation**: Avoiding information leakage
    - **Combinatorial Purged Cross-Validation**: Testing multiple strategies
    
    This page demonstrates backtesting a simple strategy based on the labeling techniques.
    """)
    
    # Select forex pair
    pair = st.selectbox("Select Forex Pair", list(data_dict.keys()))
    data = data_dict[pair]
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.slider(
            "Horizon (periods)",
            min_value=1,
            max_value=20,
            value=5,
            key="bt_horizon"
        )
    with col2:
        threshold = st.slider(
            "Return Threshold (%)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            key="bt_threshold"
        ) / 100
    with col3:
        transaction_cost = st.slider(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.01,
            key="bt_tc"
        ) / 100
    
    # Strategy selection
    strategy_type = st.radio(
        "Strategy Type",
        ["Fixed-Time Horizon", "Triple-Barrier Method"]
    )
    
    # Generate signals based on selected strategy
    if strategy_type == "Fixed-Time Horizon":
        # Apply fixed-time horizon labeling
        labels = fixed_time_horizon(data['Close'], horizon=horizon, threshold=threshold)
        
        # Create signals
        signals = pd.Series(0, index=data.index)
        signals[labels == 1] = 1
        signals[labels == -1] = -1
    else:
        # Calculate volatility
        returns = calculate_returns(data['Close'])
        volatility = calculate_volatility(returns, window=20)
        
        # Triple-barrier parameters
        tb_col1, tb_col2, tb_col3 = st.columns(3)
        with tb_col1:
            upper_barrier = st.slider(
                "Upper Barrier",
                min_value=0.5,
                max_value=3.0,
                value=2.0,
                step=0.1,
                key="tb_upper"
            )
        with tb_col2:
            lower_barrier = st.slider(
                "Lower Barrier",
                min_value=0.5,
                max_value=3.0,
                value=2.0,
                step=0.1,
                key="tb_lower"
            )
        with tb_col3:
            max_horizon = st.slider(
                "Maximum Horizon",
                min_value=5,
                max_value=30,
                value=10,
                key="tb_max_horizon"
            )
        
        # Apply triple-barrier method
        tb_labels = triple_barrier_labeling(
            data['Close'], 
            volatility, 
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            max_horizon=max_horizon
        )
        
        # Create signals
        signals = pd.Series(0, index=data.index)
        signals[tb_labels['Label'] == 1] = 1
        signals[tb_labels['Label'] == -1] = -1
    
    # Run backtest
    backtest_results = backtest_strategy(
        data['Close'],
        signals,
        transaction_cost=transaction_cost
    )
    
    # Display backtest results
    st.subheader("Backtest Results")
    
    # Plot equity curve
    fig_equity = plot_equity_curve(backtest_results)
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Performance metrics
    metrics = calculate_performance_metrics(backtest_results)
    
    # Display metrics
    st.markdown("#### Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Return", f"{metrics['Total Return']:.2%}")
        st.metric("Annualized Return", f"{metrics['Annualized Return']:.2%}")
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    with col2:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
        st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
    with col3:
        st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
        st.metric("Average Win", f"{metrics['Average Win']:.2%}")
        st.metric("Average Loss", f"{metrics['Average Loss']:.2%}")
    
    # Trade analysis
    st.markdown("#### Trade Analysis")
    
    # Calculate trade statistics
    position_changes = backtest_results['Position'].diff().fillna(0)
    trades = position_changes[position_changes != 0]
    
    # Count trades
    n_trades = len(trades)
    n_buys = len(trades[trades > 0])
    n_sells = len(trades[trades < 0])
    
    # Display trade counts
    trade_cols = st.columns(3)
    with trade_cols[0]:
        st.metric("Total Trades", n_trades)
    with trade_cols[1]:
        st.metric("Buy Trades", n_buys)
    with trade_cols[2]:
        st.metric("Sell Trades", n_sells)
    
    # Plot trades on price chart
    fig_trades = plot_price_with_signals(
        data['Close'],
        backtest_results['Position'],
        title=f"{pair} with Trading Signals"
    )
    st.plotly_chart(fig_trades, use_container_width=True)

# Portfolio Construction page
elif page == "Portfolio Construction":
    st.header("Portfolio Construction")
    
    st.markdown("""
    López de Prado introduces advanced portfolio construction techniques:
    
    - **Hierarchical Risk Parity (HRP)**: Clustering-based portfolio allocation
    - **Nested Clustered Optimization (NCO)**: Extension of HRP
    - **Meta-Labeling for Bet Sizing**: Two-step approach for position sizing
    
    This page demonstrates bet sizing techniques based on meta-labeling.
    """)
    
    # Select forex pairs
    selected_pairs = st.multiselect(
        "Select Forex Pairs",
        list(data_dict.keys()),
        default=list(data_dict.keys())[:2]
    )
    
    if not selected_pairs:
        st.warning("Please select at least one forex pair.")
    else:
        # Bet sizing parameters
        st.subheader("Bet Sizing Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            win_rate = st.slider(
                "Win Rate",
                min_value=0.3,
                max_value=0.7,
                value=0.5,
                step=0.05
            )
        with col2:
            win_loss_ratio = st.slider(
                "Win/Loss Ratio",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1
            )
        
        # Calculate Kelly criterion
        kelly = kelly_criterion(win_rate, win_loss_ratio)
        
        # Display Kelly result
        st.markdown(f"#### Kelly Criterion: {kelly:.2%}")
        st.markdown("""
        The Kelly Criterion suggests the optimal fraction of capital to bet
        to maximize long-term growth rate.
        """)
        
        # Meta-labeling bet sizing
        st.subheader("Meta-Labeling for Bet Sizing")
        
        st.markdown("""
        Meta-labeling is a two-step approach:
        1. Primary model predicts the side (buy/sell)
        2. Secondary model predicts the probability of success
        
        The bet size is then determined based on the probability of success.
        """)
        
        # Simulate meta-labeling probabilities
        np.random.seed(42)
        
        for pair in selected_pairs:
            data = data_dict[pair]
            
            # Simulate primary model signals
            signals = pd.Series(0, index=data.index)
            signals.iloc[::10] = 1  # Buy every 10th bar
            signals.iloc[::15] = -1  # Sell every 15th bar
            
            # Simulate meta-labeling probabilities
            probs = pd.Series(np.random.uniform(0.3, 0.9, size=len(signals)), index=signals.index)
            probs[signals == 0] = 0.5  # Neutral when no signal
            
            # Calculate bet sizes
            threshold = st.slider(
                f"Probability Threshold for {pair}",
                min_value=0.5,
                max_value=0.8,
                value=0.6,
                step=0.05
            )
            
            fraction = st.slider(
                f"Maximum Position Size for {pair} (Kelly Fraction)",
                min_value=0.1,
                max_value=1.0,
                value=kelly,
                step=0.1
            )
            
            bet_sizes = meta_labeling_bet_sizing(probs, threshold=threshold, fraction=fraction)
            
            # Combine signals and bet sizes
            positions = signals * bet_sizes
            
            # Plot positions
            st.markdown(f"#### {pair} Positions")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=["Price", "Position Size"])
            
            # Add price
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add positions
            fig.add_trace(
                go.Bar(
                    x=positions.index,
                    y=positions,
                    name='Position',
                    marker=dict(
                        color=positions.apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray')
                    )
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=500, title=f"{pair} Price and Positions")
            st.plotly_chart(fig, use_container_width=True)
            
            # Run backtest with sized positions
            backtest_results = backtest_strategy(
                data['Close'],
                positions,
                transaction_cost=0.0001
            )
            
            # Display backtest results
            metrics = calculate_performance_metrics(backtest_results)
            
            # Display metrics
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Total Return", f"{metrics['Total Return']:.2%}")
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            with metric_cols[1]:
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
            with metric_cols[2]:
                st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
                st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
            
            # Plot equity curve
            fig_equity = plot_equity_curve(backtest_results)
            st.plotly_chart(fig_equity, use_container_width=True)

# Chat with Expert page
elif page == "Chat with Expert":
    st.header("Chat with Financial ML Expert")
    
    st.markdown("""
    This chat interface allows you to ask questions about Marcos López de Prado's 
    financial machine learning techniques and concepts from his book.
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Financial Machine Learning Expert. Ask me anything about Marcos López de Prado's techniques, financial machine learning concepts, or how to apply them to forex markets."}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Function to generate responses
    def generate_response(prompt):
        # Dictionary of common questions and answers
        qa_pairs = {
            "what is the triple barrier method": """
                The Triple-Barrier Method is a labeling technique introduced by López de Prado that defines three barriers:
                
                1. **Upper barrier**: If price reaches this level, label as positive (1)
                2. **Lower barrier**: If price reaches this level, label as negative (-1)
                3. **Vertical barrier**: If neither upper nor lower barriers are touched within a specified time period, label based on the price at that time
                
                This method is more informative than traditional fixed-time horizon labeling because it considers both price movement and time, creating path-dependent labels that better reflect trading scenarios.
            """,
            
            "what are information driven bars": """
                Information-Driven Bars are alternative data structures that sample based on information arrival rather than fixed time intervals. Two main types are:
                
                1. **Tick Imbalance Bars (TIBs)**: Created when the absolute tick imbalance exceeds a threshold
                2. **Volume Imbalance Bars (VIBs)**: Created when the absolute volume imbalance exceeds a threshold
                
                These bars help reduce serial correlation and create more homogeneous price changes, improving the performance of machine learning models by sampling more frequently during informative market periods and less frequently during quiet periods.
            """,
            
            "what is fractional differentiation": """
                Fractional Differentiation is a technique to make financial time series stationary while preserving memory. Traditional differencing (d=1) makes series stationary but removes all memory, while no differencing (d=0) preserves memory but keeps non-stationarity.
                
                Fractional differentiation uses a fractional d value between 0 and 1, finding the optimal balance between:
                
                1. Making the series stationary enough for ML models (smaller p-value in stationarity tests)
                2. Preserving as much memory as possible (higher correlation with the original series)
                
                This technique is crucial for financial ML because it addresses the dilemma between stationarity requirements and the need to preserve predictive information in the data.
            """,
            
            "what is meta labeling": """
                Meta-Labeling is a two-step approach for trading strategy development:
                
                1. **Primary model**: Predicts the direction/side of the trade (buy, sell, or hold)
                2. **Secondary model (meta-labeler)**: Predicts the probability of success for the primary model's predictions
                
                Benefits include:
                
                - Addressing class imbalance by focusing the secondary model only on instances where the primary model predicts a trade
                - Separating side prediction from size prediction, allowing for more nuanced position sizing
                - Reducing false positives by filtering out low-probability trades
                - Improving strategy performance by dynamically adjusting position sizes based on predicted probabilities
                
                This approach is particularly useful for bet sizing in trading strategies.
            """,
            
            "what is hierarchical risk parity": """
                Hierarchical Risk Parity (HRP) is a portfolio optimization technique that:
                
                1. Uses hierarchical clustering to identify groups of similar assets
                2. Allocates risk (rather than capital) across these clusters
                3. Distributes risk within each cluster based on asset volatility
                
                Advantages over traditional methods like Mean-Variance Optimization:
                
                - More robust to estimation errors in expected returns and covariances
                - No matrix inversion required, avoiding numerical instability
                - Better out-of-sample performance with more diversified portfolios
                - Less concentrated positions and lower turnover
                
                HRP leverages the hierarchical structure of asset correlations to create more balanced portfolios.
            """,
            
            "what is purged cross validation": """
                Purged Cross-Validation is a modification of standard k-fold cross-validation for financial data that:
                
                1. **Purges**: Removes training samples that overlap in time with test samples to prevent information leakage
                2. **Embargoes**: Adds a buffer period after test samples to further prevent leakage
                
                This technique is crucial in financial ML because:
                
                - Financial data often has overlapping labels due to multiple prediction horizons
                - Standard cross-validation would leak future information into the training set
                - Performance estimates from standard CV are overly optimistic
                
                Purged CV provides more realistic performance estimates and helps prevent overfitting.
            """
        }
        
        # Check for matches in the QA pairs
        for question, answer in qa_pairs.items():
            if any(keyword in prompt.lower() for keyword in question.split()):
                return answer.strip()
        
        # Default response for questions not in the dictionary
        return """
            That's an interesting question about financial machine learning. While I don't have a pre-defined answer for this specific query, I can suggest exploring López de Prado's book "Advances in Financial Machine Learning" for detailed information on this topic.
            
            The book covers data structures, labeling techniques, sample weights, feature engineering, model evaluation, and portfolio construction techniques specifically designed for financial applications.
            
            You can also explore the other sections of this dashboard to see practical implementations of many of these concepts with forex market data.
        """
    
    # Chat input
    if prompt := st.chat_input("Ask a question about financial machine learning"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        response = generate_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
