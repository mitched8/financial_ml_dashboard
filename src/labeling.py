"""
Implementation of labeling techniques from López de Prado's book
"Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd


def fixed_time_horizon(prices, horizon, threshold):
    """
    Apply fixed-time horizon labeling.
    
    Parameters:
    -----------
    prices : pandas.Series
        Series of asset prices
    horizon : int
        Number of periods to look ahead
    threshold : float
        Return threshold for considering a price move significant
        
    Returns:
    --------
    pandas.Series
        Labels: 1 (positive move), -1 (negative move), 0 (no significant move)
    """
    # Calculate future returns
    future_returns = prices.pct_change(horizon).shift(-horizon)
    
    # Apply threshold to determine labels
    labels = np.zeros(len(prices))
    labels[future_returns > threshold] = 1
    labels[future_returns < -threshold] = -1
    
    return pd.Series(labels, index=prices.index)


def triple_barrier_labeling(prices, volatility, upper_barrier=2, lower_barrier=2, 
                           max_horizon=10, min_ret=0):
    """
    Apply the Triple-Barrier Method for labeling.
    
    Parameters:
    -----------
    prices : pandas.Series
        Series of asset prices
    volatility : pandas.Series
        Series of volatility estimates
    upper_barrier : float
        Upper barrier multiplier (in terms of volatility)
    lower_barrier : float
        Lower barrier multiplier (in terms of volatility)
    max_horizon : int
        Maximum number of periods to look ahead (vertical barrier)
    min_ret : float
        Minimum return threshold for considering a price move significant
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - 'Label': 1 (upper barrier hit), -1 (lower barrier hit), 0 (vertical barrier hit)
        - 'HitTime': Time when a barrier was hit
    """
    labels = []
    
    for i in range(len(prices) - 1):
        # Skip if we're too close to the end of the series
        if i + max_horizon >= len(prices):
            labels.append({'Label': np.nan, 'HitTime': np.nan})
            continue
        
        # Current price and volatility
        current_price = prices.iloc[i]
        current_vol = volatility.iloc[i]
        
        # Calculate barriers
        upper_price = current_price * (1 + upper_barrier * current_vol)
        lower_price = current_price * (1 - lower_barrier * current_vol)
        
        # Look ahead up to max_horizon
        hit_upper = False
        hit_lower = False
        hit_time = max_horizon
        
        for j in range(1, max_horizon + 1):
            future_price = prices.iloc[i + j]
            
            # Check if price hit upper barrier
            if future_price >= upper_price:
                hit_upper = True
                hit_time = j
                break
            
            # Check if price hit lower barrier
            if future_price <= lower_price:
                hit_lower = True
                hit_time = j
                break
        
        # Determine label
        if hit_upper:
            label = 1
        elif hit_lower:
            label = -1
        else:
            # Vertical barrier hit
            future_return = (prices.iloc[i + max_horizon] / current_price) - 1
            if abs(future_return) < min_ret:
                label = 0
            else:
                label = np.sign(future_return)
        
        labels.append({
            'Label': label,
            'HitTime': hit_time
        })
    
    # Add NaN for the last point
    labels.append({'Label': np.nan, 'HitTime': np.nan})
    
    return pd.DataFrame(labels, index=prices.index)


def get_barrier_touches(prices, volatility, upper_barrier=2, lower_barrier=2, max_horizon=10):
    """
    Get the times when price touches each barrier.
    
    Parameters:
    -----------
    prices : pandas.Series
        Series of asset prices
    volatility : pandas.Series
        Series of volatility estimates
    upper_barrier : float
        Upper barrier multiplier (in terms of volatility)
    lower_barrier : float
        Lower barrier multiplier (in terms of volatility)
    max_horizon : int
        Maximum number of periods to look ahead (vertical barrier)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns for each barrier touch time
    """
    touches = []
    
    for i in range(len(prices) - 1):
        # Skip if we're too close to the end of the series
        if i + max_horizon >= len(prices):
            touches.append({
                'UpperTouch': np.nan,
                'LowerTouch': np.nan,
                'VerticalTouch': max_horizon
            })
            continue
        
        # Current price and volatility
        current_price = prices.iloc[i]
        current_vol = volatility.iloc[i]
        
        # Calculate barriers
        upper_price = current_price * (1 + upper_barrier * current_vol)
        lower_price = current_price * (1 - lower_barrier * current_vol)
        
        # Look ahead up to max_horizon
        upper_touch = np.nan
        lower_touch = np.nan
        
        for j in range(1, max_horizon + 1):
            if i + j >= len(prices):
                break
                
            future_price = prices.iloc[i + j]
            
            # Check if price hit upper barrier
            if np.isnan(upper_touch) and future_price >= upper_price:
                upper_touch = j
            
            # Check if price hit lower barrier
            if np.isnan(lower_touch) and future_price <= lower_price:
                lower_touch = j
            
            # If both barriers are hit, we can stop
            if not np.isnan(upper_touch) and not np.isnan(lower_touch):
                break
        
        touches.append({
            'UpperTouch': upper_touch,
            'LowerTouch': lower_touch,
            'VerticalTouch': max_horizon
        })
    
    # Add NaN for the last point
    touches.append({
        'UpperTouch': np.nan,
        'LowerTouch': np.nan,
        'VerticalTouch': np.nan
    })
    
    return pd.DataFrame(touches, index=prices.index)


def meta_labeling(primary_model_probs, prices, volatility, upper_barrier=2, 
                 lower_barrier=2, max_horizon=10):
    """
    Apply meta-labeling to determine bet size.
    
    Parameters:
    -----------
    primary_model_probs : pandas.Series
        Probabilities from the primary model (side prediction)
    prices : pandas.Series
        Series of asset prices
    volatility : pandas.Series
        Series of volatility estimates
    upper_barrier : float
        Upper barrier multiplier (in terms of volatility)
    lower_barrier : float
        Lower barrier multiplier (in terms of volatility)
    max_horizon : int
        Maximum number of periods to look ahead (vertical barrier)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with meta-labels for bet sizing
    """
    # Get side from primary model (1 for long, -1 for short)
    side = np.sign(primary_model_probs - 0.5)
    
    # Apply triple barrier method
    tb_labels = triple_barrier_labeling(
        prices, volatility, upper_barrier, lower_barrier, max_horizon
    )
    
    # Create meta-labels
    meta_labels = []
    
    for i in range(len(side)):
        if np.isnan(tb_labels['Label'].iloc[i]):
            meta_labels.append(np.nan)
            continue
        
        # If side matches direction and barrier was hit, label as 1
        if side.iloc[i] * tb_labels['Label'].iloc[i] > 0:
            meta_labels.append(1)
        else:
            meta_labels.append(0)
    
    return pd.Series(meta_labels, index=prices.index)
