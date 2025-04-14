"""
Implementation of sample weights and feature engineering techniques from López de Prado's book
"Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
from scipy import stats


def get_sample_weights(labels, num_concurrent=None, average_uniqueness=False, return_num_concurrent=False):
    """
    Calculate sample weights based on label uniqueness.
    
    Parameters:
    -----------
    labels : pandas.Series
        Series of labels
    num_concurrent : pandas.Series, optional
        Pre-calculated number of concurrent labels
    average_uniqueness : bool
        Whether to return the average uniqueness over label lifespan
    return_num_concurrent : bool
        Whether to return the number of concurrent labels
        
    Returns:
    --------
    pandas.Series or tuple
        Sample weights and optionally number of concurrent labels
    """
    if num_concurrent is None:
        # Count number of labels active at each time step
        num_concurrent = pd.Series(0, index=labels.index)
        
        for idx, label in labels.iteritems():
            if np.isnan(label):
                continue
                
            # Find the end of this label's lifespan
            end_idx = idx
            for future_idx in labels.index[labels.index > idx]:
                if np.isnan(labels[future_idx]):
                    end_idx = future_idx
                    break
            
            # Increment count for all points in this label's lifespan
            num_concurrent.loc[idx:end_idx] += 1
    
    # Calculate uniqueness as reciprocal of concurrent labels
    uniqueness = 1 / num_concurrent
    
    if average_uniqueness:
        # Calculate average uniqueness over each label's lifespan
        avg_uniqueness = pd.Series(np.nan, index=labels.index)
        
        for idx, label in labels.iteritems():
            if np.isnan(label):
                continue
                
            # Find the end of this label's lifespan
            end_idx = idx
            for future_idx in labels.index[labels.index > idx]:
                if np.isnan(labels[future_idx]):
                    end_idx = future_idx
                    break
            
            # Calculate average uniqueness for this label
            avg_uniqueness.loc[idx] = uniqueness.loc[idx:end_idx].mean()
        
        uniqueness = avg_uniqueness
    
    if return_num_concurrent:
        return uniqueness, num_concurrent
    else:
        return uniqueness


def get_sequential_bootstrap_weights(labels, sample_weights=None, batch_size=None):
    """
    Apply sequential bootstrapping to generate sample weights.
    
    Parameters:
    -----------
    labels : pandas.Series
        Series of labels
    sample_weights : pandas.Series, optional
        Initial sample weights
    batch_size : int, optional
        Size of each bootstrap batch, defaults to 10% of data
        
    Returns:
    --------
    pandas.Series
        Updated sample weights
    """
    if sample_weights is None:
        sample_weights = pd.Series(1.0, index=labels.index)
    
    if batch_size is None:
        batch_size = max(int(len(labels) * 0.1), 1)
    
    # Remove NaN labels
    valid_indices = labels.dropna().index
    valid_weights = sample_weights.loc[valid_indices]
    
    # Normalize weights
    valid_weights = valid_weights / valid_weights.sum()
    
    # Apply sequential bootstrapping
    updated_weights = pd.Series(0.0, index=labels.index)
    
    for _ in range(int(len(valid_indices) / batch_size)):
        # Sample indices based on current weights
        sampled_indices = np.random.choice(
            valid_indices, size=batch_size, replace=True, p=valid_weights
        )
        
        # Count occurrences of each index
        for idx in sampled_indices:
            updated_weights.loc[idx] += 1
        
        # Update weights to reduce probability of resampling
        for idx in np.unique(sampled_indices):
            valid_weights.loc[idx] *= 0.5
        
        # Renormalize weights
        valid_weights = valid_weights / valid_weights.sum()
    
    # Normalize final weights
    updated_weights = updated_weights / updated_weights.sum()
    
    return updated_weights


def get_time_decay_weights(timestamps, decay_factor=0.5, normalize=True):
    """
    Apply time decay to sample weights.
    
    Parameters:
    -----------
    timestamps : pandas.DatetimeIndex or pandas.Series
        Timestamps for each sample
    decay_factor : float
        Factor by which weights decay (0 to 1)
    normalize : bool
        Whether to normalize weights to sum to 1
        
    Returns:
    --------
    pandas.Series
        Time-decayed weights
    """
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.index
    
    # Convert to numeric for calculation
    if isinstance(timestamps, pd.DatetimeIndex):
        numeric_time = (timestamps - timestamps.min()).total_seconds()
    else:
        numeric_time = np.arange(len(timestamps))
    
    # Calculate time decay weights
    max_time = numeric_time.max()
    weights = np.power(decay_factor, numeric_time / max_time)
    
    # Create Series with original index
    weights_series = pd.Series(weights, index=timestamps)
    
    # Normalize if requested
    if normalize and weights_series.sum() > 0:
        weights_series = weights_series / weights_series.sum()
    
    return weights_series


def fractional_differentiation(series, d, threshold=1e-5):
    """
    Apply fractional differentiation to make a time series stationary
    while preserving memory.
    
    Parameters:
    -----------
    series : pandas.Series
        Time series to differentiate
    d : float
        Differentiation order (between 0 and 1)
    threshold : float
        Threshold for truncating the weights
        
    Returns:
    --------
    pandas.Series
        Fractionally differentiated series
    """
    # Get weights
    weights = get_weights_frac_diff(d, series, threshold)
    width = len(weights)
    
    # Apply weights to series
    df_diff = pd.Series(index=series.index)
    for i in range(width, len(series)):
        # Slice the series for the current window
        window = series.iloc[i-width:i]
        # Apply the weights (need to reverse the window to match the weights)
        df_diff.iloc[i] = np.dot(weights, window[::-1])
    
    return df_diff.dropna()


def get_weights_frac_diff(d, series, threshold=1e-5):
    """
    Compute weights for fractional differentiation.
    
    Parameters:
    -----------
    d : float
        Differentiation order (between 0 and 1)
    series : pandas.Series
        Time series to differentiate (used only for length)
    threshold : float
        Threshold for truncating the weights
        
    Returns:
    --------
    numpy.ndarray
        Weights for fractional differentiation
    """
    weights = [1.0]
    k = 1
    
    while k < len(series):
        weight = weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1
    
    return np.array(weights[::-1])


def find_optimal_d(series, d_range=np.arange(0, 1, 0.1), threshold=1e-5):
    """
    Find the optimal differentiation order that achieves stationarity
    while preserving maximum memory.
    
    Parameters:
    -----------
    series : pandas.Series
        Time series to differentiate
    d_range : numpy.ndarray
        Range of d values to test
    threshold : float
        Threshold for truncating the weights
        
    Returns:
    --------
    float
        Optimal differentiation order
    dict
        Dictionary with test results for each d value
    """
    results = {}
    
    for d in d_range:
        # Apply fractional differentiation
        diff_series = fractional_differentiation(series, d, threshold)
        
        # Skip if too few values
        if len(diff_series) < 10:
            results[d] = {
                'adf_pvalue': np.nan,
                'correlation': np.nan,
                'stationary': np.nan
            }
            continue
        
        # Test for stationarity using ADF test
        adf_result = stats.adfuller(diff_series.dropna())
        
        # Calculate correlation with original series
        common_idx = diff_series.index.intersection(series.index)
        correlation = np.corrcoef(
            diff_series.loc[common_idx], 
            series.loc[common_idx]
        )[0, 1]
        
        results[d] = {
            'adf_pvalue': adf_result[1],
            'correlation': correlation,
            'stationary': adf_result[1] < 0.05
        }
    
    # Find minimum d that achieves stationarity
    stationary_ds = [d for d, res in results.items() 
                    if res['stationary'] is True]
    
    if len(stationary_ds) > 0:
        optimal_d = min(stationary_ds)
    else:
        optimal_d = max(d_range)
    
    return optimal_d, results
