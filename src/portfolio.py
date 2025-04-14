"""
Implementation of bet sizing and portfolio construction techniques
from López de Prado's book "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def kelly_criterion(win_rate, win_loss_ratio):
    """
    Calculate the optimal bet size using the Kelly Criterion.
    
    Parameters:
    -----------
    win_rate : float
        Probability of winning (between 0 and 1)
    win_loss_ratio : float
        Ratio of average win to average loss
        
    Returns:
    --------
    float
        Optimal fraction of capital to bet
    """
    # Kelly formula: f* = p - (1-p)/r
    # where p is win rate, r is win/loss ratio
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    
    # Limit to [0, 1] range
    return max(0, min(1, kelly))


def meta_labeling_bet_sizing(probabilities, threshold=0.5, fraction=1.0):
    """
    Calculate bet sizes based on meta-labeling probabilities.
    
    Parameters:
    -----------
    probabilities : pandas.Series
        Probabilities from meta-labeling model
    threshold : float
        Probability threshold for taking a bet
    fraction : float
        Maximum fraction of capital to bet (Kelly fraction)
        
    Returns:
    --------
    pandas.Series
        Bet sizes (between 0 and fraction)
    """
    # Calculate bet sizes
    bet_sizes = pd.Series(0.0, index=probabilities.index)
    
    # Only take bets above threshold
    mask = probabilities > threshold
    
    # Scale bet size by probability
    bet_sizes[mask] = (probabilities[mask] - threshold) / (1 - threshold) * fraction
    
    return bet_sizes


def hierarchical_risk_parity(returns, correlation_method='pearson', linkage_method='single'):
    """
    Implement the Hierarchical Risk Parity (HRP) portfolio allocation.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Asset returns
    correlation_method : str
        Method for calculating correlation matrix
    linkage_method : str
        Method for hierarchical clustering
        
    Returns:
    --------
    pandas.Series
        Portfolio weights
    """
    # Calculate correlation matrix
    corr = returns.corr(method=correlation_method)
    
    # Convert correlation to distance
    distance = np.sqrt(0.5 * (1 - corr))
    
    # Perform hierarchical clustering
    condensed_dist = squareform(distance)
    z = linkage(condensed_dist, method=linkage_method)
    
    # Get cluster assignments
    num_assets = len(returns.columns)
    clusters = fcluster(z, t=0.7 * np.max(z[:, 2]), criterion='distance')
    
    # Sort assets by cluster
    sort_idx = np.argsort(clusters)
    
    # Calculate inverse variance (quasi-diagonalization)
    var = np.diag(returns.cov())
    inv_var = 1 / var
    
    # Calculate cluster variances
    cluster_var = {}
    for i in range(1, max(clusters) + 1):
        cluster_indices = np.where(clusters == i)[0]
        cluster_var[i] = np.sum(inv_var[cluster_indices])
    
    # Calculate weights
    weights = pd.Series(0, index=returns.columns)
    
    for i in range(1, max(clusters) + 1):
        cluster_indices = np.where(clusters == i)[0]
        cluster_weight = cluster_var[i] / sum(cluster_var.values())
        
        # Distribute cluster weight among assets
        for idx in cluster_indices:
            asset = returns.columns[idx]
            asset_weight = cluster_weight * (inv_var[idx] / np.sum(inv_var[cluster_indices]))
            weights[asset] = asset_weight
    
    return weights


def minimum_variance_portfolio(returns):
    """
    Calculate minimum variance portfolio weights.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Asset returns
        
    Returns:
    --------
    pandas.Series
        Portfolio weights
    """
    n = len(returns.columns)
    cov = returns.cov()
    
    # Define objective function (portfolio variance)
    def objective(weights):
        return np.dot(weights.T, np.dot(cov, weights))
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: no short selling
    bounds = tuple((0, 1) for _ in range(n))
    
    # Initial guess: equal weights
    initial_weights = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    # Return weights as Series
    return pd.Series(result['x'], index=returns.columns)


def equal_risk_contribution(returns):
    """
    Calculate Equal Risk Contribution (ERC) portfolio weights.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Asset returns
        
    Returns:
    --------
    pandas.Series
        Portfolio weights
    """
    n = len(returns.columns)
    cov = returns.cov().values
    
    # Define objective function (variance from equal risk contribution)
    def objective(weights):
        weights = np.array(weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        risk_contribution = weights * np.dot(cov, weights) / portfolio_risk
        target_risk = portfolio_risk / n
        return np.sum((risk_contribution - target_risk)**2)
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: no short selling
    bounds = tuple((0, 1) for _ in range(n))
    
    # Initial guess: equal weights
    initial_weights = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    # Return weights as Series
    return pd.Series(result['x'], index=returns.columns)


def dynamic_position_sizing(signals, volatility, target_volatility=0.01, max_leverage=1.0):
    """
    Implement dynamic position sizing based on volatility targeting.
    
    Parameters:
    -----------
    signals : pandas.Series
        Trading signals (-1, 0, 1)
    volatility : pandas.Series
        Asset volatility estimates
    target_volatility : float
        Target portfolio volatility
    max_leverage : float
        Maximum allowed leverage
        
    Returns:
    --------
    pandas.Series
        Position sizes
    """
    # Calculate position sizes
    position_sizes = signals * (target_volatility / volatility)
    
    # Limit leverage
    position_sizes = position_sizes.clip(-max_leverage, max_leverage)
    
    return position_sizes
