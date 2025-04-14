"""
Implementation of cross-validation, feature importance, and backtesting techniques
from López de Prado's book "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def purged_kfold_cv(X, y, embargo_size, n_splits=5):
    """
    Implement Purged K-fold Cross-Validation.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Labels with datetime index
    embargo_size : int
        Number of samples to embargo after each test set
    n_splits : int
        Number of folds
        
    Returns:
    --------
    list
        List of tuples (train_indices, test_indices)
    """
    # Convert to numpy arrays if pandas objects
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        index = X.index
        X_values = X.values
    else:
        index = np.arange(len(X))
        X_values = X
        
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Initialize standard K-fold CV
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    # Generate splits
    splits = []
    for train_indices, test_indices in kf.split(X_values):
        # Get the times for train and test sets
        test_times = index[test_indices]
        
        # Determine purged train indices
        train_times = index[train_indices]
        purged_train_indices = []
        
        for idx, train_time in zip(train_indices, train_times):
            # Check if this train sample overlaps with any test sample
            overlap = False
            for test_time in test_times:
                # If train time is within embargo period of test time, exclude it
                if abs((train_time - test_time).total_seconds()) < embargo_size:
                    overlap = True
                    break
            
            if not overlap:
                purged_train_indices.append(idx)
        
        splits.append((purged_train_indices, test_indices))
    
    return splits


def feature_importance_MDI(model, feature_names):
    """
    Calculate Mean Decrease Impurity (MDI) feature importance.
    
    Parameters:
    -----------
    model : sklearn tree-based model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pandas.Series
        Feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def feature_importance_MDA(model, X, y, cv_splits, feature_names, scoring='accuracy'):
    """
    Calculate Mean Decrease Accuracy (MDA) feature importance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Labels
    cv_splits : list
        List of tuples (train_indices, test_indices)
    feature_names : list
        List of feature names
    scoring : str
        Scoring metric ('accuracy', 'precision', 'recall', 'f1')
        
    Returns:
    --------
    pandas.Series
        Feature importances
    """
    # Define scoring function
    if scoring == 'accuracy':
        score_func = accuracy_score
    elif scoring == 'precision':
        score_func = precision_score
    elif scoring == 'recall':
        score_func = recall_score
    elif scoring == 'f1':
        score_func = f1_score
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")
    
    # Calculate baseline scores
    baseline_scores = []
    for train_indices, test_indices in cv_splits:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = score_func(y_test, y_pred)
        baseline_scores.append(score)
    
    baseline_score = np.mean(baseline_scores)
    
    # Calculate feature importances
    importances = {}
    
    for feature in feature_names:
        # Create shuffled versions of X for each fold
        feature_scores = []
        
        for train_indices, test_indices in cv_splits:
            X_train, X_test = X.iloc[train_indices].copy(), X.iloc[test_indices].copy()
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            # Shuffle the feature
            X_test_shuffled = X_test.copy()
            X_test_shuffled[feature] = np.random.permutation(X_test[feature].values)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_shuffled)
            score = score_func(y_test, y_pred)
            feature_scores.append(score)
        
        # Calculate importance as decrease in score
        importance = baseline_score - np.mean(feature_scores)
        importances[feature] = importance
    
    return pd.Series(importances).sort_values(ascending=False)


def backtest_strategy(prices, positions, transaction_cost=0.0001):
    """
    Backtest a trading strategy.
    
    Parameters:
    -----------
    prices : pandas.Series
        Asset prices
    positions : pandas.Series
        Position sizes (-1 for short, 0 for neutral, 1 for long)
    transaction_cost : float
        Transaction cost as a fraction of price
        
    Returns:
    --------
    pandas.DataFrame
        Backtest results with columns:
        - 'Position': Position size
        - 'Price': Asset price
        - 'Return': Period return
        - 'Strategy': Strategy return
        - 'Equity': Cumulative strategy return
        - 'DrawdownPct': Drawdown percentage
    """
    # Ensure positions and prices have the same index
    positions = positions.reindex(prices.index)
    positions = positions.fillna(0)
    
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Calculate transaction costs
    position_changes = positions.diff().fillna(0)
    transaction_costs = abs(position_changes) * transaction_cost
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1) * returns - transaction_costs
    
    # Calculate equity curve
    equity = (1 + strategy_returns).cumprod()
    
    # Calculate drawdowns
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Position': positions,
        'Price': prices,
        'Return': returns,
        'Strategy': strategy_returns,
        'Equity': equity,
        'DrawdownPct': drawdown
    })
    
    return results


def calculate_performance_metrics(backtest_results):
    """
    Calculate performance metrics from backtest results.
    
    Parameters:
    -----------
    backtest_results : pandas.DataFrame
        Backtest results from backtest_strategy function
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Extract relevant series
    strategy_returns = backtest_results['Strategy']
    equity = backtest_results['Equity']
    drawdown = backtest_results['DrawdownPct']
    
    # Calculate metrics
    total_return = equity.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(equity)) - 1
    annualized_volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    max_drawdown = drawdown.min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
    
    win_rate = (strategy_returns > 0).mean()
    loss_rate = (strategy_returns < 0).mean()
    
    avg_win = strategy_returns[strategy_returns > 0].mean() if any(strategy_returns > 0) else 0
    avg_loss = strategy_returns[strategy_returns < 0].mean() if any(strategy_returns < 0) else 0
    
    profit_factor = abs(strategy_returns[strategy_returns > 0].sum() / 
                        strategy_returns[strategy_returns < 0].sum()) if any(strategy_returns < 0) else np.inf
    
    # Compile metrics
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Win Rate': win_rate,
        'Loss Rate': loss_rate,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Profit Factor': profit_factor
    }
    
    return metrics


def walk_forward_validation(X, y, model, window_size, step_size=1):
    """
    Perform walk-forward validation.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Labels
    model : sklearn model
        Model to train and validate
    window_size : int
        Size of the rolling window for training
    step_size : int
        Number of steps to move forward for each validation
        
    Returns:
    --------
    pandas.Series
        Out-of-sample predictions
    """
    predictions = pd.Series(index=y.index, dtype=float)
    
    for i in range(window_size, len(X), step_size):
        # Define train and test sets
        X_train = X.iloc[i-window_size:i]
        y_train = y.iloc[i-window_size:i]
        
        # Define test point(s)
        test_end = min(i + step_size, len(X))
        X_test = X.iloc[i:test_end]
        
        # Skip if no test data
        if len(X_test) == 0:
            continue
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        preds = model.predict(X_test)
        
        # Store predictions
        predictions.iloc[i:test_end] = preds
    
    return predictions
