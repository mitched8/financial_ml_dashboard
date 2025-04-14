# Key Concepts from "Advances in Financial Machine Learning" by Marcos López de Prado

## 1. Financial Data Structures
- **Time bars**: Sample at fixed time intervals (not recommended due to poor statistical properties)
- **Tick bars**: Sample after every x transactions (better statistical properties)
- **Volume bars**: Sample every time x units of the asset are exchanged
- **Dollar bars**: Sample every time $x worth of an asset are exchanged
- **Information-driven bars**:
  - Tick imbalance bars (TIBs): Produced more frequently when there is informed trading
  - Volume/dollar imbalance bars: Replace the sign with the signed volume/dollar
  - Run bars: Analyze sequence of buys in overall volume

## 2. Labeling Techniques
- **Fixed-horizon method**: Simple but naive approach based on returns
- **Triple-Barrier method**: Uses three barriers (stop loss, profit take, expiration)
- **Meta-labeling**: Used to learn the size of bets after knowing the side

## 3. Sample Weights
- **Label uniqueness**: Calculate uniqueness of labels to avoid redundant information
- **Sequential bootstrapping**: Continually modifies probability distribution to reduce overlaps
- **Time decay weighting**: Apply decay to weight based on uniqueness

## 4. Fractionally Differentiated Features
- **Integer differentiation**: Removes 'memory' from price series but loses information
- **Fractional differentiation**: Preserves memory while achieving stationarity
- **Backshift operator**: Used to implement fractional differentiation

## 5. Ensemble Methods
- **Bootstrap aggregation (bagging)**: Fits estimators on different training sets
- **Random forests**: Decision tree ensemble method (needs adjustments for financial data)
- **Boosting**: Addresses underfitting but less dangerous than overfitting in finance

## 6. Cross-validation
- **Purged k-fold CV**: Removes training observations whose labels overlap with test set
- **Combinatorial purged CV**: Allows multiple paths to be tested with ordered groups

## 7. Feature Importance
- **Mean decrease impurity (MDI)**: For tree methods, measures node impurity reduction
- **Mean decrease accuracy (MDA)**: Out-of-sample estimate that permutes features
- **Single feature importance (SFI)**: Computes out-of-sample importance of individual features

## 8. Hyperparameter Tuning
- **Proper CV**: Must use purged K-fold CV + bagging
- **Random search CV**: For optimization over many parameters
- **Log uniform distribution**: For sampling parameters with non-linear effects

## 9. Bet Sizing
- **Kelly criterion**: Optimal bet sizing based on edge and odds
- **Meta-labeling approach**: Secondary model to determine bet size

## 10. Backtesting
- **Seven deadly sins**: Survivorship bias, lookahead bias, storytelling, data snooping, etc.
- **Walk-forward testing**: Common but easy to overfit
- **CV backtesting**: Tests multiple alternative histories
- **Synthetic data backtesting**: Reduces overfitting by generating multiple scenarios

## 11. Backtest Statistics
- **General characteristics**: Time range, AUM, leverage, bet frequency, etc.
- **Performance metrics**: PnL, annualized return, hit ratio, time-weighted return
- **Runs and drawdowns**: Measure downside risk and concentration of returns
- **Implementation shortfall**: Fees/slippage per portfolio turnover
- **Efficiency**: Sharpe Ratio, Deflated SR, probabilistic SR

## 12. Strategy Risk
- **Strategy capacity**: Maximum AUM before returns deteriorate
- **Strategy risk**: Probability of strategy failure
- **Overfitting risk**: Probability that backtest is overfit

## 13. Machine Learning Asset Allocation
- **Hierarchical Risk Parity (HRP)**: Allocates based on hierarchical clusters
- **Nested clustered optimization (NCO)**: Improves on HRP

## 14. Structural Breaks
- **CUSUM test**: Detects structural breaks in time series
- **Chow test**: Tests if coefficients in two linear regressions are equal

## 15. Entropy Features
- **Shannon entropy**: Measures uncertainty in a random variable
- **Mutual information**: Measures dependence between variables

## 16. Microstructural Features
- **Price sequences**: Patterns in tick data
- **Strategic models**: Model market participants' behavior
- **Sequential models**: Analyze order flow
