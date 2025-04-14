# Financial Machine Learning Dashboard

This interactive dashboard explores the key concepts from Marcos López de Prado's book "Advances in Financial Machine Learning" applied to forex market data.

## Features

- **Data Structures**: Interactive exploration of alternative bar types (time bars, tick bars, volume bars)
- **Labeling Techniques**: Implementation of fixed-time horizon and triple-barrier methods
- **Sample Weights & Features**: Visualization of sample weights and fractional differentiation
- **Backtesting**: Strategy testing with performance metrics and equity curves
- **Portfolio Construction**: Kelly criterion and meta-labeling for bet sizing
- **Chat with Expert**: Q&A interface for questions about financial machine learning concepts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Data

The dashboard uses real forex data from Yahoo Finance API with fallback to synthetic data when needed.

## Requirements

See requirements.txt for a full list of dependencies.

## Author

Created as a demonstration of López de Prado's financial machine learning concepts.
