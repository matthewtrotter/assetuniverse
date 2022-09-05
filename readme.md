Asset Universe
==============

The asset universe downloads historical daily prices and returns of user-specified stocks, futures, and currencies. It downloads historical data from the following sources:

- [Interactive Brokers Trader Workstation](https://www.interactivebrokers.com/en/trading/tws.php)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Federal Reserve Economic Database](https://fred.stlouisfed.org)

## Using
```python
import datetime
from assetuniverse import Asset, AssetUniverse

# Set start date, end date, and assets to download
days = 2*365    # 2 years
end = datetime.date.today()
start = end - datetime.timedelta(days=days)
assets = [
    Asset(start, end, 'AAPL'),
    Asset(start, end, 'CL=F', readable_name='Oil'),
    Asset(start, end, 'EURUSD=X'),
]

# Download the daily returns of the assets
AU = AssetUniverse(start, end, assets)
AU.download()

# Plot price history in a webpage.
# Prices are normalized to start at $1.
AU.plot_prices()

# Print covariance and correlation matrices
print(AU.correlation_matrix())  # correlation matrix over entire history
print(AU.correlation_matrix(
    ['AAPL', 'CL=F'],   # only of these two assets
    start=end - datetime.timedelta(days=30) # over past month
    ))
print(AU.covariance_matrix(
    ['AAPL', 'CL=F']
    ))
```

## Testing
In the project root directory, run `pytest test.py`