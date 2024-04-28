Asset Universe
==============

![Python Package](https://github.com/matthewtrotter/assetuniverse/workflows/Python%20package/badge.svg)

Download historical daily prices and returns of stocks, futures, cryptocurrencies, and fiat currencies. Plot historical returns and calculate correlation and covariance matrices. Asset Universe downloads historical data from the following sources:

- [Interactive Brokers Trader Workstation](https://www.interactivebrokers.com/en/trading/tws.php)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Federal Reserve Economic Database](https://fred.stlouisfed.org)

## Installing
Install the [package with pip](https://pypi.org/project/assetuniverse/0.1.0/#description):
```bash
pip install assetuniverse
```

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
cashasset = Asset(start, end, 'VFISX', readable_name='Cash')

# Download the daily returns of the assets
AU = AssetUniverse(start, end, assets, cashasset)
AU.download()

# Print returns and prices
print(AU.returns())
print(AU.prices())

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

### Cash Asset
The `AssetUniverse` class requires you to specify a cash asset. This is the risk-free asset that the portfolio can invest in. For example, the Vanguard fund `VFISX` is a money market fund that invests in short-term U.S. government securities. 
```python
cashasset = Asset(start, end, 'VFISX', readable_name='Cash')
AU = AssetUniverse(start, end, assets, cashasset)
```

### Margin Borrowing Rate
The `AssetUniverse` class calculates a typical margin borrowing rate for each asset. The default rate is calculated as the 30-day Federal Funds Effective Rate plus a 1.5% spread. The rate is used to calculate the daily cost of borrowing money on margin in order to buy stock above the portfolio net asset value or short a stock. You can specify your own spread:
    
```python
# Set the borrow spread to 3.0%
AU = AssetUniverse(start, end, assets, cashasset, borrow_spread=3.0)
```
