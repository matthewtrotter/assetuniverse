Asset Universe
==============

The asset universe downloads historical daily prices and returns of user-specified stocks, futures, and currencies. It downloads historical data from the following sources:

- [Yahoo Finance](https://finance.yahoo.com/)
- [Federal Reserve Economic Database](https://fred.stlouisfed.org/series/GOLDAMGBD228NLBM)
## Installation

Install Asset Universe using `pip`:

```bash
git clone URL
cd assetuniverse
python3 -m pip install .
```

## Using
```python
days = 2*365    # 2 years
end = datetime.date.today()
start = end - datetime.timedelta(days=days)
assets = [
    Asset(start, end, 'AAPL'),
    Asset(start, end, 'CL=F', display_name='Oil'),
    Asset(start, end, 'EURUSD=X'),
]

AU = AssetUniverse(start, end, assets)
AU.plot_prices()

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
In the project root directory, run `python3 -m pytest tests`