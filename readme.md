Asset Universe
==============

The asset universe is a collection of historical daily returns of user-specified stocks and alternetive assets. It downloads historical data from the following sources:

- [Interactive Brokers](https://www.interactivebrokers.com/en/home.php)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Federal Reserve Economic Database](https://fred.stlouisfed.org/series/GOLDAMGBD228NLBM)

## Setup

Install the [IB API](https://interactivebrokers.github.io/tws-api/initial_setup.html#install) to your Python environment.

Open and log in to [Interactive Brokers Trader Workstation](https://www.interactivebrokers.com/en/index.php?f=14099) with the API enabled:
1. In Configuration, go to API > Settings
2. Check "Enable ActiveX and Socket Clients"
3. Check "Read-Only API"
4. Set the "Socket port" to 7496

## Installation

Install Asset Universe using `pip`:

```bash
git clone URL
cd assetuniverse
python3 -m pip install .
```

## Using

Define your contracts (stocks and/or futures) as a `list` of `AssetUniverse.AssetUniverseContract` class:
```python
contracts = []
contract = AssetUniverseContract(
    secType = 'FUT',
    currency = 'USD',
    exchange = 'GLOBEX',
    localSymbol = 'ESU1', # "Local Name" on the IB details page (link below)
    data_source = 'TWS'
)
contracts.append(contract)

contract = AssetUniverseContract(
    secType='FUT',
    currency='USD',
    exchange='ECBOT',
    localSymbol='ZB   SEP 21',
    data_source='TWS'
)
contracts.append(contract)

contract = AssetUniverseContract(
    symbol='SPY',
    secType='STK',
    currency='USD',
    exchange='SMART',
    data_source='TWS'
)
contracts.append(contract)

contract = AssetUniverseContract(
    symbol='AAPL',
    secType='STK',
    currency='USD',
    exchange='SMART',
    data_source='TWS' 
)
contracts.append(contract)

contract = AssetUniverseContract(
    symbol='SBUX',
    secType='STK',
    currency='USD',
    exchange='SMART',
    data_source='Yahoo Finance' # Alternate data source is Yahoo Finance
)
contracts.append(contract)

# Instantiate asset universe - will start the download
days = 365
end = datetime.date.today()
start = end - datetime.timedelta(days=days)
AU = AssetUniverse(start, end, contracts, offline=False)

# Plot and calculate correlations
AU.plotprices()
print(AU.correlation_matrix())
print(AU.correlation_matrix(['SPY', 'ESU1']))
```

### Local Name
[Here](https://misc.interactivebrokers.com/cstools/contract_info/v3.10/index.php?action=CONTRACT_DETAILS&clt=1&detlev=2&site=IB&sess=1630107982&mid=001&conid=428520022) is an example of the details for the ES futures contract.