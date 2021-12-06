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
asdf

## Testing
In the project root directory, run `python3 -m pytest tests`