Asset Universe
==============

The asset universe is a collection of historical daily returns of user-specified stocks and alternetive assets. It downloads historical data from the following sources:

- [Alpha Vantage](https://www.alphavantage.co/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Quandl Wiki Continuous Futures](https://www.quandl.com/data/CHRIS-Wiki-Continuous-Futures/)
- [Federal Reserve Economic Database](https://fred.stlouisfed.org/series/GOLDAMGBD228NLBM)

## Installation

Installation is done using ```pip``` or ```pip3```:

```bash
git clone URL
cd assetuniverse
pip install .
```

## Substitutions

The asset universe will download substitute data for certain assets that are either unavailable or receive data with too many NaNs.

As of this writing (February 2, 2018), the following substitutes exist:
- FLBAX: TLT
- VDIGX: VIG
- FSIVX: ACWX
- DFEMX: EEM
- VBLTX: BLV
- VGSLX: VNQ
- FNCMX: ONEQ
- BRK-B: BRK-A
- INCO: INDA
- EWS: SGF
- VOO: SPY