#!/usr/bin/env python

"""
Title: Asset Universe Class Definition
Author: Matthew Trotter
Description:
    This asset universe class will download and parse stock and
    financial asset data from various online sources. You can use
    the price DataFrame or daily returns DataFrame for further
    analysis and testing.

Copyright 2019 Matthew Trotter
"""

import copy
import datetime
import numpy as np
from pandas import DataFrame, to_datetime, date_range
import pandas as pd
import plotly.express as px
from typing import List, Optional

from assetuniverse.asset import Asset

class AssetUniverse:
    def __init__(self, start, end, assets: List[Asset], offline=False, borrow_spread=1.5):
        self.start = start
        self.end = end
        self.assets = assets
        self.offline = offline
        self.cashsym = Asset(start=start, end=end, ticker='VFISX')
        self.ratesym = Asset(start=start, end=end, ticker='Fed Funds Rate')
        self.borrow_spread = borrow_spread      # Percentage points above Fed Funds Rate
        self.download()


    def download(self):
        """Download all price and return data
        """
        print('Downloading asset universe data... ', flush=True)
        YahooFinanceTickers = [a.ticker for a in self.assets if a.data_source == 'Yahoo Finance']
        FredTickers = [a.ticker for a in self.assets if a.data_source == 'FRED']
        OfflineTickers = [a.ticker for a in self.assets if a.data_source == 'Offline']



        if self.offline:
            closes = self.generateOfflineData(self.sym)
            closesCash = self.generateOfflineData([self.cashsym])
            annualBorrowRate = self.generateOfflineData([self.ratesym])
        else:
            closes = self.downloadFromTws(self.sym)
            closesYahoo = self.downloadFromYahoo(self.sym)
            closesCash = self.downloadFromYahoo([self.cashsym])
            annualBorrowRate = self.downloadFromFred([self.ratesym])

            # Join all closes together on same dates
            if closes.size:
                if closesYahoo.size:
                    closes = closes.join(closesYahoo, how="inner")
            elif closesYahoo.size:
                closes = closesYahoo

            # Forward-fill cash closes
            idx = date_range(self.start, self.end)
            closesCash = closesCash.reindex(idx, method='ffill')

        # Join all closes together
        closes = closes.join(closesCash, how="inner")
        closes = closes.join(annualBorrowRate, how="left")
        self.index_names = []
        if self.indices is not None:
            for index in self.indices:
                closes = closes.join(index.p, how="inner")
            self.index_names = list(index.name for index in self.indices)

        # Forward fill all the NaNs and zeros
        closes[closes == 0] = np.nan
        closes.fillna(method="ffill", inplace=True)
        closes.dropna(axis=0, how="any", inplace=True)

        # Separate closes from borrow rate
        self.rborrow = closes.loc[:, self.freddic[self.ratesym.symbol]] + self.borrow_spread
        closes = closes.drop(self.freddic[self.ratesym.symbol], axis="columns")

        # Normalize price to start from $1
        self.originalprices = copy.deepcopy(closes)
        for i in range(0, len(closes.columns)):
            closes.iloc[:,i] = closes.iloc[:,i]/closes.iloc[0,i]

        # Separate cash from assets
        string_syms = [s.get_symbol() for s in self.sym]
        self.p = closes.loc[:, string_syms+self.index_names]
        self.pcash = closes.loc[:, self.cashsym.get_symbol()]

        # Calculate daily returns
        self.r = self.p.pct_change()
        self.r = self.r.iloc[1:, :]  # delete first row - pct_change() returns first row as NaN
        self.rcash = self.pcash.pct_change()
        self.rcash = self.rcash.iloc[1:]  # delete first row - pct_change() returns first row as NaN

        self.allsym = self.r.columns
        print('Done.', flush=True)

    def tickers(self, include_cash=True, include_borrow_rate=True) -> List[str]:
        """Get a list of all ticker symbols

        Parameters
        ----------
        include_cash : bool, optional
            Include the cash asset in the result, by default True
        include_borrow_rate : bool, optional
            Include the borrow rate in the result, by default True

        Returns
        -------
        List[str]
            List of ticker symbols in the asset universe
        """
        pass

    def returns(self, tickers: List[str], start: datetime.date = None, end: datetime.date = None, normalize=True) -> DataFrame:
        """Get the daily returns between start and end dates.

        Parameters
        ----------
        tickers : List[str]
            Tickers to include, by default all tickers in the AssetUniverse
        start : datetime.date, optional
            Start date, by default the start date of the AssetUniverse
        end : datetime.date, optional
            End date, by default the start date of the AssetUniverse

        Returns
        -------
        DataFrame
            Daily returns
        """
        pass

    def prices(self, tickers: List[str], start: datetime.date = None, end: datetime.date = None, normalize=True) -> DataFrame:
        """Get the daily prices between start and end dates.

        Parameters
        ----------
        tickers : List[str]
            Tickers to include, by default all tickers in the AssetUniverse
        start : datetime.date, optional
            Start date, by default the start date of the AssetUniverse
        end : datetime.date, optional
            End date, by default the start date of the AssetUniverse
        normalize : bool, optional
            Normalize start prices to $1, by default True

        Returns
        -------
        DataFrame
            Daily prices
        """
        pass


    def delete(self, tickers: List[str] = []):
        """Delete the tickers from the asset universe

        Parameters
        ----------
        tickers : List[str], optional
            Tickers to delete, by default []
        """
        pass


    def plot_prices(self, tickers: List[str], start: datetime.date = None, end: datetime.date = None, normalize=True) -> None:
        """Plot asset prices over time

        Parameters
        ----------
        tickers : List[str]
            Tickers to include, by default all tickers in the AssetUniverse
        start : datetime.date, optional
            Start date, by default the start date of the AssetUniverse
        end : datetime.date, optional
            End date, by default the start date of the AssetUniverse
        normalize : bool, optional
            Normalize start prices to $1, by default True
        """
        prices = self.p.copy(deep=True)
        prices['Date'] = prices.index
        fig = px.line(prices, x="Date", y=self.p.columns)
        # fig = px.line(prices, x="Date", y=self.p.columns,
        #       hover_data={"Date": "|%B %d, %Y"})
        fig.update_yaxes(
            type='log'
        )
        fig.show()


    def __add__(self, other):
        # Combine two asset universes into one asset universe
        combinedAU = copy.deepcopy(self)
        if other is not None:
            # Check for compatibility
            if self.start != other.start or self.end != other.end:
                raise RuntimeError('Start or end dates do not match!')
            if self.offline != other.offline:
                raise RuntimeError('Offline statuses do not match!')

            # Merge symbols, prices, and returns
            combinedAU.sym = combinedAU.sym + other.sym
            combinedAU.p = combinedAU.p.join(other.p, how='inner')  # merge to have same dates
            combinedAU.r = combinedAU.r.join(other.r, how='inner')
            combinedAU.originalprices = combinedAU.originalprices.join(other.originalprices, how='inner')
        return combinedAU


    def correlation_matrix(self, tickers: List[str]) -> np.ndarray:
        """Calculate the correlation matrix

        Parameters
        ----------
        tickers : List[str]
            Tickers to include

        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        pass
        # if len(symbols):
        #     return self.r[symbols].loc[keep_indices].corr(method='pearson').values
        # else:
        #     return self.r.loc[keep_indices].corr(method='pearson').values


    def covariance_matrix(self, tickers: List[str]) -> np.ndarray:
        """Calculate the correlation matrix

        Parameters
        ----------
        tickers : List[str]
            Tickers to include

        Returns
        -------
        np.ndarray
            Covariance matrix
        """
        pass


if __name__ == "__main__":
    """end = datetime.datetime.today()
    start = end - datetime.timedelta(days=60)
    symbols = ['RDS-B', 'BRK-B', 'Palladium', 'AAPL', 'GOOG', 'Gold']
    a = AssetUniverse(start, end, symbols)
    a.plotprices()"""

    days = 365
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    # sym = _get_test_contracts()
    # sym = _get_bond_futures_contracts()
    assets = pd.read_excel('examples/assets.xlsx')
    sym = parse_to_contracts(assets)

    AU = AssetUniverse(start, end, sym, offline=True)
    AU.plotprices()
    # AU.correlation_histogram(sym[0], sym[1])
    print(AU.correlation_matrix())
    print(AU.correlation_matrix(['GOOG', 'UBT']))
    print(AU.correlation_matrix(['BRK B', 'ARKW', 'AMZN']))




