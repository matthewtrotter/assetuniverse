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

from .utils.asset import Asset
from .utils.downloaders import YahooFinanceDownloader, FredDownloader, OfflineDownloader

# from assetuniverse import Asset
# from assetuniverse.downloaders import YahooFinanceDownloader, FredDownloader, OfflineDownloader

class AssetUniverse:
    def __init__(self, start, end, assets: List[Asset], offline=False, borrow_spread=1.5):
        self.start = start
        self.end = end
        self.assets = {asset.ticker: asset for asset in assets}
        self.offline = offline
        self.cashasset = Asset(start=start, end=end, ticker='VFISX')
        self.borrowrate = Asset(start=start, end=end, ticker='Fed Funds Rate', data_source='FRED')
        self.borrow_spread = borrow_spread      # Percentage points above Fed Funds Rate
        self.download()


    def download(self) -> None:
        """Download all price and return data for all assets
        """
        print('Downloading asset universe data... ', flush=True)
        
        # Separate tickers into separate downloader lists
        YahooFinanceTickers = [a.ticker for a in self.assets.values() if a.data_source == 'Yahoo Finance']
        YahooFinanceTickers = YahooFinanceTickers + [self.cashasset.ticker]
        FredTickers = [a.ticker for a in self.assets.values() if a.data_source == 'FRED']
        FredTickers = FredTickers + [self.borrowrate.ticker]
        OfflineTickers = [a.ticker for a in self.assets.values() if a.data_source == 'Offline']

        # Download
        yfd = YahooFinanceDownloader(self.start, self.end, YahooFinanceTickers)
        fd = FredDownloader(self.start, self.end, FredTickers)
        od = OfflineDownloader(self.start, self.end, OfflineTickers)
        prices_list = [
            yfd.download(),
            fd.download(),
            od.download()
        ]

        # Join all closes together on same date axis
        joined_prices = self._join_prices(prices_list)

        # Rename cash and borrow rate
        cashname = 'Cash'
        borrowname = 'Borrow Rate'
        joined_prices = joined_prices.rename(columns={self.cashasset.ticker: cashname, self.borrowrate.ticker: borrowname})
        self.cashasset.ticker = cashname
        self.borrowrate.ticker = borrowname

        # Add spread to borrow rate
        joined_prices.loc[:, self.borrowrate.ticker] = joined_prices.loc[:, self.borrowrate.ticker] + self.borrow_spread

        # Assign prices to each asset individually
        for asset in self.assets.values():
            asset.assign_prices(joined_prices[asset.ticker])
        self.cashasset.assign_prices(joined_prices[self.cashasset.ticker])
        self.borrowrate.assign_prices(joined_prices[self.borrowrate.ticker])
        print('Done.', flush=True)

    def _join_prices(self, prices_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Join the prices of the raw downloaded data to have the same dates.

        Parameters
        ----------
        prices : List[pd.DataFrame]
            List of prices

        Returns
        -------
        pd.DataFrame
            Joined prices on same date axis
        """
        # Join all closes together
        joined_prices = pd.DataFrame()
        joined_prices = prices_list[0]
        for prices in prices_list[1:]:
            if not prices.empty:
                joined_prices = joined_prices.join(prices, how='inner')
        # closes = closes.join(annualBorrowRate, how='left')    # Do I need this for the borrow rate? Different than how='inner'

        # Forward fill all the NaNs and zeros
        joined_prices[joined_prices == 0] = np.nan
        joined_prices.fillna(method="ffill", inplace=True)
        joined_prices.dropna(axis=0, how="any", inplace=True)

        # # Forward-fill cash closes - Do I need this part?
        # idx = date_range(self.start, self.end)
        # closesCash = closesCash.reindex(idx, method='ffill')

        return joined_prices

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
        tickers = list(self.assets.keys())
        if include_cash:
            tickers = tickers + [self.cashasset.ticker]
        if include_borrow_rate:
            tickers = tickers + [self.borrowrate.ticker]
        return tickers

    def returns(self, tickers: List[str]=[], start: datetime.date = None, end: datetime.date = None, normalize=True) -> DataFrame:
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
            Selected daily returns
        """
        if len(tickers) == 0:
            tickers = list(self.assets.keys()) + [self.cashasset.ticker, self.borrowrate.ticker]
        selected_returns = []
        for ticker in tickers:
            if ticker in self.assets.keys():
                returns = self.assets[ticker].returns
            elif ticker == self.cashasset.ticker:
                returns = self.cashasset.returns
            elif ticker == self.borrowrate.ticker:
                returns = self.borrowrate.returns
            selected_returns.append(returns)
        return pd.concat(selected_returns, axis=1, join='inner')

    def prices(self, tickers: List[str]=[], start: datetime.date = None, end: datetime.date = None, normalize=True) -> DataFrame:
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
        if len(tickers) == 0:
            tickers = list(self.assets.keys()) + [self.cashasset.ticker, self.borrowrate.ticker]
        selected_prices = []
        if normalize == True:
            for ticker in tickers:
                if ticker in self.assets.keys():
                    prices_normalized = self.assets[ticker].prices_normalized
                elif ticker == self.cashasset.ticker:
                    prices_normalized = self.cashasset.prices_normalized
                elif ticker == self.borrowrate.ticker:
                    prices_normalized = self.borrowrate.prices_normalized
                selected_prices.append(prices_normalized)
        else:
            for ticker in tickers:
                if ticker in self.assets.keys():
                    prices = self.assets[ticker].prices
                elif ticker == self.cashasset.ticker:
                    prices = self.cashasset.prices
                elif ticker == self.borrowrate.ticker:
                    prices = self.borrowrate.prices
                selected_prices.append(prices)
        return pd.concat(selected_prices, axis=1, join='inner')


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
    assets = [
        Asset(start, end, 'AAPL'),
        Asset(start, end, 'CL=F'),
        Asset(start, end, 'EURUSD=X'),
    ]

    AU = AssetUniverse(start, end, assets)
    AU.plotprices()
    # AU.correlation_histogram(sym[0], sym[1])
    print(AU.correlation_matrix())
    print(AU.correlation_matrix(['GOOG', 'UBT']))
    print(AU.correlation_matrix(['BRK B', 'ARKW', 'AMZN']))
