#!/usr/bin/env python

import copy
import datetime
import numpy as np
from pandas import DataFrame, to_datetime, date_range
import pandas as pd
import plotly.express as px
from typing import List

# Use these when importing assetuniverse as a library
from .utils.asset import Asset
from .utils.downloaders import InteractiveBrokersDownloader, YahooFinanceDownloader, FredDownloader, OfflineDownloader

# Use these when running core.py as main
# from utils.asset import Asset
# from utils.downloaders import YahooFinanceDownloader, FredDownloader, OfflineDownloader

# from assetuniverse import Asset
# from assetuniverse.downloaders import YahooFinanceDownloader, FredDownloader, OfflineDownloader

class AssetUniverse:
    def __init__(self, start, end, assets: List[Asset], cashasset:Asset, offline=False, borrow_spread=1.5):
        self.start = start
        self.end = end
        self.assets = {asset.ticker: asset for asset in assets}
        self.offline = offline
        self.cashasset = cashasset#Asset(start=start, end=end, ticker='VFISX', display_name='Cash')
        self.borrowrate = Asset(start=start, end=end, ticker='Fed Funds Rate', readable_name='Borrow Rate', data_source='FRED')
        self.borrow_spread = borrow_spread      # Percentage points above Fed Funds Rate

    def download(self) -> None:
        """Download all price and return data for all assets
        """
        print('Downloading asset universe data... ', flush=True)
        
        # Separate tickers into separate downloader lists
        InteractiveBrokersTickers = [a.ticker for a in self.assets.values() if a.data_source == 'Interactive Brokers']
        InteractiveBrokersCurrencies = [a.currency for a in self.assets.values() if a.data_source == 'Interactive Brokers']
        InteractiveBrokersExchanges = [a.exchange for a in self.assets.values() if a.data_source == 'Interactive Brokers']
        YahooFinanceTickers = [a.ticker for a in self.assets.values() if a.data_source == 'Yahoo Finance']
        if self.cashasset.ticker:
            YahooFinanceTickers = YahooFinanceTickers + [self.cashasset.ticker]
        FredTickers = [a.ticker for a in self.assets.values() if a.data_source == 'FRED']
        FredTickers = FredTickers + [self.borrowrate.ticker]
        OfflineTickers = [a.ticker for a in self.assets.values() if a.data_source == 'Offline']

        # Download
        ibd = InteractiveBrokersDownloader(self.start, self.end, InteractiveBrokersTickers, InteractiveBrokersCurrencies, InteractiveBrokersExchanges)
        yfd = YahooFinanceDownloader(self.start, self.end, YahooFinanceTickers)
        fd = FredDownloader(self.start, self.end, FredTickers)
        od = OfflineDownloader(self.start, self.end, OfflineTickers)
        prices_list = [
            ibd.download(),
            yfd.download(),
            fd.download(),
            od.download()
        ]
        ibd.shutdown()

        # Join all closes together on same date axis
        joined_prices = self._join_prices(prices_list)
        d = [d for d in [ibd.last_dates_downloaded, yfd.last_dates_downloaded, fd.last_dates_downloaded, od.last_dates_downloaded] if d is not None and not d.empty]
        self.last_dates_downloaded = pd.concat(d)

        # Rename cash and borrow rate
        cashname = 'Cash'
        borrowname = 'Borrow Rate'
        joined_prices = joined_prices.rename(columns={self.borrowrate.ticker: borrowname})
        self.borrowrate.ticker = borrowname
        if self.cashasset.ticker:
            joined_prices = joined_prices.rename(columns={self.cashasset.ticker: cashname})
            self.cashasset.ticker = cashname

        # Add spread to borrow rate and convert to "borrow price"
        joined_prices.loc[:, self.borrowrate.ticker] = joined_prices.loc[:, self.borrowrate.ticker] + self.borrow_spread
        daily_borrow_rate = (1 + joined_prices.loc[:, self.borrowrate.ticker]/100)**(1/252)
        joined_prices.loc[:, self.borrowrate.ticker] = daily_borrow_rate.cumprod()

        # Assign prices to each asset individually
        for asset in self.assets.values():
            asset.assign_prices(joined_prices[asset.ticker])
        if self.cashasset.ticker:
            self.cashasset.assign_prices(joined_prices[self.cashasset.ticker])
        else:
            ones = joined_prices[self.borrowrate.ticker]*0 + 1.0
            ones = ones.rename(cashname)
            self.cashasset.assign_prices(ones)
            self.cashasset.ticker = cashname
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
        for i, prices in enumerate(prices_list):
            if not prices.empty and joined_prices.empty:
                # Find first non empty prices
                joined_prices = prices
                continue
            if not joined_prices.empty:
                # Join prices with previous prices
                joined_prices = joined_prices.join(prices, how='inner')
        # closes = closes.join(annualBorrowRate, how='left')    # Do I need this for the borrow rate? Different than how='inner'

        # Forward fill all the NaNs and zeros
        joined_prices[joined_prices == 0] = np.nan
        joined_prices.ffill(inplace=True)
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

    def returns(self, tickers: List[str]=[], start: datetime.date = None, end: datetime.date = None) -> DataFrame:
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
            else:
                raise ValueError(f'Ticker {ticker} not in the asset universe: {self.tickers()}')
            selected_returns.append(returns)
        all_returns = pd.concat(selected_returns, axis=1, join='inner')
        return all_returns.loc[start:end,:]

    def prices(self, tickers: List[str]=[], start: datetime.date = None, end: datetime.date = None, normalize=True) -> DataFrame:
        """Get the daily prices between start and end dates.

        Parameters
        ----------
        tickers : List[str]
            Tickers to include, by default all tickers in the AssetUniverse
        start : datetime.date, optional
            Start date, by default the start date of the AssetUniverse
        end : datetime.date, optional
            End date, by default the end date of the AssetUniverse
        normalize : bool, optional
            Normalize start prices to $1, by default True

        Returns
        -------
        DataFrame
            Selected daily prices
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
                else:
                    raise ValueError(f'Ticker {ticker} not in the asset universe: {self.tickers()}')
                selected_prices.append(prices_normalized)
        else:
            for ticker in tickers:
                if ticker in self.assets.keys():
                    prices = self.assets[ticker].prices
                elif ticker == self.cashasset.ticker:
                    prices = self.cashasset.prices
                elif ticker == self.borrowrate.ticker:
                    prices = self.borrowrate.prices
                else:
                    raise ValueError(f'Ticker {ticker} not in the asset universe: {self.tickers()}')
                selected_prices.append(prices)
        all_prices = pd.concat(selected_prices, axis=1, join='inner')
        return all_prices.loc[start:end,:]


    def delete(self, tickers: List[str] = []):
        """Delete the tickers from the asset universe

        Parameters
        ----------
        tickers : List[str], optional
            Tickers to delete, by default []
        """
        if not isinstance(tickers, list):
            raise TypeError('Provide a list of tickers to delete.')
        for ticker in tickers:
            if ticker == self.cashasset.ticker:
                raise ValueError(f'Cannot delete cash asset: {ticker}')
            if ticker == self.borrowrate.ticker:
                raise ValueError(f'Cannot delete borrow rate: {ticker}')
            try:
                self.assets.pop(ticker)
            except KeyError as exc:
                raise KeyError(f'Could not delete requested ticker {ticker} because it doesn\'t exist in the asset universe')


    def plot_prices(self, tickers: List[str] = [], start: datetime.date = None, end: datetime.date = None, normalize=True) -> None:
        """Plot asset prices over time

        Parameters
        ----------
        tickers : List[str]
            Tickers to include, by default all tickers in the AssetUniverse except the borrow rate
        start : datetime.date, optional
            Start date, by default the start date of the AssetUniverse
        end : datetime.date, optional
            End date, by default the end date of the AssetUniverse
        normalize : bool, optional
            Normalize start prices to $1, by default True
        """
        if len(tickers) == 0:
            tickers = self.tickers(include_borrow_rate=False)
        prices = self.prices(tickers, start, end, normalize)
        renames = {}
        for ticker in prices.columns:
            asset = self.assets.get(ticker, None)
            if ticker == self.cashasset.ticker:
                asset = self.cashasset
            if ticker == self.borrowrate.ticker:
                asset = self.borrowrate
            if asset:
                if asset.display_name:
                    renames[ticker] = asset.display_name
        if len(renames):
            prices = prices.rename(columns=renames)
        prices['Date'] = prices.index
        fig = px.line(prices, x="Date", y=prices.columns)
        # fig = px.line(prices, x="Date", y=prices.columns,
        #       hover_data={"Date": "|%B %d, %Y"})
        fig.update_yaxes(
            type='log'
        )
        # update y axis name
        fig.update_yaxes(title_text='Price (Normalized)' if normalize else 'Price')
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


    def correlation_matrix(self, tickers: List[str] = [], start: datetime.date = None, end: datetime.date = None) -> np.ndarray:
        """Calculate the correlation matrix

        Parameters
        ----------
        tickers : List[str]
            Tickers to include, by default all non-cash assets
        start : datetime.date, optional
            Start date, by default the start date of the AssetUniverse
        end : datetime.date, optional
            End date, by default the end date of the AssetUniverse

        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        if len(tickers) == 0:
            tickers = self.tickers(include_cash=False, include_borrow_rate=False)
        returns = self.returns(tickers, start, end)
        return returns.corr(method='pearson')


    def covariance_matrix(self, tickers: List[str] = [], start: datetime.date = None, end: datetime.date = None) -> np.ndarray:
        """Calculate the correlation matrix

        Parameters
        ----------
        tickers : List[str]
            Tickers to include, by default all non-cash assets
        start : datetime.date, optional
            Start date, by default the start date of the AssetUniverse
        end : datetime.date, optional
            End date, by default the end date of the AssetUniverse

        Returns
        -------
        np.ndarray
            Covariance matrix
        """
        if len(tickers) == 0:
            tickers = self.tickers(include_cash=False, include_borrow_rate=False)
        returns = self.returns(tickers, start, end)
        return returns.cov()
