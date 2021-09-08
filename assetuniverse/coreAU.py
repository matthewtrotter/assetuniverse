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
import pandas_datareader.data as web
import plotly.express as px
from typing import List
import yfinance as yf

from assetuniverse.ContractSamples import AssetUniverseContract
from assetuniverse.twsapi import TwsDownloadApp

class AssetUniverse:

    def __init__(self, start, end, symbols: List[AssetUniverseContract], indices=None, offline=False, borrow_spread=0.75):
        self.start = start
        self.end = end
        self.sym = symbols
        self.indices = indices
        self.offline = offline
        self.cashsym = AssetUniverseContract(
            symbol='VFISX',
            data_source='Yahoo Finance'
        )
        self.ratesym = AssetUniverseContract(
            symbol='Fed Funds Rate',
            data_source='FRED'
        )
        self.borrow_spread = borrow_spread      # Percentage points above Fed Funds Rate

        self.freddic = {'5-Year Breakeven Inflation Rate': 'T5YIE',
                        '10-Year Breakeven Inflation Rate': 'T10YIE',
                        'Fed Funds Rate': 'DFF',
                        '3-Month Treasury Constant Maturity Rate': 'GS3M',
                        '1-Year Treasury Constant Maturity Rate': 'DGS1'}        
        self.downloadData()


    def downloadData(self):
        #download total return price series from yahoo, FRED, and Quandl and convert to returns.
        print('Downloading asset universe data... ', flush=True)

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


    def downloadFromTws(self, sym: AssetUniverseContract):
        """Download historical daily closing prices from Interactive Brokers TWS API
        """
        twssym = [s for s in sym if s.data_source == 'TWS']
        num_weeks = np.ceil((self.end - self.start).days/7) + 1
        start = datetime.datetime.combine(self.start, datetime.datetime.min.time())
        end = datetime.datetime.combine(self.end, datetime.datetime.min.time())
        tws = TwsDownloadApp(twssym, start, end, False, f'{num_weeks} W', '1 day', 'TRADES')
        tws.connect("127.0.0.1", 7496, clientId=0)
        tws.run()
        return tws.closes.dropna()

    
    def downloadFromYahoo(self, sym):
        # Only download assets that are not in FRED or Quandl dictionaries
        yahooFinanceSymbols = [s.symbol for s in sym if s.data_source == 'Yahoo Finance']
        yahooFinanceSymbols = [s for s in yahooFinanceSymbols if s not in self.freddic.keys()]
        closes = DataFrame()
        if len(yahooFinanceSymbols):
            data = yf.download(yahooFinanceSymbols, interval="1d", auto_adjust=True, prepost=False, threads=True,
                               start=self.start, end=self.end)
            if len(yahooFinanceSymbols) > 1:
                closes = DataFrame(data["Close"][yahooFinanceSymbols])
            else:
                closes = DataFrame(data["Close"])
                closes = closes.rename(columns={"Close": yahooFinanceSymbols[0]})

        return closes


    def downloadFromFred(self, sym):
        # Only download assets that are in the FRED dictionary
        fredSymbols = [s.symbol for s in sym if s.data_source == 'FRED']
        closes = DataFrame()
        for symbol in fredSymbols:
            closes = web.DataReader(self.freddic[symbol], 'fred', self.start, self.end+datetime.timedelta(1))
            self.__rates = True
        return closes


    def generateOfflineData(self, sym):
        # Generate random prices
        diff = self.end - self.start
        offlineSym = list()
        offlineSym = [self.freddic.get(s.symbol, s.symbol) for s in sym]
        r = np.exp(np.random.randn(diff.days + 1, len(offlineSym))/100) + 0.00001
        #closes = DataFrame({'Date': date_range(self.start, self.end, freq='D'),
        #                    self.sym: np.cumprod(r, axis=0)})
        closes = DataFrame(data=np.cumprod(r, axis=0), columns=offlineSym)
        closes["Date"] = date_range(self.start, self.end, freq='D')
        closes = closes.set_index('Date')
        return closes


    def deleteassets(self, assets):
        # Delete the price and return data for the symbols specified
        for x in assets:
            self.p = self.p.drop(x, 1)
            self.r = self.r.drop(x, 1)
            self.sym.remove(x)


    def plotprices(self):
        """Plot asset prices over time.
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


    def getindex(self, index_name):
        indices = [index for index in self.indices if index.name == index_name]
        return indices


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


    def correlation_matrix(self, symbols=[], rand_drop_percent:float=0):
        """Calculate the correlation matrix
        """
        num_keep = int((1-rand_drop_percent)*self.r.shape[0])
        keep_indices = np.random.choice(self.r.index, num_keep, replace=False)
        if len(symbols):
            return self.r[symbols].loc[keep_indices].corr(method='pearson').values
        else:
            return self.r.loc[keep_indices].corr(method='pearson').values


    def covariance_matrix(self, symbols=[], rand_drop_percent:float=0):
        """Calculate the covariance matrix
        """
        num_keep = int((1-rand_drop_percent)*self.r.shape[0])
        keep_indices = np.random.choice(self.r.index, num_keep, replace=False)
        if len(symbols):
            return self.r[symbols].loc[keep_indices].cov().values
        else:
            return self.r.loc[keep_indices].cov().values


    def correlation_histogram(self, sym1:str, sym2:str, num_trials=1000):
        """Calculate the histogram of the correlation coefficient between two symbols.
        The algorithm randomly drops 10% of the dates on each iteration.
        """
        num_keep = int(0.9*self.r.shape[0])
        correlations = np.zeros(num_trials)
        for i in range(num_trials):
            keep_indices = np.random.choice(self.r.index, num_keep, replace=False)
            correlations[i] = self.r[[sym1, sym2]].loc[keep_indices].corr(method='pearson').values[0,1]
        fig = px.histogram(
            correlations, 
            histnorm='probability density', 
            nbins=20,
            range_x=[-1, 1]
        )
        fig.layout.xaxis.title = f'{sym1} / {sym2}'
        fig.layout.title = 'Correlation Coefficient Histogram'
        fig.show()


def _get_test_contracts() -> List[AssetUniverseContract]:
    contracts = []
    contract = AssetUniverseContract(
        secType = 'FUT',
        currency = 'USD',
        exchange = 'GLOBEX',
        localSymbol = 'ESU1',   # "Local Name" on the IB details page
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
        secType = 'STK',
        currency = 'USD',
        exchange = 'SMART',
        data_source = 'TWS'
    )
    contracts.append(contract)

    contract = AssetUniverseContract(
        symbol='AAPL',
        secType = 'STK',
        currency = 'USD',
        exchange = 'SMART',
        data_source = 'Yahoo Finance'
    )
    contracts.append(contract)

    contract = AssetUniverseContract(
        symbol='BRK B',
        secType = 'STK',
        currency = 'USD',
        exchange = 'SMART',
        data_source = 'TWS'
    )
    contracts.append(contract)

    return contracts


def _get_bond_futures_contracts() -> List[AssetUniverseContract]:
    contracts = []

    contract = AssetUniverseContract(
        secType = 'FUT',
        currency = 'USD',
        exchange = 'ECBOT',
        localSymbol = 'UB   SEP 21',   # "Local Name" on the IB details page
        data_source = 'TWS'
    )
    contracts.append(contract)

    contract = AssetUniverseContract(
        secType = 'FUT',
        currency = 'USD',
        exchange = 'ECBOT',
        localSymbol = 'TN   SEP 21',   # "Local Name" on the IB details page
        data_source = 'TWS'
    )
    contracts.append(contract)

    # contract = AssetUniverseContract(
    #     secType = 'FUT',
    #     currency = 'CAD',
    #     exchange = 'CDE',
    #     localSymbol = 'CGBU21',   # "Local Name" on the IB details page
    #     data_source = 'TWS'
    # )
    # contracts.append(contract)

    # contract = AssetUniverseContract(
    #     secType = 'FUT',
    #     currency = 'CAD',
    #     exchange = 'CDE',
    #     localSymbol = 'CGFU21',   # "Local Name" on the IB details page
    #     data_source = 'TWS'
    # )
    # contracts.append(contract)

    # contract = AssetUniverseContract(
    #     secType = 'FUT',
    #     currency = 'MXN',
    #     exchange = 'MEXDER',
    #     localSymbol = 'DVM10 SP21',   # "Local Name" on the IB details page
    #     data_source = 'TWS'
    # )
    # contracts.append(contract)

    # contract = AssetUniverseContract(
    #     secType = 'FUT',
    #     currency = 'KRW',
    #     exchange = 'KSE',
    #     localSymbol = '1671U',   # "Local Name" on the IB details page
    #     data_source = 'TWS'
    # )
    # contracts.append(contract)

    contract = AssetUniverseContract(
        secType = 'FUT',
        currency = 'EUR',
        exchange = 'DTB',
        localSymbol = 'FBTP SEP 21',   # "Local Name" on the IB details page
        data_source = 'TWS'
    )
    contracts.append(contract)

    contract = AssetUniverseContract(
        secType = 'FUT',
        currency = 'EUR',
        exchange = 'DTB',
        localSymbol = 'FGBX SEP 21',   # "Local Name" on the IB details page
        data_source = 'TWS'
    )
    contracts.append(contract)

    contract = AssetUniverseContract(
        secType = 'FUT',
        currency = 'EUR',
        exchange = 'DTB',
        localSymbol = 'FGBL SEP 21',   # "Local Name" on the IB details page
        data_source = 'TWS'
    )
    contracts.append(contract)

    contract = AssetUniverseContract(
        secType = 'FUT',
        currency = 'CHF',
        exchange = 'SOFFEX',
        localSymbol = 'CONF SEP 21',   # "Local Name" on the IB details page
        data_source = 'TWS'
    )
    contracts.append(contract)

    return contracts


def parse_to_contracts(assets: pd.DataFrame):
    """Parse the symbols in the dataframe into assetuniverse contracts

    Parameters
    ----------
    assets : pd.DataFrame
        import from excel with assets
    """
    contracts = list()
    for _, asset in assets.iterrows():
        au_contract = AssetUniverseContract(
            symbol=asset['symbol'],
            localSymbol=None,#asset['localSymbol'],
            secType=asset['secType'],
            currency=asset['currency'],
            exchange=asset['exchange'],
            data_source=asset['data_source']
        )
        contracts.append(au_contract)
    return contracts


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




