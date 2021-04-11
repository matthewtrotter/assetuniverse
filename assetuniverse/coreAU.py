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

import yfinance as yf
from pandas import DataFrame, to_datetime, date_range
import pandas_datareader.data as web
import numpy as np
import datetime
import quandl as q
import plotly.express as px
import copy
from alpha_vantage.timeseries import TimeSeries

class AssetUniverse:

    def __init__(self, start, end, symbols, indices=None, offline=False, cashsym="VFISX", borrow_spread=1.0):
        self.start = start
        self.end = end
        self.sym = symbols
        self.indices = indices
        self.offline = offline
        self.cashsym = [cashsym,]
        self.ratesym = ["Fed Funds Rate",]
        self.borrow_spread = borrow_spread      # Percentage points above Fed Funds Rate

        # Alpha Vantage
        self.ts = TimeSeries(key='WLG07CIPK2R59FQW', output_format='pandas')

        # Quandl
        #self.quandlAPItoken = 'JUxys2zYgMrMMNxGABfZ'
        q.ApiConfig.api_key = "JUxys2zYgMrMMNxGABfZ"

        # import substitutions and symbol dictionaries, then download
        self.importdicts()
        self.downloadData()

    def importdicts(self):
        # Import dictionaries from CSV files
        self.subdict = {'FLBAX': 'TLT',
                        'VDIGX': 'VIG',
                        'FSIVX': 'ACWX',
                        'DFEMX': 'EEM',
                        'VBLTX': 'BLV',
                        'VGSLX': 'VNQ',
                        'FNCMX': 'ONEQ',
                        'BRK-B': 'BRK-A',
                        'INCO': 'INDA',
                        'EWS': 'SGF',
                        'VOO': 'SPY',
                        'Gold': 'GLD'}

        self.quandldic = {'US Corp Bonds': 'ML/TRI',
                          'EM Corp Bonds': 'ML/IGEM',
                          'Eurodollar': 'CHRIS/CME_ED1',
                          '2-Year Note': 'CHRIS/CME_TU1',
                          '10-Year Note': 'CHRIS/CME_TY1',
                          'Oil': 'CHRIS/CME_CL1',
                          'Natural Gas': 'CHRIS/CME_NG1',
                          'Gold': 'LBMA/GOLD',
                          'Silver': 'CHRIS/CME_SI1',
                          'Platinum': 'CHRIS/CME_PL1',
                          'Palladium': 'CHRIS/CME_PA1',
                          'Copper': 'CHRIS/CME_HG1',
                          'Sugar': 'CHRIS/ICE_SB1',
                          'Lumber': 'CHRIS/CME_LB1',
                          'Corn': 'CHRIS/CME_C1',
                          'Cocoa': 'CHRIS/LIFFE_C1',
                          'Soybeans': 'CHRIS/CME_S1',
                          'EURUSD': 'FRED/DEXUSEU',
                          'USDCHF': 'FRED/DEXSZUS',
                          'AUDUSD': 'FRED/DEXUSAL',
                          'USDJPY': 'FRED/DEXJPUS',
                          'USDINR': 'FRED/DEXINUS',
                          'USDCAD': 'FRED/DEXCAUS',
                          'GBPUSD': 'FRED/DEXUSUK',
                          'BTCUSD': 'BITFINEX/BTCUSD',
                          'LTCUSD': 'BITFINEX/LTCUSD',
                          'ETHUSD': 'BITFINEX/ETHUSD',
                          'China Stocks': 'NASDAQOMX/NQCN'}

        self.freddic = {'5-Year Breakeven Inflation Rate': 'T5YIE',
                        '10-Year Breakeven Inflation Rate': 'T10YIE',
                        'Fed Funds Rate': 'DFF',
                        '3-Month Treasury Constant Maturity Rate': 'GS3M',
                        '1-Year Treasury Constant Maturity Rate': 'DGS1'}

    def downloadData(self):
        #download total return price series from yahoo, FRED, and Quandl and convert to returns.
        print('Downloading asset universe data... ', flush=True)

        if self.offline:
            closes = self.generateOfflineData(self.sym)
            closesCash = self.generateOfflineData(self.cashsym)
            annualBorrowRate = self.generateOfflineData(self.ratesym)
        else:
            closes = self.downloadYahooFinanceAssets(self.sym)
            closesCash = self.downloadYahooFinanceAssets(self.cashsym)
            closesQuandl = self.downloadQuandlAssets(self.sym)
            closesFred = self.downloadFredAssets(self.sym)
            annualBorrowRate = self.downloadFredAssets(self.ratesym)

            if closes.size:
                if closesQuandl.size:
                    closes = closes.join(closesQuandl, how="inner")
                if closesFred.size:
                    closes = closes.join(closesFred, how="inner")
            elif closesQuandl.size:
                closes = closesQuandl
                if closesFred.size:
                    closes = closes.join(closesFred, how="inner")
            elif closesFred.size:
                closes = closesFred

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
        # closes = closes.replace(to_replace=0.0, method='ffill')
        closes.dropna(axis=0, how="any", inplace=True)

        # Separate closes from borrow rate
        self.rborrow = closes.loc[:, self.freddic[self.ratesym[0]]] + self.borrow_spread
        closes = closes.drop(self.freddic[self.ratesym[0]], axis="columns")

        # Normalize price to start from $1
        self.originalprices = copy.deepcopy(closes)
        for i in range(0, len(closes.columns)):
            closes.iloc[:,i] = closes.iloc[:,i]/closes.iloc[0,i]

        # Separate cash from assets
        self.p = closes.loc[:, self.sym+self.index_names]
        self.pcash = closes.loc[:, self.cashsym]

        # Calculate daily returns
        self.r = self.p.pct_change()
        self.r = self.r.iloc[1:, :]  # delete first row - pct_change() returns first row as NaN
        self.rcash = self.pcash.pct_change()
        self.rcash = self.rcash.iloc[1:, :]  # delete first row - pct_change() returns first row as NaN

        self.allsym = self.r.columns

        print('Done.', flush=True)

    def downloadYahooFinanceAssets(self, sym):
        # Only download assets that are not in FRED or Quandl dictionaries
        yahooFinanceSymbols = [s for s in sym if (s not in self.quandldic.keys() and s not in self.freddic.keys())]
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

    def downloadQuandlAssets(self, sym):
        # Only download assets that are in the Quandl dictionary
        quandlSymbols = [s for s in sym if s in self.quandldic.keys()]
        closes = DataFrame()
        for symbol in quandlSymbols:
            single = self.downloadQuandlSingle(symbol)
            closes[symbol] = single.iloc[:,0]
        return closes

    def downloadFredAssets(self, sym):
        # Only download assets that are in the FRED dictionary
        fredSymbols = [s for s in sym if s in self.freddic.keys()]
        closes = DataFrame()
        for symbol in fredSymbols:
            closes = web.DataReader(self.freddic[symbol], 'fred', self.start, self.end+datetime.timedelta(1))
            self.__rates = True
        return closes

    def generateOfflineData(self, sym):
        # Generate random prices
        diff = self.end - self.start
        r = np.exp(np.random.randn(diff.days + 1, len(sym))/100) + 0.00001
        #closes = DataFrame({'Date': date_range(self.start, self.end, freq='D'),
        #                    self.sym: np.cumprod(r, axis=0)})
        closes = DataFrame(data=np.cumprod(r, axis=0), columns=sym)
        closes["Date"] = date_range(self.start, self.end, freq='D')
        closes = closes.set_index('Date')
        return closes

    def downloadQuandlSingle(self, symbol):
        # Download this symbol from quandl
        sdate = self.start.strftime('%Y-%m-%d')
        edate = self.end.strftime('%Y-%m-%d')
        closes = q.get(self.quandldic[symbol], start_date=sdate, end_date=edate)

        CHRIScols = ['Open', 'High', 'Low', 'Change', 'Settle', 'Volume', 'Open Interest']
        NASDAQOMXcols = ['Index Value', 'High', 'Low', 'Total Market Value', 'Dividend Market Value']
        MLcols = ['VALUE']

        if np.array_equal(closes.columns, CHRIScols):
            dropcols = ['Open', 'High', 'Low', 'Change', 'Volume', 'Open Interest']
        elif np.array_equal(closes.columns, NASDAQOMXcols):
            dropcols = ['High', 'Low', 'Total Market Value', 'Dividend Market Value']
        else:
            dropcols = closes.columns[1:]
        closes = closes.drop(dropcols, axis=1)
        closes.name = symbol

        return closes

    def downloadalphavantage(self, symbol):
        # Download this symbol from alpha vantage
        closes, metadata = self.ts.get_daily_adjusted(symbol, 'full')
        dropcols = ['3. low', '6. volume', '1. open', '2. high', '8. split coefficient', '4. close', '7. dividend amount']
        closes = closes.drop(dropcols, axis=1)
        closes.columns = [symbol]
        closes.index = to_datetime(closes.index)
        closes = closes[(self.start-datetime.timedelta(1)):self.end]
        retvals = self.checknansandsubstitute(symbol, closes, 3)
        newsym = retvals[0]
        closes = retvals[1]

        delta = datetime.timedelta(round((self.end - self.start).days*0.1))
        if closes.index[0] >= (self.start + delta):
            raise AttributeError

        try:
            closes = self.addtodaysdata(newsym, closes)
        except:
            print('Could not download intraday data')

        return newsym, closes

    def addtodaysdata(self, symbol, closes):
        # Check the last date of the historical data, if it's not today,
        # then download today's data.
        if closes.index[-1].date() != datetime.datetime.today().date():
            # Download today's most recent price
            formattedsymbol = symbol.replace('-', '.')
            x = self.ts.get_batch_stock_quotes([formattedsymbol])
            x = x[0]
            lastprice = DataFrame(x['2. price'], to_datetime(x['4. timestamp']))
            lastprice = lastprice.rename(columns={lastprice.columns[0]: closes.columns[0]}, copy=True)

            # If the last price date is more recent than the historical data,
            # then append it to the history
            if lastprice.index[-1].date() > closes.index[-1].date():
                closes = closes.append(lastprice)

        return closes

    def checknansandsubstitute(self, symbol, closes, n):
        # If data series has more than n NaNs, then substitute with another asset
        numnans = int(closes.isnull().sum())
        if numnans > n and symbol in self.subdict:
            substitutesymbol = self.subdict[symbol]
            retvals = self.downloadSingle(substitutesymbol)
            #temp = retvals[0]
            closessub = retvals[1]
            numsubnans = int(closessub.isnull().sum())
            if numsubnans < numnans:
                symbol = substitutesymbol + ' (subst. for ' + symbol + ')'
                closes = closessub.copy()

        return symbol, closes

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
        fig = px.line(prices, x="Date", y=self.p.columns,
            
              hover_data={"Date": "|%B %d, %Y"})
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


if __name__ == "__main__":
    """end = datetime.datetime.today()
    start = end - datetime.timedelta(days=60)
    symbols = ['RDS-B', 'BRK-B', 'Palladium', 'AAPL', 'GOOG', 'Gold']
    a = AssetUniverse(start, end, symbols)
    a.plotprices()"""

    days = 365.25 * 0.25
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=days)
    #sym = ["VWELX", "DODBX", "Gold"] # Longest history
    sym = ["XOM", "AAPL", "MSFT"]
    AU = AssetUniverse(start, end, sym, offline=False)
    # AU.plotprices()
    # AU.correlation_histogram(sym[0], sym[1])
    print(AU.correlation_matrix())
    print(AU.correlation_matrix(['XOM', 'MSFT']))




