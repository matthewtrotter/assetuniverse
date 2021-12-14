import yfinance as yf
import pandas_datareader.data as web

import datetime
from pandas import DataFrame, date_range
import numpy as np

class Downloader:
    def __init__(self, start, end, tickers) -> None:
        self.start = start
        self.end = end
        self.tickers = tickers


class YahooFinanceDownloader(Downloader):
    """Download from Yahoo Finance
    """
    def download(self):
        closes = DataFrame()
        if len(self.tickers) == 0:
            raise ValueError('There are no tickers to download.')
        data = yf.download(self.tickers, interval="1d", auto_adjust=True, prepost=False, threads=True,
                            start=self.start, end=self.end)
        if len(self.tickers) > 1:
            closes = DataFrame(data["Close"][self.tickers])
        else:
            closes = DataFrame(data["Close"])
            closes = closes.rename(columns={"Close": self.tickers[0]})
        return closes


class FredDownloader(Downloader):
    """Download from FRED
    """
    def __init__(self, start, end, tickers) -> None:
        super().__init__(start, end, tickers)
        self.freddic = {'5-Year Breakeven Inflation Rate': 'T5YIE',
                        '10-Year Breakeven Inflation Rate': 'T10YIE',
                        'Fed Funds Rate': 'DFF',
                        '3-Month Treasury Constant Maturity Rate': 'GS3M',
                        '1-Year Treasury Constant Maturity Rate': 'DGS1'}        
        
    def download(self):
        closes = DataFrame()
        for ticker in self.tickers:
            closes = web.DataReader(
                self.freddic.get(ticker, ticker), 
                'fred', 
                self.start, 
                self.end+datetime.timedelta(1)
                )
            closes = closes.rename(columns={closes.columns[0]: ticker})

            # Reindex tickers that aren't updated often
            if ticker in ['Fed Funds Rate',]:
                idx = date_range(start=self.start, end=self.end, freq='D')
                closes = closes.reindex(idx)
                closes.fillna(method="ffill", inplace=True)
        return closes


class OfflineDownloader(Downloader):
    """Generates random prices and returns for situations without internet access
    """
    def download(self):
        diff = self.end - self.start
        offlineSym = list()
        r = np.exp(np.random.randn(diff.days + 1, len(offlineSym))/100) + 0.00001
        closes = DataFrame(data=np.cumprod(r, axis=0), columns=offlineSym)
        closes["Date"] = date_range(self.start, self.end, freq='D')
        closes = closes.set_index('Date')
        return closes
