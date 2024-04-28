import appdirs
import diskcache
from time import sleep, time
import ib_insync
import yfinance as yf
import pandas_datareader.data as web

import datetime
from pandas import DataFrame, date_range, Series
import numpy as np

class Downloader:
    def __init__(self, start, end, tickers) -> None:
        self.start = start
        self.end = end
        self.tickers = tickers
        self.last_dates_downloaded = None

    def _calculate_last_date_downloaded(self, closes: DataFrame) -> None:
        self.last_dates_downloaded = closes.apply(Series.last_valid_index)

class InteractiveBrokersDownloader(Downloader):
    def __init__(self, start, end, tickers, currencies, exchanges) -> None:
        super().__init__(start, end, tickers)
        self.currencies = currencies
        self.exchanges = exchanges
        self.ib = None
    
    def connect(self) -> None:
        """Instantiate connection Trader Workstation
        """
        self.ib = ib_insync.IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1, readonly=True)
        user_data = appdirs.user_cache_dir('assetuniverse', 'interactivebrokersdownloader')
        self.cache = diskcache.Cache(user_data)
        self.cache_validity_seconds = 20*60

    def shutdown(self) -> None:
        """Close connections
        """
        if self.ib:
            self.ib.disconnect()
    
    def download(self):
        closes = DataFrame()
        if self.tickers:
            self.connect()
            for symbol, currency, exchange in zip(self.tickers, self.currencies, self.exchanges):
                # Load from cache if already exists
                key = f'{symbol}{currency}{exchange}'
                cached = self.cache.get(key, None)
                download = True
                data = None
                if cached:
                    last_downloaded = cached.get('last_downloaded', 0)
                    if last_downloaded > time() - self.cache_validity_seconds:
                        delay = time() - last_downloaded
                        print(f'IB TWS: Loaded {symbol} data from cache.\tWas last downloaded {round(delay/60, 1)} minutes ago...')
                        data = cached['data']
                        download = False
                
                # Download if cache did not have recent data
                if download:
                    print(f'IB TWS: Downloading {symbol} in {currency} currency from {exchange} exchange...')
                    data = self._download_cont_future(symbol, currency, exchange)
                    self.cache[key] = {
                        'data': data,
                        'last_downloaded': time()
                    }

                # Join with other closes
                if closes.empty:
                    closes = data
                else:
                    closes = closes.join(data)
            super()._calculate_last_date_downloaded(closes)
        return closes
    
    def _download_cont_future(self, symbol:str, currency:str, exchange:str) -> DataFrame:
        """Download historical data for a continuous futures contract

        Parameters
        ----------
        symbol : str
            Symbol of the continuous futures contract
        currency : str
            Base currency of the contract
        exchange : str
            Exchange to target

        Returns
        -------
        DataFrame
            Historical closing prices. Example:
                    date        open        high         low       close  volume  average  barCount
            0    2019-12-27  116.210938  116.437500  115.886719  116.433594    -1.0     -1.0        -1
            1    2019-12-30  116.074219  116.218750  116.074219  116.175781    -1.0     -1.0        -1
            2    2019-12-31  116.179688  116.207031  116.144531  116.207031    -1.0     -1.0        -1
            3    2020-01-02  116.378906  116.531250  116.378906  116.421875    -1.0     -1.0        -1
        
        Raises
        ------
        ValueError
            _description_
        """
        contract = ib_insync.ContFuture(
            symbol=symbol,
            exchange=exchange,
            currency=currency
            )
        days = (self.end - self.start).days + 1
        duration = f'{days} D'
        if days > 365:
            duration = f'{int(np.ceil(days/365))} Y'
        for i in range(5):
            bars = self.ib.reqHistoricalData(
                contract, 
                endDateTime=None, 
                durationStr=duration,
                barSizeSetting='1 day', 
                whatToShow='ADJUSTED_LAST', 
                useRTH=True
                )
            if bars:
                break
            if i < 4:
                sleep(2)
                print(f'IB TWS: Downloading {symbol} in {currency} currency from {exchange} exchange... try #{i+2}')
        bars = ib_insync.util.df(bars)
        bars = DataFrame(data=bars['close'].values, index=bars['date'], columns=[symbol,])
        return bars
        


class YahooFinanceDownloader(Downloader):
    """Download from Yahoo Finance
    """
    def download(self):
        closes = DataFrame()
        if len(self.tickers) == 0:
            raise ValueError('There are no tickers to download.')
        if self.end == datetime.date.today():
            data = yf.download(self.tickers, interval="1d", auto_adjust=True, prepost=False, threads=True,
                                start=self.start)
        else:
            data = yf.download(self.tickers, interval="1d", auto_adjust=True, prepost=False, threads=True,
                                start=self.start, end=self.end)
        if len(self.tickers) > 1:
            closes = DataFrame(data["Close"][self.tickers])
        else:
            closes = DataFrame(data["Close"])
            closes = closes.rename(columns={"Close": self.tickers[0]})
        super()._calculate_last_date_downloaded(closes)
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
        super()._calculate_last_date_downloaded(closes)
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
        super()._calculate_last_date_downloaded(closes)
        return closes
