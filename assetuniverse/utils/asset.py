import datetime
from typing import Dict, List
import pandas as pd

VALID_DATA_SOURCES = ['Yahoo Finance', 'FRED', 'Offline', 'Interactive Brokers']
VALID_SEC_TYPES = ['Stock', 'Future']

class Asset:
    def __init__(self, 
        start: datetime.date = datetime.date.today(),
        end: datetime.date = datetime.date.today()-datetime.timedelta(days=180),
        ticker: str = 'AAPL',
        alternate_tickers: List[str] = [],
        exchange: str = 'SMART',
        currency: str = 'USD',
        sectype: str = None,
        readable_name: str = None,
        downloader_definition: Dict = {},
        data_source: str = 'Yahoo Finance'
        ) -> None:

        self.start = start
        self.end = end
        self.ticker = ticker
        self.alternate_tickers = alternate_tickers
        self.exchange = exchange
        self.currency = currency
        if not sectype:
            sectype = 'Stock'
        if sectype not in VALID_SEC_TYPES:
            raise ValueError(f'Secrurity type {sectype} is not valid. Must be one of: {VALID_SEC_TYPES}')
        self.sectype = sectype
        self.readable_name = readable_name or ticker
        self.display_name = f'{ticker} - {self.readable_name}'
        self.downloader_definition = downloader_definition
        if data_source not in VALID_DATA_SOURCES:
            raise ValueError(f'Data source {data_source} is not valid. Must be one of: {VALID_DATA_SOURCES}')
        self.data_source = data_source
        self.prices = pd.DataFrame()
        self.prices_normalized = pd.DataFrame()
        self.returns = pd.DataFrame()

    def assign_prices(self, prices: pd.Series) -> None:
        """Assign the unnormalized daily closing prices. This function will automatically
        cal`cu`late the returns and normalized prices.

        Parameters
        ----------
        prices : pd.Series
            Unnormalized daily closing prices.
        """
        self.prices = prices
        self.prices_normalized = self.prices/self.prices.iloc[0] # Normalize price to start from $1
        self.returns = self.prices.pct_change()
        self.returns = self.returns[1:]  # delete first row - pct_change() returns first row as NaN

    def __str__(self):
        result = f'{self.ticker}, '
        if len(self.alternate_tickers):
            alternate_names = ', '.join([t for t in self.alternate_tickers])
            result = result + f'\t\talternate names: {alternate_names},\t'
        result = result + f'\t\tdisplays as {self.display_name}'
        return result
    
    def __repr__(self):
        return f'Asset({self.start}, {self.end}, \'{self.ticker}\')'