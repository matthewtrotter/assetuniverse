import datetime
from typing import Dict, List

VALID_DATA_SOURCES = ['Yahoo Finance']

class Asset:
    def __init__(self, 
        start: datetime.date = datetime.date.today(),
        end: datetime.date = datetime.date.today()-datetime.timedelta(days=180),
        ticker: str = 'AAPL',
        alternate_tickers: List[str] = [],
        downloader_definition: Dict = {},
        data_source: str = 'Yahoo Finance'
        ) -> None:
        self.start = start
        self.end = end
        self.ticker = ticker
        self.alternate_tickers = alternate_tickers
        self.downloader_definition = downloader_definition
        if data_source not in VALID_DATA_SOURCES:
            raise ValueError(f'Datasource {data_source} is not valid. Must be one of: {VALID_DATA_SOURCES}')
        self.data_source = data_source
        self.prices
        self.prices_normalized
        self.returns
        self.returns_normalized