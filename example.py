from assetuniverse import AssetUniverse
from assetuniverse import Asset
import datetime


days = 1*365
end = datetime.date.today()
start = end - datetime.timedelta(days=days)
cashasset = Asset(start, end, 'VFISX')
# tickers = ['AAPL', 'MSFT', 'SPY', 'TLT', 'UUP', 'GLD']
# assets = [Asset(start, end, ticker) for ticker in tickers]

assets = [
    Asset(start, end, 'GLD', readable_name='Gold'),
    Asset(start, end, 'USO', readable_name='Oil'),
    Asset(start, end, 'AAPL', readable_name='Apple'),
    Asset(start, end, 'MSFT', readable_name='Microsoft'),
    Asset(start, end, 'SPY', readable_name='S&P 500'),
    Asset(start, end, 'TLT', readable_name='30-year Treasury Bonds'),
    Asset(start, end, 'UUP', readable_name='US Dollar'),
    # Asset(start, end, 'YK', readable_name='Corn', exchange='ECBOT', data_source='Interactive Brokers'),
    # Asset(start, end, 'MCD', readable_name='CAD.USD', exchange='GLOBEX', data_source='Interactive Brokers', ),
]

AU = AssetUniverse(start, end, assets, cashasset=cashasset)
AU.download()
AU.plot_prices()

print(AU.returns())

print(AU.last_dates_downloaded)
