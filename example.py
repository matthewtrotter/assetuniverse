from assetuniverse import AssetUniverse
from assetuniverse import Asset
import datetime


days = 365
end = datetime.date.today()
start = end - datetime.timedelta(days=days)
cashasset = Asset(start, end, 'VFISX')
tickers = ['AAPL', 'CL=F', 'EURUSD=X']
assets = [Asset(start, end, ticker) for ticker in tickers]

# assets = [
#     Asset(start, end, 'GLD', readable_name='Gold'),
#     Asset(start, end, 'SLV', readable_name='Silver'),
#     Asset(start, end, 'SPY', readable_name='S&P 500'),
#     Asset(start, end, 'TLT', readable_name='30-year Treasury Bonds'),
#     Asset(start, end, 'FXI', readable_name='Chinese Stocks'),
#     Asset(start, end, 'AAPL', readable_name='Apple'),
#     Asset(start, end, 'VHT', readable_name='Healthcare'),
#     Asset(start, end, 'VNQ', readable_name='US Real Estate'),
#     Asset(start, end, 'QQQ', readable_name='Technology'),
#     Asset(start, end, 'XLE', readable_name='Energy Stocks'),
#     # Asset(start, end, 'UUP', readable_name='US Dollar'),
# ]

AU = AssetUniverse(start, end, assets, cashasset=cashasset)
AU.download()
AU.plot_prices()

print(AU.returns())
