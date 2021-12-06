import datetime

import pytest

import assetuniverse.assetuniverse
# from assetuniverse import AssetUniverse
# from assetuniverse import Asset

def test_basic():
    print(dir(assetuniverse))
    days = 365
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    tickers = ['AAPL', 'CL=F', 'EURUSD=X']
    assets = [assetuniverse.assetuniverse.Asset(start, end, ticker) for ticker in tickers]

    AU = assetuniverse.assetuniverse.AssetUniverse(start, end, assets)

    for au_ticker, ticker in zip(AU.assets.keys(), tickers):
        assert au_ticker == ticker, f'Asset {au_ticker} did not have correct ticker {ticker}'

    for i in range(1, len(tickers)):
        prices = AU.prices(tickers[:i])
        assert prices.shape[0] > 100, f'Prices for assets {tickers[:i]} did not download correctly.'
        assert prices.shape[1] == i,  f'Prices for asset {tickers[:i]} did not join correctly'
        returns = AU.prices(tickers[:i])
        assert returns.shape[0] > 100, f'Returns for asset {tickers[:i]} did not download correctly.'
        assert returns.shape[1] == i,  f'Returns for asset {tickers[:i]} did not join correctly'
    
    p = AU.prices(normalize=False)
    assert not (p.iloc[0,:] == 1.0).any(), 'Prices should not start at $1'
    p = AU.prices(normalize=True)
    assert (p.iloc[0,:] == 1.0).all(), 'Normalized prices should start at $1'

