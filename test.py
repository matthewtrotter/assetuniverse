import datetime

import pytest

from assetuniverse import Asset, AssetUniverse


def test_basic():
    days = 365
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    tickers = ['AAPL', 'CL=F', 'EURUSD=X']
    assets = [Asset(start, end, ticker) for ticker in tickers]
    cashasset = Asset(start, end, 'VFISX')

    AU = AssetUniverse(start, end, assets, cashasset)
    AU.download()

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
    with pytest.raises(ValueError, match='not in the asset universe'):
        AU.prices(['ABCDEFG'])
    with pytest.raises(ValueError, match='not in the asset universe'):
        AU.returns(['ABCDEFG'])

    pre_delete_tickers = AU.tickers()
    AU.delete(tickers[:2])
    assert AU.tickers() == pre_delete_tickers[2:], f'Did not delete {tickers[0]} correctly.'
    with pytest.raises(KeyError, match='Could not delete requested ticker*'):
        AU.delete(['does not exist'])


def test_fail():
    assert False, "Test Fails Here!"