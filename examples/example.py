import datetime
import pandas as pd
from assetuniverse import AssetUniverseContract, AssetUniverse

def parse_to_contracts(assets: pd.DataFrame):
    """Parse the symbols in the dataframe into assetuniverse contracts

    Parameters
    ----------
    assets : pd.DataFrame
        import from excel with assets
    """
    contracts = list()
    for _, asset in assets.iterrows():
        if asset['secType'] == 'FUT':
            localSymbol = asset['localSymbol']
        elif asset['secType'] in ['STK', 'IND', 'CONTFUT']:
            localSymbol = None
        au_contract = AssetUniverseContract(
            symbol=asset['symbol'],
            localSymbol=localSymbol,
            secType=asset['secType'],
            currency=asset['currency'],
            exchange=asset['exchange'],
            data_source=asset['data_source']
        )
        contracts.append(au_contract)
        print(au_contract)
    return contracts


assets = pd.read_excel('examples/assets.xlsx')
assetslist = parse_to_contracts(assets)
end = datetime.date.today()
start = end - datetime.timedelta(days=365)
au = AssetUniverse(start, end, assetslist)
au.plotprices()