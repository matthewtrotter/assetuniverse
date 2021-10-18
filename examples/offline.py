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
        au_contract = AssetUniverseContract(
            symbol=asset['symbol'],
            localSymbol=None,#asset['localSymbol'],
            secType=asset['secType'],
            currency=asset['currency'],
            exchange=asset['exchange'],
            data_source=asset['data_source']
        )
        contracts.append(au_contract)
    return contracts


assets = pd.read_excel('examples/assets.xlsx')
assetslist = parse_to_contracts(assets)
end = datetime.date.today()
start = end - datetime.timedelta(days=365)
au = AssetUniverse(start, end, assetslist, offline=True)

au.plotprices()

