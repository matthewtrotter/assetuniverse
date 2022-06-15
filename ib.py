from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1, readonly=True)

contract = ContFuture(
    symbol='ZF',
    exchange='ECBOT',
    )
# contract = Stock(
#     symbol='AAPL',
#     exchange='SMART',
#     currency='USD'
#     )
# contract = Contract(
#     secType='STK',
#     symbol='XOM',
#     exchange='SMART',
#     currency='USD'
# )
bars = ib.reqHistoricalData(
    contract, 
    endDateTime=None, 
    durationStr='50 Y',
    barSizeSetting='1 day', 
    whatToShow='ADJUSTED_LAST', 
    useRTH=True
    )

# convert to pandas dataframe:
df = util.df(bars)
print(df)
