from ib_insync import IB, Index
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # adjust host/port/clientId

# SPX is an index, not a stock — must use Index contract
contract = Index('SPX', 'CBOE', 'USD')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='20 Y',
    barSizeSetting='1 day',
    whatToShow='TRADES',   # try 'MIDPOINT' if this errors out
    useRTH=True,
    formatDate=1
)

df = pd.DataFrame(bars)[['date', 'close']]
df.to_csv('spx_history.csv', index=False)
print(df.head(), df.tail())

ib.disconnect()
