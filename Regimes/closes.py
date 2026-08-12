from ib_async import IB, Index
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # adjust host/port/clientId

contract = Index('SPX', 'CBOE', 'USD')
ib.qualifyContracts(contract)

# IBKR limits how much history you can pull per request.
# For daily bars, '20 Y' is usually about the max in one call for indices.
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',        # '' = now
    durationStr='20 Y',    # how far back
    barSizeSetting='1 day',
    whatToShow='TRADES',   # indices often need 'TRADES' or 'MIDPOINT' depending on data sub
    useRTH=True,
    formatDate=1
)

df = pd.DataFrame(bars)[['date', 'close']]
df.to_csv('spx_history.csv', index=False)
print(df.head(), df.tail())

ib.disconnect()