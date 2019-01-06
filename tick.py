import math
import pandas as pd
import time

class Tick(object):
    def __init__(self, csvfile):
        headers = ["date", "time", "price", "bid", "ask", "size"]
        self.data = pd.read_csv(csvfile, chunksize=10000, names=headers, parse_dates={"Datetime" : [0,1]})
        self.data.set_index(self.data['Datetime'], inplace=True)
        self.data = self.data.drop('Datetime', axis=1)

    def agregate(self, period='1s'):
        start_time = time.time()
        secs_first = self.data['price'].resample(period).first()
        secs_close = self.data['price'].resample(period).last()
        lastClose = float('nan')
        secs_open = pd.Series(index=secs_close.index, name='price', dtype=float)
        secs_open.fillna(0)
        for index in range(0, secs_close.size):
            if math.isnan(secs_close[index]) is False:
                if math.isnan(lastClose):
                    secs_open.iloc[index] = secs_first[index]
                else:
                    secs_open.iloc[index] = lastClose
                lastClose = secs_close[index]
        secs_low = self.data['price'].resample(period).min()
        secs_high = self.data['price'].resample(period).max()
        secs_volume = self.data['size'].resample(period).sum()

        df = pd.DataFrame(index=secs_low.index)
        df['Open'] = secs_open.values
        df['High'] = secs_high.values
        df['Low'] = secs_low.values
        df['Close'] = secs_close.values
        df['Volume'] = secs_volume.values
        df = df.dropna()
        print("agregate --- %s seconds ---" % (time.time() - start_time))
        return df