import pandas as pd
import time
import cProfile as profile

def date_convert(date, time):
    datetime = date + " " + time
    dates = lambda v: pd.to_datetime(v, format='%m/%d/%Y %H:%M:%S')
    return dates(datetime)

class Tick(object):
    def __init__(self, csvfile):
        headers = ["date", "time", "price", "bid", "ask", "size"]
        self.reader = pd.read_csv(csvfile, chunksize=100000, names=headers, parse_dates={"Datetime" : [0,1]}, date_parser=date_convert)

    def agregate(self, period='1s'):
        start_time = delta_time = time.time()
#        pr = profile.Profile()
#        pr.enable()
        ohlcv = None
        for data in self.reader:
            delta_time = time.time()
            data.set_index(data['Datetime'], inplace=True)
            data = data.drop('Datetime', axis=1)
            secs_open = data['price'].resample(period).first()
            secs_close = data['price'].resample(period).last()
            secs_low = data['price'].resample(period).min()
            secs_high = data['price'].resample(period).max()
            secs_volume = data['size'].resample(period).sum()
            delta_time = time.time()
            print("agregate next {} rows from {}".format(data.__len__(), data.iloc[0].name))
            df = pd.DataFrame(index=secs_low.index)
            df['Open'] = secs_open.values
            df['High'] = secs_high.values
            df['Low'] = secs_low.values
            df['Close'] = secs_close.values
            df['Volume'] = secs_volume.values
            df = df.dropna()
            df['Open'] = df['Open'].round(4)
            df['High'] = df['High'].round(4)
            df['Low'] = df['Low'].round(4)
            df['Close'] = df['Close'].round(4)
            if ohlcv is None:
                ohlcv = df
            else:
                if df.iloc[0].name == ohlcv.iloc[-1].name:
                    if df.iloc[0]["High"] > ohlcv.iloc[-1]["High"]:
                        ohlcv.loc[ohlcv.iloc[-1].name, "High"] = df.iloc[0]["High"]
                    if df.iloc[0]["Low"] < ohlcv.iloc[-1]["Low"]:
                        ohlcv.loc[ohlcv.iloc[-1].name, "Low"] = df.iloc[0]["Low"]
                    ohlcv.loc[ohlcv.iloc[-1].name, "Close"] = df.iloc[0]["Close"]
                    ohlcv.loc[ohlcv.iloc[-1].name, "Volume"] = ohlcv.iloc[-1]["Volume"] + df.iloc[0]["Volume"]
                    df = df.drop(df.index[0])
                ohlcv = ohlcv.append(df)
#        pr.disable()
#        pr.dump_stats('profile.pstat')
        print("agregate --- %s seconds ---" % (time.time() - start_time))
        return ohlcv