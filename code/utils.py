import pandas as pd
import numpy as np
import datetime

def timestamp_to_datetime(timestamp, unit="milli", timezone = None):
    """ convert a exchange timestamp to a datetime object

    :param timestamp: timestamp in unit
    :param unit: str, unit of timestamp, supported values are ["sec", "milli", "micro"]
    :return: datetime.datetime object
    """
    if unit not in ["sec", "milli", "micro"]:
        print("invalid timestamp unit")
        assert False
    if timezone == "UTC": timezone = pytz.UTC
    timestamp = int(timestamp)
    if unit == "sec":
        return datetime.datetime.fromtimestamp(timestamp,tz=timezone)

    if unit == "milli":
        milliseconds = timestamp % 1000
        t = datetime.datetime.fromtimestamp(timestamp//1000,tz=timezone)
        return t + datetime.timedelta(milliseconds=milliseconds)

    if unit == "micro":
        microseconds = timestamp % 1000000
        t = datetime.datetime.fromtimestamp(timestamp//1000000,tz=timezone)
        return t + datetime.timedelta(microseconds=microseconds)


def agg_to_kline(df, interval, offset=0, add_time=False):
    """ converts aggregate trades to kline with specified interval

    :param df: aggregate trade dataframe
    :param interval: int, unit is second
    :param offset: int, unit is second, has to be smaller than interval; if none-zero,
                    offset seconds will be added to all relevant aggregate trades before
                    aggregating, intended to decrease fixed time scale sampling bias
    :param add_time: an additional column of human readable time will be added if set to True
    :return:
    """
    if df.shape[0] == 0:
        return pd.DataFrame()
    if offset < 0 or offset > interval:
        print("offset has to be in [0, interval)")
        assert False
    df = df[["Timestamp", "Price", "months_to_exp"]]
    df = df.sort_values(by=["Timestamp"]).reset_index(drop=True)
    df["group"] = df["Timestamp"] // interval
    res = pd.DataFrame()
    res["OpenTime"] = df["group"] * interval
    res['Num_Trades'] = df["group"].apply(lambda x: df['group'].value_counts().loc[x])
    res["group"] = df["group"]
    res = res.groupby(["group"]).agg({"OpenTime": lambda x: x.iloc[0],
                                      "Num_Trades": lambda x: x.sum(),
                                      "group": lambda x: x.iloc[0]}).reset_index(drop=True)
    cols = ["OpenTime", "Num_Trades","group"]
    if add_time:
        res["Time"] = res["OpenTime"].apply(lambda x: timestamp_to_datetime(x,unit="sec"))
        cols.append("Time")
    return res[cols]


def string_to_date(x):
    try:
        x = datetime.datetime.strptime(x,'%m/%d/%Y')
        return x
    except:
        return x


def string_to_date2(x):
    try:
        x = datetime.datetime.strptime(x[:10],'%Y-%m-%d')
        return x
    except:
        return x
    
def string_to_date3(x):
    try:
        x = datetime.datetime.strptime(x[:10],'%Y-%m-%d')
        return x
    except:
        return x
    
def reshape_into_train(X,y):
    
    word_per_sec = 4
    target_length = 15
    reminder = X.shape[0]%(word_per_sec*target_length)
    X = X[:-reminder,:].reshape((-1,word_per_sec,target_length))
    y = y[target_length::target_length]
    
    return X,y

def reshape_into_train2(X,S,y):
    word_per_sec = 4
    target_length = 15
    reminder = X.shape[0]%(word_per_sec*target_length)
    X = X[:-reminder,:].reshape((-1,word_per_sec,target_length))
    S = S[:-reminder].reshape((-1,800,target_length))
    y = y[target_length::target_length]
    return X,S,y
