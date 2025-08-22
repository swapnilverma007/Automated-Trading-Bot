import pandas as pd
import numpy as np
import talib

def atr_trailing_stop_loss(data, Atr=5, Hhv=10, Mult=2.5):

    df = data.copy()

    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=Atr)

    def ta_highest(source, length):
        return source.rolling(window=length, min_periods=1).max()

    df['Prev'] = ta_highest(df['high'] - Mult * df['ATR'], Hhv)

    def ta_barssince(condition):
        result = np.full(len(condition), np.nan)
        last_true_index = -1
        for i in range(len(condition)):
            if condition.iloc[i]:
                last_true_index = i
            if last_true_index != -1:
                result[i] = i - last_true_index
        return result

    condition = (df['close'] > df['Prev']) & (df['close'] > df['close'].shift(1))
    df['bars_since'] = ta_barssince(condition)

    df['cum_1'] = np.arange(1, len(df) + 1)

    df['highest_1'] = ta_highest(df['high'] - Mult * df['ATR'], Hhv)
    df['highest_2'] = ta_highest(df['high'] - Mult * df['ATR'], Hhv)

    df['iff_1'] = np.where(
        (df['close'] > df['highest_1']) & (df['close'] > df['close'].shift(1)),
        df['highest_2'],
        df['Prev']
    )

    df['TS'] = np.where(df['cum_1'] < 16, df['close'], df['iff_1'])

    return df[['TS']]