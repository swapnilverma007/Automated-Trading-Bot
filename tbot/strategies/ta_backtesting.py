from binance.client import Client
import pandas as pd
import talib
import vectorbt as vbt
import numpy as np
from .ma import atr_trailing_stop_loss

def get_historical_data(client, symbol, interval, start_date):
    data = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_date)
    
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
        'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    df.set_index('open_time', inplace=True)
    
    float_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    df[float_columns] = df[float_columns].astype(float)
    
    df.drop(columns=['close_time', 'quote_asset_volume', 'number_of_trades', 
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], 
            inplace=True)
    
    return df

def backtest_MA_Strategy(data, atr_period=5, hhv_period=10, atr_multiplier=2.5):
    data = data.copy()
    data['SMA20'] = talib.SMA(data['close'], 20)
    data['SMA50'] = talib.SMA(data['close'], 50)

    ts_data = atr_trailing_stop_loss(data, Atr=atr_period, Hhv=hhv_period, Mult=atr_multiplier)
    data['TS'] = ts_data['TS']
    data.dropna(inplace=True)

    if len(data) < 50:
        raise ValueError("Not enough data points after calculating indicators. Try a longer date range.")

    long_entries = (data['SMA20'] > data['SMA50']) & (data['close'] > data['TS']) & \
                   (data['close'].shift(1) > data['TS'].shift(1)) & \
                   (data['close'].shift(2) > data['TS'].shift(2)) & \
                   (data['close'].shift(3) > data['TS'].shift(3))
                   
    long_exits = (data['close'] < data['TS']) & (data['close'].shift(1) < data['TS'].shift(1)) & \
                 (data['close'].shift(2) < data['TS'].shift(2))

    portfolio = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=long_entries,
        exits=long_exits,
        init_cash=1000,
    )

    # print("Portfolio stats:", portfolio.stats())
    stats_dict = portfolio.stats()
    stats = stats_dict.to_dict()  # Convert Series to dictionary for safer access
    
    metrics = {
        'start_value': float(stats.get('Start Value', 0.0)),
        'end_value': float(stats.get('End Value', 0.0)),
        'total_return': float(stats.get('Total Return [%]', 0.0)),
        'total_trades': int(stats.get('Total Trades', 0)),
        'win_rate': float(stats.get('Win Rate [%]', 0.0)),
        'best_trade': float(stats.get('Best Trade [%]', 0.0)),
        'worst_trade': float(stats.get('Worst Trade [%]', 0.0)),
        'sharpe_ratio': float(stats.get('Sharpe Ratio', 0.0)),
    }
    
    backtest_data = {
        'dates': data.index.strftime('%Y-%m-%d %H:%M').tolist(),
        'close_prices': data['close'].tolist(),
        'sma20': data['SMA20'].tolist(),
        'sma50': data['SMA50'].tolist(),
        'trailing_stop': data['TS'].tolist(),
        'equity_curve': portfolio.value().tolist(),
        'drawdown_curve': portfolio.drawdown().tolist(),
    }
    
    trades = portfolio.trades
    trade_data = []
    
    if len(trades.records_arr) > 0:
        # Use vectorbt's built-in method to get trade records as a DataFrame
        trades_df = trades.records
        
        for i in range(len(trades_df)):
            trade_record = trades_df.iloc[i]
            
            entry_time = trade_record['entry_time'] if 'entry_time' in trade_record and pd.notnull(trade_record['entry_time']) else None
            exit_time = trade_record['exit_time'] if 'exit_time' in trade_record and pd.notnull(trade_record['exit_time']) else None
            
            trade_data.append({
                'entry_time': entry_time.strftime('%Y-%m-%d %H:%M') if entry_time else '',
                'exit_time': exit_time.strftime('%Y-%m-%d %H:%M') if exit_time else '',
                'entry_price': float(trade_record.get('entry_price', 0.0)) if 'entry_price' in trade_record else 0.0,
                'exit_price': float(trade_record.get('exit_price', 0.0)) if 'exit_price' in trade_record else 0.0,
                'pnl': float(trade_record.get('pnl', 0.0)) if 'pnl' in trade_record else 0.0,
                'return': float(trade_record.get('return', 0.0)) if 'return' in trade_record else 0.0,
                'status': str(trade_record.get('status', 'unknown')),
                'position_size': float(trade_record.get('size', 0.0)) if 'size' in trade_record else 0.0,
                'duration': float(trade_record.get('duration', pd.Timedelta(0)).total_seconds() / 3600) 
                            if 'duration' in trade_record and pd.notnull(trade_record['duration']) else 0.0,
            })
    
    return metrics, backtest_data, trade_data