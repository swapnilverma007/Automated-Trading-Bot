from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.mail import send_mail
from binance.client import Client
from binance.enums import *
import json
import os
from .binance_data import get_binance_data
from .utils import fetch_and_save_news, analyze_sentiment
from .models import NewsItem, WishlistItem, PriceAlert
import threading
import pandas as pd
import talib
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from .custom_indicators import atr_trailing_stop_loss2
from .strategies.ta_backtesting import get_historical_data, backtest_MA_Strategy
import traceback
import numpy as np
import time
from datetime import timedelta, datetime
from django.utils import timezone
from .utils import calculate_market_sentiment
import tensorflow as tf
import joblib
from .strategies.ma import atr_trailing_stop_loss
from django.conf import settings

def sync_binance_time(client_instance):
    try:
        server_time = client_instance.get_server_time()
        timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)
        client_instance.timestamp_offset = timestamp_offset
        print(f"Time synced with Binance. Offset: {timestamp_offset}ms")
        return True
    except Exception as e:
        print(f"Failed to sync time: {str(e)}")
        return False

API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET')

client = Client(API_KEY, API_SECRET, testnet=True)
client2 = Client(API_KEY, API_SECRET)

sync_binance_time(client)
sync_binance_time(client2)

# Create your views here.
def index(request):
    return render(request, 'index.html')

def automate(request):
    # Sync time with Binance server
    sync_binance_time(client)
    
    try:
        all_tickers = client.get_all_tickers()
        usdt_pairs = [ticker for ticker in all_tickers if ticker['symbol'].endswith('USDT')]
        ongoing_trades = client.futures_get_open_orders(recvWindow=60000)
        context = {
            'cryptocurrencies': usdt_pairs,
            'ongoing_trades': ongoing_trades,
        }
        return render(request, 'automate.html', context)
    except Exception as e:
        print(f"Error in automate view: {str(e)}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        context = {
            'error': f"API Error: {str(e)}"
        }
        return render(request, 'error.html', context)

def place_trade(request):
    if request.method == 'POST':
        sync_binance_time(client)
        symbol = request.POST['crypto']
        order_type = request.POST['order_type']
        price = request.POST.get('price')
        stop_loss = request.POST['stop_loss']
        take_profit = request.POST['take_profit']
        quantity = request.POST['size']
        side = request.POST['trade_direction']
        leverage = request.POST['leverage']
        opp_side='SELL' if side=='BUY' else 'BUY'

        try:
            client.futures_change_leverage(symbol=symbol, leverage=int(leverage))
            if order_type == 'MARKET':
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
            elif order_type == 'LIMIT':
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=price
                )
            if stop_loss:
                client.futures_create_order(
                    symbol=symbol,
                    side=opp_side,
                    type='STOP_MARKET',
                    stopPrice=stop_loss,
                    quantity=quantity
                )
            if take_profit:
                client.futures_create_order(
                    symbol=symbol,
                    side=opp_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=take_profit,
                    quantity=quantity
                )
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': True})
            else:
                return redirect('automate')
        except Exception as e:
            error_message = str(e)
            # Special handling for the notional value error
            if 'APIError(code=-4164): Order\'s notional must be no smaller than' in error_message:
                error_message = "Order's notional must be no smaller than 5. Please increase your order size."
                
            return JsonResponse({'error': error_message})

def stop_trade(request, trade_id):
    if request.method == 'POST':
        try:
            client.futures_cancel_order(orderId=trade_id)
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'error': str(e)})
        
def fetch_market_price(request):
    symbol=request.GET.get('symbol')
    if not symbol:
        return JsonResponse({'error': 'Symbol is required'},status=400)
    try:
        ticker=client.get_symbol_ticker(symbol=symbol)
        return JsonResponse({'price':ticker['price']})
    except Exception as e:
        return JsonResponse({'error':str(e)},status=500)
    
def chart_view(request):
    all_tickers = client.get_all_tickers()
    usdt_pairs = [ticker for ticker in all_tickers if ticker['symbol'].endswith('USDT')]
    symbol = request.GET.get('symbol', 'BTCUSDT')
    interval = request.GET.get('interval', '15m')
    indicator = request.GET.get('indicator', None)
    data = get_binance_data(symbol, interval)
    # print("Columns in DataFrame:", data.columns)
    if 'timestamp' not in data.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")
    
    chart_data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
    sma20_data = []
    sma50_data = []
    rsi_data = []
    atr_ts_data = []
    if indicator == 'SMA':
        data['SMA20'] = talib.SMA(data['close'], timeperiod=20)
        data['SMA50'] = talib.SMA(data['close'], timeperiod=50)
        sma20_data = data[['timestamp', 'SMA20']].dropna().to_dict(orient='records')
        sma50_data = data[['timestamp', 'SMA50']].dropna().to_dict(orient='records')
    elif indicator == 'RSI':
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        rsi_data = data[['timestamp', 'RSI']].dropna().to_dict(orient='records')
    elif indicator == 'ATR_TS':
        ts_data = atr_trailing_stop_loss2(data)
        atr_ts_data = ts_data.dropna().to_dict(orient='records')

    context = {
        'symbol': symbol,
        'interval': interval,
        'indicator': indicator,
        'chart_data': json.dumps(chart_data),
        'sma20_data' : json.dumps(sma20_data),
        'sma50_data' : json.dumps(sma50_data),
        'rsi_data' : json.dumps(rsi_data),
        'atr_ts_data' : json.dumps(atr_ts_data),
        'cryptocurrencies': usdt_pairs
    }
    return render(request, 'chart.html', context)

@ensure_csrf_cookie
def news_view(request):
    selected_coin = request.GET.get('coin', 'ALL')
    
    news_data = NewsItem.objects.all().order_by('-published_at')
    
    if selected_coin != 'ALL':
        news_data = news_data.filter(currencies__contains=selected_coin)
    
    news_data = news_data[:100]

    latest_timestamp = news_data.first().published_at if news_data.exists() else None

    if latest_timestamp:
        api_key = ''
        fetch_and_save_news(api_key, latest_timestamp)

    return render(request, 'news.html', {
        'news_data': news_data,
        'selected_coin': selected_coin
    })

def fetch_new_news(request):
    selected_coin = request.GET.get('coin', 'ALL')
    latest_timestamp = NewsItem.objects.first().published_at if NewsItem.objects.exists() else None
    
    if latest_timestamp:
        api_key = ''
        fetch_and_save_news(api_key, latest_timestamp)

    news_data = NewsItem.objects.all().order_by('-published_at')
    
    # Apply filtering in the response data but return all news
    # The filtering will happen client-side for better UX
    news_list = [
        {
            'title': item.title,
            'published_at': item.published_at.strftime('%Y-%m-%d %H:%M') if item.published_at else '',
            'currencies': item.currencies,
            'sentiment': item.sentiment,
            'score': item.score,
        }
        for item in news_data[:100]
    ]
    
    return JsonResponse({'news_data': news_list})

def analyze_user_input(request):
    if request.method == "POST":
        user_title = request.POST.get('user_title')
        if not user_title:
            return JsonResponse({'error': 'No headline provided'}, status=400)
            
        try:
            sentiment, score = analyze_sentiment(user_title)
            return JsonResponse({'sentiment': sentiment})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def strategies_view(request):
    """View for displaying trading strategy cards"""
    strategies = [
        {
            'id': 'lstm',
            'name': 'LSTM Neural Network',
            'icon': 'brain',
            'description': 'Deep learning strategy using Long Short-Term Memory networks to predict price movements based on historical patterns.',
            'best_for': 'BTC, ETH, Large-cap coins',
            'category': 'AI & Machine Learning',
            'difficulty': 'Advanced',
            'color': 'from-purple-500 to-indigo-600'
        },
        {
            'id': 'ma',
            'name': 'Moving Average Crossover',
            'icon': 'chart-line',
            'description': 'Classic technical strategy that generates signals when shorter-term moving averages cross above or below longer-term moving averages.',
            'best_for': 'All cryptocurrency pairs',
            'category': 'Technical Analysis',
            'difficulty': 'Beginner',
            'color': 'from-blue-500 to-cyan-600'
        }
    ]
    
    return render(request, 'strategies.html', {'strategies': strategies})

def ma_backtest(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol')
            interval = data.get('interval')
            start_date = data.get('start_date')
            
            historical_data = get_historical_data(client2, symbol, interval, start_date)
            
            if historical_data.empty:
                return JsonResponse({'success': False, 'error': 'No data returned from Binance API for the specified parameters.'})
            
            metrics, backtest_data, trade_data = backtest_MA_Strategy(historical_data)
            
            # Check for NaN values that would cause JSON serialization issues
            for key, value in metrics.items():
                if pd.isna(value) or np.isinf(value):
                    metrics[key] = 0.0
            
            return JsonResponse({
                'success': True,
                'metrics': metrics,
                'backtest_data': backtest_data,
                'trade_data': trade_data,
            })
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'success': False, 'error': str(e)})
    
    elif request.method == 'GET':
        all_tickers = client.get_all_tickers()
        usdt_pairs = [ticker for ticker in all_tickers if ticker['symbol'].endswith('USDT')]
        
        context = {
            'cryptocurrencies': usdt_pairs,
        }
        
        return render(request, 'ma_backtest.html', context)

def get_market_sentiment(request):
    coin = request.GET.get('coin', 'ALL')
    
    news_data = NewsItem.objects.all().order_by('-published_at')[:100]
    
    if coin != 'ALL':
        filtered_news = [item for item in news_data if coin in item.currencies]
        
        # If there are fewer than 10 news items for a specific coin
        if len(filtered_news) < 10:
            return JsonResponse({
                'coin': coin,
                'sentiment': None,
                'insufficient_data': True,
                'message': "Insufficient data for analysis. Try checking overall market sentiment or Bitcoin sentiment."
            })
        
        result = calculate_market_sentiment(filtered_news, coin_focus=coin)
    else:
        result = calculate_market_sentiment(news_data, coin_focus=None)
    
    return JsonResponse({
        'coin': coin,
        'sentiment': result['category'],
        'score': round(result.get('score', 0), 2)
    })

def lstm_predict(request):
    all_tickers = client.get_all_tickers()
    usdt_pairs = [ticker for ticker in all_tickers if ticker['symbol'].endswith('USDT')]
    
    available_models = []
    lstm_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tbot', 'models', 'lstm')
    
    os.makedirs(lstm_model_dir, exist_ok=True)
    
    for ticker in usdt_pairs:
        symbol = ticker['symbol']
        coin = symbol.replace('USDT', '').lower()
        model_path = os.path.join(lstm_model_dir, f'{coin}_model.h5')
        
        if os.path.exists(model_path):
            available_models.append(symbol)
    
    context = {
        'cryptocurrencies': usdt_pairs,
        'available_models': available_models,
        'available_models_json': json.dumps(available_models)
    }
    
    return render(request, 'lstm_predict.html', context)

def run_lstm_prediction(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid request method'})
    
    try:
        data = json.loads(request.body)
        symbol = data.get('symbol')
        
        coin = symbol.replace('USDT', '').lower()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'tbot', 'models', 'lstm', f'{coin}_model.h5')
        scaler_path = os.path.join(base_dir, 'tbot', 'models', 'lstm', f'{coin}_scaler.pkl')
        seq_len_path = os.path.join(base_dir, 'tbot', 'models', 'lstm', f'{coin}_sequence_len.pkl')
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(seq_len_path)):
            return JsonResponse({
                'success': False, 
                'error': f'Model for {symbol} not available yet. Please try another cryptocurrency.'
            })
            
        historical_data = get_binance_data(symbol, interval=Client.KLINE_INTERVAL_1HOUR,limit=200)
        
        if historical_data.empty:
            return JsonResponse({'success': False, 'error': 'Failed to fetch historical data'})
        
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        sequence_len = joblib.load(seq_len_path)
        
        df_for_prediction = historical_data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        data_scaled = scaler.transform(df_for_prediction)
        
        X = np.array([data_scaled[-sequence_len:]])
        
        prediction_scaled = model.predict(X)
        
        prediction_copies = np.repeat(prediction_scaled, df_for_prediction.shape[1], axis=1)
        prediction = scaler.inverse_transform(prediction_copies)[0, 3]  # Get the close price
        
        last_timestamp = int(historical_data['timestamp'].iloc[-1])
        current_price = float(historical_data['close'].iloc[-1])
        
        next_time = int(last_timestamp + 3600)
        
        chart_data = []
        for _, row in historical_data.iterrows():
            chart_data.append({
                'timestamp': int(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        return JsonResponse({
            'success': True,
            'current_price': float(current_price),
            'predicted_price': float(prediction),
            'prediction_time': next_time,
            'last_timestamp': last_timestamp,
            'chart_data': chart_data
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({'success': False, 'error': str(e)})
    
def wishlist_view(request):
    """View for displaying the wishlist page with automated trading setups"""

    wishlist_items = WishlistItem.objects.all().order_by('-created_at')
    
    all_tickers = client.get_all_tickers()
    usdt_pairs = [ticker for ticker in all_tickers if ticker['symbol'].endswith('USDT')]
    
    context = {
        'wishlist_items': wishlist_items,
        'cryptocurrencies': usdt_pairs,
    }
    
    return render(request, 'wishlist.html', context)

def add_wishlist_item(request):
    """Add a coin to the wishlist with selected strategy and options"""
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
        strategy = request.POST.get('strategy')
        use_sentiment = request.POST.get('use_sentiment') == 'on'
        
        try:
            trade_amount = float(request.POST.get('trade_amount', 0))
            leverage = int(request.POST.get('leverage', 1))
            
            if trade_amount < 5:
                return JsonResponse({
                    'success': False, 
                    'error': 'Trade amount must be at least 5 USDT'
                })
        except ValueError:
            return JsonResponse({
                'success': False, 
                'error': 'Invalid trade amount or leverage'
            })
        
        if WishlistItem.objects.filter(symbol=symbol).exists():
            return JsonResponse({
                'success': False, 
                'error': 'This coin is already in your wishlist'
            })
        
        WishlistItem.objects.create(
            symbol=symbol,
            strategy=strategy,
            use_sentiment=use_sentiment,
            active=True,
            trade_amount=trade_amount,
            leverage=leverage
        )
        
        return JsonResponse({'success': True})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def remove_wishlist_item(request, item_id):
    """Remove an item from the wishlist"""
    if request.method == 'POST':
        try:
            item = WishlistItem.objects.get(id=item_id)
            
            if item.in_position:
                print(f"Closing position for {item.symbol} before removal")
                
            item.delete()
            return JsonResponse({'success': True})
        except WishlistItem.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Item not found'})
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def check_signals(request):
    """Check for trading signals on all active wishlist items and execute trades"""
    wishlist_items = WishlistItem.objects.filter(active=True)
    
    results = []
    
    for item in wishlist_items:
        result = {
            'id': item.id,
            'symbol': item.symbol,
            'strategy': item.strategy,
            'signals': {},
            'in_position': item.in_position,
            'position_side': item.position_side,
            'action_taken': None,
            'trade_amount': item.trade_amount,
            'order_response': None
        }
        
        if item.strategy == 'MA':
            data = get_binance_data(item.symbol, interval='15m')
            
            data = data.copy()
            data['SMA20'] = talib.SMA(data['close'], 20)
            data['SMA50'] = talib.SMA(data['close'], 50)

            ts_data = atr_trailing_stop_loss(data, Atr=5, Hhv=10, Mult=2.5)
            data['TS'] = ts_data['TS']
            data.dropna(inplace=True)
            
            long_entry = (data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1]) and \
                        (data['close'].iloc[-1] > data['TS'].iloc[-1]) and \
                        (data['close'].iloc[-2] > data['TS'].iloc[-2]) and \
                        (data['close'].iloc[-3] > data['TS'].iloc[-3])
            
            long_exit = (data['close'].iloc[-1] < data['TS'].iloc[-1]) and \
                        (data['close'].iloc[-2] < data['TS'].iloc[-2])
            
            sentiment_valid = True
            sentiment_category = "neutral"

            if item.use_sentiment:
                news_data = NewsItem.objects.all().order_by('-published_at')[:100]
                sentiment_result = calculate_market_sentiment(news_data)
                sentiment_category = sentiment_result['category']
                
                if sentiment_category == 'bearish' and long_entry:
                    sentiment_valid = False
                elif sentiment_category == 'bullish' and long_exit and item.in_position and item.position_side == "LONG":
                    sentiment_valid = False
            
            action_taken = None
            order_response = None
            current_price = float(data['close'].iloc[-1]) if len(data) > 0 else None
            
            if item.trade_amount >= 5:
                try:
                    sync_binance_time(client)
                    
                    if not item.in_position and long_entry and sentiment_valid:
                        action_taken = "LONG_ENTRY"

                        quantity = round_step_size(
                            (item.trade_amount * item.leverage) / current_price,
                            get_quantity_precision(client, item.symbol)
                        )
                        
                        client.futures_change_leverage(symbol=item.symbol, leverage=item.leverage)
                        
                        order = client.futures_create_order(
                            symbol=item.symbol,
                            side='BUY',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        order_response = {
                            "orderId": order.get('orderId'),
                            "quantity": quantity,
                            "price": current_price
                        }
                        
                        item.in_position = True
                        item.position_side = "LONG"
                        item.entry_price = current_price
                        item.entry_time = timezone.now()
                        item.save()

                    elif item.in_position and item.position_side == "LONG" and long_exit and sentiment_valid:
                        action_taken = "LONG_EXIT"
                        
                        quantity = round_step_size(
                            (item.trade_amount * item.leverage) / item.entry_price,
                            get_quantity_precision(client, item.symbol)
                        )
                        
                        order = client.futures_create_order(
                            symbol=item.symbol,
                            side='SELL',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        order_response = {
                            "orderId": order.get('orderId'),
                            "quantity": quantity,
                            "price": current_price
                        }
                        
                        item.in_position = False
                        item.position_side = None
                        item.save()

                except Exception as e:
                    print(f"Error executing trade for {item.symbol}: {str(e)}")
                    order_response = {"error": str(e)}
            
            result['signals'] = {
                'long_entry': bool(long_entry),
                'long_exit': bool(long_exit),
                'sentiment_category': sentiment_category,
                'sentiment_valid': sentiment_valid
            }
            
            result['action_taken'] = action_taken
            result['current_price'] = current_price
            result['order_response'] = order_response
            
        results.append(result)
    
    return JsonResponse({'results': results})

def execute_trade(request):
    """Execute a trade based on a wishlist item signal"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            item_id = data.get('item_id')
            action = data.get('action')
            
            item = WishlistItem.objects.get(id=item_id)
            
            ticker = client.get_symbol_ticker(symbol=item.symbol)
            current_price = float(ticker['price'])
            
            if action == 'LONG_ENTRY':
                # Place a market buy order (simplified)
                item.in_position = True
                item.position_side = "LONG"
                item.entry_price = current_price
                item.entry_time = timezone.now()
                item.save()
                
                return JsonResponse({
                    'success': True,
                    'message': f'Successfully entered LONG position for {item.symbol} at {current_price}'
                })
            
            elif action == 'LONG_EXIT':
                # Close the position (simplified)
                item.in_position = False
                item.position_side = None
                item.save()
                
                return JsonResponse({
                    'success': True,
                    'message': f'Successfully exited LONG position for {item.symbol} at {current_price}'
                })
                
            return JsonResponse({
                'success': False,
                'error': 'Invalid action specified'
            })
                
        except WishlistItem.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Wishlist item not found'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })

def get_quantity_precision(client, symbol):
    """Get the quantity precision for a symbol"""
    info = client.futures_exchange_info()
    for x in info['symbols']:
        if x['symbol'] == symbol:
            for f in x['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    precision = 0
                    while step_size < 1:
                        step_size *= 10
                        precision += 1
                    return precision
    return 3

def round_step_size(quantity, precision):
    """Round quantity to valid step size based on precision"""
    return round(quantity, precision)

def alerts_view(request):
    """View for displaying and managing price alerts"""
    
    alerts = PriceAlert.objects.filter(active=True).order_by('-created_at')
    
    all_tickers = client.get_all_tickers()
    usdt_pairs = [ticker for ticker in all_tickers if ticker['symbol'].endswith('USDT')]
    
    context = {
        'alerts': alerts,
        'cryptocurrencies': usdt_pairs,
    }
    
    return render(request, 'alerts.html', context)

@csrf_exempt
def add_price_alert(request):
    """Add a new price alert"""
    if request.method == 'POST':
        try:
            symbol = request.POST.get('symbol')
            target_price = float(request.POST.get('target_price'))
            condition = request.POST.get('condition')
            
            email = settings.PRICE_ALERT_EMAIL
            
            if not all([symbol, target_price, condition, email]):
                return JsonResponse({
                    'success': False, 
                    'error': 'All fields are required'
                })
                
            if target_price <= 0:
                return JsonResponse({
                    'success': False, 
                    'error': 'Target price must be greater than 0'
                })
                
            PriceAlert.objects.create(
                symbol=symbol,
                target_price=target_price,
                condition=condition,
                email=email,
                active=True
            )

            return JsonResponse({'success': True})
            
        except ValueError:
            return JsonResponse({
                'success': False, 
                'error': 'Invalid price value'
            })
            
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def delete_price_alert(request, alert_id):
    """Delete a price alert"""
    if request.method == 'POST':
        try:
            alert = PriceAlert.objects.get(id=alert_id)
            alert.delete()
            return JsonResponse({'success': True})
        except PriceAlert.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Alert not found'})
            
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def check_price_alerts():
    """Check all active price alerts and send notifications if triggered"""
    active_alerts = PriceAlert.objects.filter(active=True, triggered=False)
    
    if not active_alerts:
        return
    
    symbols = list(set(alert.symbol for alert in active_alerts))
    
    for symbol in symbols:
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            for alert in active_alerts.filter(symbol=symbol):
                alert.check_and_notify(current_price)
                
        except Exception as e:
            print(f"Error checking price alert for {symbol}: {str(e)}")

def start_price_alert_checker():
    def run_checker():
        while True:
            check_price_alerts()
            time.sleep(60)  # Check every 60 seconds
    
    alert_thread = threading.Thread(target=run_checker, daemon=True)
    alert_thread.start()
    return alert_thread

# Start the alert checker thread when Django starts
price_alert_thread = start_price_alert_checker()