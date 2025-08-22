from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('automate/', views.automate, name='automate'),
    path('place_trade/', views.place_trade, name='place_trade'),
    path('stop_trade/<str:trade_id>/', views.stop_trade, name='stop_trade'),
    path('__debug__/', include('debug_toolbar.urls')),
    path('fetch_market_price/', views.fetch_market_price, name='fetch_market_price'),
    path('chart_view/',views.chart_view,name='chart_view'),
    path('news_view/',views.news_view,name='news_view'),
    path('fetch_new_news/', views.fetch_new_news, name='fetch_new_news'),
    path('analyze_user_input/',views.analyze_user_input,name='analyze_user_input'),
    path('strategies/',views.strategies_view,name='strategies_view'),
    path('strategies/ma_backtest/', views.ma_backtest, name='ma_backtest'),
    path('strategies/lstm_predict/', views.lstm_predict, name='lstm_predict'),
    path('run_lstm_prediction/', views.run_lstm_prediction, name='run_lstm_prediction'),
    path('get_market_sentiment/', views.get_market_sentiment, name='get_market_sentiment'),
    path('wishlist/', views.wishlist_view, name='wishlist'),
    path('wishlist/add/', views.add_wishlist_item, name='add_wishlist_item'),
    path('wishlist/remove/<int:item_id>/', views.remove_wishlist_item, name='remove_wishlist_item'),
    path('wishlist/check_signals/', views.check_signals, name='check_signals'),
    path('wishlist/execute_trade/', views.execute_trade, name='execute_trade'),
    path('alerts/', views.alerts_view, name='alerts'),
    path('alerts/add/', views.add_price_alert, name='add_price_alert'),
    path('alerts/delete/<int:alert_id>/', views.delete_price_alert, name='delete_price_alert'),
]