from django.db import models
from django.core.mail import send_mail
from django.conf import settings

# Create your models here.
class NewsItem(models.Model):
    title = models.CharField(max_length=255)
    published_at = models.DateTimeField()
    currencies = models.CharField(max_length=255)
    sentiment = models.CharField(max_length=50)
    score = models.FloatField()

    def __str__(self):
        return self.title
    
class WishlistItem(models.Model):
    symbol = models.CharField(max_length=20)
    strategy = models.CharField(max_length=50)
    use_sentiment = models.BooleanField(default=False)
    active = models.BooleanField(default=True)
    in_position = models.BooleanField(default=False)
    position_side = models.CharField(max_length=10, null=True, blank=True)  # "LONG" or "SHORT" or None
    entry_price = models.FloatField(null=True, blank=True)
    entry_time = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    trade_amount = models.FloatField(default=0.0)  # Amount in USDT to use for each trade
    leverage = models.IntegerField(default=1)      # Leverage to use for trades

    def __str__(self):
        return f"{self.symbol} - {self.strategy}"
    
class PriceAlert(models.Model):
    CONDITION_CHOICES = [
        ('above', 'Price above'),
        ('below', 'Price below'),
    ]
    
    symbol = models.CharField(max_length=20)
    target_price = models.FloatField()
    condition = models.CharField(max_length=10, choices=CONDITION_CHOICES)
    email = models.EmailField(default=settings.DEFAULT_FROM_EMAIL)
    active = models.BooleanField(default=True)
    triggered = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.symbol} {self.condition} {self.target_price}"
    
    def check_and_notify(self, current_price):
        """Check if alert should be triggered and send email if needed"""
        if not self.active or self.triggered:
            return False
            
        triggered = False
        if self.condition == 'above' and current_price > self.target_price:
            triggered = True
        elif self.condition == 'below' and current_price < self.target_price:
            triggered = True
        
        if triggered:
            self.send_alert_email(current_price)
            self.triggered = True
            self.active = False
            self.save()
            return True
            
        return False
    
    def send_alert_email(self, current_price):
        """Send email notification for triggered alert"""
        subject = f'Price Alert Triggered: {self.symbol}'
        message = f"""
        Your price alert has been triggered!
        
        Coin: {self.symbol}
        Alert Condition: Price {self.get_condition_display()} {self.target_price}
        Current Price: {current_price}
        
        This is an automated message from your TradingBot.
        """
        
        notification_email = settings.PRICE_ALERT_EMAIL
        
        try:
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                [notification_email],
                fail_silently=False,
            )
            return True
        except Exception as e:
            print(f"Failed to send email alert: {str(e)}")
            return False