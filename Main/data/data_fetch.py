import requests
import pandas as pd
from config import ALPHA_VANTAGE_API_KEY

def fetch_market_data(symbol, interval='1day'):
    """Fetches historical market data from Alpha Vantage"""
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'compact'
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. adjusted close': 'adj_close',
        '6. volume': 'volume'
    }).astype(float)
    df.index = pd.to_datetime(df.index)
    return df