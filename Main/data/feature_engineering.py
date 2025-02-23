# feature_engineering.py
def add_features(df):
    """Adds derived features like moving averages and volatility"""
    df['returns'] = df['adj_close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['moving_avg_50'] = df['adj_close'].rolling(window=50).mean()
    return df