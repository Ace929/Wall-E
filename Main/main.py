# main.py
import sys
import os

from Main.data.data_fetch import fetch_market_data
from Main.data.data_cleaning import clean_data
from Main.data.feature_engineering import add_features
from Main.data.database import store_data

def main():
    symbol = 'AAPL'
    data = fetch_market_data(symbol)
    clean_data = clean_data(data)
    final_data = add_features(clean_data)
    store_data(final_data)
    print("Data pipeline completed.")

if __name__ == "__main__":
    main()