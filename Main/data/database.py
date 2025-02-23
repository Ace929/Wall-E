# database.py
import sqlite3

def store_data(df, table_name='market_data'):
    """Stores data in an SQLite database"""
    conn = sqlite3.connect('investment_data.db')
    df.to_sql(table_name, conn, if_exists='replace', index=True)
    conn.close()