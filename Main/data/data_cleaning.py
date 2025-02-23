# data_cleaning.py
def clean_data(df):
    """Cleans the dataset by handling missing values and outliers"""
    df.dropna(inplace=True)
    df = df[df.apply(lambda x: (x - x.mean()).abs() / x.std() < 3).all(axis=1)]
    return df