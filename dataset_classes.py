import torch
from torch.utils.data import Dataset
import requests
import numpy as np

class dataset(Dataset):
    def __init__(self, api_key, symbol, interval):
        self.api_key = api_key
        self.symbol = symbol
        self.interval = interval

        # Pull data from AlphaVantage
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.symbol}&interval={self.interval}&apikey={self.api_key}'
        response = requests.get(url)
        data = response.json()

        # Extract time series data
        time_series_data = data.get('Time Series (5min)', {})

        # Create an empty NumPy array with dtype=object to handle mixed data types
        data_type = [('Date', 'O'), ('Open', 'O'), ('High', 'O'), ('Low', 'O'), ('Close', 'O'), ('Volume', 'O')]
        self.stock_array = np.empty(len(time_series_data), dtype=data_type)

        # Fill the array with data
        for i, (date, stock_data) in enumerate(time_series_data.items()):
            self.stock_array[i] = (date,
                                   stock_data.get('1. open'),
                                   stock_data.get('2. high'),
                                   stock_data.get('3. low'),
                                   stock_data.get('4. close'),
                                   stock_data.get('5. volume'))

    def __len__(self):
        return len(self.stock_array)

    def __getitem__(self, idx):
        return self.stock_array[idx]

