import torch
from torch.utils.data import Dataset
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Financial_Dataset(Dataset):
    def __init__(self, api_key, symbol, interval='5min', sequence_length=10):
        self.api_key = api_key
        self.symbol = symbol
        self.interval = interval
        self.sequence_length = sequence_length

        # Pull data from AlphaVantage
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.symbol}&interval={self.interval}&apikey={self.api_key}'
        response = requests.get(url)
        data = response.json()

        # Extract time series data
        time_series_data = data.get('Time Series (5min)', {})

        # Convert data to a numpy array
        stock_data = []
        for date, stock_info in time_series_data.items():
            stock_data.append([
                float(stock_info['1. open']),
                float(stock_info['2. high']),
                float(stock_info['3. low']),
                float(stock_info['4. close']),
                float(stock_info['5. volume'])
            ])
        
        # Reverse the order to have chronological order
        stock_data = np.array(stock_data)[::-1]

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.normalized_data = scaler.fit_transform(stock_data)

        # Create sequences
        self.dataX, self.dataY = self.create_sequences(self.normalized_data, self.sequence_length)

    def create_sequences(self, data, sequence_length):
        dataX, dataY = [], []
        for i in range(len(data) - sequence_length):
            seq_in = data[i:(i + sequence_length)]
            seq_out = data[i + sequence_length, -2]  # Assuming 'close' price is the target
            dataX.append(seq_in)
            dataY.append(seq_out)
        return np.array(dataX), np.array(dataY)

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        return torch.tensor(self.dataX[idx], dtype=torch.float), torch.tensor(self.dataY[idx], dtype=torch.float)
