import requests
import numpy as np

# Pull data from aplhvantage this in particular is IBM data, can be changed
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=KVWI7K2RP426FT5P'
r = requests.get(url)
data = r.json()

# Extract time series data
time_series_data = data.get('Time Series (5min)', {})

# Create an empty NumPy array with dtype=object to handle mixed data types
dtype = [('Date', 'O'), ('Open', 'O'), ('High', 'O'), ('Low', 'O'), ('Close', 'O'), ('Volume', 'O')]
stock_array = np.empty(len(time_series_data), dtype=dtype)

# Fill the array with data
for i, (date, stock_data) in enumerate(time_series_data.items()):
    stock_array[i] = (date,
                      stock_data.get('1. open'),
                      stock_data.get('2. high'),
                      stock_data.get('3. low'),
                      stock_data.get('4. close'),
                      stock_data.get('5. volume'))

# Print the NumPy array
print(stock_array)