from src.dataset_classes import dataset

# Replace 'YOUR_API_KEY' with your actual AlphaVantage API key
api_key = 'KVWI7K2RP426FT5P'
symbol = 'IBM'

# Initialize the dataset
dataset = dataset(api_key, symbol)

# Print an element to verify the data
print(dataset[0])
