from dataset_classes import dataset


api_key = 'KVWI7K2RP426FT5P'


symbol = input("Enter stock symbol: ")
interval = input("Enter time frame: ")


dataset = dataset(api_key, symbol, interval)

print(dataset[0])