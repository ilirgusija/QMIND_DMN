import torch

# Function to calculate the Simple Moving Average (SMA)
def simple_moving_average(data, window_size):
   
    if window_size > data.size(0):
        raise ValueError("Window size must be less than or equal to the number of data points")
    
    sma = torch.zeros(data.size(0) - window_size + 1)
    
    for i in range(sma.size(0)):
        sma[i] = data[i:i + window_size].mean()
    return sma

# Function to calculate the Exponential Moving Average (EMA)
def exponential_moving_average(data, alpha):
    ema = torch.zeros_like(data)
    
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        # Calculate the EMA using the formula: EMA = alpha * current_value + (1 - alpha) * previous_EMA
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

# Function to calculate the Moving Average Convergence Divergence (MACD)
def moving_average_convergence_divergence(data, short_window, long_window):
    short_ema = exponential_moving_average(data, 2 / (short_window + 1))
    
    long_ema = exponential_moving_average(data, 2 / (long_window + 1))
    
    macd = short_ema - long_ema
    return macd

# Function to calculate Lagged Returns
def lagged_returns(data, lag):
    returns = torch.zeros_like(data)
    
    returns[lag:] = (data[lag:] - data[:-lag]) / data[:-lag]
    return returns

# Function to calculate the Relative Strength Index (RSI)
def relative_strength_index(data, window_size):
    delta = data[1:] - data[:-1]
    
    gain = torch.where(delta > 0, delta, torch.tensor(0.0))
    loss = torch.where(delta < 0, -delta, torch.tensor(0.0))
    
    avg_gain = torch.zeros_like(data)
    avg_loss = torch.zeros_like(data)
    
    avg_gain[window_size] = gain[:window_size].mean()
    avg_loss[window_size] = loss[:window_size].mean()
    
    for i in range(window_size + 1, len(data)):
        avg_gain[i] = (avg_gain[i - 1] * (window_size - 1) + gain[i - 1]) / window_size
        avg_loss[i] = (avg_loss[i - 1] * (window_size - 1) + loss[i - 1]) / window_size
    
    rs = avg_gain / avg_loss
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# To use these functions, define a data tensor to calculate the time series features, can use data from preprocessing
