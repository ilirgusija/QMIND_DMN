import numpy as np

def sharpe_ratio(returns, risk_free_rate):
    """
    Calculate the Sharpe Ratio.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.
    - risk_free_rate (float): Risk-free rate.

    Returns:
    - float: Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def sortino_ratio(returns, risk_free_rate):
    """
    Calculate the Sortino Ratio.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.
    - risk_free_rate (float): Risk-free rate.

    Returns:
    - float: Sortino Ratio.
    """
    downside_returns = np.where(returns < risk_free_rate, returns - risk_free_rate, 0)
    downside_deviation = np.std(downside_returns)
    return np.mean(returns - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan

def max_drawdown(returns):
    """
    Calculate the Maximum Drawdown.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.

    Returns:
    - float: Maximum Drawdown.
    """
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    return np.max(drawdown)

def beta(returns, market_returns):
    """
    Calculate the Beta.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.
    - market_returns (array-like): Array-like object containing the returns of the market.

    Returns:
    - float: Beta.
    """
    covariance = np.cov(returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else np.nan

def alpha(returns, market_returns, risk_free_rate):
    """
    Calculate the Alpha.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.
    - market_returns (array-like): Array-like object containing the returns of the market.
    - risk_free_rate (float): Risk-free rate.

    Returns:
    - float: Alpha.
    """
    beta_value = beta(returns, market_returns)
    excess_returns = returns - (risk_free_rate + beta_value * (market_returns - risk_free_rate))
    return np.mean(excess_returns)

# Example usage:
if __name__ == "__main__":
    returns = np.random.normal(0.05, 0.1, 100)  # Example returns data
    risk_free_rate = 0.03  # Example risk-free rate
    market_returns = np.random.normal(0.06, 0.15, 100)  # Example market returns data

    print("Sharpe Ratio:", sharpe_ratio(returns, risk_free_rate))
    print("Sortino Ratio:", sortino_ratio(returns, risk_free_rate))
    print("Max Drawdown:", max_drawdown(returns))
    print("Beta:", beta(returns, market_returns))
    print("Alpha:", alpha(returns, market_returns, risk_free_rate))
