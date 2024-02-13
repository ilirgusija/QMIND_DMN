import numpy as np
import torch

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
    return torch.mean(excess_returns) / torch.std(excess_returns)

def sortino_ratio(returns, risk_free_rate):
    """
    Calculate the Sortino Ratio.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.
    - risk_free_rate (float): Risk-free rate.

    Returns:
    - float: Sortino Ratio.
    """
    downside_returns = torch.where(returns < risk_free_rate, returns - risk_free_rate, torch.tensor(0.0))
    downside_deviation = torch.std(downside_returns)
    return torch.mean(returns - risk_free_rate) / downside_deviation if downside_deviation != 0 else torch.tensor(float('nan'))

def max_drawdown(returns):
    """
    Calculate the Maximum Drawdown.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.

    Returns:
    - float: Maximum Drawdown.
    """
    cumulative_returns = torch.cumprod(1 + returns, dim=0)
    peak, _ = torch.cummax(cumulative_returns, dim=0)
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = torch.max(drawdown)
    return max_drawdown.item()

def beta(returns, market_returns):
    """
    Calculate the Beta.

    Args:
    - returns (array-like): Array-like object containing the returns of the investment.
    - market_returns (array-like): Array-like object containing the returns of the market.

    Returns:
    - float: Beta.
    """

    min_size = min(len(returns), len(market_returns))
    returns = returns[:min_size]
    market_returns = market_returns[:min_size]

    mean_returns = torch.mean(returns)
    mean_market_returns = torch.mean(market_returns)
    covariance = torch.mean((returns - mean_returns) * (market_returns - mean_market_returns))
    market_variance = torch.var(market_returns)

    # Calculate beta
    if market_variance != 0:
        beta_value = covariance / market_variance
    else:
        beta_value = float('nan')

    return beta_value.clone().detach()

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
    min_size = min(len(returns), len(market_returns))
    returns = returns[:min_size]
    market_returns = market_returns[:min_size]

    beta_value = beta(returns, market_returns)
    excess_returns = returns - (risk_free_rate + beta_value * (market_returns - risk_free_rate))
    return torch.mean(excess_returns)

# Example usage:
if __name__ == "__main__":
    returns = torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))  # Example returns data
    risk_free_rate = 0.03  # Example risk-free rate
    market_returns = torch.normal(mean=0.5, std=torch.arange(1., 6.)) # Example market returns data

    print("Sharpe Ratio:", sharpe_ratio(returns, risk_free_rate))
    print("Sortino Ratio:", sortino_ratio(returns, risk_free_rate))
    print("Max Drawdown:", max_drawdown(returns))
    print("Beta:", beta(returns, market_returns))
    print("Alpha:", alpha(returns, market_returns, risk_free_rate))