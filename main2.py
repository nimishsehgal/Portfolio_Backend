from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.optimize import minimize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/equal-weight-cumsum-plot/")
async def equal_weight_cumsum_plot(
    selected_stocks: List[str] = Form(...),
    investment_amount: float = Form(...),
    investment_period: int = Form(...)
):
    df_full = pd.read_csv("data/nifty_df_close.csv", index_col=0, parse_dates=True)
    df = df_full[selected_stocks]

    if len(df) < investment_period + 1:
        return {"error": "Not enough data for the specified investment period."}

    df = df.iloc[-(investment_period + 1):]
    simple_returns = df.pct_change().iloc[1:]
    mean_daily_returns = simple_returns.mean(axis=1)
    log_returns = np.log1p(mean_daily_returns)
    cumulative_log_returns = log_returns.cumsum()

    plt.figure(figsize=(12, 7))
    plt.plot(df.index[1:], cumulative_log_returns, label="Equal-Weighted Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.title("Equal-Weighted Portfolio: Cumulative Log Return")
    plt.grid(True)
    plt.legend()

    # Fix overlapping x-axis dates
    import matplotlib.dates as mdates
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=40)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

    final_cum_log_return = cumulative_log_returns.iloc[-1]
    percent_return = (np.exp(final_cum_log_return) - 1) * 100
    final_value = investment_amount * np.exp(final_cum_log_return)

    # Compute per-stock investment
    equal_weight = 1 / len(selected_stocks)
    per_stock_investment = {
        stock: round(equal_weight * investment_amount, 2)
        for stock in selected_stocks
    }

    return {
        "plot_base64": plot_base64,
        "percent_return": percent_return,
        "final_value": final_value,
        "per_stock_investment": per_stock_investment
    }

def solve_mean_variance(mu, cov, target_variance):
    n = len(mu)

    def objective(w):
        return -w @ mu

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: target_variance - w.T @ cov @ w}
    ]

    bounds = [(0, 1) for _ in range(n)]
    best_result = None
    best_return = -np.inf

    # Try multiple starting points to explore the space better
    for _ in range(10):
        initial = np.random.dirichlet(np.ones(n), size=1)[0]
        result = minimize(objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            ret = result.x @ mu
            if ret > best_return:
                best_return = ret
                best_result = result

    return best_result.x if best_result is not None else None

@app.post("/mean-variance-optimization/")
async def mean_variance_optimization(
    selected_stocks: List[str] = Form(...),
    investment_amount: float = Form(...),
    investment_period: int = Form(...),
    target_risk: float = Form(...)
):
    df_full = pd.read_csv("data/nifty_df_close.csv", index_col=0, parse_dates=True)
    df = df_full[selected_stocks]

    if len(df) < 2 * investment_period:
        return {"error": f"Not enough data. Need at least {2 * investment_period} days."}

    # Define training and testing periods
    train_data = df.iloc[-(2 * investment_period):-investment_period]
    test_data = df.iloc[-investment_period:]

    # Compute returns
    train_returns = train_data.pct_change().dropna()
    test_returns = test_data.pct_change().dropna()

    mu = train_returns.mean()
    cov = train_returns.cov()  

    # Solve for optimal weights under risk constraint
    weights = solve_mean_variance(mu, cov, target_risk ** 2)
    if weights is None:
        return {"error": "Optimization failed. Try a different risk level or check data."}

    portfolio_returns = (test_returns * weights).sum(axis=1)
    log_returns_mv = np.log1p(portfolio_returns)
    cumulative_returns = log_returns_mv.cumsum()

    # Plot: cumulative return and portfolio weights
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(test_returns.index, investment_amount * cumulative_returns, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value (₹)")
    plt.grid(True)

    import matplotlib.dates as mdates
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    # plt.bar(selected_stocks, weights)
    percent_weights = [w * 100 for w in weights]
    # investment_values = [w * investment_amount for w in weights]

    bars = plt.bar(selected_stocks, percent_weights)
    for bar, pct in zip(bars, percent_weights):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f"{pct:.1f}%", 
                ha='center', va='bottom', fontsize=9)

    plt.title("Optimal Weights")
    plt.xlabel("Stock")
    plt.ylabel("Weight")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Final return calculations
    final_cum_log_return_mv = cumulative_returns.iloc[-1]
    percent_return = (np.exp(final_cum_log_return_mv) - 1) * 100
    final_value = investment_amount * np.exp(final_cum_log_return_mv)

    investment_breakdown = {
        stock: {
            "weight_percent": round(w * 100, 2),
            "amount_invested": round(w * investment_amount, 2)
        }
        for stock, w in zip(selected_stocks, weights)
    }

    return {
        "plot_base64": plot_base64,
        "final_value": final_value,
        "total_return": percent_return,
        # "weights": {stock: float(w) for stock, w in zip(selected_stocks, weights)}
        "investment_breakdown": investment_breakdown
    }

def solve_min_variance(cov):
    n = cov.shape[0]

    def objective(w):
        return w.T @ cov @ w  # Minimize portfolio variance

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # sum of weights = 1
    ]

    bounds = [(0, 1) for _ in range(n)]  # no short-selling

    best_result = None
    best_variance = np.inf

    for _ in range(10):
        initial = np.random.dirichlet(np.ones(n), size=1)[0]
        result = minimize(objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            var = result.fun
            if var < best_variance:
                best_variance = var
                best_result = result

    return best_result.x if best_result is not None else None

@app.post("/minimum-risk-optimization/")
async def minimum_risk_optimization(
    selected_stocks: List[str] = Form(...),
    investment_amount: float = Form(...),
    investment_period: int = Form(...)
):
    df_full = pd.read_csv("data/nifty_df_close.csv", index_col=0, parse_dates=True)
    df = df_full[selected_stocks]

    if len(df) < 2 * investment_period:
        return {"error": f"Not enough data. Need at least {2 * investment_period} days."}

    # Define training and testing periods
    train_data = df.iloc[-(2 * investment_period):-investment_period]
    test_data = df.iloc[-investment_period:]

    # Compute returns
    train_returns = train_data.pct_change().dropna()
    test_returns = test_data.pct_change().dropna()

    cov = train_returns.cov()  

    # Solve for minimum variance weights
    weights = solve_min_variance(cov)
    
    portfolio_returns = (test_returns * weights).sum(axis=1)
    log_returns_mv = np.log1p(portfolio_returns)
    cumulative_returns = log_returns_mv.cumsum()

    # Plot: cumulative return and portfolio weights
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(test_returns.index, investment_amount * cumulative_returns, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value (₹)")
    plt.grid(True)

    # Fix overlapping x-axis dates
    import matplotlib.dates as mdates
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=40)

    plt.subplot(1, 2, 2)
    # plt.bar(selected_stocks, weights)
    percent_weights = [w * 100 for w in weights]
    # investment_values = [w * investment_amount for w in weights]

    bars = plt.bar(selected_stocks, percent_weights)
    for bar, pct in zip(bars, percent_weights):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f"{pct:.1f}%",  
                ha='center', va='bottom', fontsize=9)

    plt.title("Optimal Weights")
    plt.xlabel("Stock")
    plt.ylabel("Weight")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Final return calculations
    final_cum_log_return_mv = cumulative_returns.iloc[-1]
    percent_return = (np.exp(final_cum_log_return_mv) - 1) * 100
    final_value = investment_amount * np.exp(final_cum_log_return_mv)

    investment_breakdown = {
        stock: {
            "weight_percent": round(w * 100, 2),
            "amount_invested": round(w * investment_amount, 2)
        }
        for stock, w in zip(selected_stocks, weights)
    }

    return {
        "plot_base64": plot_base64,
        "final_value": final_value,
        "total_return": percent_return,
        # "weights": {stock: float(w) for stock, w in zip(selected_stocks, weights)}
        "investment_breakdown": investment_breakdown
    }