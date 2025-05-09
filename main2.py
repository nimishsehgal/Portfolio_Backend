from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend origin
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
    # Load data
    df_full = pd.read_csv("data/stock_data.csv", index_col=0, parse_dates=True)
    df = df_full[selected_stocks]

    # Ensure we have enough data
    if len(df) < investment_period + 1:
        return {"error": "Not enough data for the specified investment period."}

    # Slice the last (investment_period + 1) rows
    df = df.iloc[-(investment_period + 1):]

    # Compute simple returns for each stock: (p[t+1] - p[t]) / p[t]
    simple_returns = df.pct_change().iloc[1:]  # Shape: (investment_period, n_stocks)

    # Average return across stocks for each day
    mean_daily_returns = simple_returns.mean(axis=1)  # Shape: (investment_period,)

    # Convert to log returns: log(1 + r)
    log_returns = np.log1p(mean_daily_returns)

    # Cumulative sum of log returns
    cumulative_log_returns = log_returns.cumsum()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[1:], cumulative_log_returns, label="Equal-Weighted Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.title("Equal-Weighted Portfolio: Cumulative Log Return")
    plt.grid(True)
    plt.legend()

    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Final return calculations
    final_cum_log_return = cumulative_log_returns.iloc[-1]
    percent_return = (np.exp(final_cum_log_return) - 1) * 100
    final_value = investment_amount * np.exp(final_cum_log_return)

    return {
        "plot_base64": plot_base64,
        "percent_return": percent_return,
        "final_value": final_value
    }
