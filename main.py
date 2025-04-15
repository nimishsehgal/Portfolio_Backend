# backend/main.py
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def ret(r, w): return np.dot(r, w)
def vol(w, cov): return np.sqrt(np.dot(w.T, np.dot(cov, w)))

@app.post("/equal-weight-portfolio/")
async def build_equal_weight_portfolio(
    selected_stocks: List[str] = Form(...),
    investment_amount: float = Form(...),
    investment_period: int = Form(...)
):
    # Load pre-existing CSV file
    df_full = pd.read_csv("data/stock_data.csv", index_col=0, parse_dates=True)

    # Filter selected stocks
    df2 = df_full[selected_stocks]
    total_days = len(df2)
    window_size = investment_period
    step_size = investment_period

    results_4 = []
    w_equal = np.ones(df2.shape[1])
    w_equal /= np.sum(w_equal)
    w_opt = w_equal

    for start in range(0, total_days - window_size, step_size):
        end = start + window_size
        future_end = end + step_size
        df_train = df2.iloc[start:end]

        r_train = np.mean(df_train, axis=0) * 252
        covar_train = df_train.cov()

        if future_end < total_days:
            df_test = df2.iloc[end:future_end]
            r_test = np.mean(df_test, axis=0) * 252

            realized_return = ret(r_test, w_opt)
            realized_risk = vol(w_opt, covar_train)

            results_4.append({
                "Start Date": str(df2.index[start].date()),
                "End Date": str(df2.index[end-1].date()),
                "Next Period Start": str(df2.index[end].date()),
                "Next Period End": str(df2.index[future_end-1].date()),
                "Optimized Return": ret(r_train, w_opt),
                "Optimized Risk": vol(w_opt, covar_train),
                "Realized Return": realized_return,
                "Realized Risk": realized_risk
            })

    return {"results": results_4}