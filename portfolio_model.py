import pandas as pd
import numpy as np

def ret(r, w): return np.dot(w, r)
def vol(w, cov): return np.sqrt(np.dot(w.T, np.dot(cov, w)))

def run_equal_weight_model(df1, df2, window_size=60, step_size=30):
    total_days = df2.shape[0]
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
                "Start Date": df1.index[start].strftime('%Y-%m-%d'),
                "End Date": df1.index[end-1].strftime('%Y-%m-%d'),
                "Next Period Start": df1.index[end].strftime('%Y-%m-%d'),
                "Next Period End": df1.index[future_end-1].strftime('%Y-%m-%d'),
                "Optimized Return": float(ret(r_train, w_opt)),
                "Optimized Risk": float(vol(w_opt, covar_train)),
                "Realized Return": float(realized_return),
                "Realized Risk": float(realized_risk)
            })

    return results_4
