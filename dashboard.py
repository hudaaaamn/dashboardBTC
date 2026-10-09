# dashboard.py — Bitcoin Price Dashboard / TA Huda
# Revisi 8 Oktober 2026 — deployment SATU FILE.
# Cukup ganti seluruh isi dashboard.py di repository Streamlit Anda.
# Fungsi model, UI, dan hasil Colab terverifikasi sudah berada di file ini.
# Tidak perlu dashboard_core.py, dashboard_results.py, atau folder dashboard_data.
# Paket pihak ketiga sama dengan dashboard sebelumnya; SciPy adalah dependensi
# hmmlearn. Rentang lama Streamlit >=1.40 didukung melalui adapter UI.
# Jalankan: python -m streamlit run dashboard.py
# Komparasi adalah hasil eksperimen tersimpan; prediksi live melakukan refit.
# Netflow stale dan batasan statistik ditampilkan secara eksplisit.

import base64
import hashlib
import inspect
import io
import json
import os
import zlib
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy.special import logsumexp

# Frozen Colab result: primary raw run, 8 metric rows, 15 tests (block 14).
# Base64 encodes compressed JSON data only; it never executes code.
_EXPERIMENTS_SHA256 = "8b375db9099072c533da99f81b2b1f255a631e1b56c8fc9f6522b17c8030696d"
_EXPERIMENTS_B64 = (
    "eNq9WF1z28YV/SscPLVTCd5791t9chw78TSKXclNO9PJcJYkKMEGQQ0B2kk9/l197y/ruaBELCVnbCWybc8kxGKBPefcPfcs3hfd/LJapenbat"
    "PV67Y4oaNiU3Xbpu/kWr2sq8V0kfqqOClYsTsmdax8gZu2bVecvC826d10WTd9tZFfTZpVDW79W9p2qZls+7RKk//9d4K7Jru76vZCLvjJizf9"
    "RJ6IZ72p2wUmdRhrKvx+mzZ1antcyp5+VGB12uujYlmlfruppl2fNv1uXfZY2WOiYhyshkcOS5ahgCHcflH1t0fiDswActqs52+yQVbyxLSqm1"
    "+nXf0fkED2qJit133Xb9LVFEyl1VVTgQlW+HNUzNfyuwdp235+wNkr4hPLJ0aXZPCX/qLUiZLnX1M/sNluV1e/yrSSSo2xq9QuUjdc4OFCN6+H"
    "G6gkt7vQp75brRdVI7epkmwpD/3lAqvshB1dmlKIaeqLy/5itsIlU7rhpsvVqqnSph0m6uFxfdV2682yWb/bvZSGG/HWN3V/fHMzYT4VH46KVd"
    "Vv6jle/O/3xamsAYP/+u6b4c2jXqePn4I4x740ylqrjXLKeHdUnJ2eY4SJY+mMMjZGxWy9zHgpU0pvgtEUfbQmWmUNpnBxosoYrHWBgjbeso64"
    "/m29eTwH4SaUPmrrMKydcSR6PVmD4nQB9dpt0xwVj99e/LNe9Jc3v1/W7Sw1zQ/rWxe+r3cXPhyN6H4QFr/75vQuPLyZjI/MKljvzQiPlSlV8B"
    "HgPWqEbYYPCMhjudp5rY0e8RlryOoQlXKaQ4YvlsqRc0wx6EiGHxbeGeptvZo8W6Oy70poULzBambPkJGUvcEIiXSpiMhBDCw+hhEjlHDQyXli"
    "0iaMEL2D8E5FD+4ofj0Jz386u42MlQ2l0hwNXip1qPfqRQLjBoUYUMFW7YsTe0MF5xUBnQrGSNleI/McgiOPqg1kZfU3yCwGddDaB0UOZRIeuD"
    "bPX51+RDPsdZSlCd5AOnKjZsGWno22IllQljLNvAMlJpIQY4c5N6IZy4G0c9DFfb26HP47+Qf6UmrvQvSwPXhKYKdQN5yVZYwlJIpaWe1iiDqD"
    "iIvekBa5rfMZQvYwKYrOKU+ZdijwgJqPKAjrh9ofAUZVHoCnHK0ngrdCco4KZhb8AXSWBzP2PordM6FyDohgTSVDAo5YkBShz2l5ldqrNDmrLu"
    "pVNfnT96enf/4IOb4kLFc5joFRmQfkKKwKSnoDJnxOTpR/IcCdSGvO2UGtw6yc9spn+lsqjRTYniPz+fwo7HgpMxMxrLW7w48j7CSWx8N2bvMD"
    "7Z13LsTIHmoe8PNj1Usrm5xWbZ8u73KDHcDaAlNk5/Zlg1aOJu1IM3oMBt1IDIoswiYci54QNidG43kgzGKn5cRwCRdQsD7ARHO7Dy/KgxdHqB"
    "pwLus74MWgtK2GSsZgLUJ4zgt6KjOURf4IztrgP/y8Cwt119+0bAkriFqd5L6sb++6uuw6sDRyNtuFI8KLFvVyWW2qdg4Mx2R9ifLELgwKNmMZ"
    "5TKvp0OEOAa9pYWFa2PYSzH5YfASbRSjDmHFYNfCELE5htpHqErvhFK0SoeKcB7kOYYXSeleTS/XzWo3DNzYiQgARJIBjATX19UcmaffbCspgw"
    "N8Wee+L0CP5mCQNiIUxNvEKUeAFkTDRLThiFq1xuYASaE8DTNQALtVYhvXCE2JDQhw2IGAqb2tjnejI0IYFAhA1yH4bkQS+ATE2937fjjZlQxz"
    "l4CGQoUluRwm0g2qiL2KWkftKEPppNQUBxSAxDXWmYqQlQI6iUJkMDANMyJEtBzxLFPT3QW0a9X3lAu1XkapepgR9o7P1fIO4UIZNERyESlN52"
    "IZiIWJ6AwoSQuYX1Kr62Z9T4mQJIFMa2QHNjR0oT02bVCIyFLeyj5z7DJsHp6uPP44QgNnNplEGmlLiFIobwy6eyn00Q50T8E86IM9oEPD5VAj"
    "OSi0ThAfGBaKRIgukIEC7ehDiMgk6QXKaMr9A78xVUMzbSR25XpZgxTG6FUa9qmZ/hjMa6//JNKhf1gvTmc87WLyHqkqA7ww4KgDKqwEtgyq6I"
    "4m4L1YehwizrjFcMAIIQ6pRaTVDyjgvlV9EpuSV92Aufmx12m/XNr9/wMt77rh/V3ZzygzqB4ZDoT4je1hXcz9Dcdtp9FI0C819gDiwAEARsDH"
    "kVMHhDAcVmLI+EduQX5AfeLZHvbnvgDAaD+nuiTIyYEKC8GpShLBKAjcFz0WV4ECZpwXF8l5C8ixG6RNkz5sw5LncDDCyRWPDS7Hh2FCpI0IN3"
    "J6Qoqx9AnfuxPJ7p02Sjk2opcg+gV/ywJ9CQjKYjOI2fFBlyoxw8pRx8GevaSUUUJvFA472HsQnwPbe0n425A+1xhQYC6o6JCBr0N03nrLKA1l"
    "zIkZKF1K4IK26GfyseAgQeEQrSEL2rkGpEAPBOp3esJdg4sHW+zu4O+0jN9e+b3sAmYd5Kgnny5iVMblsdaW8GA0q+BkW9zotYPCpTg0lEIwxj"
    "TK4xDaDlBqpCxsld28B0X2OT7BOKnj2BARgLASbHpzIJJsIJwHvXLon6yiPmhD6JpoujiT4Lzj5dg8GgWOPcYiy8NjkDIU5TZxa4xvI/35Bsu0"
    "W283ssriZdVebF/XqX103qdZ3dR96qbDwe7R2bad7r4wT5+smzSbyldWRG3/6KenZ8+fPX/67fT06auz50/Op09enL58fPb8/MWP5bx7W+RHoD"
    "/4qlePp/PdwPiRWi7W7TXT04H/6f5r8fUC2vXwMf3Z/mv4Kr3eThZYQmon0DqBiGozQSeYLKD8BFPrtpycpXeTRd1Xfbp6gxu7apYuUj1JbWrq"
    "ru52n9r/OunmqakW++Guaru6r98KorL48AH4B9DT7jKhgOWT8xfh+aTQ5Bd2huNyFWbzConaVHGxRKpZBiQfaxNRWjrn/cxz4qVfLOUIFZRbYj"
    "RZcZevKMtJQZWtfEyLxXxulti0tEjzWQpx7mYVknHimal4ZpHvlzZxQuBMYRl5UZnk3cJYkPt/pPgIxw=="
)

FEATURE_COLS = [
    "log_return", "volatility_7d", "volatility_14d", "volatility_30d",
    "price_to_ma7", "price_to_ma30", "return_lag_1", "return_lag_2", "return_lag_3",
    "exchange_netflow", "netflow_change", "netflow_ma_7", "polarity",
    "sentiment_ma_7", "netflow_weighted", "regime_labeled",
]
CORE_VERSION = "2026.10.08-singlefile-v4"


def filter_hmm(model, observations):
    """P(state_t | observations_0:t), with fixed, train-only parameters.

    Matches the verified Colab recurrence. The emission helper is private in
    hmmlearn 0.3.3; that exact version is required and tested against endpoint
    posterior probabilities. No backward pass or Viterbi decoding of test data.
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or not len(observations) or not np.isfinite(observations).all():
        raise ValueError("Input HMM harus berupa matriks observasi finite yang tidak kosong.")
    emission = model._compute_log_likelihood(observations)
    with np.errstate(divide="ignore"):
        transition = np.log(model.transmat_)
        forward = np.log(model.startprob_) + emission[0]
    result = np.empty_like(emission)
    forward -= logsumexp(forward)
    result[0] = np.exp(forward)
    for t in range(1, len(observations)):
        forward = emission[t] + logsumexp(forward[:, None] + transition, axis=0)
        forward -= logsumexp(forward)
        result[t] = np.exp(forward)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Probabilitas filtering HMM tidak valid.")
    return result


def map_train_states(model, train_observations):
    """Bear/Sideways/Bull ordering from the fit block only, including empty states."""
    path = model.predict(train_observations)
    means = []
    for state in range(model.n_components):
        selected = train_observations[:, 0][path == state]
        means.append(float(selected.mean()) if len(selected) else float(model.means_[state, 0]))
    return {int(state): int(rank) for rank, state in enumerate(np.argsort(means, kind="stable"))}


def split_forecast_frame(frame):
    """Chronological 70/10/20; purge a feature date at each horizon-1 boundary."""
    n = len(frame)
    train_end, calib_end = int(n * .70), int(n * .80)
    if train_end < 60 or calib_end - train_end < 20 or n - calib_end < 20:
        raise ValueError("Data lengkap belum cukup untuk train, kalibrasi, dan test.")
    train = frame.iloc[:train_end - 1].copy()
    calibration = frame.iloc[train_end:calib_end - 1].copy()
    test = frame.iloc[calib_end:].copy()
    if not (train.target_date.max() < calibration.index.min()
            and calibration.target_date.max() < test.index.min()):
        raise ValueError("Tanggal target melintasi batas split.")
    return train, calibration, test


def conformal_margin(scores, alpha=.10):
    """Keep the quantile convention of the saved Colab runs (NumPy linear)."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < alpha < 1:
        raise ValueError("Skor kalibrasi tidak valid.")
    q = min(np.ceil((1 - alpha) * (len(scores) + 1)) / len(scores), 1.)
    return float(np.quantile(scores, q, method="linear"))


def load_experiments():
    """Decode verified results embedded in this file; no filesystem read."""
    raw = zlib.decompress(base64.b64decode(_EXPERIMENTS_B64, validate=True))
    if hashlib.sha256(raw).hexdigest() != _EXPERIMENTS_SHA256:
        raise ValueError("Integritas data eksperimen tertanam tidak sesuai.")
    data = json.loads(raw)
    if data.get("schema_version") != 1 or not data.get("runs"):
        raise ValueError("Paket hasil eksperimen tidak valid.")
    for run in data["runs"].values():
        names = {row["Model"] for row in run["metrics"]}
        if "Model Usulan" not in names or len(names) != len(run["metrics"]):
            raise ValueError("Identitas model pada paket hasil tidak valid.")
        for row in run["metrics"]:
            if not np.isfinite(row["MAE"]) or row["n"] != run["n"]:
                raise ValueError("Metrik atau ukuran sampel tidak valid.")
        for row in run["statistics"]:
            if row["n"] != run["n"]:
                raise ValueError("Ukuran sampel statistik berbeda dari metrik.")
            if run["family_size"] and not (0 < row["p_holm"] <= 1):
                raise ValueError("p Holm tidak valid.")
    return data


def build_forecast(prices, sentiment, onchain, onchain_live=False):
    """Live refit, separate from frozen Colab evaluation tables.

    Raw filtering is fixed in advance. Horizon-1 target purge follows the newer
    multiwindow protocol. Therefore live splits/weights are not a reproduction
    of the single-window historical run even on an identical end date.
    """
    from importlib.metadata import version
    from hmmlearn.hmm import GaussianHMM
    from xgboost import XGBRegressor

    if sentiment.empty or "polarity" not in sentiment:
        raise ValueError("Sentimen riil tidak tersedia. Prediksi tidak dijalankan.")
    if onchain.empty or onchain.exchange_netflow.dropna().empty:
        raise ValueError("Observasi netflow riil tidak tersedia.")
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Tanggal harga harus unik, berurutan, dan tidak kosong.")
    if not (prices.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Harga memiliki tanggal yang hilang; horizon satu hari belum dapat dihitung.")
    valid_dates = onchain.exchange_netflow.dropna().index
    observed = valid_dates[valid_dates <= prices.index.max()]
    if not len(observed):
        raise ValueError("Tidak ada netflow yang tersedia pada atau sebelum tanggal harga.")
    last_real = observed.max()
    df = prices[["close"]].join(sentiment[["polarity"]], how="left")
    df = df.join(onchain[["exchange_netflow"]], how="left")
    df[["polarity", "exchange_netflow"]] = df[["polarity", "exchange_netflow"]].ffill()
    df = df.dropna(subset=["close", "polarity", "exchange_netflow"])
    if (df.close <= 0).any():
        raise ValueError("Harga harus positif.")
    df["log_return"] = np.log(df.close / df.close.shift(1))
    for days in [7, 14, 30]:
        df[f"volatility_{days}d"] = df.log_return.rolling(days).std()
    for days in [7, 30]:
        df[f"price_to_ma{days}"] = df.close / df.close.rolling(days).mean()
    for lag in [1, 2, 3]:
        df[f"return_lag_{lag}"] = df.log_return.shift(lag)
    df["netflow_change"] = df.exchange_netflow.diff()
    df.loc[df.index > last_real, "netflow_change"] = 0.
    df["netflow_ma_7"] = df.exchange_netflow.rolling(7).mean()
    df["sentiment_ma_7"] = df.polarity.rolling(7).mean()
    df["netflow_weighted"] = df.exchange_netflow * (1 + df.polarity)
    df["target_return"] = df.log_return.shift(-1)
    df["target_date"] = df.index.to_series().shift(-1)
    usable = df.dropna(subset=FEATURE_COLS[:-1] + ["target_return", "target_date"])
    train_base, _, _ = split_forecast_frame(usable)

    hdata = df.dropna(subset=["log_return", "volatility_14d"])
    X_hmm = hdata[["log_return", "volatility_14d"]].to_numpy()
    fit_n = min(int(len(hdata) * .70), hdata.index.searchsorted(train_base.index[-1], side="right"))
    if fit_n < 60:
        raise ValueError("Blok train HMM belum cukup.")
    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, tol=.01, random_state=42)
    hmm.fit(X_hmm[:fit_n])
    mapping = map_train_states(hmm, X_hmm[:fit_n])
    probabilities = filter_hmm(hmm, X_hmm)
    labels = np.array([mapping[int(state)] for state in probabilities.argmax(axis=1)])
    df["regime_labeled"] = pd.Series(labels, index=hdata.index)
    usable = df.dropna(subset=FEATURE_COLS + ["target_return", "target_date"])
    train, calibration, test = split_forecast_frame(usable)
    live = df.iloc[[-1]]
    if not np.isfinite(live[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Fitur prediksi terakhir tidak lengkap.")
    params = dict(n_estimators=300, learning_rate=.05, max_depth=5,
                  random_state=42, verbosity=0, tree_method="hist", n_jobs=1)
    models = [XGBRegressor(**params, objective="reg:quantileerror", quantile_alpha=q)
              for q in [.05, .50, .95]]
    for model in models:
        model.fit(train[FEATURE_COLS].to_numpy(), train.target_return.to_numpy())
    lo_calib = models[0].predict(calibration[FEATURE_COLS].to_numpy())
    hi_calib = models[2].predict(calibration[FEATURE_COLS].to_numpy())
    scores = np.maximum(lo_calib - calibration.target_return.to_numpy(),
                        calibration.target_return.to_numpy() - hi_calib)
    margin = conformal_margin(scores)
    lo, med, hi = [float(model.predict(live[FEATURE_COLS].to_numpy())[0]) for model in models]
    close = float(live.close.iloc[0])
    pred_lo, pred_med, pred_hi = close * np.exp([lo - margin, med, hi + margin])
    test_X = test[FEATURE_COLS].to_numpy()
    hist_lo = test.close.to_numpy() * np.exp(models[0].predict(test_X) - margin)
    hist_med = test.close.to_numpy() * np.exp(models[1].predict(test_X))
    hist_hi = test.close.to_numpy() * np.exp(models[2].predict(test_X) + margin)
    if not np.isfinite([pred_lo, pred_med, pred_hi]).all() or not np.isfinite(hist_lo).all() or not np.isfinite(hist_hi).all():
        raise ValueError("Prediksi harga tidak finite.")
    if pred_lo > pred_hi or np.any(hist_lo > hist_hi):
        raise ValueError("Batas interval saling silang; hasil tidak ditampilkan sebagai interval valid.")
    history = list(hmm.monitor_.history)
    metadata = dict(version=CORE_VERSION, regime_method="raw_filter", hmm_fit_n=fit_n,
        hmm_fit_end=str(hdata.index[fit_n - 1].date()), hmm_iterations=int(hmm.monitor_.iter),
        hmm_last_delta=float(history[-1] - history[-2]) if len(history) > 1 else None,
        train_n=len(train), calibration_n=len(calibration), test_n=len(test),
        train_end=str(train.index[-1].date()), calibration_end=str(calibration.index[-1].date()),
        purge_days=1, quantile_method="linear", feature_count=len(FEATURE_COLS),
        versions={name: version(name) for name in ["numpy", "pandas", "scipy", "hmmlearn", "xgboost"]})
    return dict(df=df.dropna(subset=FEATURE_COLS), df_test=test, hist_target_dates=test.target_date,
        pred_date=live.index[0] + pd.Timedelta(days=1), pred_lo=float(pred_lo), pred_med=float(pred_med),
        pred_hi=float(pred_hi), last_close=close, last_date=live.index[0],
        last_regime=int(live.regime_labeled.iloc[0]), last_sent=float(live.polarity.iloc[0]),
        last_netflow=float(live.exchange_netflow.iloc[0]), hist_lo=hist_lo, hist_med=hist_med, hist_hi=hist_hi,
        hist_true=test.close.to_numpy() * np.exp(test.target_return.to_numpy()), conf_margin=margin,
        onchain_live=onchain_live, onchain_last_real_date=last_real, onchain_last_csv_date=onchain.index.max(),
        onchain_staleness_days=(prices.index.max() - last_real).days, pipeline_metadata=metadata)

BASELINES = ["XGBoost", "LightGBM", "Random Forest", "SVR", "LSTM", "Model Usulan"]
METRICS = ["MAE", "RMSE", "MAPE", "R2", "DirAcc", "Coverage", "AvgWidth", "PinballLo", "PinballHi"]


def p_text(value):
    if value is None or pd.isna(value):
        return "—"
    return "<0.0002 (arsip dibulatkan)" if value == 0 else f"{value:.6f}"


def display_metrics(frame):
    result = frame.copy()
    for column in METRICS:
        if column not in result.columns:
            continue
        if column in ["MAE", "RMSE", "AvgWidth"]:
            fmt = lambda x: f"${x:,.2f}"
        elif column in ["MAPE", "DirAcc", "Coverage"]:
            fmt = lambda x: f"{x:.2f}%"
        elif column == "R2":
            fmt = lambda x: f"{x:.4f}"
        else:
            fmt = lambda x: f"{x:.2f}"
        result[column] = result[column].map(lambda x: fmt(x) if pd.notna(x) else "—")
    return result


def statistic_table(statistics, adjusted=True):
    rows = []
    for row in statistics:
        reject = row["reject"] if adjusted else row["p_raw"] < .05
        rows.append({"Pembanding": row["comparison"], "Metrik": row["metric"],
            "n": row["n"], "Δ FULL − pembanding": f"{row['difference']:+.4f}",
            "CI95 pointwise": f"[{row['ci_low']:+.4f}; {row['ci_high']:+.4f}]",
            "p mentah": p_text(row["p_raw"]), "p Holm": p_text(row["p_holm"]),
            "Keputusan": "Perbedaan terdeteksi" if reject else "Belum terdeteksi"})
    return pd.DataFrame(rows)


def main_comparison_frame(run):
    """Six familiar model rows, always from the fixed primary experiment."""
    roles = {"XGBoost": "Gradient boosting", "LightGBM": "Gradient boosting",
             "Random Forest": "Ensemble tree", "SVR": "Support vector",
             "LSTM": "Deep learning sekuensial",
             "Model Usulan": "HMM + XGB Quantile + Conformal"}
    metrics = pd.DataFrame(run["metrics"]).set_index("Model")
    main = metrics.loc[BASELINES[:6]].reset_index()
    main.insert(0, "No", range(1, len(main) + 1))
    main.insert(2, "Peran", main.Model.map(roles))
    main.insert(3, "Sumber", "Colab (offline)")
    return main[["No", "Model", "Peran", "Sumber"] + METRICS[:7]]


def selected_statistics(run, block=14):
    rows = [row for row in run["statistics"] if row["block"] == block] if run["family_size"] else run["statistics"]
    return statistic_table(rows, bool(run["family_size"]))


def render_inference_note(run, block):
    st.caption(f"Stationary bootstrap {run['bootstrap_resamples']:,} resample · blok {block} hari · "
               f"Holm{run['family_size']} · CI95 basic pointwise, bukan simultan.")
    st.caption("Selisih adalah Model Usulan (FULL) dikurangi pembanding. MAE/pinball dalam USD; "
               "DirAcc/coverage dalam poin persentase. Tidak terdeteksi bukan bukti kesetaraan.")


def render_ablation(run, table, render_table, suffix=""):
    metrics = pd.DataFrame(run["metrics"])
    ablation = metrics[metrics.Model.isin(["Model Usulan", "Tanpa Regime (HMM)", "Netflow Mentah"])]
    render_table(display_metrics(ablation[["Model", "n"] + METRICS]), label="Metrik ablasi" + suffix)
    render_table(table[table.Pembanding.isin(["Tanpa Regime (HMM)", "Netflow Mentah"])],
                 label="Uji ablasi" + suffix)
    st.caption("Ablasi dan statistik memakai run serta tanggal uji yang sama. Baca arah selisih bersama p Holm. "
               "Bootstrap hari bersyarat pada model yang telah difit dan tidak mencakup seluruh variasi pelatihan ulang.")


def render_metric_charts(run, render_chart, accent):
    metrics = pd.DataFrame(run["metrics"])
    main = metrics[metrics.Model.isin(BASELINES)]
    st.markdown("<div class='section-label'>Visualisasi Metrik</div>", unsafe_allow_html=True)
    chart_tabs = st.tabs(["MAE & MAPE", "R² & DirAcc", "Probabilistik"])
    colors = [accent if name == "Model Usulan" else "#aab4bf" for name in main.Model]
    for chart_tab, pairs in zip(chart_tabs[:2], [
        [("MAE", "MAE (USD) · lebih rendah lebih baik", "${:,.2f}"), ("MAPE", "MAPE (%)", "{:.2f}%")],
        [("R2", "R²", "{:.4f}"), ("DirAcc", "Directional Accuracy (%)", "{:.2f}%")],
    ]):
        with chart_tab:
            columns = st.columns(2)
            for column, (metric, title, fmt) in zip(columns, pairs):
                with column:
                    fig = go.Figure(go.Bar(y=main.Model, x=main[metric], orientation="h", marker_color=colors,
                        text=[fmt.format(v) for v in main[metric]], textposition="outside"))
                    fig.update_layout(title=title, height=350, margin=dict(r=90))
                    if metric == "DirAcc":
                        fig.add_vline(x=50, line_dash="dash", line_color="#637080")
                    render_chart(fig)
    with chart_tabs[2]:
        full = metrics[metrics.Model.eq("Model Usulan")].iloc[0]
        for column, metric, label in zip(st.columns(4), ["Coverage", "AvgWidth", "PinballLo", "PinballHi"],
                                          ["Coverage", "Avg Width", "Pinball Q05", "Pinball Q95"]):
            value = f"{full[metric]:.2f}%" if metric == "Coverage" else f"${full[metric]:,.2f}"
            column.metric(label, value)
        st.caption("Coverage empiris pada eksperimen utama; target nominal 90%. Ini tidak menjamin cakupan live "
                   "atau cakupan sama pada setiap periode, khususnya ketika netflow tertinggal.")


def render_experiments(render_table, render_chart, accent):
    """Present only the verified primary causal raw experiment."""
    try:
        data = load_experiments()
    except (OSError, ValueError, KeyError) as error:
        st.error("Data eksperimen tertanam tidak valid. Salin ulang seluruh isi dashboard.py versi terbaru.")
        st.caption(type(error).__name__)
        return
    primary = data["runs"]["raw_filter"]
    st.markdown(f"""
    <div class='page-header'>
        <h2>Komparasi Model Prediksi</h2>
        <p>Evaluasi pada test set · {primary['n']} tanggal uji bersama</p>
    </div>
    <details class='info-box'>
        <summary>Referensi eksperimen · hasil utama terbaru · kausal raw filtering</summary>
        <div class='notice-body'>
            Seluruh model pada tabel dan grafik memakai hasil Google Colab yang sama:
            <b>{escape(primary['label'])}</b>.<br>
            Tanggal fitur {escape(primary['feature_start'])}–{escape(primary['feature_end'])};
            target terakhir {escape(primary['target_end'])}.
            Data dikunci pada {escape(primary['run_date_lock'])} (eksklusif).
            Angka ini merupakan hasil historis tersimpan. Prediksi live pada tab Prediksi
            melakukan refit dengan data dan bobot tersendiri; coverage historis tidak menjamin coverage live.
        </div>
    </details>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Hasil Evaluasi Model</div>", unsafe_allow_html=True)
    main = main_comparison_frame(primary)[["Model"] + METRICS[:7]]
    render_table(display_metrics(main).rename(columns={
        "R2": "R²", "DirAcc": "Akurasi arah", "Coverage": "Cakupan", "AvgWidth": "Lebar rentang",
    }), label="Hasil evaluasi utama · raw kausal")
    st.caption("MAE, RMSE, dan MAPE mengukur kesalahan prediksi; nilai lebih kecil berarti kesalahan lebih rendah. "
               "Cakupan dan lebar rentang tersedia untuk Model Usulan.")

    # The main analysis uses the fixed block-14 protocol; no experiment selector.
    with st.expander("Uji Signifikansi Statistik (stationary bootstrap, N=20000, Holm15, CI 95%)"):
        block = 14
        render_inference_note(primary, block)
        table = selected_statistics(primary, block)
        render_table(table[table.Metrik.eq("MAE") & table.Pembanding.isin(BASELINES)], label="Uji MAE utama")

    with st.expander("Ablation Study (Kontribusi HMM & Sentiment-Weighted Netflow)"):
        st.caption("Hasil kausal utama · 373 tanggal bersama · blok 14 hari · Holm15.")
        render_ablation(primary, selected_statistics(primary, 14), render_table)

    render_metric_charts(primary, render_chart, accent)


def render_prediction(result, render_chart):
    """Put the forecast first, with concise model information below the chart."""
    last_close, last_date = result["last_close"], result["last_date"]
    pred_med, pred_lo, pred_hi = result["pred_med"], result["pred_lo"], result["pred_hi"]
    pred_date = result["pred_date"]
    delta_pct = (pred_med - last_close) / last_close * 100
    delta_class = "delta-positive" if delta_pct >= 0 else "delta-negative"
    regime = int(result["last_regime"])
    regime_name = REGIME_NAMES[regime]
    regime_class = {0: "regime-bear", 1: "regime-side", 2: "regime-bull"}[regime]
    regime_help = {0: "Kondisi pasar melemah", 1: "Pergerakan relatif mendatar", 2: "Kondisi pasar menguat"}[regime]
    width = pred_hi - pred_lo
    median_position = 50 if width == 0 else min(100, max(0, (pred_med - pred_lo) / width * 100))
    price_age = (pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timestamp(last_date.date())).days
    chain_age = result["onchain_staleness_days"]
    st.markdown(f"""
    <header class='page-header'>
        <h2>Prediksi harga Bitcoin</h2>
        <p>Estimasi harga dan rentang prediksi untuk {format_date_id(pred_date)}.</p>
    </header>
    <section class='forecast-overview' aria-label='Ringkasan prediksi Bitcoin'>
        <div class='forecast-main'>
            <div class='estimate'>
                <span class='metric-label'>Estimasi harga · {format_date_id(pred_date)}</span>
                <div class='estimate-number'>${pred_med:,.0f}</div>
                <div class='estimate-change'><span class='{delta_class}'>{delta_pct:+.2f}%</span> dari harga terakhir</div>
            </div>
            <div class='estimate-range'>
                <span class='metric-label'>Rentang prediksi · target cakupan 90%</span>
                <div class='range-number'><span>${pred_lo:,.0f}</span><span class='range-separator'>s.d.</span><span>${pred_hi:,.0f}</span></div>
                <div class='range-scale' aria-hidden='true'><span class='range-end'></span><span class='range-track' style='--median-position:{median_position:.2f}%'></span><span class='range-end'></span></div>
                <span class='metric-foot'>Harga aktual dapat berada di luar rentang ini.</span>
            </div>
        </div>
        <div class='forecast-context'>
            <div><span>Harga terakhir</span><strong>${last_close:,.0f}</strong></div>
            <div><span>Data acuan</span><strong>{format_date_id(last_date)}</strong></div>
            <div class='regime-context'><span>Regime pasar</span><strong class='{regime_class}'>{regime_name}</strong></div>
        </div>
    </section>
    """, unsafe_allow_html=True)

    if price_age > 1:
        st.markdown(f"""
        <details class='warn-box'><summary>{WARN_ICON}<span class='notice-title'>Harga acuan tertinggal {price_age} hari.</span></summary>
        <div class='notice-body'>Prediksi mengikuti harga penutupan terakhir, {format_date_id(last_date)}.
        Tekan <b>Refresh Data</b> untuk memeriksa ketersediaan harga yang lebih baru.</div></details>
        """, unsafe_allow_html=True)
    if chain_age > 1:
        st.markdown(f"""
        <details class='warn-box'><summary>{WARN_ICON}<span class='notice-title'>Data on-chain memakai catatan terakhir {format_date_id(result['onchain_last_real_date'])}.</span></summary>
        <div class='notice-body'>Nilai netflow tertinggal {chain_age} hari dari harga acuan dan diteruskan
        dari observasi terakhir (forward-fill). Nilai tersebut tidak menggambarkan aktivitas bursa terbaru.
        Hasil pengujian historis belum mengukur akurasi prediksi dalam kondisi ini.</div></details>
        """, unsafe_allow_html=True)

    heading, control = st.columns([3, 1], vertical_alignment="bottom")
    with heading:
        st.markdown("<div class='section-label'>Harga aktual & estimasi</div>", unsafe_allow_html=True)
        st.caption("Area oranye menunjukkan rentang prediksi. Garis putus-putus menunjukkan estimasi harga.")
    with control:
        period = st.selectbox("Rentang grafik", ["90 hari terakhir", "30 hari terakhir", "Seluruh test set"],
                              key="forecast_period", label_visibility="collapsed")
    dates = pd.DatetimeIndex(result["hist_target_dates"])
    actual, med, low, high = (result[name] for name in ["hist_true", "hist_med", "hist_lo", "hist_hi"])
    n = min(len(dates), len(actual), len(med), len(low), len(high))
    dates = dates[:n]
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=list(dates) + list(dates[::-1]),
        y=list(high[:n]) + list(low[:n][::-1]), fill="toself", fillcolor=TEAL_SOFT,
        line=dict(color="rgba(0,0,0,0)"), name="Rentang 90%", hoverinfo="skip"))
    chart.add_trace(go.Scatter(x=dates, y=actual[:n], mode="lines", name="Harga aktual",
        line=dict(color=TEXT, width=1.7)))
    chart.add_trace(go.Scatter(x=dates, y=med[:n], mode="lines", name="Estimasi harga",
        line=dict(color="#b36a15", width=1.6, dash="dash")))
    chart.add_trace(go.Scatter(x=[pred_date], y=[pred_med], mode="markers",
        name=f"Estimasi {pred_date.strftime('%d %b')}", showlegend=False,
        marker=dict(color="#b36a15", size=8, line=dict(color=SURFACE, width=2))))
    chart.add_trace(go.Scatter(x=[pred_date, pred_date], y=[pred_lo, pred_hi], mode="lines",
        name="Rentang prediksi berikutnya", showlegend=False, line=dict(color="#b36a15", width=2.5)))
    chart.update_layout(template="plotly_white", height=360,
        margin=dict(l=16, r=24, t=52, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(size=11)),
        yaxis=dict(tickprefix="$", tickformat=",", side="right", nticks=5),
        xaxis=dict(showgrid=False, tickformat="%d %b", nticks=5), hovermode="x unified")
    if period != "Seluruh test set":
        days = 90 if period == "90 hari terakhir" else 30
        chart.update_xaxes(range=[pred_date - timedelta(days=days), pred_date + timedelta(days=2)])
    chart.update_traces(hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>", selector=dict(mode="lines"))
    render_chart(chart)
    st.caption("Grafik memakai tanggal harga yang diprediksi. Hasil perhitungan ini terpisah dari evaluasi pada tab Komparasi Model.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("<div class='section-label'>Memahami rentang prediksi</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <section class='detail-panel' aria-label='Detail rentang prediksi'>
        <dl>
            <div class='detail-row'><dt>Batas bawah</dt><dd>${pred_lo:,.0f}</dd></div>
            <div class='detail-row'><dt>Estimasi tengah (median)</dt><dd>${pred_med:,.0f}</dd></div>
            <div class='detail-row'><dt>Batas atas</dt><dd>${pred_hi:,.0f}</dd></div>
            <div class='detail-row'><dt>Lebar rentang</dt><dd>${width:,.0f} · {width / pred_med * 100:.1f}% dari estimasi</dd></div>
        </dl>
        <p class='detail-note'>Rentang yang lebih lebar menunjukkan ketidakpastian model yang lebih besar.
        Target cakupan 90% tidak menjamin setiap harga aktual masuk ke dalam rentang.</p>
        </section>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-label'>Kondisi pasar pada data acuan</div>", unsafe_allow_html=True)
        polarity, netflow = result["last_sent"], result["last_netflow"]
        sentiment = ("Extreme Greed" if polarity > .6 else "Greed" if polarity > .2
                     else "Neutral" if polarity > -.2 else "Fear" if polarity > -.6 else "Extreme Fear")
        direction = "Arus keluar bersih dari bursa" if netflow > 0 else "Arus masuk bersih ke bursa" if netflow < 0 else "Arus bursa seimbang"
        chain_context = f"Catatan terakhir {format_date_id(result['onchain_last_real_date'])}." if chain_age > 1 else "Data on-chain tersedia hingga tanggal acuan."
        st.markdown(f"""
        <section class='detail-panel' aria-label='Kondisi pasar'>
            <div class='market-item'><div class='market-heading'><span>Regime pasar</span><strong>{regime_name}</strong></div><p>{regime_help} berdasarkan klasifikasi model.</p></div>
            <div class='market-item'><div class='market-heading'><span>Sentimen pasar</span><strong>{sentiment}</strong></div><p>Indeks Fear &amp; Greed · skor polaritas {polarity:+.2f}.</p></div>
            <div class='market-item'><div class='market-heading'><span>Netflow bursa</span><strong>{netflow:+,.0f} BTC</strong></div><p>{direction}. {chain_context}</p></div>
        </section>
        """, unsafe_allow_html=True)

    with st.expander("Informasi model"):
        st.markdown("Estimasi memakai **HMM**, **XGBoost Quantile**, dan **Conformal Prediction** "
                    "untuk memprediksi harga satu hari setelah tanggal data acuan.")
        st.caption("HMM membaca kondisi pasar dengan raw filtering kausal. Model dilatih ulang saat data "
                   "diperbarui. Data dan bobotnya dapat berbeda dari eksperimen historis di Colab.")
    st.markdown("<div class='disclaimer-box'>Prototipe penelitian Tugas Akhir, Sekolah Vokasi UGM. "
                "Prediksi merupakan estimasi model dan bukan nasihat investasi.</div>", unsafe_allow_html=True)

def stretch_kwargs(widget):
    """Support both existing Streamlit >=1.40 and the tested 1.64 runtime."""
    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def page_tabs(comparison_only):
    labels = ["Prediksi", "Komparasi Model", "Analisis Data"]
    if "default" in inspect.signature(st.tabs).parameters:
        return st.tabs(labels, default="Komparasi Model" if comparison_only else "Prediksi")
    # Older Streamlit starts on the first tab. Keep offline results accessible
    # immediately, and return containers in the same semantic order as above.
    if comparison_only:
        comp, pred, analysis = st.tabs([labels[1], labels[0], labels[2]])
        return pred, comp, analysis
    return st.tabs(labels)


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Bitcoin Price Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PRESENTATION — restrained research workspace, Bitcoin orange.
# Native Streamlit controls retain their keyboard and screen-reader support.
# ============================================================
BG = "#f7f8fa"
SURFACE = "#ffffff"
SURFACE_2 = "#eef0f3"
BORDER = "#dfe3e8"
TEXT = "#202730"
TEXT_DIM = "#586473"
TEXT_MUTE = "#637080"
ACCENT = "#ee9b32"
ACCENT_SOFT = "rgba(248,161,56,0.14)"
# Forecasts use the product accent; red also marks data warnings.
TEAL = ACCENT
TEAL_SOFT = ACCENT_SOFT
GREEN = "#287451"
RED = "#b63e3e"
AMBER = "#875b16"
GRID = "#edf0f3"
REGIME_COLORS = {0: RED, 1: TEXT_MUTE, 2: GREEN}
REGIME_NAMES = {0: "Bear", 1: "Sideways", 2: "Bull"}
WARN_ICON = "<span class='notice-icon' aria-hidden='true'>!</span>"
MONTHS_ID = ("Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember")

def format_date_id(value):
    return f"{value.day} {MONTHS_ID[value.month - 1]} {value.year}"

st.markdown("""
<style>
/* Source stylesheet; embedded into dashboard.py at build time. */
:root {
  --font-sans: 'Segoe UI Variable', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'Cascadia Code', 'SFMono-Regular', Consolas, monospace;
  --ink: #202730;
  --muted: #586473;
  --line: #e1e5ea;
  --orange-ink: #9e570e;
  --orange: #ee9b32;
  --warning: #b42318;
  color-scheme: light;
}
html, body, .stApp { background: #f7f8fa; color: var(--ink); }
.stApp, button, input, select, textarea { font-family: var(--font-sans); }
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stMetric"], [role="tab"] { font-family: var(--font-sans); }
h1, h2, h3, h4 { font-family: var(--font-sans); color: var(--ink); text-wrap: balance; }
[data-testid="stHeader"] { background: #f7f8fa; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.block-container { max-width: 1240px; padding: 3.5rem 2.5rem 2rem; }
[data-testid="stVerticalBlock"] { gap: .9rem; }
[data-testid="stColumn"], [data-baseweb="tab-panel"] { min-width: 0; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: var(--muted); line-height: 1.65; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
[data-testid="stMarkdownContainer"] strong { color: var(--ink); }
code, .stMarkdown code { color: var(--orange-ink); background: #fff1df; }
.tnum { font-variant-numeric: tabular-nums; }

.topbar { display: flex; align-items: center; justify-content: space-between; gap: 20px; min-height: 46px; }
.topbar-brand { display: flex; align-items: center; gap: 12px; }
.topbar-coin { display: grid; place-items: center; width: 35px; height: 35px; background: var(--orange); color: #202730; border-radius: 9px; font-size: 24px; font-weight: 600; }
.topbar-title h1 { font-size: 17px; font-weight: 650; letter-spacing: -.5px; padding: 0; margin: 0; line-height: 1.3; }
.topbar-title p { margin: 2px 0 0; font-size: 12px; }
.topbar-meta { color: var(--muted); font: 11px var(--font-mono); white-space: nowrap; }
.masthead-rule { height: 1px; background: var(--line); margin: 4px 0 2px; }
.research-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding-top: 20px; border-top: 1px solid var(--line); margin-top: 24px; color: var(--muted); font-size: 11px; }
.welcome { padding: 28px 0 18px; }
.eyebrow { color: var(--orange-ink); font-size: 11px; font-weight: 600; letter-spacing: 1.1px; text-transform: uppercase; margin-bottom: 16px; }
.welcome h2 { font-size: clamp(30px, 3.1vw, 42px); font-weight: 600; line-height: 1.15; letter-spacing: -1.5px; margin: 0 0 18px; padding: 0; max-width: 620px; }
.welcome p { max-width: 440px; font-size: 15px; line-height: 1.7; margin: 0; }
.ready-panel { padding: 22px 24px; background: #fff; border: 1px solid var(--line); border-radius: 12px; margin-top: 26px; }
.ready-panel h3 { font-size: 15px; font-weight: 600; margin: 0 0 5px; padding: 0; }
.ready-panel .panel-caption { font-size: 12px; margin: 0 0 14px; }
.source-row { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 13px 0; border-top: 1px solid #edf0f3; }
.source-name { display: block; font-size: 13px; font-weight: 600; }
.source-detail { display: block; font-size: 12px; color: var(--muted); margin-top: 3px; }
.source-state { font-size: 11px; color: var(--muted); white-space: nowrap; }
.method-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px; padding: 22px 0; border-top: 1px solid var(--line); margin: 28px 0 0; }
.method-step { display: flex; gap: 12px; }
.method-index { font: 12px var(--font-mono); color: var(--orange-ink); padding-top: 3px; }
.method-step h3 { font-size: 13px; font-weight: 600; margin: 0 0 7px; padding: 0; }
.method-step p { font-size: 12px; margin: 0; max-width: 280px; }

.stTabs [role="tablist"] { gap: 26px; background: transparent; border-bottom: 1px solid var(--line); margin-bottom: 16px; }
.stTabs [role="tab"] { height: 43px; padding: 0 1px; border-radius: 0; color: var(--muted); background: transparent; font-weight: 550; }
.stTabs [role="tab"] p { font-size: 13px; white-space: nowrap; }
.stTabs [role="tab"][aria-selected="true"], .stTabs [role="tab"][aria-selected="true"] p { color: var(--orange-ink); }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"] { background: var(--orange) !important; height: 2px !important; }
.stTabs [data-baseweb="tab-border"] { background: transparent; }
.stTabs .stTabs [role="tablist"] { gap: 20px; margin-bottom: 10px; }
.stTabs .stTabs [role="tab"] { height: 36px; }
.page-header { margin: 0 0 5px; }
.page-header h2 { font-size: 27px; font-weight: 600; letter-spacing: -.8px; line-height: 1.25; padding: 0; margin: 0 0 8px; }
.page-header p { font-size: 13px; margin: 0; max-width: 65ch; }
.data-status { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; padding: 0 0 2px; color: var(--muted); font-size: 11px; }
.status-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; }
.status-pill .dot { width: 5px; height: 5px; border-radius: 50%; background: #637080; }
.status-live .dot { background: #287451; }
.status-stale .dot { background: #875b16; }
.status-stale { color: #805411; }
.status-date { margin-left: auto; font-size: 11px; }

/* One forecast surface; the primary estimate is intentionally dominant. */
.forecast-overview { background: #fff; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin: 4px 0 2px; }
.forecast-main { display: grid; grid-template-columns: 1fr 1fr; padding: 28px 30px; gap: 30px; }
.estimate { border-right: 1px solid #edf0f3; padding-right: 28px; }
.metric-label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 9px; }
.estimate-number { color: var(--ink); font-size: clamp(38px, 4vw, 50px); font-weight: 600; line-height: 1.1; letter-spacing: -2px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.estimate-change { margin-top: 11px; color: var(--muted); font-size: 12px; }
.estimate-change > span { font-weight: 600; margin-right: 5px; }
.range-number { display: flex; flex-wrap: wrap; align-items: baseline; gap: 7px; font-size: clamp(23px, 2.5vw, 30px); font-weight: 550; line-height: 1.3; letter-spacing: -.8px; font-variant-numeric: tabular-nums; }
.range-number > span { white-space: nowrap; }
.range-separator { color: var(--muted); font-size: 12px; font-weight: 400; letter-spacing: 0; }
.range-scale { display: flex; align-items: center; height: 18px; margin-top: 12px; gap: 7px; max-width: 310px; }
.range-end { height: 9px; width: 1px; background: #d7a369; }
.range-track { position: relative; height: 2px; flex: 1; background: #f0d2af; }
.range-track::after { content: ''; position: absolute; width: 7px; height: 7px; top: -2.5px; left: var(--median-position, 50%); transform: translateX(-50%); border-radius: 50%; background: #c3741d; }
.metric-foot { display: block; color: var(--muted); font-size: 12px; margin-top: 7px; line-height: 1.6; }
.forecast-context { display: flex; align-items: center; gap: 30px; padding: 16px 30px; background: #fafbfc; border-top: 1px solid #edf0f3; font-size: 12px; }
.forecast-context > div { display: flex; align-items: baseline; gap: 9px; }
.forecast-context span { color: var(--muted); }
.forecast-context strong { font-weight: 600; font-variant-numeric: tabular-nums; }
.forecast-context .regime-context { margin-left: auto; }
.delta-positive, .regime-bull { color: #287451; }
.delta-negative, .regime-bear { color: #ae3c3c; }
.regime-side { color: var(--muted); }
.section-label { color: var(--ink); font-size: 15px; font-weight: 600; margin: 18px 0 3px; letter-spacing: -.2px; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #fff; }
.detail-panel { padding: 4px 0 10px; }
.detail-panel dl { margin: 0; }
.detail-row { display: flex; justify-content: space-between; align-items: baseline; gap: 20px; padding: 12px 0; border-bottom: 1px solid #e8ecf0; font-size: 13px; }
.detail-row dt { color: var(--muted); }
.detail-row dd { margin: 0; text-align: right; color: var(--ink); font-weight: 550; font-variant-numeric: tabular-nums; }
.detail-note { color: var(--muted); font-size: 12px; line-height: 1.65; margin: 13px 0 0; max-width: 65ch; }
.market-item { padding: 12px 0; border-bottom: 1px solid #e8ecf0; }
.market-item:last-child { border: 0; }
.market-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; font-size: 13px; color: var(--muted); }
.market-heading strong { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.market-item p { font-size: 12px; margin: 5px 0 0; }

.info-box { padding: 14px 18px; background: #eef1f4; border-radius: 8px; color: var(--muted); font-size: 12px; line-height: 1.7; margin: 3px 0 6px; }
.info-box b { color: var(--ink); }
.info-box summary { cursor: pointer; font-weight: 500; color: var(--ink); }
.info-box .notice-body { margin-top: 10px; max-width: 90ch; }
/* Data warnings use a red symbol, a neutral surface, and readable dark text. */
.warn-box { padding: 14px 0; border-bottom: 1px solid var(--line); background: transparent; color: var(--ink); font-size: 13px; line-height: 1.6; margin: 0; }
.warn-box summary { display: flex; align-items: flex-start; gap: 10px; list-style: none; cursor: pointer; font-weight: 600; border-radius: 3px; }
.warn-box summary::-webkit-details-marker { display: none; }
.warn-box summary::marker { content: ''; }
.warn-box .notice-icon { flex: 0 0 17px; width: 17px; height: 17px; margin: 2px 0 0; color: var(--warning); font-size: 12px; line-height: 1; }
.warn-box .notice-title { flex: 1; min-width: 0; }
.warn-box summary::after { content: ''; flex: 0 0 6px; width: 6px; height: 6px; border-right: 1.5px solid var(--muted); border-bottom: 1.5px solid var(--muted); transform: rotate(45deg); margin: 5px 3px 0 8px; transition: transform .16s ease; }
.warn-box[open] summary::after { transform: rotate(225deg); margin-top: 8px; }
.warn-box summary:hover .notice-title { text-decoration: underline; text-underline-offset: 3px; }
.warn-box .notice-body { margin: 8px 24px 0 27px; max-width: 75ch; color: var(--muted); }
.about-dashboard { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr); gap: 24px 40px; padding: 8px 0 4px; }
.about-dashboard h3 { font-size: 13px; font-weight: 600; margin: 0 0 12px; padding: 0; }
.about-dashboard p { font-size: 13px; margin: 0 0 10px; max-width: 60ch; }
.about-sources { margin: 0; font-size: 13px; line-height: 1.65; }
.about-sources > div { display: grid; grid-template-columns: 72px minmax(0, 1fr); gap: 12px; margin-bottom: 6px; }
.about-sources dt { color: var(--muted); }
.about-sources dd { color: var(--ink); margin: 0; }
.disclaimer-box { color: var(--muted); font-size: 11px; line-height: 1.7; padding: 14px 0 0; margin-top: 8px; border-top: 1px solid var(--line); max-width: 100ch; }
.notice-icon { display: inline-grid; place-items: center; width: 13px; height: 13px; border: 1px solid currentColor; border-radius: 50%; font-size: 10px; font-weight: 650; margin-right: 6px; }
[data-testid="stExpander"] { background: transparent; border-color: var(--line); border-radius: 8px; }
[data-testid="stExpander"] summary { color: var(--ink); font-size: 12px; }
[data-testid="stExpander"] summary:hover { background: #eef1f4; }
[data-testid="stMetric"] { padding: 18px 20px; border-left: 2px solid #e4e8ed; background: #fff; }
[data-testid="stMetricValue"] { color: var(--ink); font-size: 27px; font-variant-numeric: tabular-nums; letter-spacing: -.7px; }
[data-testid="stMetricValue"] > div { white-space: normal; overflow-wrap: anywhere; }
.stButton button { border-radius: 7px; min-height: 42px; padding: 8px 17px; transition: background .16s ease, border-color .16s ease, transform .16s ease; }
.stButton button p { white-space: nowrap; font-size: 12px; }
.stButton button[kind="primary"] { background: var(--orange); border-color: var(--orange); color: #202730; }
.stButton button[kind="primary"] p { color: #202730; font-weight: 600; }
.stButton button[kind="primary"]:hover { background: #e38f26; border-color: #e38f26; }
.stButton button[kind="secondary"] { background: #fff; border-color: var(--line); color: var(--ink); }
.stButton button[kind="secondary"]:hover { background: #eef1f4; border-color: #b2bac4; }
.stButton button:active { transform: translateY(1px); }
button:focus-visible, [tabindex]:focus-visible, summary:focus-visible { outline: 2px solid #9e570e !important; outline-offset: 3px; }
[data-testid="stSelectbox"] label { color: var(--muted); font-size: 12px; }
[data-baseweb="select"] > div { background: #fff; border-color: var(--line); border-radius: 7px; }
[data-baseweb="select"] { color: var(--ink); font-size: 12px; }
[data-testid="stSpinner"] { color: var(--muted); padding: 24px 0; }
.table-wrap { max-width: 100%; overflow-x: auto; background: #fff; border: 1px solid var(--line); border-radius: 10px; margin: 6px 0 10px; }
table.analytics-table { width: 100%; min-width: 1020px; border-collapse: collapse; font-size: 12px; }
table.analytics-table th { text-align: left; font-weight: 500; color: var(--muted); background: #f0f3f5; padding: 12px 14px; white-space: nowrap; }
table.analytics-table td { padding: 14px; color: var(--ink); border-top: 1px solid #edf0f3; white-space: nowrap; }
table.analytics-table td:nth-child(n+5), table.analytics-table th:nth-child(n+5) { text-align: right; font-variant-numeric: tabular-nums; }
table.analytics-table tbody tr:has(td.model-usulan) { background: #fff7ec; }
table.analytics-table td.model-usulan { color: var(--orange-ink); font-weight: 650; }
table.compact-table { min-width: 850px; }
table.compact-table td:nth-child(n+2), table.compact-table th:nth-child(n+2) { text-align: right; font-variant-numeric: tabular-nums; }
table.compact-table th:first-child, table.compact-table td:first-child { min-width: 150px; }
table.compact-table tr:has(td.model-usulan) td:first-child { background: #fff7ec; }
table.analytics-table tbody tr:hover { background: #f2f4f6; }
table.analytics-table th:first-child, table.analytics-table td:first-child { position: sticky; left: 0; background: #fff; }
table.analytics-table th:first-child { background: #f0f3f5; }
[data-testid="stMarkdownContainer"]:has(table) { overflow-x: auto; }
[data-testid="stMarkdownContainer"] table:not(.analytics-table) { min-width: 500px; font-size: 12px; }
[data-testid="stMarkdownContainer"] table th, [data-testid="stMarkdownContainer"] table td { border-color: var(--line); }

@media (max-width: 1000px) {
  .block-container { padding: 3.5rem 1.5rem 2rem; }
  .forecast-main { padding: 24px; gap: 22px; }
  .forecast-context { padding: 15px 24px; gap: 20px; }
  .estimate-number { font-size: 42px; }
  .range-number { font-size: 25px; }
  .forecast-context > div { flex-wrap: wrap; gap: 3px 7px; }
  .topbar-meta { display: none; }
}
@media (max-width: 640px) {
  .block-container { padding: 3rem 1rem 1.5rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100% !important; width: 100% !important; min-width: 0 !important; }
  [data-testid="stHorizontalBlock"]:has(.topbar) { flex-wrap: nowrap !important; align-items: center; }
  [data-testid="stHorizontalBlock"]:has(.topbar) > [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; width: auto !important; }
  [data-testid="stHorizontalBlock"]:has(.topbar) > [data-testid="stColumn"]:last-child { flex: 0 0 108px !important; width: 108px !important; }
  .topbar-title p { display: none; }
  .topbar-title h1 { font-size: 16px; }
  .welcome { padding: 16px 0 4px; }
  .welcome h2 { font-size: 31px; letter-spacing: -1px; }
  .ready-panel { margin-top: 8px; padding: 20px; }
  .method-strip { grid-template-columns: 1fr; gap: 20px; margin-top: 12px; }
  .method-step p { max-width: none; }
  .forecast-main { grid-template-columns: 1fr; gap: 22px; padding: 22px; }
  .estimate { border-right: 0; border-bottom: 1px solid #edf0f3; padding: 0 0 20px; }
  .estimate-number { font-size: 43px; }
  .range-number { font-size: 26px; }
  .forecast-context { flex-wrap: wrap; padding: 15px 22px; gap: 14px 24px; }
  .forecast-context > div { flex-direction: column; gap: 4px; }
  .forecast-context .regime-context { margin-left: 0; }
  .page-header h2 { font-size: 24px; }
  .stTabs [role="tablist"] { gap: 20px; }
  .stTabs [role="tab"] p { font-size: 12px; }
  .data-status { gap: 8px 12px; }
  .status-date { margin-left: 0; width: 100%; }
  .detail-row { gap: 15px; font-size: 12px; }
  .about-dashboard { grid-template-columns: 1fr; gap: 22px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}

</style>
""", unsafe_allow_html=True)


def render_table(df: pd.DataFrame, label='Hasil evaluasi model'):
    table_class = "analytics-table compact-table" if df.columns[0] == "Model" else "analytics-table"
    html = df.to_html(index=False, escape=True, classes=table_class, border=0)
    html = html.replace("<td>Model Usulan</td>", "<td class='model-usulan'>Model Usulan</td>")
    st.markdown(
        f"<div class='table-wrap' role='region' aria-label='{escape(label, quote=True)}' tabindex='0'>{html}</div>",
        unsafe_allow_html=True,
    )


def render_chart(fig, **kwargs):
    """One visual language for charts; only presentation settings change."""
    margins = fig.layout.margin.to_plotly_json()
    for side, minimum in {"l": 16, "r": 16, "t": 40, "b": 16}.items():
        margins[side] = max(margins.get(side) or 0, minimum)
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=margins,
        font=dict(family="Segoe UI, sans-serif", color=TEXT, size=12),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_color=TEXT),
        modebar=dict(bgcolor="rgba(255,255,255,0)", color=TEXT_MUTE, activecolor=ACCENT),
    )
    fig.update_xaxes(zeroline=False, showline=False, gridcolor=GRID, automargin=True)
    fig.update_yaxes(zeroline=False, showline=False, gridcolor=GRID, automargin=True)
    st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart), theme=None,
                    config={"displaylogo": False, "scrollZoom": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, **kwargs)


# ============================================================
# FUNGSI FETCH DATA (di-cache 1 jam)
# ============================================================
@st.cache_data(ttl=3600)
def fetch_price(start="2021-01-01"):
    import yfinance as yf
    cache_dir = Path(__file__).resolve().parent / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    df = yf.download("BTC-USD", start=start, interval="1d", progress=False)
    if df.empty:
        raise RuntimeError("Yahoo Finance tidak mengembalikan data BTC-USD")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[["date","Open","High","Low","Close","Volume"]].set_index("date")
    df.columns = ["open","high","low","close","volume"]
    df = df.dropna()

    today_utc = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df.index < today_utc]  # buang baris "hari ini" (live/belum final)
    return df

@st.cache_data(ttl=3600)
def fetch_sentiment():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=15)
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df = df[["date","value","value_classification"]].set_index("date").sort_index()
        mapping = {"Extreme Fear":-1,"Fear":-0.5,"Neutral":0,"Greed":0.5,"Extreme Greed":1}
        df["polarity"] = df["value_classification"].map(mapping)
        df["polarity"] = df["polarity"].fillna((df["value"].astype(int)-50)/50)
        return df, True
    except Exception:
        return pd.DataFrame(), False

@st.cache_data(ttl=3600)
def fetch_onchain(start="2021-01-01"):
    """
    Mengambil data on-chain (exchange netflow).

    Prioritas:
    1. Jika COINMETRICS_API_KEY tersedia (st.secrets atau environment
       variable) -> pakai endpoint CoinMetrics Pro (live, harian).
    2. Jika tidak -> fallback ke CSV Community di GitHub. Kesegaran metrik
       diperiksa dari nilai netflow tidak kosong yang benar-benar tersedia;
       tanggal baris CSV terakhir dapat lebih baru daripada nilai netflow.

    Return: (df, is_live)
        df       : dataframe dengan kolom exchange_netflow, FlowInExNtv, FlowOutExNtv
        is_live  : True jika data diambil dari jalur live/premium
    """
    api_key = None
    try:
        api_key = st.secrets.get("COINMETRICS_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("COINMETRICS_API_KEY")

    if api_key:
        try:
            url = "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
            params = {
                "assets": "btc",
                "metrics": "FlowInExNtv,FlowOutExNtv",
                "start_time": start,
                "frequency": "1d",
                "page_size": 10000,
                "api_key": api_key,
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()["data"]
            if len(payload) > 0:
                df = pd.DataFrame(payload)
                df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                df = df.set_index("time").sort_index()
                df["FlowInExNtv"]  = pd.to_numeric(df["FlowInExNtv"], errors="coerce")
                df["FlowOutExNtv"] = pd.to_numeric(df["FlowOutExNtv"], errors="coerce")
                df["exchange_netflow"] = df["FlowOutExNtv"] - df["FlowInExNtv"]
                df = df[["exchange_netflow","FlowInExNtv","FlowOutExNtv"]].loc[start:]
                if len(df.dropna()) > 0:
                    return df, True
        except Exception:
            pass  # jatuh ke fallback gratis di bawah

    # --- Fallback: CSV Community gratis (mungkin sudah berhenti update) ---
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    r = requests.get(url, timeout=30)
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.set_index("time")
    df["exchange_netflow"] = df["FlowOutExNtv"] - df["FlowInExNtv"]
    return df[["exchange_netflow","FlowInExNtv","FlowOutExNtv"]].loc[start:], False

# ============================================================
# FUNGSI FEATURE ENGINEERING + MODEL
# ============================================================
@st.cache_data(ttl=3600)
def build_features_and_predict():
    prices = fetch_price()
    sentiment, real = fetch_sentiment()
    if not real:
        raise RuntimeError("Sentimen riil tidak tersedia. Prediksi tidak dijalankan.")
    onchain, live = fetch_onchain()
    return build_forecast(prices, sentiment, onchain, live)

# ============================================================
# STATE — apakah pipeline/model sudah pernah dijalankan di sesi ini
# ------------------------------------------------------------
# Model baru dijalankan setelah tombol ditekan. Komparasi tersimpan
# dapat dibuka terpisah tanpa mengakses API atau melatih model.
# ============================================================
if "comparison_only" not in st.session_state:
    st.session_state.comparison_only = False

if "dashboard_started" not in st.session_state:
    st.session_state.dashboard_started = False

# ============================================================
# TOPBAR — branding (bagian yang tidak butuh data dulu)
# ============================================================
def start_dashboard():
    if st.session_state.dashboard_started:
        st.cache_data.clear()
    st.session_state.dashboard_started = True
    st.session_state.comparison_only = False


def open_comparison():
    st.session_state.comparison_only = True


topbar_col, refresh_col = st.columns([4, 1], vertical_alignment="center")
with topbar_col:
    st.markdown("""
    <header class='topbar'>
        <div class='topbar-brand'>
            <div class='topbar-coin' aria-hidden='true'>₿</div>
            <div class='topbar-title'>
                <h1>BTC Dashboard</h1>
                <p>Estimasi harga &amp; kondisi pasar</p>
            </div>
        </div>
        <span class='topbar-meta'>BTC / USD</span>
    </header>
    """, unsafe_allow_html=True)
with refresh_col:
    if st.session_state.dashboard_started:
        st.button("Refresh Data", **stretch_kwargs(st.button), on_click=start_dashboard,
                  help="Ambil ulang data dan jalankan model tanpa menunggu cache 1 jam.")
    elif st.session_state.comparison_only:
        st.button("Jalankan Prediksi", on_click=start_dashboard, **stretch_kwargs(st.button))
    else:
        st.markdown("<div class='topbar-meta' style='text-align:right'>UGM</div>",
                    unsafe_allow_html=True)
st.markdown("<div class='masthead-rule'></div>", unsafe_allow_html=True)

if not st.session_state.dashboard_started and not st.session_state.comparison_only:
    intro, readiness = st.columns([1.45, 1], gap="large")
    with intro:
        st.markdown("""
        <section class='welcome'>
            <div class='eyebrow'>Prediksi 1 hari ke depan</div>
            <h2>Harga Bitcoin berikutnya,<br>dengan rentang prediksinya.</h2>
            <p>Lihat estimasi harga satu hari ke depan, rentang prediksi,
            dan kondisi pasar dari data yang tersedia.</p>
        </section>
        """, unsafe_allow_html=True)
        st.button("Muat & Jalankan Model", type="primary", on_click=start_dashboard,
                  help="Mengambil data harga, sentimen, dan on-chain, lalu melatih model prediksi.")
        st.button("Lihat Hasil Eksperimen", on_click=open_comparison,
                  help="Buka hasil Colab terverifikasi tanpa mengambil data atau melatih model.")
        st.caption("Data diperiksa saat prediksi dijalankan. Komparasi model dapat dibuka langsung.")
    with readiness:
        st.markdown("""
        <aside class='ready-panel' aria-label='Kesiapan sumber data'>
            <h3>Data yang digunakan</h3>
            <p class='panel-caption'>Diperiksa saat Anda menjalankan prediksi.</p>
            <div class='source-row'><div><span class='source-name'>Harga Bitcoin</span>
            <span class='source-detail'>Yahoo Finance · BTC-USD</span></div>
            <span class='source-state'>Belum dimuat</span></div>
            <div class='source-row'><div><span class='source-name'>Sentimen pasar</span>
            <span class='source-detail'>Crypto Fear &amp; Greed Index</span></div>
            <span class='source-state'>Belum dimuat</span></div>
            <div class='source-row'><div><span class='source-name'>Aktivitas on-chain</span>
            <span class='source-detail'>CoinMetrics · Exchange netflow</span></div>
            <span class='source-state'>Belum dimuat</span></div>
        </aside>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <section class='method-strip' aria-label='Alur model prediksi'>
        <article class='method-step'><span class='method-index'>01</span><div>
        <h3>Kenali kondisi pasar</h3><p>Lihat klasifikasi kondisi pasar dari
        data harga historis.</p></div></article>
        <article class='method-step'><span class='method-index'>02</span><div>
        <h3>Estimasi rentang harga</h3><p>Baca estimasi tengah bersama batas
        bawah dan atas prediksi.</p></div></article>
        <article class='method-step'><span class='method-index'>03</span><div>
        <h3>Pahami ketidakpastian</h3><p>Rentang prediksi memakai target
        cakupan 90%; hasil aktual dapat berbeda.</p></div></article>
    </section>
    <div class='disclaimer-box'>
    {WARN_ICON}<b>Disclaimer:</b> Dashboard ini merupakan prototipe akademik sebagai bagian dari
    Tugas Akhir Program Studi Teknologi Rekayasa Perangkat Lunak, Universitas Gadjah Mada.
    Prediksi yang ditampilkan <b>bukan merupakan nasihat investasi</b> dan tidak boleh
    dijadikan dasar keputusan finansial.
    </div>
    <footer class='research-footer'><span>Huda Muhammad Nur · Sekolah Vokasi UGM</span>
    <span>Penelitian Tugas Akhir · 2026</span></footer>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================
# LOAD DATA (dengan spinner)
# ============================================================
DATA_OK = False
if st.session_state.dashboard_started:
    with st.spinner("Mengambil data dan menyiapkan prediksi..."):
        try:
            result = build_features_and_predict()
            df_full = result["df"]
            sent_real = True
            DATA_OK = True
        except Exception as error:
            st.error("Data belum berhasil dimuat. Periksa koneksi, lalu tekan Refresh Data untuk mencoba lagi.")
            st.caption(f"Jenis kendala: {type(error).__name__}. Komparasi offline tetap tersedia.")

if DATA_OK:
    price_last_date = result["last_date"]
    price_staleness_days = (pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timestamp(price_last_date.date())).days
    is_price_fresh = price_staleness_days <= 1
    onchain_stale_days = result["onchain_staleness_days"]
    is_onchain_fresh = onchain_stale_days <= 1
    price_status = "Harga terbaru" if is_price_fresh else f"Harga tertinggal {price_staleness_days} hari"
    chain_status = "On-chain terbaru" if is_onchain_fresh else f"Netflow historis · {onchain_stale_days} hari"
    st.markdown(f"""
    <div class='data-status' aria-label='Kesegaran data'>
        <span class='status-pill {"status-live" if is_price_fresh else "status-stale"}'>{price_status}</span>
        <span class='status-pill {"status-live" if is_onchain_fresh else "status-stale"}'>{chain_status}</span>
        <span class='status-date'>Data acuan · {format_date_id(price_last_date)}</span>
    </div>
    """, unsafe_allow_html=True)

tab_pred, tab_comp, tab_data = page_tabs(st.session_state.comparison_only)

# ============================================================
# HALAMAN 1: PREDIKSI
# ============================================================
with tab_pred:
    if not DATA_OK:
        st.info("Jalankan prediksi untuk memuat data terbaru. Hasil Colab tetap tersedia pada tab Komparasi Model.")
    else:
        render_prediction(result, render_chart)

# ============================================================
# HALAMAN 2: KOMPARASI MODEL
# ============================================================
with tab_comp:
    render_experiments(render_table, render_chart, ACCENT)

# ============================================================
# HALAMAN 3: ANALISIS DATA
# ============================================================
with tab_data:
    if not DATA_OK:
        st.info("Jalankan prediksi untuk memuat data terbaru. Hasil Colab tetap tersedia pada tab Komparasi Model.")
    else:
        st.markdown("""
        <div class='page-header'>
            <h2>Analisis Data Historis</h2>
            <p>Harga BTC · Sentimen Fear & Greed · On-Chain Netflow · Regime HMM</p>
        </div>
        """, unsafe_allow_html=True)

        col_r1, col_r2 = st.columns([3, 1])
        with col_r2:
            period = st.selectbox("Periode", ["6 Bulan","1 Tahun","2 Tahun","Semua"], index=1)

        period_map = {"6 Bulan":180, "1 Tahun":365, "2 Tahun":730, "Semua":9999}
        days       = period_map[period]
        cutoff     = df_full.index[-1] - timedelta(days=days)
        df_view    = df_full[df_full.index >= cutoff].copy()

        st.markdown("<div class='section-label'>Harga & Regime Pasar (HMM)</div>",
                    unsafe_allow_html=True)

        regime_colors = REGIME_COLORS
        regime_names  = REGIME_NAMES

        fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.75, 0.25],
                                   vertical_spacing=0.03)
        fig_price.add_trace(
            go.Scatter(x=df_view.index, y=df_view["close"],
                       mode="lines", name="Harga BTC",
                       line=dict(color=TEXT, width=1.5)),
            row=1, col=1
        )
        for regime_id, color in regime_colors.items():
            mask = df_view["regime_labeled"] == regime_id
            segments = df_view[mask]
            if len(segments) > 0:
                fig_price.add_trace(
                    go.Bar(x=segments.index,
                           y=[1]*len(segments),
                           name=regime_names[regime_id],
                           marker_color=color, opacity=0.6,
                           showlegend=True),
                    row=2, col=1
                )

        fig_price.update_layout(
            template="plotly_white", paper_bgcolor=BG,
            plot_bgcolor=BG, height=420,
            font=dict(color=TEXT),
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(color=TEXT_DIM)),
            yaxis=dict(tickprefix="$", tickformat=",", gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
            yaxis2=dict(showticklabels=False, gridcolor=GRID),
            xaxis2=dict(gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
            barmode="stack"
        )
        render_chart(fig_price)

        col_s, col_n = st.columns(2)

        with col_s:
            st.markdown("<div class='section-label'>Sentimen — Fear & Greed Index</div>",
                        unsafe_allow_html=True)
            sent_view = df_view.dropna(subset=["polarity"])
            fig_sent  = go.Figure()
            fig_sent.add_trace(go.Bar(
                x=sent_view.index,
                y=sent_view["polarity"],
                marker_color=[GREEN if v > 0 else RED
                              for v in sent_view["polarity"]],
                name="Polarity"
            ))
            fig_sent.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTE)
            fig_sent.update_layout(
                template="plotly_white", paper_bgcolor=BG,
                plot_bgcolor=BG, height=280,
                font=dict(color=TEXT),
                margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(title="Polarity (-1 to +1)", gridcolor=GRID, tickfont=dict(color=TEXT_DIM),
                           tickvals=[-1,-0.5,0,0.5,1],
                           ticktext=["Ext Fear","Fear","Neutral","Greed","Ext Greed"]),
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT_DIM))
            )
            if not sent_real:
                st.warning("Sentimen riil tidak tersedia")
            render_chart(fig_sent)

        with col_n:
            st.markdown("<div class='section-label'>On-Chain — Exchange Netflow</div>",
                        unsafe_allow_html=True)
            st.caption("Konvensi fitur penelitian: outflow − inflow; nilai positif berarti arus keluar bersih dari bursa.")
            if not is_onchain_fresh:
                st.warning(
                    f"Data on-chain historis terakhir: "
                    f"{format_date_id(result['onchain_last_real_date'])}. "
                    f"Bagian setelah tanggal ini adalah nilai forward-fill "
                    f"(bukan data riil baru)."
                )
            nf_view = df_view.dropna(subset=["exchange_netflow"])
            fig_nf  = go.Figure()
            fig_nf.add_trace(go.Bar(
                x=nf_view.index,
                y=nf_view["exchange_netflow"],
                marker_color=[GREEN if v > 0 else RED if v < 0 else TEXT_MUTE
                              for v in nf_view["exchange_netflow"]],
                name="Netflow"
            ))
            fig_nf.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTE)
            if not is_onchain_fresh and result["onchain_last_real_date"] >= cutoff:
                last_real_dt = result["onchain_last_real_date"].to_pydatetime()
                fig_nf.add_shape(
                    type="line", xref="x", yref="paper",
                    x0=last_real_dt, x1=last_real_dt, y0=0, y1=1,
                    line=dict(dash="dot", color=AMBER, width=1.5)
                )
                fig_nf.add_annotation(
                    x=last_real_dt, y=1, xref="x", yref="paper",
                    text="Data riil terakhir", showarrow=False,
                    yanchor="bottom", font=dict(color=AMBER, size=11)
                )
            fig_nf.update_layout(
                template="plotly_white", paper_bgcolor=BG,
                plot_bgcolor=BG, height=280,
                font=dict(color=TEXT),
                margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(title="Netflow (BTC)", gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT_DIM))
            )
            render_chart(fig_nf)

        st.markdown("<div class='section-label'>Fitur Usulan — Sentiment-Weighted Netflow</div>",
                    unsafe_allow_html=True)
        nfw_view = df_view.dropna(subset=["netflow_weighted"])
        fig_nfw  = go.Figure()
        fig_nfw.add_trace(go.Scatter(
            x=nfw_view.index, y=nfw_view["netflow_weighted"],
            mode="lines", fill="tozeroy",
            line=dict(color=TEAL, width=1),
            fillcolor=TEAL_SOFT,
            name="Netflow × (1 + Polarity)"
        ))
        fig_nfw.add_trace(go.Scatter(
            x=nfw_view.index, y=nfw_view["exchange_netflow"],
            mode="lines", line=dict(color=TEXT_DIM, width=1, dash="dot"),
            name="Netflow mentah"
        ))
        fig_nfw.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTE)
        fig_nfw.update_layout(
            template="plotly_white", paper_bgcolor=BG,
            plot_bgcolor=BG, height=260,
            font=dict(color=TEXT),
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(color=TEXT_DIM)),
            yaxis=dict(title="BTC", gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
            xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT_DIM))
        )
        render_chart(fig_nfw)
        st.markdown(f"""
        <div class='info-box'>
        <b>Fitur Usulan — Sentiment-Weighted Netflow</b>: Netflow on-chain dikalikan
        dengan bobot sentimen <code>(1 + polarity)</code>. Ketika sentimen Extreme Fear
        (polarity = −1), bobot = 0 sehingga sinyal netflow dilemahkan. Ketika Extreme Greed
        (polarity = +1), bobot = 2 sehingga magnitudo fitur diperbesar. Ini
        adalah transformasi fitur yang diuji melalui ablasi, bukan bukti bahwa
        polaritas menentukan penyebab atau arah perpindahan BTC.
        </div>
        """, unsafe_allow_html=True)

with st.expander("Tentang dashboard ini"):
    st.markdown(f"""
    <section class='about-dashboard' aria-label='Tentang dashboard'>
        <div>
            <h3>Sumber data</h3>
            <dl class='about-sources'>
                <div><dt>Harga</dt><dd>Yahoo Finance</dd></div>
                <div><dt>Sentimen</dt><dd>Crypto Fear &amp; Greed Index</dd></div>
                <div><dt>On-chain</dt><dd>CoinMetrics</dd></div>
            </dl>
        </div>
        <div>
            <h3>Model prediksi</h3>
            <p>HMM dengan raw filtering kausal, XGBoost Quantile, dan Conformal Prediction.</p>
            <p>Prediksi dihitung saat model dijalankan. Tab Komparasi Model menampilkan hasil eksperimen historis.</p>
        </div>
    </section>
    <div class='disclaimer-box'>
    {WARN_ICON}<b>Bukan nasihat investasi.</b> Dashboard ini merupakan prototipe
    akademik bagian dari Tugas Akhir Program Studi Teknologi Rekayasa
    Perangkat Lunak, Universitas Gadjah Mada.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<footer class='research-footer'><span>Huda Muhammad Nur · Sekolah Vokasi UGM</span>
<span>Penelitian Tugas Akhir · 2026</span></footer>
""", unsafe_allow_html=True)
