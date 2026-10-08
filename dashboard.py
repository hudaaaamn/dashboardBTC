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

# Frozen Colab results: 10 runs, 84 metric rows, 410 statistical rows.
# Base64 encodes compressed JSON data only; it never executes code.
_EXPERIMENTS_SHA256 = "d8d6969ee5546e05252e0a545128d37f728a308baeb655b861e48bd852d61c2a"
_EXPERIMENTS_B64 = (
    "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"
    "zCk="
)

FEATURE_COLS = [
    "log_return", "volatility_7d", "volatility_14d", "volatility_30d",
    "price_to_ma7", "price_to_ma30", "return_lag_1", "return_lag_2", "return_lag_3",
    "exchange_netflow", "netflow_change", "netflow_ma_7", "polarity",
    "sentiment_ma_7", "netflow_weighted", "regime_labeled",
]
CORE_VERSION = "2026.10.08-singlefile-v2"


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

BASELINES = ["XGBoost", "LightGBM", "Random Forest", "SVR", "LSTM", "Model Usulan", "Naive"]
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
    if run["family_size"]:
        st.caption(f"Stationary bootstrap {run['bootstrap_resamples']:,} resample · blok {block} hari · "
                   f"Holm{run['family_size']} · CI95 basic pointwise, bukan simultan. "
                   "Blok 14 adalah analisis utama; 7/28 adalah sensitivitas.")
        if run["kind"] == "multiwindow":
            st.caption("Holm45 mencakup tiga fold per varian; memilih tahun tidak menghitung ulang koreksi. "
                       "Naive tidak termasuk keluarga uji inferensial.")
    else:
        st.warning("Arsip bootstrap iid harian N=5000 tanpa koreksi uji berganda. "
                   "Kesimpulan nominal bersifat eksploratif dan berbeda dari stationary bootstrap/Holm.")
    st.caption("Selisih adalah Model Usulan (FULL) dikurangi pembanding. MAE/pinball dalam USD; "
               "DirAcc/coverage dalam poin persentase. Tidak terdeteksi bukan bukti kesetaraan.")


def render_ablation(run, table, render_table, suffix=""):
    if run["kind"] == "archive":
        st.info("Ablasi arsip awal memakai 387 baris; tidak dicampur dengan tabel 373 tanggal. "
                "Pilih hasil kausal atau reproduksi historis untuk ablasi pada tanggal bersama.")
        return
    metrics = pd.DataFrame(run["metrics"])
    ablation = metrics[metrics.Model.isin(["Model Usulan", "Tanpa Regime (HMM)", "Netflow Mentah"])]
    render_table(display_metrics(ablation[["Model", "n"] + METRICS]), label="Metrik ablasi" + suffix)
    render_table(table[table.Pembanding.isin(["Tanpa Regime (HMM)", "Netflow Mentah"])],
                 label="Uji ablasi" + suffix)
    st.caption("Ablasi dan statistik memakai run serta tanggal uji yang sama. Baca arah selisih bersama p Holm. "
               "Bootstrap hari bersyarat pada model yang telah difit dan tidak mencakup seluruh variasi pelatihan ulang.")


def render_multiwindow(data, variant, render_table, render_chart, accent, chart_key="multiwindow_primary"):
    summary = []
    for run in data["runs"].values():
        if run["kind"] == "multiwindow" and run["variant"] == variant:
            full = next(row for row in run["metrics"] if row["Model"] == "Model Usulan")
            naive = next(row for row in run["metrics"] if row["Model"] == "Naive")
            summary.append({"Tahun": run["year"], "n": run["n"], "MAE FULL (USD)": full["MAE"],
                "MAE Naive (USD)": naive["MAE"], "Δ MAE (USD)": full["MAE"] - naive["MAE"],
                "Coverage FULL (%)": full["Coverage"]})
    overview = pd.DataFrame(summary).sort_values("Tahun")
    render_table(overview.round(4), label="Ringkasan multiwindow")
    if (overview["Δ MAE (USD)"] > 0).all():
        st.warning("Model Usulan belum mengungguli baseline harga tetap secara nominal pada ketiga periode. "
                   "Perbandingan Naive bersifat deskriptif, belum diuji signifikansinya.")
    st.caption("Naive memakai harga hari ini sebagai prediksi besok. DirAcc 0% pada artefak mengikuti "
               "aturan kecocokan tanda untuk return prediksi nol; bukan baseline tebak acak 50%.")
    coverage = go.Figure(go.Bar(x=overview.Tahun.astype(str), y=overview["Coverage FULL (%)"],
                                marker_color=accent, text=overview["Coverage FULL (%)"].round(2), textposition="outside"))
    coverage.add_hline(y=90, line_dash="dash", line_color="#637080")
    coverage.update_layout(title="Coverage per fold · garis target 90%", xaxis_type="category",
                           xaxis_title="Tahun", yaxis_ticksuffix="%", height=300)
    render_chart(coverage, key=chart_key)


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
        st.caption("Coverage empiris pada run terpilih; target nominal 90%. Ini tidak menjamin cakupan live "
                   "atau cakupan sama pada setiap periode, khususnya ketika netflow tertinggal.")


def render_experiments(render_table, render_chart, accent):
    """Present one main result; disclose sensitivity/history only on request."""
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
    render_table(display_metrics(main_comparison_frame(primary)), label="Hasil evaluasi utama · raw kausal")

    # Keep technical selectors inside collapsed sections, as in the original UI.
    with st.expander("Uji Signifikansi Statistik (stationary bootstrap, N=20000, Holm15, CI 95%)"):
        block = st.selectbox("Panjang blok rata-rata (hari)", [14, 7, 28], key="main_bootstrap_block")
        render_inference_note(primary, block)
        table = selected_statistics(primary, block)
        render_table(table[table.Metrik.eq("MAE") & table.Pembanding.isin(BASELINES)], label="Uji MAE utama")

    with st.expander("Ablation Study (Kontribusi HMM & Sentiment-Weighted Netflow)"):
        st.caption("Hasil kausal utama · 373 tanggal bersama · blok 14 hari · Holm15.")
        render_ablation(primary, selected_statistics(primary, 14), render_table)

    render_metric_charts(primary, render_chart, accent)

    with st.expander("Uji lintas periode 2023–2025 dan baseline Naive"):
        st.caption("Pemeriksaan tambahan hasil kausal raw pada tiga backtest. Setiap fold memiliki 170 tanggal uji. "
                   "Angka fold tidak digabung ke tabel utama 373 tanggal. Rincian inferensial memakai Holm45.")
        render_multiwindow(data, "raw_filter", render_table, render_chart, accent)

    with st.expander("Rincian eksperimen, sensitivitas, dan sumber data"):
        st.caption("Bagian ini untuk memeriksa varian, arsip, dan asal angka. Pilihan di sini tidak mengubah "
                   "tabel/grafik utama di atas atau melatih model live.")
        run_id = st.selectbox("Hasil eksperimen", list(data["runs"]),
                             format_func=lambda key: data["runs"][key]["label"], key="experiment_run")
        run = data["runs"][run_id]
        st.caption(f"{run['label']} · {run['n']} tanggal fitur · {run['feature_start']}–{run['feature_end']} · "
                   f"target terakhir {run['target_end']} · data dikunci {run['run_date_lock']} (eksklusif).")
        st.caption(run["note"])
        metrics = pd.DataFrame(run["metrics"])
        main = metrics[metrics.Model.isin(BASELINES)]
        render_table(display_metrics(main[["Model", "n"] + METRICS]), label="Rincian evaluasi · " + run_id)
        st.download_button("Unduh metrik CSV", metrics.to_csv(index=False).encode("utf-8"),
                           file_name=f"metrics_{run_id}.csv", mime="text/csv", key="download_metrics")
        block = st.selectbox("Blok sensitivitas (hari)", [14, 7, 28], key="bootstrap_block") if run["family_size"] else None
        render_inference_note(run, block)
        table = selected_statistics(run, block)
        render_table(table[table.Metrik.eq("MAE") & table.Pembanding.isin(BASELINES)], label="Uji MAE rincian")
        render_ablation(run, table, render_table, " · rincian")
        if run["kind"] == "multiwindow":
            render_multiwindow(data, run["variant"], render_table, render_chart, accent, chart_key="multiwindow_detail")
        st.code(json.dumps({key: run[key] for key in ["completed_utc", "metric_source", "statistics_source", "versions"]},
                           indent=2, ensure_ascii=False), language="json")
        st.code(json.dumps({path: data["source_sha256"][path] for path in [run["metric_source"], run["statistics_source"]]},
                           indent=2), language="json")

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
ACCENT = "#f8a138"
ACCENT_SOFT = "rgba(248,161,56,0.14)"
# Forecasts use the same product accent; red/green only encode market states.
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

st.markdown(f"""
<style>
:root {{
    --font-sans: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'Cascadia Code', 'SFMono-Regular', Consolas, monospace;
    color-scheme: light;
}}
html, body, .stApp {{ background:{BG}; color:{TEXT}; }}
.stApp, button, input, select, textarea {{ font-family:var(--font-sans); }}
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stMetric"], [role="tab"] {{ font-family:var(--font-sans); }}
h1, h2, h3, h4 {{ font-family:var(--font-sans); color:{TEXT}; }}
[data-testid="stHeader"] {{ background:{BG}; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none; }}
.block-container {{ max-width:1320px; padding:4rem 3rem 3rem; }}
[data-testid="stVerticalBlock"] {{ gap:1rem; }}
[data-testid="stColumn"], [data-baseweb="tab-panel"] {{ min-width:0; }}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ color:{TEXT_DIM}; line-height:1.65; }}
[data-testid="stMarkdownContainer"] strong {{ color:{TEXT}; }}
[data-testid="stCaptionContainer"] {{ color:{TEXT_MUTE}; }}
[data-testid="stCaptionContainer"] p {{ color:{TEXT_MUTE}; }}
code, .stMarkdown code {{ color:{ACCENT}; background:{ACCENT_SOFT}; }}
.tnum {{ font-variant-numeric:tabular-nums; }}

/* A quiet masthead, not another floating card. */
.topbar {{ display:flex; align-items:center; justify-content:space-between;
    gap:24px; min-height:48px; margin-bottom:0; }}
.topbar-brand {{ display:flex; align-items:center; gap:13px; }}
.topbar-coin {{ display:grid; place-items:center; width:40px; height:40px;
    background:{ACCENT}; color:{TEXT}; border-radius:11px; font-size:27px; font-weight:600; }}
.topbar-title h1 {{ font-size:19px; font-weight:650; letter-spacing:-.6px;
    padding:0; margin:0; line-height:1.3; }}
.topbar-title p {{ margin:3px 0 0; font-size:12px; color:{TEXT_DIM}; }}
.topbar-meta {{ color:{TEXT_MUTE}; font:12px var(--font-mono); white-space:nowrap; }}
.masthead-rule {{ height:1px; background:{BORDER}; margin:0 0 12px; }}
.research-footer {{ display:flex; justify-content:space-between; flex-wrap:wrap;
    gap:10px; padding-top:22px; border-top:1px solid {BORDER}; margin-top:24px;
    color:{TEXT_MUTE}; font-size:12px; }}

/* Start state: honest data readiness, no invented quotes or prices. */
.welcome {{ padding:38px 0 22px; }}
.eyebrow {{ color:{ACCENT}; font:600 11px var(--font-mono);
    letter-spacing:1.3px; text-transform:uppercase; margin-bottom:17px; }}
.welcome h2 {{ font-size:clamp(30px,3.4vw,45px); font-weight:600; line-height:1.16;
    letter-spacing:-1.8px; margin:0 0 20px; padding:0; max-width:650px; }}
.welcome p {{ max-width:510px; font-size:15px; line-height:1.7; margin:0; }}
.ready-panel {{ padding:26px 28px; background:{SURFACE}; border:1px solid {BORDER};
    border-radius:12px; margin-top:34px; }}
.ready-panel h3 {{ font-size:15px; font-weight:600; margin:0 0 6px; padding:0; }}
.ready-panel .panel-caption {{ font-size:12px; margin:0 0 16px; color:{TEXT_MUTE}; }}
.source-row {{ display:flex; justify-content:space-between; align-items:center;
    gap:16px; padding:14px 0; border-top:1px solid {GRID}; }}
.source-name {{ display:block; font-size:13px; font-weight:600; }}
.source-detail {{ display:block; font-size:12px; color:{TEXT_MUTE}; margin-top:3px; }}
.source-state {{ font:11px var(--font-mono); color:{TEXT_MUTE}; white-space:nowrap; }}
.method-strip {{ display:grid; grid-template-columns:repeat(3,1fr); gap:28px;
    padding:26px 0; border-top:1px solid {BORDER}; border-bottom:1px solid {BORDER};
    margin:28px 0 12px; }}
.method-step {{ display:flex; gap:14px; }}
.method-index {{ font:12px var(--font-mono); color:{ACCENT}; padding-top:3px; }}
.method-step h3 {{ font-size:14px; font-weight:600; margin:0 0 7px; padding:0; }}
.method-step p {{ font-size:12.5px; margin:0; max-width:280px; }}

/* Native navigation: one continuous baseline and a clear active state. */
.stTabs [role="tablist"] {{ gap:26px; background:transparent;
    border-bottom:1px solid {BORDER}; margin-bottom:18px; }}
.stTabs [role="tab"] {{ height:48px; padding:0 2px; border-radius:0;
    color:{TEXT_DIM}; background:transparent; font-weight:550; }}
.stTabs [role="tab"] p {{ font-size:13px; white-space:nowrap; }}
.stTabs [role="tab"][aria-selected="true"] {{ color:{ACCENT}; }}
.stTabs [role="tab"][aria-selected="true"] p {{ color:{ACCENT}; }}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"] {{ background:{ACCENT}; height:2px; }}
.stTabs [data-baseweb="tab-border"] {{ background:transparent; }}
.stTabs .stTabs [role="tablist"] {{ gap:20px; margin-bottom:10px; }}
.stTabs .stTabs [role="tab"] {{ height:38px; }}
.page-header {{ margin:0 0 4px; }}
.page-header h2 {{ font-size:29px; font-weight:600; letter-spacing:-1px;
    line-height:1.25; padding:0; margin:0 0 8px; }}
.page-header p {{ font-size:13px; margin:0; color:{TEXT_DIM}; }}
.data-status {{ display:flex; align-items:center; gap:18px; flex-wrap:wrap;
    padding:0 0 8px; color:{TEXT_MUTE}; font-size:12px; }}
.status-pill {{ display:inline-flex; align-items:center; gap:7px; font-size:12px; }}
.status-pill .dot {{ width:6px; height:6px; border-radius:50%; background:{TEXT_MUTE}; }}
.status-live .dot {{ background:{GREEN}; }}
.status-stale .dot {{ background:{AMBER}; }}
.status-stale {{ color:{AMBER}; }}
.status-date {{ margin-left:auto; font:11px var(--font-mono); }}

/* The forecast has priority; supporting metrics share a single surface. */
.forecast-strip {{ display:grid; grid-template-columns:1.05fr 1.15fr 1.25fr .8fr;
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
    overflow:hidden; margin:4px 0 8px; }}
.forecast-cell {{ padding:20px 24px; position:relative; }}
.forecast-cell + .forecast-cell {{ border-left:1px solid {BORDER}; }}
.forecast-primary {{ background:#fff8ef; }}
.metric-label {{ display:block; color:{TEXT_DIM}; font-size:12px; margin-bottom:13px; }}
.metric-number {{ font-size:clamp(24px,2.5vw,34px); line-height:1.2; font-weight:600;
    letter-spacing:-1.4px; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.forecast-primary .metric-number {{ color:{ACCENT}; }}
.metric-foot {{ display:block; color:{TEXT_MUTE}; font-size:12px; margin-top:10px; }}
.metric-range {{ display:flex; flex-wrap:nowrap; align-items:baseline; gap:4px;
    font-family:var(--font-sans); font-size:clamp(19px,2.1vw,30px);
    line-height:1.2; font-weight:600; letter-spacing:-1.2px;
    font-variant-numeric:tabular-nums; white-space:nowrap; }}
.metric-range > span {{ white-space:nowrap; }}
.range-separator {{ color:{TEXT_MUTE}; font-size:11px; font-weight:400;
    letter-spacing:0; padding:0 2px; }}
.metric-regime {{ font-size:27px; letter-spacing:-.8px; font-weight:550; }}
.delta-positive {{ color:{GREEN}; }}
.delta-negative {{ color:{RED}; }}
.regime-bull {{ color:{GREEN}; }}
.regime-bear {{ color:{RED}; }}
.regime-side {{ color:{TEXT_DIM}; }}
[data-testid="stMetric"] {{ padding:18px 20px; border:1px solid {BORDER};
    border-radius:10px; background:{SURFACE}; }}
[data-testid="stMetricLabel"] {{ color:{TEXT_DIM}; }}
[data-testid="stMetricValue"] {{ color:{TEXT}; font-size:27px;
    font-variant-numeric:tabular-nums; letter-spacing:-1px; }}
[data-testid="stMetricValue"] > div {{ white-space:normal; overflow-wrap:anywhere; }}
.section-label {{ color:{TEXT}; font-size:15px; font-weight:600;
    margin:20px 0 3px; letter-spacing:-.25px; }}
[data-testid="stPlotlyChart"] {{ border:1px solid {BORDER}; border-radius:12px;
    overflow:hidden; background:{SURFACE}; }}
.detail-panel {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
    padding:8px 22px 18px; }}
.detail-row {{ display:flex; justify-content:space-between; align-items:baseline;
    gap:20px; padding:12px 0; border-bottom:1px solid {GRID}; font-size:13px; }}
.detail-row dt {{ color:{TEXT_DIM}; }}
.detail-row dd {{ margin:0; text-align:right; color:{TEXT}; font-weight:550;
    font-variant-numeric:tabular-nums; }}
.detail-panel dl {{ margin:0; }}
.detail-note {{ color:{TEXT_MUTE}; font-size:12px; line-height:1.6; margin:14px 0 0; }}
.market-item {{ padding:16px 0; border-bottom:1px solid {GRID}; }}
.market-item:last-child {{ border:0; padding-bottom:0; }}
.market-heading {{ display:flex; justify-content:space-between; align-items:center;
    gap:16px; font-size:13px; color:{TEXT_DIM}; }}
.market-heading strong {{ color:{TEXT}; font-weight:600; font-variant-numeric:tabular-nums; }}
.market-item p {{ font-size:12px; margin:6px 0 0; }}

/* Disclosures remain visible but do not bury the forecast. */
.info-box {{ padding:17px 20px; background:{SURFACE_2}; border-radius:8px;
    color:{TEXT_DIM}; font-size:13px; line-height:1.75; margin:4px 0 10px; }}
.info-box b {{ color:{TEXT}; }}
.info-box summary {{ cursor:pointer; font-weight:550; color:{TEXT}; }}
.info-box .notice-body {{ margin-top:12px; }}
.warn-box {{ padding:14px 18px; border-left:3px solid {AMBER}; background:#faf6ed;
    border-radius:0 8px 8px 0; color:#755119; font-size:12.5px; line-height:1.7; margin:0 0 8px; }}
.warn-box summary {{ cursor:pointer; color:#755119; font-weight:500; }}
.warn-box summary:focus-visible {{ outline:2px solid {ACCENT}; outline-offset:5px; }}
.warn-box .notice-body {{ margin-top:12px; color:{TEXT_DIM}; }}
.disclaimer-box {{ color:{TEXT_MUTE}; font-size:12px; line-height:1.7;
    padding:16px 0; margin-top:18px; border-top:1px solid {BORDER}; }}
.disclaimer-box b {{ color:{TEXT_DIM}; }}
.notice-icon {{ display:inline-grid; place-items:center; width:14px; height:14px;
    border:1px solid currentColor; border-radius:50%; font-size:10px; font-weight:700;
    margin-right:6px; vertical-align:1px; }}
[data-testid="stExpander"] {{ background:transparent; border-color:{BORDER}; border-radius:8px; }}
[data-testid="stExpander"] summary {{ color:{TEXT}; font-size:13px; }}
[data-testid="stExpander"] summary:hover {{ background:{SURFACE_2}; }}

/* Controls: readable tooltips, keyboard focus, pressed feedback. */
.stButton button {{ border-radius:7px; min-height:42px; padding:8px 18px;
    transition:background .16s ease, border-color .16s ease, transform .16s ease; }}
.stButton button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; color:{TEXT}; }}
.stButton button[kind="primary"] p {{ color:{TEXT}; font-weight:600; font-size:13px; }}
.stButton button[kind="primary"]:hover {{ background:#e89028; border-color:#e89028; }}
.stButton button[kind="secondary"] {{ background:{SURFACE}; border-color:{BORDER}; color:{TEXT}; }}
.stButton button[kind="secondary"]:hover {{ background:{SURFACE_2}; border-color:#b2bac4; }}
.stButton button:active {{ transform:translateY(1px); }}
button:focus-visible, [tabindex]:focus-visible {{ outline:2px solid {ACCENT} !important; outline-offset:3px; }}
[data-testid="stTooltipContent"] {{ background:{TEXT}; color:white; border-radius:6px; }}
[data-testid="stSelectbox"] label {{ color:{TEXT_DIM}; font-size:12px; }}
[data-baseweb="select"] > div {{ background:{SURFACE}; border-color:{BORDER}; border-radius:7px; }}
[data-baseweb="select"] {{ color:{TEXT}; }}
[data-testid="stSpinner"] {{ color:{TEXT_DIM}; padding:24px 0; }}

/* Tables scroll within their own region, never across the page. */
.table-wrap {{ max-width:100%; overflow-x:auto; background:{SURFACE};
    border:1px solid {BORDER}; border-radius:10px; margin:6px 0 10px; }}
table.analytics-table {{ width:100%; min-width:1050px; border-collapse:collapse; font-size:12px; }}
table.analytics-table th {{ text-align:left; font-weight:500; color:{TEXT_DIM};
    background:#f1f3f5; padding:13px 16px; white-space:nowrap; }}
table.analytics-table td {{ padding:15px 16px; color:{TEXT}; border-top:1px solid {GRID}; white-space:nowrap; }}
table.analytics-table td:nth-child(n+5), table.analytics-table th:nth-child(n+5) {{
    text-align:right; font-variant-numeric:tabular-nums; }}
table.analytics-table tbody tr:has(td.model-usulan) {{ background:#fff8ef; }}
table.analytics-table td.model-usulan {{ color:{ACCENT}; font-weight:650; }}
table.analytics-table tbody tr:hover {{ background:{SURFACE_2}; }}
table.analytics-table th:first-child, table.analytics-table td:first-child {{
    position:sticky; left:0; background:{SURFACE}; }}
table.analytics-table th:first-child {{ background:#f1f3f5; }}
table.analytics-table td.model-usulan {{ background:#fff8ef; }}
[data-testid="stMarkdownContainer"]:has(table) {{ overflow-x:auto; }}
[data-testid="stMarkdownContainer"] table:not(.analytics-table) {{ min-width:560px; font-size:13px; }}
[data-testid="stMarkdownContainer"] table th,
[data-testid="stMarkdownContainer"] table td {{ border-color:{BORDER}; }}

@media (max-width:1000px) {{
    .block-container {{ padding:4rem 1.5rem 2rem; }}
    .forecast-strip {{ grid-template-columns:1fr 1fr; }}
    .forecast-cell:nth-child(3) {{ border-left:0; }}
    .forecast-cell:nth-child(n+3) {{ border-top:1px solid {BORDER}; }}
    .metric-number {{ font-size:32px; }}
    .metric-range {{ font-size:clamp(19px,2.7vw,27px); }}
    .topbar-meta {{ display:none; }}
}}
@media (max-width:640px) {{
    .block-container {{ padding:4rem 1rem 2rem; }}
    [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap !important; }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        flex:1 1 100% !important; width:100% !important; min-width:0 !important; }}
    .topbar {{ min-height:54px; margin:0; }}
    .topbar-title h1 {{ font-size:17px; }}
    .topbar-title p {{ font-size:12px; }}
    .welcome {{ padding:20px 0 4px; }}
    .welcome h2 {{ font-size:33px; letter-spacing:-1.1px; }}
    .ready-panel {{ margin-top:8px; padding:22px; }}
    .method-strip {{ grid-template-columns:1fr; gap:22px; margin-top:12px; }}
    .method-step p {{ max-width:none; }}
    .forecast-strip {{ grid-template-columns:1fr; }}
    .forecast-cell {{ padding:18px 14px; }}
    .forecast-cell + .forecast-cell {{ border-left:0; border-top:1px solid {BORDER}; }}
    .metric-number {{ font-size:27px; letter-spacing:-1px; }}
    .metric-range {{ font-size:clamp(19px,6vw,27px); letter-spacing:-1px; }}
    .range-separator {{ padding:0; }}
    .metric-label {{ font-size:12px; }}
    .metric-foot {{ font-size:10px; }}
    .metric-regime {{ font-size:25px; }}
    .page-header h2 {{ font-size:25px; }}
    .page-header p {{ font-size:12px; }}
    .stTabs [role="tablist"] {{ gap:18px; }}
    .stTabs [role="tab"] p {{ font-size:12px; }}
    .data-status {{ gap:10px 16px; }}
    .status-date {{ margin-left:0; width:100%; }}
    .detail-panel {{ padding:6px 16px 16px; }}
}}
@media (prefers-reduced-motion:reduce) {{
    *, *::before, *::after {{ transition:none !important; animation:none !important; }}
}}
</style>
""", unsafe_allow_html=True)


def render_table(df: pd.DataFrame, label='Hasil evaluasi model'):
    html = df.to_html(index=False, escape=True, classes="analytics-table", border=0)
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
                <p>Prediksi Probabilistik</p>
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
            <h2>Prediksi Bitcoin, dengan<br>ukuran ketidakpastian.</h2>
            <p>Amati estimasi harga, interval prediksi 90%, dan kondisi pasar
            melalui model HMM, XGBoost Quantile, dan Conformal Prediction.</p>
        </section>
        """, unsafe_allow_html=True)
        st.button("Muat & Jalankan Model", type="primary", on_click=start_dashboard,
                  help="Mengambil data harga, sentimen, dan on-chain, lalu melatih model prediksi.")
        st.button("Lihat Hasil Eksperimen", on_click=open_comparison,
                  help="Buka hasil Colab terverifikasi tanpa mengambil data atau melatih model.")
        st.caption("Prediksi terbaru memproses data setelah tombol dijalankan. Hasil eksperimen dapat langsung dibaca.")
    with readiness:
        st.markdown("""
        <aside class='ready-panel' aria-label='Kesiapan sumber data'>
            <h3>Sumber data</h3>
            <p class='panel-caption'>Status ketersediaan diperiksa saat model dijalankan.</p>
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
        <h3>Kenali kondisi pasar</h3><p>HMM mengidentifikasi regime Bear, Sideways,
        atau Bull dari data historis.</p></div></article>
        <article class='method-step'><span class='method-index'>02</span><div>
        <h3>Estimasi rentang harga</h3><p>XGBoost Quantile menghasilkan batas bawah,
        median, dan batas atas prediksi.</p></div></article>
        <article class='method-step'><span class='method-index'>03</span><div>
        <h3>Kalibrasi interval</h3><p>Conformal Prediction menyesuaikan interval
        dengan target cakupan 90%.</p></div></article>
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
    with st.spinner("Memuat data dan menjalankan model kausal..."):
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
    price_status = "Harga up-to-date" if is_price_fresh else f"Harga tertinggal {price_staleness_days} hari"
    chain_status = "On-chain terbaru" if is_onchain_fresh else f"On-chain tertinggal {onchain_stale_days} hari"
    st.markdown(f"""
    <div class='data-status' aria-label='Kesegaran data'>
        <span class='status-pill {"status-live" if is_price_fresh else "status-stale"}'>{price_status}</span>
        <span class='status-pill {"status-live" if is_onchain_fresh else "status-stale"}'>{chain_status}</span>
        <span class='status-date'>Data acuan · {price_last_date.strftime('%d %b %Y')}</span>
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
        st.markdown("""
        <div class='page-header'>
            <h2>Prediksi harga Bitcoin</h2>
            <p>Horizon 1 hari · HMM raw filtering kausal + XGBoost Quantile + Conformal Prediction</p>
        </div>
        """, unsafe_allow_html=True)

        last_close  = result["last_close"]
        last_date   = result["last_date"]
        pred_med    = result["pred_med"]
        pred_lo     = result["pred_lo"]
        pred_hi     = result["pred_hi"]
        pred_date   = result["pred_date"]
        last_regime = result["last_regime"]
        last_sent   = result["last_sent"]
        last_netflow= result["last_netflow"]

        st.caption("Pipeline live memakai raw filtering dan purge satu tanggal pada batas split. "
                   "Data/bobot live berbeda dari artefak Colab; waktu publikasi intraday sumber belum diaudit.")
        with st.expander("Metode dan versi pipeline live"):
            st.json(result["pipeline_metadata"])
            st.caption("Parameter dan nama state HMM berasal dari train saja. Filtering sesudah cutoff tidak "
                       "memakai observasi masa depan. Transformasi fitur train tetap in-sample. "
                       "Konvensi kuantil CQR dipertahankan dari Colab; target 90% bukan jaminan cakupan deret waktu.")
            delta = result["pipeline_metadata"]["hmm_last_delta"]
            if delta is not None and delta < 0:
                st.warning("Likelihood HMM turun pada iterasi terakhir. Baca prediksi bersama keterbatasan numerik ini.")

        delta_pct = (pred_med - last_close) / last_close * 100
        regime_label = REGIME_NAMES[last_regime]
        regime_class = {0: "regime-bear", 1: "regime-side", 2: "regime-bull"}[last_regime]
        delta_class = "delta-positive" if delta_pct >= 0 else "delta-negative"
        st.markdown(f"""
        <section class='forecast-strip' aria-label='Ringkasan prediksi Bitcoin'>
            <div class='forecast-cell'>
                <span class='metric-label'>Harga terakhir</span>
                <div class='metric-number'>${last_close:,.0f}</div>
                <span class='metric-foot'>Penutupan · {last_date.strftime('%d %b %Y')}</span>
            </div>
            <div class='forecast-cell forecast-primary'>
                <span class='metric-label'>Prediksi median</span>
                <div class='metric-number'>${pred_med:,.0f}</div>
                <span class='metric-foot'><span class='{delta_class}'>{delta_pct:+.2f}%</span>
                · {pred_date.strftime('%d %b %Y')}</span>
            </div>
            <div class='forecast-cell'>
                <span class='metric-label'>Interval prediksi 90%</span>
                <div class='metric-range'><span>${pred_lo:,.0f}</span><span class='range-separator'>s.d.</span><span>${pred_hi:,.0f}</span></div>
                <span class='metric-foot'>Kuantil 5% sampai 95% · Terkalibrasi</span>
            </div>
            <div class='forecast-cell'>
                <span class='metric-label'>Regime pasar</span>
                <div class='metric-regime {regime_class}'>{regime_label}</div>
                <span class='metric-foot'>Filtering maju HMM</span>
            </div>
        </section>
        """, unsafe_allow_html=True)

        if not is_price_fresh:
            st.markdown(f"""
            <details class='warn-box'><summary>Harga acuan tertinggal {price_staleness_days} hari. Prediksi mengikuti candle terakhir.</summary>
            <div class='notice-body'>{WARN_ICON}<b>Harga acuan belum ter-update ke hari ini.</b> Candle harian
            BTC-USD terakhir dari Yahoo Finance yang tersedia adalah
            <b>{price_last_date.strftime('%d %B %Y')}</b> ({price_staleness_days} hari
            lalu), sehingga prediksi di bawah ini masih berpatokan pada tanggal
            tersebut, bukan hari ini. Ini biasanya terjadi karena Yahoo Finance
            menutup candle harian instrumen crypto berdasarkan basis hari
            <b>US Eastern</b> (jauh di belakang WIB), atau candle terbaru
            memang belum di-publish sumbernya. Coba klik tombol
            <b>Refresh Data</b> di kanan atas beberapa saat lagi untuk
            mengecek ulang tanpa menunggu cache (1 jam) habis sendiri.
            </div></details>
            """, unsafe_allow_html=True)

        if not is_onchain_fresh:
            st.markdown(f"""
            <details class='warn-box'><summary>Netflow memakai forward-fill setelah {format_date_id(result['onchain_last_real_date'])}. Baca batasan data.</summary>
            <div class='notice-body'>{WARN_ICON}<b>Data on-chain tidak terkini.</b> Baris terakhir
            arsip CSV yang diambil bertanggal
            <b>{format_date_id(result['onchain_last_csv_date'])}</b>, tetapi nilai
            netflow tidak kosong terakhir bertanggal
            <b>{format_date_id(result['onchain_last_real_date'])}</b>
            ({onchain_stale_days} hari dari harga acuan). Prediksi memakai nilai netflow
            historis tersebut melalui forward-fill; nilainya tidak menggambarkan
            aktivitas bursa setelah tanggal itu. Harga dan sentimen diambil kembali
            saat pipeline dijalankan, jika sumbernya tersedia, dengan cache satu jam.
            <br><br>
            <span style='font-size:12px;color:{TEXT_DIM}'>
            <b style='color:{AMBER}'>Batasan penelitian:</b> hasil pengujian
            historis tidak mengukur akurasi prediksi saat netflow diteruskan
            dari data lama. Prediksi ini perlu dibaca dengan batasan tersebut.
            </span>
            </div></details>
            """, unsafe_allow_html=True)

        chart_title, chart_control = st.columns([3, 1], vertical_alignment="bottom")
        with chart_title:
            st.markdown("<div class='section-label'>Prediksi & harga aktual</div>", unsafe_allow_html=True)
            st.caption("Refit live pada data terbaru, terpisah dari evaluasi Colab. Sumbu tanggal adalah tanggal target t+1; area berarsir adalah interval 90%.")
        with chart_control:
            chart_period = st.selectbox("Rentang grafik", ["90 hari terakhir", "30 hari terakhir", "Seluruh test set"],
                                       key="forecast_period", label_visibility="collapsed")

        dates_test = pd.DatetimeIndex(result["hist_target_dates"])
        hist_true  = result["hist_true"]
        hist_med   = result["hist_med"]
        hist_lo    = result["hist_lo"]
        hist_hi    = result["hist_hi"]

        n_valid = min(len(dates_test), len(hist_true), len(hist_med),
                      len(hist_lo), len(hist_hi))
        dates_test = dates_test[:n_valid]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(dates_test) + list(dates_test[::-1]),
            y=list(hist_hi[:n_valid]) + list(hist_lo[:n_valid][::-1]),
            fill="toself", fillcolor=TEAL_SOFT,
            line=dict(color="rgba(0,0,0,0)"),
            name="Interval 90%", hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=dates_test, y=hist_true[:n_valid],
            mode="lines", name="Harga Aktual",
            line=dict(color=TEXT, width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=dates_test, y=hist_med[:n_valid],
            mode="lines", name="Prediksi Median",
            line=dict(color=TEAL, width=1.5, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=[pred_date], y=[pred_med],
            mode="markers", name=f"Prediksi {pred_date.strftime('%d %b')}",
            marker=dict(color=ACCENT, size=9, symbol="circle", line=dict(color=SURFACE, width=2)),
        ))
        fig.add_trace(go.Scatter(
            x=[pred_date, pred_date], y=[pred_lo, pred_hi], mode="lines",
            name="Interval prediksi berikutnya", line=dict(color=ACCENT, width=3)
        ))
        fig.update_layout(
            template="plotly_white", height=400,
            margin=dict(l=20, r=24, t=55, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                        font=dict(color=TEXT_DIM, size=11)),
            yaxis=dict(tickprefix="$", tickformat=",", gridcolor=GRID, side="right", nticks=6),
            xaxis=dict(showgrid=False, tickformat="%d %b", nticks=6),
            hovermode="x unified",
        )
        if chart_period != "Seluruh test set":
            window_days = 90 if chart_period == "90 hari terakhir" else 30
            fig.update_xaxes(range=[pred_date - timedelta(days=window_days), pred_date + timedelta(days=2)])
        fig.update_traces(hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>", selector=dict(mode="lines"))
        render_chart(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-label'>Detail prediksi berikutnya</div>",
                        unsafe_allow_html=True)
            width_pct = (pred_hi - pred_lo) / pred_med * 100
            st.markdown(f"""
            <section class='detail-panel' aria-label='Detail prediksi'>
            <dl>
            <div class='detail-row'><dt>Tanggal prediksi</dt><dd>{pred_date.strftime('%d %B %Y')}</dd></div>
            <div class='detail-row'><dt>Harga acuan · {last_date.strftime('%d %b')}</dt><dd>${last_close:,.0f}</dd></div>
            <div class='detail-row'><dt>Prediksi median</dt><dd>${pred_med:,.0f} <span class='{delta_class}'>({delta_pct:+.2f}%)</span></dd></div>
            <div class='detail-row'><dt>Batas bawah · Q05</dt><dd>${pred_lo:,.0f}</dd></div>
            <div class='detail-row'><dt>Batas atas · Q95</dt><dd>${pred_hi:,.0f}</dd></div>
            <div class='detail-row'><dt>Lebar interval</dt><dd>${pred_hi-pred_lo:,.0f} · {width_pct:.1f}%</dd></div>
            <div class='detail-row'><dt>Conformal margin</dt><dd>{result["conf_margin"]:.5f} <small>(log-scale)</small></dd></div>
            </dl>
            <p class='detail-note'>Target cakupan interval adalah 90% dalam jangka panjang.
            Harga aktual tetap dapat berada di luar rentang ini.</p>
            </section>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-label'>Kondisi pasar pada data acuan</div>",
                        unsafe_allow_html=True)
            sent_label = (
                "Extreme Greed" if last_sent > 0.6 else
                "Greed"         if last_sent > 0.2 else
                "Neutral"       if last_sent > -0.2 else
                "Fear"          if last_sent > -0.6 else
                "Extreme Fear"
            )
            sent_color = (
                GREEN if last_sent > 0.2 else
                TEXT_DIM if last_sent > -0.2 else
                RED
            )
            regime_desc = {
                0: "Bear — volatilitas tinggi, return negatif dominan",
                1: "Sideways — pasar konsolidasi, return mendekati nol",
                2: "Bull — tren naik, return positif dominan"
            }[last_regime]
            netflow_caption = (
                "(nilai historis terakhir — on-chain belum live, lihat peringatan di atas)"
                if not is_onchain_fresh else ""
            )
            st.markdown(f"""
            <section class='detail-panel' aria-label='Kondisi pasar'>
            <div class='market-item'>
                <div class='market-heading'><span>Regime HMM</span><strong class='{regime_class}'>{regime_label}</strong></div>
                <p>{regime_desc}</p>
            </div>
            <div class='market-item'>
                <div class='market-heading'><span>Sentimen pasar</span><strong style='color:{sent_color}'>{sent_label}</strong></div>
                <p>Skor polaritas <span class='tnum'>{last_sent:+.2f}</span> pada skala −1 hingga +1.</p>
            </div>
            <div class='market-item'>
                <div class='market-heading'><span>On-chain netflow</span><strong>{last_netflow:+,.0f} BTC</strong></div>
                <p>{"Arus keluar bersih dari bursa (outflow − inflow)" if last_netflow > 0
                    else "Arus masuk bersih ke bursa (outflow − inflow)" if last_netflow < 0
                    else "Arus masuk dan keluar bursa seimbang"}</p>
                <p style='color:{AMBER}'>{netflow_caption}</p>
            </div>
            <div class='market-item'>
                <div class='market-heading'><span>Netflow terbobot sentimen</span><strong>{last_netflow * (1 + last_sent):+,.0f}</strong></div>
                <p>Fitur usulan · Netflow × (1 + polarity)</p>
            </div>
            </section>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='disclaimer-box'>
        {WARN_ICON}<b>Disclaimer:</b> Dashboard ini merupakan prototipe akademik sebagai bagian dari
        Tugas Akhir Program Studi Teknologi Rekayasa Perangkat Lunak, Universitas Gadjah Mada.
        Prediksi yang ditampilkan <b>bukan merupakan nasihat investasi</b> dan tidak boleh
        dijadikan dasar keputusan finansial.
        </div>
        """, unsafe_allow_html=True)

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
    <div class='info-box' style='margin-top:0'>
    <b>Versi dashboard:</b> {CORE_VERSION}<br><br>
    <b>Sumber data:</b><br>
    • Harga: Yahoo Finance<br>
    • Sentimen: Crypto Fear & Greed Index<br>
    • On-Chain: CoinMetrics<br><br>
    <b>Model yang berjalan live di dashboard ini:</b><br>
    HMM raw filtering kausal + XGBoost Quantile + Conformal Prediction.
    <span style='font-size:12px;color:{TEXT_MUTE}'>
    Ini satu-satunya model yang dihitung ulang secara live untuk halaman prediksi saja. Semua model, dari model
    pembanding hingga usulan pada halaman "Komparasi Model" adalah referensi statis
    dari notebook eksperimen terpisah, bukan hasil live.
    </span>
    </div>
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
