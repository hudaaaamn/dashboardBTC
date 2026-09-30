# ============================================================
# dashboard.py — Dashboard Prediksi Probabilistik Harga Bitcoin
# TA: Huda Muhammad Nur — UGM Sekolah Vokasi 2026
#
# Cara menjalankan:
#   pip install -r requirements.txt
#   streamlit run dashboard.py
#
# CATATAN REVISI (Opsi B — on-chain fallback):
#   Arsip CoinMetrics Community CSV yang diambil untuk penelitian ini
#   mempunyai baris terakhir 24 Mei 2026. Kondisi arsip dapat berubah.
#   Dashboard ini
#   tetap bisa menghasilkan prediksi "hari ini
#   untuk besok" karena timeline utama sekarang mengikuti data HARGA
#   (selalu paling baru), sementara fitur on-chain di-forward-fill dari
#   nilai riil terakhir yang tersedia dan diberi badge transparansi.
#   Begitu API key premium (mis. CoinMetrics Pro) tersedia, isi
#   COINMETRICS_API_KEY di st.secrets / environment variable — fungsi
#   fetch_onchain() otomatis beralih ke jalur live tanpa perlu ubah kode lain.
#
# CATATAN CAKUPAN MODEL (hanya Model Usulan yang berjalan live di sini):
#   Dashboard ini HANYA melatih/menjalankan satu model secara live, yaitu
#   Model Usulan (HMM Regime-Switching + XGBoost Quantile + Conformal
#   Prediction) melalui build_features_and_predict() di bawah. Kelima
#   model pembanding (XGBoost, LightGBM, Random Forest, SVR, LSTM) TIDAK
#   dilatih ulang di dalam dashboard — angkanya (results_data pada halaman
#   "Komparasi Model") adalah hasil statis dari eksperimen komparasi yang
#   dijalankan terpisah di notebook Google Colab (tabel final Step 14), pada
#   373 tanggal uji bersama (11 Mei 2025 - 18 Mei 2026), lalu
#   di-hardcode di sini sebagai referensi. Keputusan desain ini disengaja:
#   Komparasi historis tersebut tidak mengukur akurasi saat fitur on-chain
#   memakai forward-fill. Jika notebook eksperimen dijalankan ulang dengan
#   data baru, angka pada results_data perlu diperbarui manual mengikuti
#   hasil notebook tsb (lihat Tabel 4.1/4.2/4.3 BAB IV).
#
# CATATAN REDESIGN TAMPILAN (September 2026):
#   Layout riset dengan ringkasan prediksi, grafik berfilter periode,
#   pemberitahuan kesegaran data, dan tabel evaluasi responsif.
#   Fungsi fetch, feature engineering, training, caching, serta angka
#   komparasi eksperimen tetap sama. Filter grafik hanya mengubah tampilan.
#
# CATATAN TAMBAHAN (Agustus 2026 — kesegaran data HARGA):
#   Ditambahkan badge kesegaran harga (mirip badge on-chain yang sudah
#   ada) + tombol "Refresh Data" manual. Ini merespons kasus nyata: harga
#   BTC-USD dari Yahoo Finance (via yfinance) kadang tampak "telat 1 hari"
#   ketika dashboard dibuka pagi/siang WIB, karena candle harian crypto di
#   Yahoo Finance sering final berdasarkan hari kalender US Eastern (jauh
#   di belakang WIB/UTC+7), ditambah cache Streamlit (ttl=3600) yang baru
#   dicek ulang saat ada kunjungan baru. Badge ini TIDAK mengubah logika
#   pipeline/model sama sekali — murni lapisan transparansi + tombol untuk
#   memaksa bypass cache tanpa menunggu ttl habis sendiri.
#
# CATATAN REVISI (Agustus 2026 — model tidak auto-run saat dibuka):
#   Sebelumnya, setiap kali dashboard DIBUKA (termasuk kunjungan pertama
#   seorang user), seluruh pipeline (fetch harga/sentimen/on-chain +
#   build_features_and_predict) langsung berjalan otomatis. Ini membuat
#   pengunjung pertama selalu menunggu proses training/prediksi walau
#   mereka belum tentu butuh data ter-refresh saat itu juga.
#   Revisi: dipakai st.session_state sebagai flag ("dashboard_started").
#   Saat halaman pertama kali dibuka, tampil pengantar dan status sumber
#   yang belum dimuat (belum ada fetch/model yang dijalankan).
#   Pipeline (fetch_price,
#   fetch_sentiment, fetch_onchain, build_features_and_predict, serta
#   seluruh tab Prediksi/Komparasi Model/Analisis Data) baru dieksekusi
#   setelah tombol "Muat & Jalankan Model" / "Refresh Data" ditekan.
#   Tombol yang sama juga dipakai untuk me-refresh ulang (bypass cache
#   1 jam) setelah dashboard pernah dijalankan sebelumnya di sesi ini.
#   Tidak ada perubahan pada logika fetch/model/caching itu sendiri —
#   perubahan murni pada KAPAN pipeline tsb dipanggil.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

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
table.analytics-table tbody tr:last-child {{ background:#fff8ef; }}
table.analytics-table tbody tr:last-child td:nth-child(2) {{ color:{ACCENT}; font-weight:650; }}
table.analytics-table tbody tr:hover {{ background:{SURFACE_2}; }}
table.analytics-table th:nth-child(2), table.analytics-table td:nth-child(2) {{
    position:sticky; left:0; background:{SURFACE}; }}
table.analytics-table th:nth-child(2) {{ background:#f1f3f5; }}
table.analytics-table tbody tr:last-child td:nth-child(2) {{ background:#fff8ef; }}
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


def render_table(df: pd.DataFrame):
    html = df.to_html(index=False, escape=True, classes="analytics-table", border=0)
    st.markdown(
        f"<div class='table-wrap' role='region' aria-label='Hasil evaluasi model' tabindex='0'>{html}</div>",
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
    st.plotly_chart(fig, use_container_width=True, theme=None,
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

    today_utc = pd.Timestamp(datetime.utcnow().date())
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
    from hmmlearn.hmm import GaussianHMM
    import xgboost as xgb

    # Fetch semua data
    df_p        = fetch_price()
    df_s, _     = fetch_sentiment()
    df_o, onchain_live = fetch_onchain()

    df = df_p[["close"]].copy()
    df = df.join(df_s[["polarity"]], how="left")
    df = df.join(df_o[["exchange_netflow"]], how="left")

    onchain_last_csv_date = df_o.index.max() if len(df_o) else None
    valid_netflow_dates = df_o["exchange_netflow"].dropna().index
    onchain_last_real_date = valid_netflow_dates.max() if len(valid_netflow_dates) else None
    last_price_date = df_p.index.max()
    onchain_staleness_days = (
        (last_price_date - onchain_last_real_date).days
        if onchain_last_real_date is not None else None
    )

    df["polarity"]         = df["polarity"].ffill()
    df["exchange_netflow"] = df["exchange_netflow"].ffill()
    df = df.dropna(subset=["close","polarity","exchange_netflow"])

    # Features
    df["log_return"]       = np.log(df["close"] / df["close"].shift(1))
    df["volatility_7d"]    = df["log_return"].rolling(7).std()
    df["volatility_14d"]   = df["log_return"].rolling(14).std()
    df["volatility_30d"]   = df["log_return"].rolling(30).std()
    df["price_to_ma7"]     = df["close"] / df["close"].rolling(7).mean()
    df["price_to_ma30"]    = df["close"] / df["close"].rolling(30).mean()
    for lag in [1,2,3]:
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
    df["netflow_change"]   = df["exchange_netflow"].diff()

    if onchain_last_real_date is not None:
        stale_mask = df.index > onchain_last_real_date
        df.loc[stale_mask, "netflow_change"] = 0.0

    df["netflow_ma_7"]     = df["exchange_netflow"].rolling(7).mean()
    df["sentiment_ma_7"]   = df["polarity"].rolling(7).mean()
    df["netflow_weighted"] = df["exchange_netflow"] * (1 + df["polarity"])

    df_hmm = df.dropna(subset=["log_return","volatility_14d"]).copy()
    X_hmm  = df_hmm[["log_return","volatility_14d"]].values
    hmm_train_end = int(len(df_hmm) * 0.70)
    hmm    = GaussianHMM(n_components=3, covariance_type="full",
                         n_iter=200, random_state=42)
    hmm.fit(X_hmm[:hmm_train_end])
    df_hmm["regime"] = hmm.predict(X_hmm)
    rm = df_hmm.iloc[:hmm_train_end].groupby("regime")["log_return"].mean().sort_values()
    regime_map = {rm.index[i]: i for i in range(3)}  # 0=bear,1=side,2=bull
    df_hmm["regime_labeled"] = df_hmm["regime"].map(regime_map)
    df = df.join(df_hmm[["regime_labeled"]], how="left")
    df["target_return"] = df["log_return"].shift(-1)

    last_row_live = df.iloc[[-1]].copy()
    df = df.dropna()

    FCOLS = ["log_return","volatility_7d","volatility_14d","volatility_30d",
             "price_to_ma7","price_to_ma30",
             "return_lag_1","return_lag_2","return_lag_3",
             "exchange_netflow","netflow_change","netflow_ma_7",
             "polarity","sentiment_ma_7","netflow_weighted","regime_labeled"]

    n         = len(df)
    train_end = int(n * 0.70)
    calib_end = int(n * 0.80)
    df_train  = df.iloc[:train_end]
    df_calib  = df.iloc[train_end:calib_end]

    X_train, y_train = df_train[FCOLS].values, df_train["target_return"].values
    X_calib, y_calib = df_calib[FCOLS].values, df_calib["target_return"].values

    ALPHA = 0.10
    pq = dict(n_estimators=300, learning_rate=0.05, max_depth=5,
              random_state=42, verbosity=0, tree_method="hist")
    mlo  = xgb.XGBRegressor(**pq, objective="reg:quantileerror", quantile_alpha=0.05)
    mmed = xgb.XGBRegressor(**pq, objective="reg:quantileerror", quantile_alpha=0.50)
    mhi  = xgb.XGBRegressor(**pq, objective="reg:quantileerror", quantile_alpha=0.95)
    mlo.fit(X_train, y_train)
    mmed.fit(X_train, y_train)
    mhi.fit(X_train, y_train)

    scores      = np.maximum(mlo.predict(X_calib)-y_calib, y_calib-mhi.predict(X_calib))
    q_level     = min(np.ceil((1-ALPHA)*(len(scores)+1))/len(scores), 1.0)
    conf_margin = np.quantile(scores, q_level)

    last_row    = last_row_live[FCOLS].values
    last_close  = float(last_row_live["close"].iloc[0])
    last_date   = last_row_live.index[0]
    last_regime = int(last_row_live["regime_labeled"].iloc[0])
    last_sent   = float(last_row_live["polarity"].iloc[0])
    last_netflow= float(last_row_live["exchange_netflow"].iloc[0])

    lo_r  = float(mlo.predict(last_row)[0])  - conf_margin
    med_r = float(mmed.predict(last_row)[0])
    hi_r  = float(mhi.predict(last_row)[0])  + conf_margin

    pred_lo  = last_close * np.exp(lo_r)
    pred_med = last_close * np.exp(med_r)
    pred_hi  = last_close * np.exp(hi_r)
    pred_date = last_date + timedelta(days=1)

    df_test  = df.iloc[calib_end:]
    X_test   = df_test[FCOLS].values
    hist_lo  = df_test["close"].values * np.exp(mlo.predict(X_test)  - conf_margin)
    hist_med = df_test["close"].values * np.exp(mmed.predict(X_test))
    hist_hi  = df_test["close"].values * np.exp(mhi.predict(X_test)  + conf_margin)
    hist_true= df_test["close"].shift(-1).values

    return {
        "df": df,
        "df_test": df_test,
        "pred_date": pred_date,
        "pred_lo": pred_lo,
        "pred_med": pred_med,
        "pred_hi": pred_hi,
        "last_close": last_close,
        "last_date": last_date,
        "last_regime": last_regime,
        "last_sent": last_sent,
        "last_netflow": last_netflow,
        "hist_lo": hist_lo,
        "hist_med": hist_med,
        "hist_hi": hist_hi,
        "hist_true": hist_true,
        "conf_margin": conf_margin,
        "onchain_live": onchain_live,
        "onchain_last_real_date": onchain_last_real_date,
        "onchain_last_csv_date": onchain_last_csv_date,
        "onchain_staleness_days": onchain_staleness_days,
    }

# ============================================================
# STATE — apakah pipeline/model sudah pernah dijalankan di sesi ini
# ------------------------------------------------------------
# Lihat "CATATAN REVISI (Agustus 2026 — model tidak auto-run saat dibuka)"
# di kepala berkas. Flag ini menentukan apakah kita berhenti setelah
# menampilkan topbar saja (belum pernah dijalankan) atau lanjut memuat
# data + menjalankan model (sudah/baru saja ditekan tombolnya).
# ============================================================
if "dashboard_started" not in st.session_state:
    st.session_state.dashboard_started = False

# ============================================================
# TOPBAR — branding (bagian yang tidak butuh data dulu)
# ============================================================
def start_dashboard():
    if st.session_state.dashboard_started:
        st.cache_data.clear()
    st.session_state.dashboard_started = True


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
        st.button("Refresh Data", use_container_width=True, on_click=start_dashboard,
                  help="Ambil ulang data dan jalankan model tanpa menunggu cache 1 jam.")
    else:
        st.markdown("<div class='topbar-meta' style='text-align:right'>UGM</div>",
                    unsafe_allow_html=True)
st.markdown("<div class='masthead-rule'></div>", unsafe_allow_html=True)

if not st.session_state.dashboard_started:
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
        st.caption("Data dan model baru diproses setelah tombol ditekan.")
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
with st.spinner("Memuat data dan menjalankan model..."):
    try:
        result = build_features_and_predict()
        df_full = result["df"]
        df_p    = fetch_price()
        df_s, sent_real = fetch_sentiment()
        df_o, onchain_live_flag = fetch_onchain()
        DATA_OK = True
    except Exception as e:
        DATA_OK = False
        st.error("Data belum berhasil dimuat. Periksa koneksi, lalu tekan Refresh Data untuk mencoba lagi.")
        with st.expander("Detail kendala pemuatan"):
            st.code(str(e), language=None)
        st.stop()

price_last_date   = result["last_date"]
price_staleness_days = (pd.Timestamp(datetime.now().date()) - pd.Timestamp(price_last_date.date())).days
is_price_fresh = price_staleness_days <= 1

onchain_stale_days = result["onchain_staleness_days"]
is_onchain_fresh = onchain_stale_days is not None and onchain_stale_days <= 1

price_status = "Harga up-to-date" if is_price_fresh else f"Harga tertinggal {price_staleness_days} hari"
chain_status = "On-chain terbaru" if is_onchain_fresh else f"On-chain tertinggal {onchain_stale_days} hari"
st.markdown(f"""
<div class='data-status' aria-label='Kesegaran data'>
    <span class='status-pill {"status-live" if is_price_fresh else "status-stale"}'>
    <span class='dot' aria-hidden='true'></span>{price_status}</span>
    <span class='status-pill {"status-live" if is_onchain_fresh else "status-stale"}'>
    <span class='dot' aria-hidden='true'></span>{chain_status}</span>
    <span class='status-date'>Data acuan · {price_last_date.strftime('%d %b %Y')}</span>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_comp, tab_data = st.tabs(["Prediksi", "Komparasi Model", "Analisis Data"])

# ============================================================
# HALAMAN 1: PREDIKSI
# ============================================================
with tab_pred:
    st.markdown("""
    <div class='page-header'>
        <h2>Prediksi harga Bitcoin</h2>
        <p>Horizon 1 hari · HMM Regime-Switching + XGBoost Quantile + Conformal Prediction</p>
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
            <span class='metric-foot'>Identifikasi HMM</span>
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
        st.caption("Hasil pada test set · Area berarsir menunjukkan interval prediksi 90%.")
    with chart_control:
        chart_period = st.selectbox("Rentang grafik", ["90 hari terakhir", "30 hari terakhir", "Seluruh test set"],
                                   key="forecast_period", label_visibility="collapsed")

    dates_test = result["df_test"].index
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
        error_y=dict(
            type="data", symmetric=False,
            array=[pred_hi - pred_med],
            arrayminus=[pred_med - pred_lo],
            color=ACCENT, thickness=2, width=8
        )
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
    st.markdown("""
    <div class='page-header'>
        <h2>Komparasi Model Prediksi</h2>
        <p>Tabel 4.1 — Evaluasi metrik pada test set (kronologis, 20% terakhir)</p>
    </div>
    """, unsafe_allow_html=True)

    results_data = [
        {"No":1, "Model":"XGBoost",      "Peran":"Gradient boosting",
         "Sumber":"Notebook (offline)",
         "MAE":1561, "RMSE":2085, "MAPE":1.68, "R2":0.9862, "DirAcc":49.6,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":2, "Model":"LightGBM",     "Peran":"Gradient boosting",
         "Sumber":"Notebook (offline)",
         "MAE":1635, "RMSE":2182, "MAPE":1.77, "R2":0.9848, "DirAcc":48.3,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":3, "Model":"Random Forest","Peran":"Ensemble tree",
         "Sumber":"Notebook (offline)",
         "MAE":1443, "RMSE":1972, "MAPE":1.57, "R2":0.9876, "DirAcc":48.5,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":4, "Model":"SVR",          "Peran":"Support vector",
         "Sumber":"Notebook (offline)",
         "MAE":2138, "RMSE":3015, "MAPE":2.29, "R2":0.9711, "DirAcc":49.6,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":5, "Model":"LSTM",         "Peran":"Deep learning sekuensial",
         "Sumber":"Notebook (offline)",
         "MAE":1434, "RMSE":1967, "MAPE":1.56, "R2":0.9877, "DirAcc":49.3,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":6, "Model":"Model Usulan", "Peran":"HMM + XGB Quantile + Conformal",
         "Sumber":"Notebook (offline)",
         "MAE":1472, "RMSE":2000, "MAPE":1.59, "R2":0.9873, "DirAcc":53.6,
         "Coverage":91.2, "AvgWidth":7209, "PinballLo":247.3, "PinballHi":233.8},
    ]
    df_res = pd.DataFrame(results_data)

    st.markdown(f"""
    <details class='info-box'><summary>Referensi eksperimen statis · 373 tanggal uji bersama · 11 Mei 2025 – 18 Mei 2026</summary>
    <div class='notice-body'>
    Angka pada tabel dan grafik di bawah ini adalah hasil evaluasi pada
    <b>373 tanggal uji yang sama</b> (11 Mei 2025 &ndash; 18 Mei 2026), dijalankan pada
    notebook eksperimen Google Colab dengan tanggal data dikunci
    <code>(RUN_DATE_LOCK=2026-05-20)</code> agar hasil dapat direproduksi.
    Semua model (baris 1&ndash;6) merupakan <b>referensi statis</b>
    dari notebook tersebut dan tidak dilatih ulang setiap dashboard dibuka.
    Hanya <b>Model Usulan</b> (baris 6) yang dihitung ulang secara live oleh
    dashboard ini setiap kali data terbaru diambil &mdash; lihat halaman
    "Prediksi" untuk hasil live tersebut. Lihat kolom
    <b>Sumber</b> pada tabel untuk penanda ini.
    </div></details>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Tabel 4.1 — Hasil Evaluasi Model</div>",
                unsafe_allow_html=True)

    df_show = df_res.copy()
    df_show["MAE"]       = df_show["MAE"].apply(lambda x: f"${x:,}")
    df_show["RMSE"]      = df_show["RMSE"].apply(lambda x: f"${x:,}")
    df_show["MAPE"]      = df_show["MAPE"].apply(lambda x: f"{x:.2f}%")
    df_show["R2"]        = df_show["R2"].apply(lambda x: f"{x:.4f}")
    df_show["DirAcc"]    = df_show["DirAcc"].apply(lambda x: f"{x:.1f}%")
    df_show["Coverage"]  = df_show["Coverage"].apply(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    df_show["AvgWidth"]  = df_show["AvgWidth"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df_show["PinballLo"] = df_show["PinballLo"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    df_show["PinballHi"] = df_show["PinballHi"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    render_table(
        df_show[["No","Model","Peran","Sumber","MAE","RMSE","MAPE",
                 "R2","DirAcc","Coverage","AvgWidth"]]
    )

    with st.expander("Uji Signifikansi Statistik (paired bootstrap, N=5000, CI 95%)"):
        st.markdown("""
        Selisih MAE antar-model diuji signifikansinya karena ukuran test set
        (373 tanggal bersama) rentan terhadap noise. Perbandingan **Model Usulan vs.
        masing-masing model pembanding**:

        | Perbandingan | Selisih MAE | p-value | Kesimpulan |
        |---|---|---|---|
        | vs XGBoost | −$89,2 | 0,0140 | Signifikan pada uji nominal |
        | vs LightGBM | −$162,6 | <0,0002 | Signifikan pada uji nominal |
        | vs SVR | −$666,3 | <0,0002 | Signifikan pada uji nominal |
        | vs Random Forest | +$28,7 | 0,1668 | Tidak signifikan |
        | vs LSTM | +$38,3 | 0,0816 | Tidak signifikan |

        Pada uji nominal, selisih MAE terhadap XGBoost, LightGBM, dan SVR
        terdeteksi; selisih terhadap Random Forest dan LSTM tidak terdeteksi.
        Ini adalah bootstrap biasa pada observasi harian berurutan, tanpa
        koreksi uji berganda. Hasilnya bersifat eksploratif. Interval model
        usulan memiliki coverage empiris 91,2% pada periode historis ini.
        """)

    with st.expander("Ablation Study (Kontribusi Regime HMM & Sentiment-Weighted Netflow)"):
        st.markdown("""
        Ablasi memakai 387 baris uji awal, sedangkan Tabel 4.1 memakai 373
        tanggal bersama untuk mengakomodasi jendela LSTM. Dua varian model
        diuji terhadap Model Usulan FULL, dengan split dan hyperparameter identik:

        | Varian | MAE | Coverage | vs FULL |
        |---|---|---|---|
        | Model Usulan (FULL) | $1.467 | 91,2% | — |
        | Tanpa Regime (HMM) | $1.469 | 91,0% | Tidak signifikan (p 0,14–0,88 di semua metrik) |
        | Netflow Mentah (tanpa bobot sentimen) | $1.490 | 89,4% | MAE dan coverage lebih buruk pada uji nominal; PinballQ95 justru lebih baik (p<0,0001) |

        **Kesimpulan RQ2:** regime HMM belum menunjukkan kontribusi
        signifikan pada data uji ini. Pembobotan sentimen memberi hasil
        campuran: MAE dan coverage lebih baik pada uji nominal, tetapi
        pinball loss kuantil atas lebih buruk. Hasil ini tidak menguji
        kinerja live ketika netflow memakai forward-fill.
        """)

    st.markdown("<div class='section-label'>Visualisasi Metrik</div>",
                unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["MAE & MAPE", "R² & DirAcc", "Probabilistik"])

    bar_colors = ["#aab4bf"]*5 + [ACCENT]

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig_mae = go.Figure(go.Bar(
                y=df_res["Model"], x=df_res["MAE"],
                orientation="h", marker_color=bar_colors,
                text=[f"${v:,}" for v in df_res["MAE"]], textposition="outside"
            ))
            fig_mae.update_layout(
                title=dict(text="MAE (USD) — lebih rendah lebih baik", font=dict(color=TEXT, size=14)),
                template="plotly_white", paper_bgcolor=BG,
                plot_bgcolor=BG, height=320,
                font=dict(color=TEXT),
                margin=dict(l=0,r=60,t=40,b=0),
                xaxis=dict(tickprefix="$", gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT))
            )
            render_chart(fig_mae)
        with col2:
            fig_mape = go.Figure(go.Bar(
                y=df_res["Model"], x=df_res["MAPE"],
                orientation="h", marker_color=bar_colors,
                text=[f"{v:.2f}%" for v in df_res["MAPE"]], textposition="outside"
            ))
            fig_mape.update_layout(
                title=dict(text="MAPE (%) — lebih rendah lebih baik", font=dict(color=TEXT, size=14)),
                template="plotly_white", paper_bgcolor=BG,
                plot_bgcolor=BG, height=320,
                font=dict(color=TEXT),
                margin=dict(l=0,r=60,t=40,b=0),
                xaxis=dict(ticksuffix="%", gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT))
            )
            render_chart(fig_mape)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_r2 = go.Figure(go.Bar(
                y=df_res["Model"], x=df_res["R2"],
                orientation="h", marker_color=bar_colors,
                text=[f"{v:.4f}" for v in df_res["R2"]], textposition="outside"
            ))
            fig_r2.update_layout(
                title=dict(text="R² — lebih tinggi lebih baik", font=dict(color=TEXT, size=14)),
                template="plotly_white", paper_bgcolor=BG,
                plot_bgcolor=BG, height=320,
                font=dict(color=TEXT),
                margin=dict(l=0,r=80,t=40,b=0),
                xaxis=dict(range=[df_res["R2"].min()-0.003, df_res["R2"].max()+0.001],
                           gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT))
            )
            render_chart(fig_r2)
        with col2:
            fig_dir = go.Figure(go.Bar(
                y=df_res["Model"], x=df_res["DirAcc"],
                orientation="h", marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in df_res["DirAcc"]], textposition="outside"
            ))
            fig_dir.add_vline(x=50, line_dash="dash", line_color=TEXT_DIM)
            fig_dir.update_layout(
                title=dict(text="Directional Accuracy (%)", font=dict(color=TEXT, size=14)),
                template="plotly_white", paper_bgcolor=BG,
                plot_bgcolor=BG, height=320,
                font=dict(color=TEXT),
                margin=dict(l=0,r=60,t=40,b=0),
                xaxis=dict(ticksuffix="%", range=[40,60], gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT))
            )
            render_chart(fig_dir)
            st.caption("Garis putus-putus abu-abu menandai baseline tebak acak (50%).")

    with tab3:
        st.info("Metrik probabilistik hanya tersedia untuk Model Usulan "
                "(Coverage, Avg Width, Pinball Loss).")
        mu = df_res[df_res["Model"]=="Model Usulan"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coverage", f"{mu['Coverage']:.1f}%",
                  delta="Target: 90%", delta_color="off")
        c2.metric("Avg Width", f"${mu['AvgWidth']:,.0f}")
        c3.metric("Pinball Q05", f"{mu['PinballLo']:.1f}")
        c4.metric("Pinball Q95", f"{mu['PinballHi']:.1f}")

        fig_cov = go.Figure()
        fig_cov.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=mu["Coverage"],
            number={"suffix":"%", "font":{"color": TEXT, "size":36}},
            delta={"reference": 90, "valueformat": ".1f", "font":{"color": TEXT_DIM, "size":14}},
            gauge={
                "axis": {"range":[75,100], "ticksuffix":"%", "tickfont":{"color": TEXT_DIM}},
                "bar":  {"color": ACCENT},
                "bgcolor": SURFACE,
                "steps":[
                    {"range":[75,90], "color": "#e5e5ea"},
                    {"range":[90,100],"color": ACCENT_SOFT.replace("0.10","0.35")}
                ],
                "threshold":{
                    "line":{"color": TEXT, "width":3},
                    "thickness":0.75, "value":90
                }
            },
            title={"text":"Coverage Interval 90%", "font":{"color": TEXT_DIM, "size":14}}
        ))
        fig_cov.update_layout(
            template="plotly_white", paper_bgcolor=BG,
            font=dict(color=TEXT),
            height=280, margin=dict(t=40,b=0,l=0,r=0)
        )
        render_chart(fig_cov)

# ============================================================
# HALAMAN 3: ANALISIS DATA
# ============================================================
with tab_data:
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
            st.warning("Sentimen: data sintetis (API tidak tersedia)")
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
    <b>Sumber data:</b><br>
    • Harga: Yahoo Finance<br>
    • Sentimen: Crypto Fear & Greed Index<br>
    • On-Chain: CoinMetrics<br><br>
    <b>Model yang berjalan live di dashboard ini:</b><br>
    HMM Regime-Switching + XGBoost Quantile + Conformal Prediction.
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
