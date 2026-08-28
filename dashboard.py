# ============================================================
# dashboard.py — Dashboard Prediksi Probabilistik Harga Bitcoin
# TA: Huda Muhammad Nur — UGM Sekolah Vokasi 2026
#
# Cara menjalankan:
#   pip install streamlit plotly yfinance xgboost hmmlearn requests
#   streamlit run dashboard.py
#
# CATATAN REVISI (Opsi B — on-chain fallback):
#   Sumber on-chain gratis (CoinMetrics Community CSV di GitHub) berhenti
#   diperbarui sejak CoinMetrics merestrukturisasi tier Community Data-nya
#   (~Mei 2026, lihat BAB I & Batasan Penelitian subbab 1.4). Dashboard ini
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
#   dijalankan terpisah di notebook Google Colab (Step 1-16), pada test
#   set n=387 (27 Apr 2025 - 18 Mei 2026, RUN_DATE_LOCK=2026-05-20), lalu
#   di-hardcode di sini sebagai referensi. Keputusan desain ini disengaja:
#   (1) melatih 5 model tambahan (termasuk LSTM) pada setiap kunjungan
#   akan membuat waktu muat dashboard jauh lebih lambat, (2) angka
#   komparasi perlu tetap pada kondisi data on-chain "bersih" (belum
#   forward-fill) seperti saat eksperimen skripsi dijalankan, sehingga
#   tidak relevan untuk dihitung ulang dengan data live yang mungkin
#   sudah forward-fill. Jika notebook eksperimen dijalankan ulang dengan
#   data baru, angka pada results_data perlu diperbarui manual mengikuti
#   hasil notebook tsb (lihat Tabel 4.1/4.2/4.3 BAB IV).
#
# CATATAN REDESIGN TAMPILAN (2026):
#   Berkas ini adalah hasil redesign UI dari versi sebelumnya. Sesuai PRD
#   redesign, TIDAK ADA perubahan pada logika backend, sumber data, alur
#   perhitungan, caching, atau nama/fungsi Python. Perubahan murni pada
#   lapisan tampilan: skema warna & tipografi baru, navigasi sidebar
#   diganti tab horizontal di bagian atas, layout dibuat responsif untuk
#   tablet/mobile, dan komponen visual (kartu, badge, expander) dirancang
#   ulang. Semua informasi & angka yang wajib tampil menurut PRD tetap ada.
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
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
import urllib.parse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
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
# DESIGN TOKENS
# ------------------------------------------------------------
# Redesign kedua: acuan visual adalah apple.com — terang, lega, hairline
# tipis pengganti border tebal, bayangan sangat halus pengganti outline,
# satu aksen warna yang dipakai hemat. Warna dasar & teks memakai palet
# neutral khas Apple (page #f5f5f7, teks #1d1d1f/#6e6e73/#86868b, hairline
# #d2d2d7). Jingga Bitcoin dipertahankan sebagai SATU-SATUNYA aksen produk
# (bukan biru generik Apple) supaya identitas "BTC Dashboard" tetap ada —
# dipakai hemat, persis prinsip restraint Apple: satu warna berani, sisanya
# netral. Teal dipertahankan sebagai aksen sekunder khusus utk prediksi.
# Tipografi: system font stack (San Francisco di Apple devices, fallback
# Inter di platform lain) untuk seluruh teks — termasuk figur numerik,
# memakai tabular-nums alih-alih font monospace terpisah, supaya angka
# tetap rapi sejajar tanpa terasa "template terminal".
# ============================================================
BG        = "#f5f5f7"   # page background khas Apple
SURFACE   = "#ffffff"   # kartu/panel
SURFACE_2 = "#f2f2f4"   # elemen sekunder di dalam kartu (bar non-highlight, dll)
BORDER    = "#d2d2d7"   # hairline khas Apple
TEXT      = "#1d1d1f"   # teks utama, hitam pekat khas Apple
TEXT_DIM  = "#6e6e73"   # teks sekunder
TEXT_MUTE = "#86868b"   # teks tersier/caption
ACCENT    = "#e8830f"   # jingga Bitcoin, digelapkan sedikit agar kontras aman di atas putih
ACCENT_SOFT = "rgba(232,131,15,0.10)"
TEAL      = "#0f9488"   # aksen sekunder (digelapkan dari versi neon agar terbaca di atas putih) — prediksi / model usulan
TEAL_SOFT = "rgba(15,148,136,0.10)"
GREEN     = "#1e8e3e"   # bull / positif — digelapkan agar teks tetap terbaca di atas putih
RED       = "#d5372b"   # bear / negatif — idem
AMBER     = "#b1740f"   # peringatan (teks); versi cerah dipakai khusus utk elemen non-teks
GRID      = "#e5e5ea"

REGIME_COLORS = {0: RED, 1: TEXT_MUTE, 2: GREEN}
REGIME_NAMES  = {0: "Bear", 1: "Sideways", 2: "Bull"}

# ============================================================
# IKON KUSTOM (pengganti emoji 📈📊🔍ℹ️⚠️)
# ------------------------------------------------------------
# Emoji bawaan platform (📈📊🔍ℹ️⚠️) dirender oleh OS/browser sendiri —
# gaya, warna, dan proporsinya beda-beda antar perangkat, dan sering
# terkesan generik/"AI slop" karena bukan bagian dari sistem desain
# halaman ini. Sebagai gantinya dipakai ikon garis custom gaya
# SF Symbols/Lucide (stroke tunggal, ujung membulat, tanpa gradient/fill
# glossy), digambar sebagai SVG lalu di-encode jadi data-URI.
#
# Untuk ikon pada st.tabs()/st.expander() (dua widget native Streamlit
# yang labelnya cuma menerima teks polos, tidak menerima HTML), ikon
# disuntik lewat CSS `mask-image` pada pseudo-element ::before — dengan
# `background-color: currentColor` warnanya otomatis ikut warna teks tab
# (redup saat idle, gelap saat aktif) tanpa perlu aset terpisah per state.
# Untuk ikon di dalam blok HTML kustom (warn-box/disclaimer-box) yang
# memang sudah lewat st.markdown(unsafe_allow_html=True), ikon ditulis
# langsung sebagai inline SVG (WARN_ICON) karena warnanya tetap (merah).
# ============================================================
def _icon_svg(paths: str) -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="white" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">' + paths + '</svg>'
    )
    return "data:image/svg+xml," + urllib.parse.quote(svg)

# Tab "Prediksi" — garis tren naik (gaya ikon "trending-up")
ICON_TREND  = _icon_svg(
    '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
    '<polyline points="17 6 23 6 23 12"/>'
)
# Tab "Komparasi Model" — batang perbandingan (gaya ikon "bar-chart")
ICON_BARS   = _icon_svg(
    '<path d="M3 3v18h18"/><path d="M18 17V9"/>'
    '<path d="M13 17V5"/><path d="M8 17v-3"/>'
)
# Tab "Analisis Data" — kaca pembesar (gaya ikon "search")
ICON_SEARCH = _icon_svg(
    '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'
)
# Expander "Tentang dashboard" — huruf i dalam lingkaran (gaya ikon "info")
ICON_INFO   = _icon_svg(
    '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'
)

# Ikon peringatan/disclaimer — lingkaran merah + tanda seru, dipakai
# langsung sebagai inline SVG (bukan mask) di dalam blok HTML kustom.
WARN_ICON = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" style="display:inline-block;'
    'vertical-align:-2.5px;margin-right:6px;flex-shrink:0;">'
    f'<circle cx="12" cy="12" r="9.4" stroke="{RED}" stroke-width="2.3"/>'
    f'<rect x="10.6" y="6.2" width="2.8" height="7.8" rx="1.4" fill="{RED}"/>'
    f'<circle cx="12" cy="17" r="1.6" fill="{RED}"/></svg>'
)

# ============================================================
# CUSTOM CSS
# ------------------------------------------------------------
# Bahasa visual: apple.com. Tidak ada gradient dekoratif, tidak ada
# border tebal berwarna — kartu dibedakan dari latar lewat bayangan
# sangat halus + hairline 1px senada latar, radius besar (18–20px),
# dan banyak ruang kosong. Satu aksen (jingga Bitcoin) dipakai hemat;
# elemen lain netral. Font: system stack (San Francisco di perangkat
# Apple, Inter di platform lain) untuk semua teks termasuk angka —
# angka memakai font-variant-numeric: tabular-nums supaya tetap rapi
# sejajar tanpa perlu font monospace terpisah.
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                 "Inter", "Helvetica Neue", Arial, sans-serif;
}}

html, body, [class*="css"] {{
    font-family: var(--font-sans);
    background-color: {BG};
    color: {TEXT};
    -webkit-font-smoothing: antialiased;
}}
.stApp {{ background-color: {BG}; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ font-family: var(--font-sans); letter-spacing:-0.01em; }}
p, span, div, label {{ font-family: var(--font-sans); }}
code, .stMarkdown code {{ font-family: "SF Mono", "IBM Plex Mono", ui-monospace, monospace; font-size:0.92em; }}
.tnum {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }}

/* Blok konten utama diberi lebar maksimum + padding lega, khas halaman produk Apple */
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1180px; }}

/* Sembunyikan sidebar bawaan — navigasi dipindah ke tab atas */
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

/* ---- Topbar (pengganti sidebar) — nav bar bersih ala apple.com ---- */
.topbar {{
    display:flex; align-items:center; justify-content:space-between;
    gap:16px; flex-wrap:wrap;
    background: rgba(255,255,255,0.82);
    backdrop-filter: saturate(180%) blur(14px);
    -webkit-backdrop-filter: saturate(180%) blur(14px);
    border:1px solid {BORDER}; border-radius:18px;
    padding:16px 26px; margin-bottom:16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}
.topbar-brand {{ display:flex; align-items:center; gap:14px; }}
.topbar-coin {{
    width:38px; height:38px; border-radius:50%;
    background: {ACCENT};
    display:flex; align-items:center; justify-content:center;
    font-size:18px; font-weight:700; color:#ffffff;
    flex-shrink:0;
}}
.topbar-title h1 {{
    color:{TEXT}; font-size:19px; font-weight:700; margin:0; line-height:1.2;
    letter-spacing:-0.015em;
}}
.topbar-title p {{ color:{TEXT_DIM}; font-size:12.5px; margin:2px 0 0; }}
.topbar-status {{ display:flex; align-items:center; gap:10px; }}

/* ---- Status pill — netral, tanpa lampu lalu lintas merah-hijau generik ---- */
.status-pill {{
    display:inline-flex; align-items:center; gap:7px;
    font-size:12.5px; font-weight:500;
    padding:6px 13px 6px 11px; border-radius:20px; white-space:nowrap;
    background:{SURFACE_2}; color:{TEXT_DIM};
    border:1px solid transparent;
}}
.status-pill .dot {{
    width:7px; height:7px; border-radius:50%; flex-shrink:0;
}}
.status-live .dot {{ background:{TEAL}; }}
.status-live {{ color:{TEXT}; }}
.status-stale .dot {{ background:{ACCENT}; }}
.status-stale {{ color:{TEXT}; background:{ACCENT_SOFT}; }}

/* ---- Tab navigasi — segmented control ala apple.com ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap:4px; background:{SURFACE_2}; border:none;
    border-radius:12px; padding:5px; margin-bottom:22px;
    width:fit-content;
}}
.stTabs [data-baseweb="tab"] {{
    height:40px; border-radius:9px; padding:0 20px;
    color:{TEXT_DIM}; font-weight:590; font-size:14px;
    background:transparent; transition: background 0.15s ease;
}}
.stTabs [aria-selected="true"] {{
    background:{SURFACE} !important; color:{TEXT} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: transparent; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* ---- Ikon tab & expander (custom line-icon, pengganti emoji) ----
   Lihat blok "IKON KUSTOM" di atas untuk alasan & definisi ICON_*.
   `background-color: currentColor` + CSS mask supaya warna ikon
   otomatis mengikuti warna teks tab (redup saat idle, gelap saat aktif)
   tanpa perlu aset terpisah per-state. */
.stTabs [data-baseweb="tab"] {{
    display:flex !important; align-items:center; gap:8px;
}}
.stTabs [data-baseweb="tab"]::before {{
    content:""; width:16px; height:16px; flex-shrink:0;
    background-color: currentColor;
    -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
    -webkit-mask-size:contain; mask-size:contain;
    -webkit-mask-position:center; mask-position:center;
}}
.stTabs [data-baseweb="tab"]:nth-of-type(1)::before {{
    -webkit-mask-image:url('{ICON_TREND}'); mask-image:url('{ICON_TREND}');
}}
.stTabs [data-baseweb="tab"]:nth-of-type(2)::before {{
    -webkit-mask-image:url('{ICON_BARS}'); mask-image:url('{ICON_BARS}');
}}
.stTabs [data-baseweb="tab"]:nth-of-type(3)::before {{
    -webkit-mask-image:url('{ICON_SEARCH}'); mask-image:url('{ICON_SEARCH}');
}}
[data-testid="stExpander"] summary {{
    display:flex !important; align-items:center; gap:9px;
}}
[data-testid="stExpander"] summary::before {{
    content:""; width:16px; height:16px; flex-shrink:0;
    background-color:{TEXT_DIM};
    -webkit-mask-image:url('{ICON_INFO}'); mask-image:url('{ICON_INFO}');
    -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
    -webkit-mask-size:contain; mask-size:contain;
    -webkit-mask-position:center; mask-position:center;
}}

/* ---- Hilangkan sisa indikator underline merah bawaan BaseWeb ----
   Tab aktif sudah ditandai lewat pill putih ({{{{[aria-selected="true"]}}}}
   di atas) ala segmented-control apple.com. Indikator underline bawaan
   Streamlit/BaseWeb (warna tema merah, lebar mengikuti teks) tidak ikut
   didesain ulang oleh pill tsb, sehingga tampil sebagai sisa garis pendek
   di bawah teks yang terlihat seperti "terpotong". Dihilangkan total
   supaya tidak dobel dengan pill putih yang sudah jadi penanda utama. */
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] * {{
    border-bottom:none !important;
    box-shadow:none !important;
}}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 18px 20px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.035);
    overflow: visible;
    height: auto;
}}
[data-testid="stMetric"] label {{
    color: {TEXT_DIM} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0;
    text-transform: none;
}}
/* Nilai metrik: Streamlit secara default memotongnya dengan "..." kalau
   kontennya panjang (mis. "$76,145 – $82,xxx"), padahal masih ada ruang.
   Diizinkan membungkus ke baris kedua dan ukuran sedikit dikecilkan +
   dibuat sedikit menyesuaikan lebar kartu, supaya nilai selalu utuh
   terbaca tanpa terpotong di layar desktop mana pun. */
[data-testid="stMetricValue"] {{
    overflow: visible !important;
}}
[data-testid="stMetricValue"] > div {{
    white-space: normal !important;
    overflow-wrap: break-word !important;
    text-overflow: unset !important;
    overflow: visible !important;
    line-height: 1.2 !important;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {{
    color: {TEXT} !important;
    -webkit-text-fill-color: {TEXT} !important;
    font-family: var(--font-sans) !important;
    font-variant-numeric: tabular-nums;
    font-size: clamp(17px, 1.9vw, 23px) !important;
    font-weight: 650 !important;
    letter-spacing: -0.01em;
    background: none !important;
    text-shadow: none !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 13px !important; font-weight:500; }}

/* ---- Page header (per tab) ---- */
.page-header {{ margin-bottom:22px; }}
.page-header h2 {{
    color:{TEXT}; font-size:26px; font-weight:700; margin:0 0 5px;
    letter-spacing:-0.02em;
}}
.page-header p {{ color:{TEXT_DIM}; font-size:14px; margin:0; }}

/* ---- Regime badge ---- */
.regime-bull {{ background:rgba(30,142,62,0.10); color:{GREEN}; padding:6px 14px;
                border-radius:20px; font-weight:600; font-size:13px; border:none; }}
.regime-bear {{ background:rgba(213,55,43,0.10); color:{RED}; padding:6px 14px;
                border-radius:20px; font-weight:600; font-size:13px; border:none; }}
.regime-side {{ background:{SURFACE_2}; color:{TEXT_DIM}; padding:6px 14px;
                border-radius:20px; font-weight:600; font-size:13px; border:none; }}

/* ---- Info / warning boxes — kartu netral, aksen warna hanya di label ---- */
.info-box {{
    background:{SURFACE}; border:1px solid {BORDER};
    border-radius:16px; padding:16px 20px; font-size:13.5px; color:{TEXT_DIM};
    line-height:1.65; margin-top:12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}
.info-box b {{ color:{TEXT}; font-weight:600; }}
.warn-box {{
    background:#fdf6ea; border:1px solid rgba(177,116,15,0.25);
    border-radius:16px;
    padding:16px 20px; font-size:13.5px; color:#7a5109; line-height:1.65;
    margin-top:12px;
}}
.warn-box b {{ color:{AMBER}; font-weight:600; }}
.disclaimer-box {{
    background:{SURFACE_2}; border:1px solid {BORDER};
    border-radius:16px; padding:15px 20px; font-size:12.5px; color:{TEXT_DIM};
    line-height:1.6; margin-top:18px;
}}
.disclaimer-box b {{ color:{TEXT}; }}

/* ---- Section label ---- */
.section-label {{
    font-family: var(--font-sans);
    font-size:13px; font-weight:600; color:{TEXT};
    letter-spacing:0; text-transform:none;
    margin: 28px 0 14px; padding-bottom:8px;
    border-bottom:1px solid {BORDER};
}}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:14px; overflow:hidden; }}

/* ---- Expander ---- */
[data-testid="stExpander"] {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px;
}}
[data-testid="stExpander"] summary {{ font-weight:500; color:{TEXT} !important; }}
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {{ color:{TEXT} !important; }}
[data-testid="stExpanderDetails"] {{ color:{TEXT_DIM}; }}

/* ---- Pengaman kontras tambahan untuk widget native ----
   Beberapa komponen Streamlit bawaan (bukan hasil st.markdown kustom di
   atas) mengambil warna teks dari base theme aplikasi. Baris ini
   memastikan semuanya tetap gelap-di-atas-terang walau base theme app
   pengguna (config.toml / preferensi sistem) berbeda dari yang kita set. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
.stCaption, small {{ color:{TEXT_DIM} !important; }}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] b {{ color:{TEXT} !important; }}
[data-testid="stNotification"] p,
[data-testid="stNotification"] div {{ color:{TEXT} !important; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ color:{TEXT} !important; }}
.stSelectbox label, .stRadio label {{ color:{TEXT} !important; }}

/* ---- Fokus & shadow saat diklik — dibuat halus, bukan kotak gelap kasar ----
   Default focus ring beberapa komponen BaseWeb mengikuti warna base theme
   (bisa tampak sebagai kotak gelap yang kontras kasar di atas latar
   terang). Diganti dengan ring tipis warna aksen, tetap terlihat jelas
   untuk aksesibilitas keyboard, tapi tidak kasar saat diklik mouse. */
button:focus, button:focus-visible,
[data-baseweb="tab"]:focus, [data-baseweb="tab"]:focus-visible,
[tabindex]:focus-visible {{
    outline: 2px solid {ACCENT} !important;
    outline-offset: 2px;
    box-shadow: none !important;
}}
/* ---- Status widget bawaan Streamlit ("Running fungsi_x()...") ----
   Muncul otomatis di kiri-atas konten saat sebuah fungsi ber-decorator
   @st.cache_data (mis. build_features_and_predict()) sedang dieksekusi.
   Ini elemen "chrome" internal Streamlit sendiri (bukan hasil st.markdown
   kita), jadi selalu memakai tema gelap bawaan Streamlit terlepas dari
   CSS halaman ini -- perlu di-override manual. Beberapa versi Streamlit
   memakai testid berbeda, jadi beberapa selector dipasang sekaligus agar
   tetap kena di versi manapun. */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] > div,
div[class*="StatusWidget"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}}
[data-testid="stStatusWidget"] *,
div[class*="StatusWidget"] * {{
    color: {TEXT_DIM} !important;
    fill: {TEXT_DIM} !important;
    -webkit-text-fill-color: {TEXT_DIM} !important;
}}
[data-testid="stStatusWidget"] code {{
    color: {TEAL} !important;
    -webkit-text-fill-color: {TEAL} !important;
    background: {TEAL_SOFT} !important;
    border-radius: 4px;
    padding: 1px 4px;
}}

[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:active,
[data-testid="stExpander"] details[open] > summary {{
    box-shadow: none !important;
    background: {SURFACE_2} !important;
    color: {TEXT} !important;
}}
[data-testid="stExpander"] summary:focus *,
[data-testid="stExpander"] summary:hover *,
[data-testid="stExpander"] summary:active *,
[data-testid="stExpander"] details[open] > summary * {{
    color: {TEXT} !important;
    fill: {TEXT} !important;
}}
.stButton > button:focus:not(:focus-visible) {{ box-shadow:none !important; outline:none !important; }}

/* ---- color-scheme: light ----
   Chrome/Edge punya fitur "auto-darken" untuk kontrol native (button,
   input, select) yang mengikuti dark-mode OS/browser TERLEPAS dari CSS
   halaman, kalau halaman tidak secara eksplisit bilang dirinya bertema
   terang. Baris ini adalah sinyal resmi ke browser: "halaman ini memang
   terang", supaya <button> (Refresh Data) tidak lagi ikut dipaksa gelap. */
:root, html {{ color-scheme: light !important; }}

/* ---- Tombol (Refresh Data) — pill khas Apple ----
   Selector diperkuat (atribut kind + testid) supaya menang dari style
   bawaan BaseWeb yang kadang punya spesifisitas lebih tinggi. */
.stButton > button,
.stButton > button[kind],
[data-testid="stBaseButton-secondary"],
[data-testid="baseButton-secondary"] {{
    border-radius:980px !important; font-weight:590 !important;
    border:1px solid {BORDER} !important; background:{SURFACE} !important;
    color:{TEXT} !important; -webkit-text-fill-color:{TEXT} !important;
    transition: background 0.15s ease;
}}
.stButton > button *, [data-testid="stBaseButton-secondary"] * {{
    color:{TEXT} !important; -webkit-text-fill-color:{TEXT} !important;
}}
.stButton > button:hover {{ background:{SURFACE_2} !important; border-color:{TEXT_MUTE} !important; }}

/* ---- Tabel HTML kustom (pengganti st.dataframe) ----
   st.dataframe dirender lewat komponen grid (canvas) milik Streamlit yang
   mengambil warnanya dari tema internal Streamlit sendiri, BUKAN dari CSS
   halaman ini — makanya walau seluruh halaman sudah terang, tabelnya bisa
   tetap gelap kalau tema Streamlit belum kebaca sebagai "light" di sisi
   pengguna. Untuk tabel yang datanya sudah final untuk ditampilkan (bukan
   butuh sortir/scroll interaktif), dipakai tabel HTML biasa yang 100%
   ikut CSS di sini — jaminan selalu terang, di browser/perangkat manapun. */
.table-wrap {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px;
    overflow-x:auto; overflow-y:hidden; margin-top:8px;
    -webkit-overflow-scrolling: touch;
}}
table.apple-table {{
    width:100%; min-width:880px; border-collapse:collapse; font-size:13.5px;
    color:{TEXT};
}}
table.apple-table thead th {{
    background:{SURFACE_2}; color:{TEXT_DIM}; font-weight:600;
    text-align:left; padding:10px 14px; border-bottom:1px solid {BORDER};
    white-space:nowrap;
}}
table.apple-table tbody td {{
    padding:9px 14px; border-bottom:1px solid {BORDER};
    color:{TEXT}; white-space:nowrap;
}}
table.apple-table tbody tr:last-child td {{ border-bottom:none; }}
table.apple-table tbody tr:hover {{ background:{BG}; }}

/* ---- Jarak antar-kartu supaya tidak "nempel" ---- */
.warn-box, .info-box, .disclaimer-box {{ margin-bottom:20px; }}

/* ---- Menyeragamkan tinggi baris kartu (metrik, dll) agar rapi sejajar ---- */
[data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
[data-testid="stHorizontalBlock"] > div {{ display:flex !important; }}
[data-testid="stHorizontalBlock"] > div > div {{ width:100%; display:flex; }}
[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {{ width:100%; }}
[data-testid="stMetric"] {{
    display:flex !important; flex-direction:column; justify-content:center;
    width:100%;
}}

/* ---- Responsif tablet/mobile ---- */
@media (max-width: 900px) {{
    .topbar {{ padding:14px 18px; border-radius:16px; }}
    .topbar-title h1 {{ font-size:16.5px; }}
    .page-header h2 {{ font-size:21px; }}
    [data-testid="stMetricValue"] {{ font-size:20px !important; }}
    [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; }}
    [data-testid="stHorizontalBlock"] > div {{ min-width: 46% !important; }}
    .stTabs [data-baseweb="tab"] {{ padding:0 12px; font-size:12.5px; }}
    .block-container {{ padding-left:0.9rem; padding-right:0.9rem; }}
}}
@media (max-width: 560px) {{
    [data-testid="stHorizontalBlock"] > div {{ min-width: 100% !important; }}
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER — render tabel sebagai HTML (bukan st.dataframe)
# ------------------------------------------------------------
# Alasan: st.dataframe dirender lewat komponen grid milik Streamlit yang
# ikut tema internal Streamlit (bisa auto dark), bukan CSS kustom di file
# ini. Untuk tabel yang murni ditampilkan (tanpa perlu sortir/scroll
# interaktif ala grid), dipakai HTML table biasa yang 100% ikut styling
# apple.com di atas — dijamin selalu terang di browser/perangkat manapun.
# Ini murni cara MENAMPILKAN data yang sama, bukan mengubah datanya.
# ============================================================
def render_table(df: pd.DataFrame):
    html = df.to_html(index=False, escape=False, classes="apple-table", border=0)
    st.markdown(f"<div class='table-wrap'>{html}</div>", unsafe_allow_html=True)

# ============================================================
# FUNGSI FETCH DATA (di-cache 1 jam)
# ============================================================
@st.cache_data(ttl=3600)
def fetch_price(start="2021-01-01"):
    import yfinance as yf
    # PERBAIKAN (Agustus 2026 — bug "tertinggal 2 hari" walau candle H-1
    # sudah final di Yahoo Finance):
    # SEBELUMNYA memakai end=datetime.now().strftime("%Y-%m-%d") (exclusive).
    # Parameter 'end' pada yfinance ternyata bisa memotong LEBIH dari satu
    # hari karena dikonversi lewat timezone internal yfinance (bukan UTC
    # bersih), bukan cuma membuang hari ini seperti yang diharapkan. Efeknya:
    # candle H-1 yang sebenarnya SUDAH final & tersedia di Yahoo Finance ikut
    # terpotong, sehingga last_date jadi H-2.
    # PERBAIKAN: jangan batasi 'end' sama sekali (ambil semua data terbaru
    # yang tersedia dari yfinance, termasuk kemungkinan baris "hari ini").
    # Lalu kita sendiri yang membuang baris hari ini secara eksplisit
    # berdasarkan tanggal UTC (bukan lewat parameter end bawaan yfinance),
    # karena BTC-USD diperdagangkan 24 jam nonstop -- baris "hari ini" itu
    # masih live/berjalan, bukan candle harian yang sudah final/closed.
    # Basis UTC dipilih supaya konsisten dengan cara Yahoo Finance sendiri
    # menandai waktu real-time-nya (lihat "As of ... UTC" di halaman Yahoo
    # Finance untuk BTC-USD).
    df = yf.download("BTC-USD", start=start, interval="1d", progress=False)
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
    2. Jika tidak -> fallback ke CSV Community gratis di GitHub. Sumber ini
       kemungkinan besar TIDAK live lagi (CoinMetrics menghentikan
       pembaruan sebagian metrik Community Data sejak ~Mei 2026), sehingga
       datanya bisa berhenti di tanggal tertentu di masa lalu.

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
    # Catatan: RobustScaler TIDAK diimpor di sini (sebelumnya diimpor tapi
    # tak pernah dipakai). Model usulan berbasis pohon (HMM Gaussian pada
    # 2 fitur skala kecil + XGBoost) tidak memerlukan scaling fitur.
    # RobustScaler pada Tabel 3.2 relevan untuk model pembanding yang
    # sensitif terhadap skala (SVR), yang dilatih terpisah di notebook
    # Colab — bukan di dalam dashboard.py ini.

    # Fetch semua data
    df_p        = fetch_price()
    df_s, _     = fetch_sentiment()
    df_o, onchain_live = fetch_onchain()

    # --- Timeline utama mengikuti HARGA (selalu paling fresh) ---
    # Sentimen & on-chain di-left-join lalu forward-fill, supaya tanggal
    # terbaru (hari ini/kemarin) tidak ikut kepotong hanya karena salah
    # satu sumber lain belum update. Ini krusial untuk prediksi "besok"
    # yang benar-benar mengacu ke harga terkini, bukan ke tanggal terakhir
    # on-chain tersedia.
    df = df_p[["close"]].copy()
    df = df.join(df_s[["polarity"]], how="left")
    df = df.join(df_o[["exchange_netflow"]], how="left")

    # Catat kapan on-chain terakhir punya nilai ASLI (sebelum forward-fill)
    onchain_last_real_date = df_o.index.max() if len(df_o) else None
    last_price_date = df_p.index.max()
    onchain_staleness_days = (
        (last_price_date - onchain_last_real_date).days
        if onchain_last_real_date is not None else None
    )

    # Forward-fill: sentimen (jaga-jaga ada gap) dan on-chain (fallback utama)
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

    # Selama periode on-chain "beku" (forward-fill), netflow_change dipaksa
    # 0 secara eksplisit. Nilainya memang otomatis ~0 karena diff() dari
    # angka yang sama, tapi dibuat eksplisit agar perilakunya jelas dan
    # tidak tergantung presisi floating point.
    if onchain_last_real_date is not None:
        stale_mask = df.index > onchain_last_real_date
        df.loc[stale_mask, "netflow_change"] = 0.0

    df["netflow_ma_7"]     = df["exchange_netflow"].rolling(7).mean()
    df["sentiment_ma_7"]   = df["polarity"].rolling(7).mean()
    df["netflow_weighted"] = df["exchange_netflow"] * (1 + df["polarity"])

    # HMM regime
    # PERBAIKAN (menghindari data leakage): HMM sebelumnya di-fit() pada
    # SELURUH baris (termasuk periode yang nanti menjadi calibration/test
    # set), sehingga parameter transisi & emisi ikut "melihat" pola
    # return/volatilitas dari periode kalibrasi dan uji sebelum split
    # dilakukan. Ini bisa membuat kualitas label regime (fitur ke-16,
    # regime_labeled) pada test set terlihat lebih baik dari yang
    # seharusnya, dan berpotensi bias jawaban RQ2 (kontribusi regime
    # detection terhadap kualitas interval). Perbaikannya: fit HMM HANYA
    # pada porsi training kronologis (70% pertama), lalu decode seluruh
    # seri (train+calib+test) memakai parameter yang SAMA (Viterbi
    # decoding via hmm.predict(), bukan fit ulang) — proporsi 70% ini
    # harus konsisten dengan split train/calib/test di bawah.
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

    # PERBAIKAN (bug off-by-one tanggal prediksi): dropna() di bawah ini
    # SEBELUMNYA langsung diterapkan ke df setelah target_return dibuat.
    # Karena target_return baris paling akhir (tanggal harga terbaru)
    # selalu NaN (belum ada harga besok untuk dihitung log-return-nya),
    # dropna() ikut membuang baris tanggal terbaru itu — akibatnya
    # "prediksi besok" dan metrik "Harga Terakhir" di dashboard diam-diam
    # mengacu ke H-1, bukan tanggal harga paling baru seperti yang
    # dirancang pada subbab 3.3.2 (Perancangan Alur Data/Pipeline).
    # Perbaikan: simpan baris terakhir (fitur lengkap, tanpa perlu
    # target_return) sebelum dropna, khusus untuk prediksi live.
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

    # Prediksi untuk besok (pakai last_row_live, bukan df.iloc[[-1]], agar
    # benar-benar mengacu ke tanggal harga TERBARU — lihat catatan
    # perbaikan off-by-one di atas — bukan ke tanggal terakhir on-chain
    # ATAU ke H-1 akibat dropna() pada target_return)
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

    # Prediksi historis untuk chart (seluruh test set)
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
        "onchain_staleness_days": onchain_staleness_days,
    }

# ============================================================
# TOPBAR — branding (bagian yang tidak butuh data dulu)
# ============================================================
topbar_col, refresh_col = st.columns([6, 1])
with topbar_col:
    st.markdown("""
    <div class='topbar'>
        <div class='topbar-brand'>
            <div class='topbar-coin'>₿</div>
            <div class='topbar-title'>
                <h1>BTC Dashboard</h1>
                <p>Prediksi probabilistik harga Bitcoin · HMM + XGBoost Quantile + Conformal</p>
            </div>
        </div>
        <div id='topbar-status-slot'></div>
    </div>
    """, unsafe_allow_html=True)
with refresh_col:
    # Tombol refresh manual: memaksa bypass cache (ttl=3600) tanpa perlu
    # menunggu 1 jam. Berguna terutama saat candle harian BTC-USD di
    # Yahoo Finance baru saja final (lihat catatan di kepala berkas soal
    # delay basis hari US Eastern vs WIB).
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("Refresh Data", use_container_width=True,
                 help="Paksa ambil ulang data harga/sentimen/on-chain terbaru, bypass cache 1 jam"):
        st.cache_data.clear()
        st.rerun()

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
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# --- Status kesegaran data HARGA (baru) + on-chain + panel info ---
# Kesegaran harga dihitung dari selisih tanggal Close terakhir (last_date,
# hasil fetch_price()) terhadap tanggal hari ini. Selisih >1 hari berarti
# candle harian BTC-USD terbaru dari Yahoo Finance belum final/ter-publish
# (biasa terjadi karena basis hari Yahoo untuk crypto condong ke US
# Eastern, jauh di belakang WIB) — BUKAN berarti dashboard/model rusak.
price_last_date   = result["last_date"]
price_staleness_days = (pd.Timestamp(datetime.now().date()) - pd.Timestamp(price_last_date.date())).days
is_price_fresh = price_staleness_days <= 1

onchain_stale_days = result["onchain_staleness_days"]
is_onchain_fresh = result["onchain_live"] or (onchain_stale_days is not None and onchain_stale_days <= 1)

status_col0, status_col1, status_col2 = st.columns([1, 1, 1])
with status_col0:
    if is_price_fresh:
        st.markdown("<span class='status-pill status-live'><span class='dot'></span>Harga Up-to-date</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f"<span class='status-pill status-stale'><span class='dot'></span>Harga tertinggal {price_staleness_days} hari</span>",
            unsafe_allow_html=True)
with status_col1:
    if is_onchain_fresh:
        st.markdown("<span class='status-pill status-live'><span class='dot'></span>On-chain Live</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-pill status-stale'><span class='dot'></span>On-chain tertinggal {onchain_stale_days} hari</span>",
                    unsafe_allow_html=True)
with status_col2:
    st.markdown("<div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

with st.expander("Tentang dashboard ini"):
    st.markdown(f"""
    <div class='info-box' style='margin-top:0'>
    <b>Sumber data:</b><br>
    • Harga: Yahoo Finance<br>
    • Sentimen: Crypto Fear & Greed Index<br>
    • On-Chain: CoinMetrics<br><br>
    <b>Model yang berjalan live di dashboard ini:</b><br>
    HMM Regime-Switching + XGBoost Quantile + Conformal Prediction.
    <span style='font-size:11.5px;color:{TEXT_MUTE}'>
    Ini satu-satunya model yang dihitung ulang secara live. Lima model
    pembanding pada halaman "Komparasi Model" adalah referensi statis
    dari notebook eksperimen terpisah, bukan hasil live.
    </span>
    </div>
    <div class='disclaimer-box'>
    {WARN_ICON}<b>Bukan nasihat investasi.</b> Dashboard ini merupakan prototipe
    akademik bagian dari Tugas Akhir Program Studi Teknologi Rekayasa
    Perangkat Lunak, Universitas Gadjah Mada.
    </div>
    """, unsafe_allow_html=True)

tab_pred, tab_comp, tab_data = st.tabs(["Prediksi", "Komparasi Model", "Analisis Data"])

# ============================================================
# HALAMAN 1: PREDIKSI
# ============================================================
with tab_pred:
    st.markdown("""
    <div class='page-header'>
        <h2>Prediksi Probabilistik Harga Bitcoin</h2>
        <p>Model usulan: HMM Regime-Switching · XGBoost Quantile · Conformal Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Peringatan transparansi jika HARGA belum up-to-date (baru) ---
    if not is_price_fresh:
        st.markdown(f"""
        <div class='warn-box'>
        {WARN_ICON}<b>Harga acuan belum ter-update ke hari ini.</b> Candle harian
        BTC-USD terakhir dari Yahoo Finance yang tersedia adalah
        <b>{price_last_date.strftime('%d %B %Y')}</b> ({price_staleness_days} hari
        lalu), sehingga prediksi di bawah ini masih berpatokan pada tanggal
        tersebut, bukan hari ini. Ini biasanya terjadi karena Yahoo Finance
        menutup candle harian instrumen crypto berdasarkan basis hari
        <b>US Eastern</b> (jauh di belakang WIB), atau candle terbaru
        memang belum di-publish sumbernya. Coba klik tombol
        <b>🔄 Refresh Data</b> di kanan atas beberapa saat lagi untuk
        mengecek ulang tanpa menunggu cache (1 jam) habis sendiri.
        </div>
        """, unsafe_allow_html=True)

    # --- Peringatan transparansi jika on-chain tidak live ---
    if not is_onchain_fresh:
        st.markdown(f"""
        <div class='warn-box'>
        {WARN_ICON}<b>Data on-chain tidak real-time.</b> Sumber gratis (CoinMetrics
        Community CSV) terakhir memiliki data riil pada
        <b>{result['onchain_last_real_date'].strftime('%d %B %Y')}</b>
        ({onchain_stale_days} hari lalu). Fitur netflow pada prediksi ini
        menggunakan nilai historis terakhir yang tersedia (forward-fill),
        sehingga tidak mencerminkan aktivitas bursa terkini. Fitur harga dan
        sentimen tetap diperbarui real-time. Lihat DAFTAR PUSTAKA/BAB III
        (Batasan Penelitian) untuk penjelasan lebih lengkap.
        <br><br>
        <span style='font-size:11px;color:{TEXT_DIM}'>
        <b style='color:{AMBER}'>Sudah divalidasi:</b> simulasi kuantitatif pada
        data historis (n=704, 8 titik cutoff) menunjukkan efek forward-fill
        terhadap akurasi model <b>tidak signifikan secara statistik</b>
        (p=0,2184) dan tidak menunjukkan tren memburuk seiring lamanya
        staleness (p=0,604). Lihat halaman "Komparasi Model" →
        "Validasi Kuantitatif Strategi Forward-Fill" untuk detail.
        </span>
        </div>
        """, unsafe_allow_html=True)

    # --- Baris metrik atas ---
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
    regime_label = {0:"🔴 Bear", 1:"⚪ Sideways", 2:"🟢 Bull"}[last_regime]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Harga Terakhir ({last_date.strftime('%d %b %Y')})",
                   f"${last_close:,.0f}")
    with col2:
        st.metric(
            f"Prediksi {pred_date.strftime('%d %b %Y')}",
            f"${pred_med:,.0f}",
            delta=f"{delta_pct:+.2f}%"
        )
    with col3:
        st.metric("Interval 90%", f"${pred_lo:,.0f} – ${pred_hi:,.0f}")
    with col4:
        st.metric("Regime Pasar", regime_label)

    st.markdown("<div class='section-label'>Grafik Prediksi vs Harga Aktual (Test Set)</div>",
                unsafe_allow_html=True)

    # --- Chart prediksi historis ---
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
    # Titik prediksi besok
    fig.add_trace(go.Scatter(
        x=[pred_date], y=[pred_med],
        mode="markers", name=f"Prediksi {pred_date.strftime('%d %b')}",
        marker=dict(color=ACCENT, size=12, symbol="star"),
        error_y=dict(
            type="data", symmetric=False,
            array=[pred_hi - pred_med],
            arrayminus=[pred_med - pred_lo],
            color=ACCENT, thickness=2, width=8
        )
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="-apple-system, Inter, sans-serif"),
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(color=TEXT_DIM)),
        yaxis=dict(tickprefix="$", tickformat=",", gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
        xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Detail prediksi & sentimen ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-label'>Detail Prediksi Besok</div>",
                    unsafe_allow_html=True)
        width_pct = (pred_hi - pred_lo) / pred_med * 100
        st.markdown(f"""
        <div class='info-box'>
        <b>Tanggal prediksi:</b> {pred_date.strftime('%A, %d %B %Y')}<br>
        <b>Harga acuan ({last_date.strftime('%d %b %Y')}):</b> ${last_close:,.0f}<br>
        <b>Prediksi median:</b> ${pred_med:,.0f}
        &nbsp;(<span style='color:{GREEN if delta_pct>=0 else RED}'>{delta_pct:+.2f}%</span>)<br>
        <b>Interval bawah (5%):</b> ${pred_lo:,.0f}<br>
        <b>Interval atas (95%):</b> ${pred_hi:,.0f}<br>
        <b>Lebar interval:</b> ${pred_hi-pred_lo:,.0f} ({width_pct:.1f}% dari median)<br>
        <b>Conformal margin:</b> {result["conf_margin"]:.5f} (log-scale)<br>
        <br>
        <span style='color:{TEXT_MUTE};font-size:11px'>
        Interval 90% berarti 90% dari waktu, harga aktual diharapkan
        jatuh di antara batas bawah dan atas. Bukan jaminan.
        </span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-label'>Kondisi Pasar Hari Ini</div>",
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
        <div class='info-box'>
        <b>Regime HMM:</b> {regime_label}<br>
        <span style='font-size:12px;color:{TEXT_MUTE}'>{regime_desc}</span><br><br>
        <b>Sentimen pasar:</b>
        <span style='color:{sent_color}'>{sent_label}</span>
        (skor: {last_sent:.2f})<br><br>
        <b>On-chain netflow:</b>
        {last_netflow:+,.0f} BTC
        <span style='font-size:11px;color:{AMBER}'>{netflow_caption}</span><br>
        <span style='font-size:12px;color:{TEXT_MUTE}'>
        {"Lebih banyak BTC keluar bursa → potensi akumulasi (bullish)" if last_netflow < 0
         else "Lebih banyak BTC masuk bursa → potensi tekanan jual (bearish)"}
        </span><br><br>
        <b>Netflow terbobot sentimen:</b>
        {last_netflow * (1 + last_sent):+,.0f}
        <span style='font-size:11px;color:{TEXT_MUTE}'>(fitur usulan TA)</span>
        </div>
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

    # Data hasil evaluasi FINAL (test set n=387, 27 Apr 2025 - 18 Mei 2026,
    # RUN_DATE_LOCK=2026-05-20). Angka ini berasal dari notebook eksperimen
    # Google Colab (Step 1-16) yang dijalankan TERPISAH dari dashboard ini
    # -- lihat "CATATAN CAKUPAN MODEL" di kepala berkas. Kolom "Sumber"
    # menandai bahwa hanya Model Usulan yang dihitung ulang secara live
    # setiap dashboard dibuka; lima model pembanding di bawah adalah
    # referensi statis dan TIDAK dilatih ulang di sini.
    results_data = [
        {"No":1, "Model":"XGBoost",      "Peran":"Gradient boosting",
         "Sumber":"Notebook (offline)",
         "MAE":1565, "RMSE":2093, "MAPE":1.68, "R2":0.9856, "DirAcc":49.4,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":2, "Model":"LightGBM",     "Peran":"Gradient boosting",
         "Sumber":"Notebook (offline)",
         "MAE":1639, "RMSE":2191, "MAPE":1.77, "R2":0.9842, "DirAcc":47.5,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":3, "Model":"Random Forest","Peran":"Ensemble tree",
         "Sumber":"Notebook (offline)",
         "MAE":1441, "RMSE":1974, "MAPE":1.56, "R2":0.9872, "DirAcc":48.6,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":4, "Model":"SVR",          "Peran":"Support vector",
         "Sumber":"Notebook (offline)",
         "MAE":2147, "RMSE":3033, "MAPE":2.29, "R2":0.9697, "DirAcc":49.4,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":5, "Model":"LSTM",         "Peran":"Deep learning sekuensial",
         "Sumber":"Notebook (offline)",
         "MAE":1461, "RMSE":1990, "MAPE":1.59, "R2":0.9874, "DirAcc":50.7,
         "Coverage":None, "AvgWidth":None, "PinballLo":None, "PinballHi":None},
        {"No":6, "Model":"Model Usulan", "Peran":"HMM + XGB Quantile + Conformal",
         "Sumber":"Live (dashboard ini)",
         "MAE":1467, "RMSE":2003, "MAPE":1.59, "R2":0.9868, "DirAcc":53.7,
         "Coverage":91.2, "AvgWidth":7245, "PinballLo":247.4, "PinballHi":235.2},
    ]
    df_res = pd.DataFrame(results_data)

    st.markdown(f"""
    <div class='info-box'>
    Angka pada tabel dan grafik di bawah ini adalah hasil evaluasi pada
    <b>test set n=387</b> (27 Apr 2025 &ndash; 18 Mei 2026), dijalankan pada
    notebook eksperimen Google Colab dengan tanggal data dikunci
    (<code>RUN_DATE_LOCK=2026-05-20</code>) agar hasil dapat direproduksi.
    Kelima model pembanding (baris 1&ndash;5) merupakan <b>referensi statis</b>
    dari notebook tersebut dan tidak dilatih ulang setiap dashboard dibuka.
    Hanya <b>Model Usulan</b> (baris 6) yang dihitung ulang secara live oleh
    dashboard ini setiap kali data terbaru diambil &mdash; lihat halaman
    "Prediksi" untuk hasil live tersebut. Lihat kolom
    <b>Sumber</b> pada tabel untuk penanda ini.
    </div>
    """, unsafe_allow_html=True)

    # --- Tabel ---
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

    # --- Ringkasan uji signifikansi & ablation study (RQ2) ---
    with st.expander("📌 Uji Signifikansi Statistik (paired bootstrap, N=5000, CI 95%)"):
        st.markdown("""
        Selisih MAE antar-model diuji signifikansinya karena ukuran test set
        (387 baris) rentan terhadap noise. Perbandingan **Model Usulan vs.
        masing-masing model pembanding**:

        | Perbandingan | Selisih MAE | p-value | Kesimpulan |
        |---|---|---|---|
        | vs XGBoost | −$97,7 | 0,0076 | **Signifikan** (Model Usulan lebih baik) |
        | vs LightGBM | −$172,2 | <0,0001 | **Signifikan** (Model Usulan lebih baik) |
        | vs SVR | −$679,5 | <0,0001 | **Signifikan** (Model Usulan lebih baik) |
        | vs Random Forest | +$26,5 | 0,1844 | Tidak signifikan |
        | vs LSTM | +$10,7 | 0,6772 | Tidak signifikan |

        Model Usulan terbukti signifikan lebih akurat dari XGBoost, LightGBM,
        dan SVR. Terhadap Random Forest dan LSTM, selisih MAE tidak terbukti
        signifikan — dari sisi MAE murni, Model Usulan **kompetitif**, bukan
        superior. Keunggulan Model Usulan terhadap keduanya terletak pada
        Directional Accuracy yang lebih tinggi dan interval probabilistik
        terkalibrasi (Coverage 91,2%) yang tidak dimiliki Random Forest
        maupun LSTM.
        """)

    with st.expander("📌 Ablation Study — Kontribusi Regime HMM & Sentiment-Weighted Netflow (RQ2)"):
        st.markdown("""
        Dua varian model diuji terhadap Model Usulan FULL, dengan split dan
        hyperparameter identik:

        | Varian | MAE | Coverage | vs FULL |
        |---|---|---|---|
        | Model Usulan (FULL) | $1.467 | 91,2% | — |
        | Tanpa Regime (HMM) | $1.469 | 91,0% | Tidak signifikan (p 0,14–0,88 di semua metrik) |
        | Netflow Mentah (tanpa bobot sentimen) | $1.490 | 89,4% | **Signifikan** lebih buruk (MAE p=0,047; Coverage p=0,02; PinballQ95 p<0,0001) |

        **Kesimpulan RQ2:** dari dua komponen usulan, hanya
        *sentiment-weighted netflow* yang terbukti berkontribusi signifikan
        secara statistik terhadap akurasi dan kalibrasi interval. Kontribusi
        kuantitatif regime HMM tidak terbukti signifikan pada data uji ini,
        meskipun regime HMM tetap bermanfaat secara konseptual sebagai
        indikator kondisi pasar (lihat halaman "Analisis Data").
        """)

    with st.expander("📌 Validasi Kuantitatif Strategi Forward-Fill (Step 14–16)"):
        st.markdown("""
        Untuk menguji apakah forward-fill data on-chain (dipakai saat sumber
        gratis tertinggal, lihat peringatan di halaman "Prediksi") aman
        digunakan, dilakukan simulasi TRUE (data riil) vs STALE (forward-fill)
        pada 8 titik cutoff historis (n=704 baris total, jendela staleness
        88 hari — gap riil sejak CoinMetrics Community tier berhenti live).

        - Efek staleness terhadap MAE **tidak signifikan** secara keseluruhan:
          diff = +$10,38, CI95% [−6,41; +26,67], **p = 0,2184**.
        - Tidak ada tren memburuk yang konsisten seiring lamanya staleness:
          slope regresi +0,185 USD/hari, CI95% [−0,453; +0,878], **p = 0,604**.

        **Kesimpulan:** strategi forward-fill pada dashboard ini terbukti
        cukup aman — model tidak kehilangan akurasi yang signifikan secara
        statistik walau fitur netflow tidak live. Ini menjadi justifikasi
        kuantitatif bagi desain badge kesegaran data biner (Live / Tertinggal
        N hari) yang dipakai pada bagian atas halaman, tanpa perlu tingkatan
        keparahan bertahap berdasarkan lama staleness.
        """)

    # --- Chart MAE ---
    st.markdown("<div class='section-label'>Visualisasi Metrik</div>",
                unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["MAE & MAPE", "R² & DirAcc", "Probabilistik"])

    bar_colors = ["#c7c7cc"]*5 + [ACCENT]   # abu netral utk 5 model pembanding, aksen jingga hanya utk Model Usulan

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
            st.plotly_chart(fig_mae, use_container_width=True)
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
            st.plotly_chart(fig_mape, use_container_width=True)

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
            st.plotly_chart(fig_r2, use_container_width=True)
        with col2:
            fig_dir = go.Figure(go.Bar(
                y=df_res["Model"], x=df_res["DirAcc"],
                orientation="h", marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in df_res["DirAcc"]], textposition="outside"
            ))
            fig_dir.add_vline(x=50, line_dash="dash", line_color=TEXT_DIM)
            fig_dir.update_layout(
                title=dict(text="Directional Accuracy — lebih tinggi lebih baik", font=dict(color=TEXT, size=14)),
                template="plotly_white", paper_bgcolor=BG,
                plot_bgcolor=BG, height=320,
                font=dict(color=TEXT),
                margin=dict(l=0,r=60,t=40,b=0),
                xaxis=dict(ticksuffix="%", range=[40,60], gridcolor=GRID, tickfont=dict(color=TEXT_DIM)),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT))
            )
            st.plotly_chart(fig_dir, use_container_width=True)
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
        st.plotly_chart(fig_cov, use_container_width=True)

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

    # Rentang waktu
    col_r1, col_r2 = st.columns([3, 1])
    with col_r2:
        period = st.selectbox("Periode", ["6 Bulan","1 Tahun","2 Tahun","Semua"], index=1)

    period_map = {"6 Bulan":180, "1 Tahun":365, "2 Tahun":730, "Semua":9999}
    days       = period_map[period]
    cutoff     = df_full.index[-1] - timedelta(days=days)
    df_view    = df_full[df_full.index >= cutoff].copy()

    # --- Chart 1: Harga + Regime ---
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
    # Warnai background per regime
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
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Chart 2: Sentimen + Netflow ---
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
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_n:
        st.markdown("<div class='section-label'>On-Chain — Exchange Netflow</div>",
                    unsafe_allow_html=True)
        if not is_onchain_fresh:
            st.warning(
                f"Data on-chain historis terakhir: "
                f"{result['onchain_last_real_date'].strftime('%d %b %Y')}. "
                f"Bagian setelah tanggal ini adalah nilai forward-fill "
                f"(bukan data riil baru)."
            )
        nf_view = df_view.dropna(subset=["exchange_netflow"])
        fig_nf  = go.Figure()
        fig_nf.add_trace(go.Bar(
            x=nf_view.index,
            y=nf_view["exchange_netflow"],
            marker_color=[RED if v > 0 else GREEN
                          for v in nf_view["exchange_netflow"]],
            name="Netflow"
        ))
        fig_nf.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTE)
        if not is_onchain_fresh and result["onchain_last_real_date"] >= cutoff:
            # NOTE: add_vline(..., annotation_text=...) pada sumbu-x bertipe
            # datetime bisa memicu TypeError ("unsupported operand type(s)
            # for +: 'int' and 'datetime.datetime'") di beberapa versi Plotly,
            # karena internalnya menghitung rata-rata (mean) posisi anotasi
            # dengan mencampur tipe int dan datetime. Konversi ke
            # to_pydatetime() saja tidak selalu cukup untuk menghindarinya.
            # Solusi yang lebih aman: gambar garis vertikal via add_shape()
            # dan label via add_annotation() secara terpisah, tanpa memakai
            # jalur otomatis add_vline() sama sekali.
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
        st.plotly_chart(fig_nf, use_container_width=True)

    # --- Chart 3: Netflow Terbobot Sentimen (fitur usulan) ---
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
    st.plotly_chart(fig_nfw, use_container_width=True)
    st.markdown(f"""
    <div class='info-box'>
    <b>Fitur Usulan — Sentiment-Weighted Netflow</b>: Netflow on-chain dikalikan
    dengan bobot sentimen <code>(1 + polarity)</code>. Ketika sentimen Extreme Fear
    (polarity = −1), bobot = 0 sehingga sinyal netflow dilemahkan. Ketika Extreme Greed
    (polarity = +1), bobot = 2 sehingga sinyal diperkuat. Logika: perpindahan BTC ke
    bursa saat panik berbeda maknanya dibanding perpindahan yang sama saat euforia.
    </div>
    """, unsafe_allow_html=True)
