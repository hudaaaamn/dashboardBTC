"""Presentation of saved, auditable experiments; no invented fallback statistics."""
from pathlib import Path
import json
import hashlib

def read_report():
    root=Path(__file__).resolve().parent/'eksperimen_forward_fill'/'results'
    try:
        summary=json.loads((root/'summary.json').read_text(encoding='utf8'))
        if hashlib.sha256((root/'predictions.csv').read_bytes()).hexdigest()!=summary['sha256_predictions']:
            return None
        return summary
    except (OSError,ValueError,KeyError):
        return None

def render_report(st):
    s=read_report()
    st.markdown('Klaim validasi sebelumnya dicabut karena notebook asal tidak memuat kode '
                'dan prediksi per tanggal yang mendukung angka tersebut. Eksperimen berikut '
                'merupakan pengujian baru dengan protokol dan snapshot data tersimpan.')
    if s is None:
        st.warning('Berkas bukti eksperimen belum tersedia atau integritasnya tidak cocok. '
                   'Keamanan forward-fill belum dapat disimpulkan.')
        return
    b=s['bootstrap'][0]
    st.markdown(f"**Eksperimen historis baru:** {s['n_pairs']} pasangan pada "
        f"{s['n_unique_dates']} tanggal unik; 8 jendela terpisah, masing-masing 88 hari. "
        'Model dibekukan pada setiap cutoff; FRESH memakai arus historis aktual dan STALE '
        'memakai arus terakhir sebelum cutoff. Harga dan sentimen sama pada kedua kondisi. '
        'Inferensi HMM kausal; 194 tanggal kalibrasi per cutoff.')
    st.table({
        'Ukuran':['MAE (USD)','RMSE (USD)','MAPE (%)','Coverage (%)','Lebar interval (USD)'],
        'FRESH':[round(s['fresh'][k],3) for k in ['MAE','RMSE','MAPE','coverage','width']],
        'STALE':[round(s['stale'][k],3) for k in ['MAE','RMSE','MAPE','coverage','width']]})
    st.markdown(f"Selisih MAE STALE − FRESH: **{b['mean_delta_usd']:+.2f} USD**, "
        f"CI95% [{b['mean_ci95'][0]:.2f}; {b['mean_ci95'][1]:.2f}], p={b['mean_p']:.4f}. "
        f"Slope terhadap umur staleness: **{b['slope_usd_per_day']:+.3f} USD/hari**, "
        f"CI95% [{b['slope_ci95'][0]:.3f}; {b['slope_ci95'][1]:.3f}], p={b['slope_p']:.4f}. "
        f"Bootstrap blok 14 hari, {b['B']:,} replikasi; p Holm masing-masing "
        f"{b['mean_p_holm']:.4f} dan {b['slope_p_holm']:.4f}.")
    st.warning('Belum ada bukti cukup untuk menolak nol pada rancangan ini. Hasil tersebut '
        'tidak membuktikan ekuivalensi, keamanan, atau ketahanan pada semua lama keterlambatan. '
        'Eksperimen memakai model tetap, sedangkan dashboard melatih ulang; dua fitting HMM '
        'mengeluarkan peringatan konvergensi. Belum merupakan validasi operasional menyeluruh.')
    st.caption(f"MAE baseline harga terakhir: {s['persistence_MAE']:.2f} USD. "
        'Protokol: eksperimen_forward_fill/PROTOKOL.md; prediksi dan metrik: '
        'eksperimen_forward_fill/results. Sensitivitas blok 7 dan 28 hari tersedia di summary.json.')
