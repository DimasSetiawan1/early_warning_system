"""
app.py — Entry Point Aplikasi Sistem Peringatan Dini
Bertugas sebagai router utama: inisialisasi, autentikasi, dan routing halaman.
Semua logika UI ada di views/, logika bisnis di controllers/, logika data di models/.
"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import db

from controllers.auth_controller import init_session_state

from views.login_view import show_login_page
from views.sidebar_view import show_sidebar
from views.dashboard_view import show_dashboard_riwayat
from views.upload_view import show_upload_page
from views.file_management_view import show_file_management_page
from views.user_management_view import show_user_management_page
from views.prediction_config_view import show_prediction_config
from views.prediction_result_view import show_prediction_results

# ── INISIALISASI ───────────────────────────────────────────────────────────────
db.init_db()

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dasbor Prediksi Mahasiswa/Siswa Dropout",
    page_icon="🎓",
    layout="wide"
)

# ── STYLING ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True

# ── SESSION STATE ──────────────────────────────────────────────────────────────
init_session_state()

# ── ROUTING UTAMA ──────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login_page()
else:
    show_sidebar()

    page = st.session_state.current_page
    role = st.session_state.role

    # Peta halaman → (fungsi, peran yang diizinkan)
    PAGE_ROUTES = {
        'Dasbor Riwayat':      (show_dashboard_riwayat,     ['BK']),
        'Konfigurasi Prediksi': (show_prediction_config,     ['BK']),
        'Hasil Prediksi':       (show_prediction_results,    ['BK', 'Guru']),
        'Unggah Berkas':        (show_upload_page,           ['Guru', 'BK']),
        'Manajemen Berkas':     (show_file_management_page,  ['Guru', 'BK']),
        'Manajemen Pengguna':   (show_user_management_page,  ['BK']),
    }

    if page in PAGE_ROUTES:
        view_fn, allowed_roles = PAGE_ROUTES[page]
        if role in allowed_roles:
            view_fn()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    else:
        # Fallback ke halaman default berdasarkan peran
        if role == 'BK':
            show_dashboard_riwayat()
        elif role == 'Guru':
            show_upload_page()
