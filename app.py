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
from views.sidebar_view import show_sidebar, show_public_sidebar
from views.dashboard_view import show_public_dashboard
from views.riwayat_prediction import show_riwayat_dashboard
from views.upload_view import show_upload_page
from views.file_management_view import show_file_management_page
from views.user_management_view import show_user_management_page
from views.prediction_config_view import show_prediction_config
from views.prediction_result_view import show_prediction_results

# ── INISIALISASI ───────────────────────────────────────────────────────────────
db.init_db()

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Peringatan Dini — SMK Tunas Teknologi",
    layout="wide"
)

# ── STYLING GLOBAL (CSS TOKEN) ──────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --color-bg: #F8F9FB;
  --color-surface: #FFFFFF;
  --color-border: #E2E6EA;
  --color-primary: #1B3A6B;
  --color-primary-light: #EBF0F9;
  --color-accent: #2563EB;
  --color-danger: #C0392B;
  --color-danger-light: #FDECEA;
  --color-success: #1A7A4A;
  --color-success-light: #E8F5EE;
  --color-text-primary: #111827;
  --color-text-secondary: #6B7280;
  --color-text-muted: #9CA3AF;
}
.stApp { background-color: var(--color-bg); }

/* Sidebar Active Button Styling */
section[data-testid="stSidebar"] button[kind="primary"] {
    background-color: var(--color-primary-light) !important;
    color: var(--color-primary) !important;
    border: none !important;
    border-left: 3px solid var(--color-accent) !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
}

/* Sidebar Inactive Button Styling */
section[data-testid="stSidebar"] button[kind="secondary"] {
    border: none !important;
    background-color: transparent !important;
    color: var(--color-text-primary) !important;
    text-align: left !important;
}

/* Hide default button borders in sidebar */
section[data-testid="stSidebar"] .stButton > button {
    display: flex;
    justify-content: flex-start;
    padding-left: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── STYLING ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True

# ── SESSION STATE ──────────────────────────────────────────────────────────────
init_session_state()

# ── ROUTING UTAMA ──────────────────────────────────────────────────────────────
page = st.session_state.get('current_page', 'Halaman Utama')
logged_in = st.session_state.get('logged_in', False)

if not logged_in:
    show_public_sidebar()
    if page == 'Masuk':
        show_login_page()
    else:
        show_public_dashboard()
else:
    show_sidebar()
    role = st.session_state.role

    # Peta halaman → (fungsi, peran yang diizinkan)
    PAGE_ROUTES = {
        'Halaman Utama':         (show_riwayat_dashboard,     ['BK', 'Guru']),
        'Dasbor Riwayat':        (show_riwayat_dashboard,     ['BK', 'Guru']),
        'Konfigurasi Prediksi':  (show_prediction_config,     ['BK']),
        'Hasil Prediksi':        (show_prediction_results,    ['BK', 'Guru']),
        'Unggah Berkas':         (show_upload_page,           ['Guru', 'BK']),
        'Manajemen Berkas':      (show_file_management_page,  ['Guru', 'BK']),
        'Manajemen Pengguna':    (show_user_management_page,  ['BK']),
    }

    if page in PAGE_ROUTES:
        view_fn, allowed_roles = PAGE_ROUTES[page]
        if role in allowed_roles:
            view_fn()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    else:
        show_riwayat_dashboard()
