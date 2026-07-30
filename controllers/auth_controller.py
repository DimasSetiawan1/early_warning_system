"""
controllers/auth_controller.py
Logika bisnis autentikasi: init session, login, logout.
"""

import streamlit as st
import db


def init_session_state():
    """Inisialisasi semua kunci session_state yang dibutuhkan aplikasi."""
    defaults = {
        'logged_in': False,
        'user_id': None,
        'username': None,
        'role': None,
        'nama_lengkap': None,
        'current_page': 'Halaman Utama',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def login(username: str, password: str) -> bool:
    """
    Coba autentikasi pengguna.
    Jika berhasil, set session_state dan return True.
    Jika gagal, return False.
    """
    user = db.authenticate_user(username, password)
    if user:
        st.session_state.logged_in = True
        st.session_state.user_id = user['id']
        st.session_state.username = user['username']
        st.session_state.role = user['role']
        st.session_state.nama_lengkap = user['nama_lengkap']
        # Halaman awal berdasarkan peran
        if user['role'] == 'Guru':
            st.session_state.current_page = 'Unggah Berkas'
        else:
            st.session_state.current_page = 'Dasbor Riwayat'
        return True
    return False


def logout():
    """Hapus semua data sesi dan kembalikan ke halaman login."""
    keys_to_clear = ['logged_in', 'user_id', 'username', 'role', 'nama_lengkap', 'current_page']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
