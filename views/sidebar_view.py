"""
views/sidebar_view.py
Tampilan sidebar navigasi berdasarkan peran pengguna.
"""

import streamlit as st
from controllers.auth_controller import logout


def show_sidebar():
    """Tampilkan sidebar dengan navigasi berdasarkan peran."""
    with st.sidebar:
        st.markdown(f"### 👋 Selamat Datang!")
        st.markdown(f"**{st.session_state.nama_lengkap}**")
        st.markdown(f"🏷️ Peran: `{st.session_state.role}`")
        st.markdown("---")

        st.markdown("### 📋 Menu Navigasi")

        role = st.session_state.role

        if role == 'BK':
            menu_items = [
                '📊 Dasbor Riwayat',
                '⚙️ Konfigurasi Prediksi',
                '👥 Manajemen Pengguna'
            ]
        elif role == 'Guru':
            menu_items = [
                '📤 Unggah Berkas',
                '📁 Manajemen Berkas'
            ]
        else:
            menu_items = []

        for item in menu_items:
            page_key = item.split(' ', 1)[1] if ' ' in item else item
            if st.button(item, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()

        st.markdown("---")

        if st.button("🚪 Keluar", use_container_width=True, type="primary"):
            logout()
            st.rerun()
