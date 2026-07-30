"""
views/sidebar_view.py
Tampilan sidebar navigasi berdasarkan peran pengguna dan akses publik.
"""

import streamlit as st
from controllers.auth_controller import logout


def show_public_sidebar():
    """Tampilkan sidebar untuk pengunjung umum / tamu (belum login)."""
    with st.sidebar:
        st.image("images.jpg", width=80)
        st.markdown("### Sistem Peringatan Dini")
        st.markdown("SMK Tunas Teknologi")
        st.markdown("---")

        curr = st.session_state.get('current_page', 'Halaman Utama')
        
        home_type = "primary" if curr == 'Halaman Utama' else "secondary"
        if st.button("Dasbor Publik", key="nav_public_home", use_container_width=True, type=home_type):
            st.session_state.current_page = 'Halaman Utama'
            st.rerun()

        login_type = "primary" if curr == 'Masuk' else "secondary"
        if st.button("Masuk", key="nav_public_login", use_container_width=True, type=login_type):
            st.session_state.current_page = 'Masuk'
            st.rerun()


def show_sidebar():
    """Tampilkan sidebar dengan navigasi pengguna yang sudah login."""
    with st.sidebar:
        st.image("images.jpg", width=80)
        st.markdown("### Sistem Peringatan Dini")
        st.markdown("SMK Tunas Teknologi")
        st.markdown("---")

        role = st.session_state.role

        if role == 'BK':
            menu_items = [
                'Dasbor Publik',
                'Riwayat Prediksi',
                'Konfigurasi Prediksi',
                'Manajemen Pengguna'
            ]
            page_mapping = {
                'Dasbor Publik': 'Halaman Utama',
                'Riwayat Prediksi': 'Dasbor Riwayat',
                'Konfigurasi Prediksi': 'Konfigurasi Prediksi',
                'Manajemen Pengguna': 'Manajemen Pengguna'
            }
        elif role == 'Guru':
            menu_items = [
                'Dasbor Publik',
                'Manajemen File',
                'Riwayat Prediksi'
            ]
            page_mapping = {
                'Dasbor Publik': 'Halaman Utama',
                'Manajemen File': 'Manajemen Berkas',
                'Riwayat Prediksi': 'Hasil Prediksi' # Wait, riwayat prediksi for Guru should be "Manajemen Berkas" or "Dasbor Riwayat"? 
            }
            # Wait, page routes in app.py:
            # 'Dasbor Riwayat' (BK, Guru), 'Hasil Prediksi' (BK, Guru), 'Unggah Berkas' (Guru, BK), 'Manajemen Berkas' (Guru, BK)
            page_mapping['Riwayat Prediksi'] = 'Dasbor Riwayat'
            
        else:
            menu_items = ['Dasbor Publik']
            page_mapping = {'Dasbor Publik': 'Halaman Utama'}

        for item in menu_items:
            page_key = page_mapping[item]
            btn_type = "primary" if st.session_state.get('current_page', 'Halaman Utama') == page_key else "secondary"
            if st.button(item, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state.current_page = page_key
                st.rerun()

        st.markdown("---")
        st.markdown(f"<div style='font-size:12px; color:var(--color-text-secondary); margin-bottom:8px;'>Masuk sebagai: <b>{st.session_state.nama_lengkap}</b></div>", unsafe_allow_html=True)

        if st.button("Keluar", use_container_width=True, type="secondary"):
            logout()
            st.session_state.current_page = 'Halaman Utama'
            st.rerun()
