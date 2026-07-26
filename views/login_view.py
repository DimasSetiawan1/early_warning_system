"""
views/login_view.py
Tampilan halaman login.
"""

import streamlit as st
from controllers.auth_controller import login


def show_login_page():
    """Tampilkan halaman login."""
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("---")
        st.markdown(
            "<h1 style='text-align: center;'>Sistem Peringatan Dini</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h4 style='text-align: center; color: gray;'>Dasbor Prediksi Siswa Putus Sekolah</h4>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Nama Pengguna", placeholder="Masukkan nama pengguna")
            password = st.text_input("🔒 Kata Sandi", type="password", placeholder="Masukkan kata sandi")
            submitted = st.form_submit_button("🔑 Masuk", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("⚠️ Nama Pengguna dan kata sandi harus diisi!")
                else:
                    if login(username, password):
                        st.rerun()
                    else:
                        st.error("❌ Nama Pengguna atau kata sandi salah!")

        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; font-size: 0.8em; color: gray;'>"
            "Akun Default:<br>"
            "BK: <code>bk</code> / <code>admin123</code><br>"
            "Guru: <code>guru</code> / <code>admin123</code>"
            "</p>",
            unsafe_allow_html=True
        )
