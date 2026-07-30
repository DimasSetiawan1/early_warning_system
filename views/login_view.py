"""
views/login_view.py
Tampilan halaman login.
"""

import streamlit as st
from controllers.auth_controller import login


def show_login_page():
    """Tampilkan halaman login."""
    
    # Custom CSS specifically for login form card
    st.markdown("""
    <style>
    [data-testid="stForm"] {
        background: var(--color-surface) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 8px !important;
        padding: 40px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        max-width: 400px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("<br><br>", unsafe_allow_html=True)

        with st.form("login_form", border=True):
            st.markdown(
                "<h1 style='font-size: 20px; font-weight: 700; color: var(--color-text-primary); margin-bottom: 4px; padding-bottom: 0;'>Masuk ke Sistem</h1>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='font-size: 13px; color: var(--color-text-secondary); margin-bottom: 32px;'>Sistem Peringatan Dini &middot; SMK Tunas Teknologi</p>",
                unsafe_allow_html=True
            )
            
            username = st.text_input("Username", placeholder="Masukkan username")
            password = st.text_input("Kata Sandi", type="password", placeholder="Masukkan kata sandi")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Masuk", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Username atau kata sandi tidak sesuai.")
                else:
                    if login(username, password):
                        st.rerun()
                    else:
                        st.error("Username atau kata sandi tidak sesuai.")

        
