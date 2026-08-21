import streamlit as st
import db
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.file_utils import format_datetime, load_file_from_path
from controllers.prediction_controller import prepare_pretrained_primary_run


def show_guru_prediction_page():
    """Halaman Uji Prediksi khusus BK (menggunakan Model Sistem)."""
    st.markdown("<h1 style='font-size: 24px; font-weight: 700; color: var(--color-primary); margin-bottom: 8px;'>Uji Data Prediksi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; color: var(--color-text-secondary); margin-bottom: 32px;'>Uji dataset menggunakan Model Sistem yang telah dilatih</p>", unsafe_allow_html=True)

    # Cek apakah model sistem tersedia
    if not os.path.exists('primary_model.pkl'):
        st.warning("⚠️ Model Sistem belum tersedia. Admin/BK perlu melakukan Konfigurasi Prediksi minimal satu kali untuk melatih Model Sistem.")
        return

    available_files = db.get_uploaded_files()
    
    # Filter hanya file yang masih ada di disk
    valid_files = [f for f in available_files if os.path.exists(f.get('file_path', ''))]

    if not valid_files:
        st.info("Belum ada file dataset yang tersedia. Silakan unggah file terlebih dahulu melalui menu Unggah Berkas.")
        return

    st.markdown("<p style='font-size: 11px; font-weight: 600; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px;'>PILIH DATASET UJI</p>", unsafe_allow_html=True)

    file_options = {}
    for f in valid_files:
        label = f"{f['original_filename']} — diunggah oleh {f['uploader_name']} pada {format_datetime(f['uploaded_at'])}"
        file_options[label] = f

    selected_file_label = st.selectbox(
        "File dataset yang akan diuji:",
        options=list(file_options.keys())
    )

    selected_file = file_options.get(selected_file_label)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
        prepare_pretrained_primary_run(selected_file)
