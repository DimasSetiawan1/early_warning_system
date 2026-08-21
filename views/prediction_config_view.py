"""
views/prediction_config_view.py
Tampilan halaman Konfigurasi Prediksi dan sub-konfigurasi per mode.
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import db
from models.file_utils import load_file_from_path, format_datetime
from models.ml_model import load_uci_artifacts
from controllers.prediction_controller import prepare_experiment_run, prepare_primary_run


def show_prediction_config():
    """Halaman konfigurasi prediksi — pilih mode dan file dataset."""
    st.markdown("<h1 style='font-size: 24px; font-weight: 700; color: var(--color-primary); margin-bottom: 8px;'>Konfigurasi Prediksi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; color: var(--color-text-secondary); margin-bottom: 32px;'>Atur parameter model dan pilih dataset</p>", unsafe_allow_html=True)

    # Load UCI artifacts
    model_uci, scaler_uci, features_uci = load_uci_artifacts()

    import os
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'fix', 'data_siswa.csv')
    
    st.markdown("<p style='font-size: 11px; font-weight: 600; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px;'>LANGKAH 1 — Dataset Pelatihan</p>", unsafe_allow_html=True)
    
    if not os.path.exists(dataset_path):
        st.error("Dataset Utama (dataset/fix/data_siswa.csv) tidak ditemukan di direktori proyek!")
        return

    st.info("Menggunakan Dataset Master: **dataset/fix/data_siswa.csv**")
    
    selected_file = {
        'original_filename': 'data_siswa.csv',
        'file_path': dataset_path
    }
    
    df_raw = load_file_from_path(dataset_path)

    if df_raw is None:
        st.error("Gagal memuat file dataset. File mungkin rusak.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; font-weight: 600; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px;'>LANGKAH 2 — Parameter Model</p>", unsafe_allow_html=True)

    mode = "Data Primer (SMK Tunas Teknologi - Train On-the-fly)"
    _config_primary_mode(df_raw, selected_file)


# ── Sub-konfigurasi Mode Eksperimen ──────────────────────────────────────────

def _config_experiment_mode(df_raw, selected_file):
    """Konfigurasi mode Eksperimen UCI (pre-trained)."""
    st.info("Model menggunakan pre-trained weights. Tidak ada parameter yang perlu diubah.")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
        prepare_experiment_run(df_raw, selected_file)


# ── Sub-konfigurasi Mode Data Primer ─────────────────────────────────────────

def _config_primary_mode(df_raw, selected_file):
    """Konfigurasi mode Data Primer (on-the-fly training)."""
    columns_list = list(df_raw.columns)
    
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        default_target_idx = 0
        for idx, col in enumerate(columns_list):
            if col.lower() in ['status', 'status_do', 'target', 'label']:
                default_target_idx = idx
                break
        target_col = st.selectbox(
            "Kolom Target (Y)",
            options=columns_list,
            index=default_target_idx
        )
        default_features = [col for col in columns_list if col != target_col]
        selected_train_features = st.multiselect(
            "Parameter Fitur (X)",
            options=default_features,
            default=default_features,
            help="Pilih fitur-fitur yang akan digunakan sebagai dasar pembelajaran C4.5."
        )

    with col_cfg2:
        test_size = st.slider("Rasio Data Uji (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
        total_valid_data = df_raw[target_col].dropna().shape[0]
        test_count = math.ceil(total_valid_data * test_size)
        train_count = total_valid_data - test_count
        st.markdown(f"<p style='font-size: 13px; color: var(--color-text-secondary); margin-top: -10px; margin-bottom: 16px;'><i>Estimasi: <b>{train_count}</b> latih | <b>{test_count}</b> uji (Total: {total_valid_data})</i></p>", unsafe_allow_html=True)
        test_file = None
        
        st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)
        max_depth = st.number_input("Kedalaman Pohon (Max Depth)", min_value=3, max_value=15, value=7)
        use_ig_selection = st.checkbox("Aktifkan Seleksi Fitur otomatis dengan Information Gain", value=False)
        ig_threshold = 0.0
        if use_ig_selection:
            ig_threshold = st.slider("Threshold Information Gain", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

    # Fase 2 — Distribusi Kelas Target
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; font-weight: 600; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px;'>LANGKAH 3 — Distribusi Kelas Target</p>", unsafe_allow_html=True)

    y_raw_preview = df_raw[target_col]
    # Kecualikan Non-Dropout
    y_raw_preview = y_raw_preview[y_raw_preview != 'Non-Dropout']
    
    target_counts = y_raw_preview.value_counts()
    target_labels = target_counts.index.tolist()
    target_values = target_counts.values.tolist()
    total_data = sum(target_values) if sum(target_values) > 0 else 1

    color_palette = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    target_colors = color_palette[:len(target_labels)]

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Distribusi Siswa Berisiko</p>", unsafe_allow_html=True)
        fig_dist, ax_dist = plt.subplots(figsize=(7, 5))
        bars = ax_dist.bar(target_labels, target_values, color=target_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, target_values):
            ax_dist.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + total_data * 0.01,
                         f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_dist.set_ylabel('Jumlah Siswa', fontsize=11)
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_dist, transparent=True)
        plt.close(fig_dist)

    with dist_col2:
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Proporsi Siswa Berisiko</p>", unsafe_allow_html=True)
        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        if target_values:
            wedges, texts, autotexts = ax_pie.pie(
                target_values, labels=target_labels, colors=target_colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
        ax_pie.axis('equal')
        plt.tight_layout()
        st.pyplot(fig_pie, transparent=True)
        plt.close(fig_pie)

    dist_summary = pd.DataFrame({
        'Kelas': target_labels,
        'Jumlah': target_values,
        'Proporsi (%)': [f"{v / total_data * 100:.1f}%" for v in target_values]
    })
    st.dataframe(dist_summary, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
        if len(selected_train_features) < 1:
            st.error("Pilih minimal 1 fitur untuk melatih model.")
        else:
            prepare_primary_run(df_raw, selected_file, {
                'target_col': target_col,
                'selected_train_features': selected_train_features,
                'test_method': "Split dari Data Latih",
                'test_size': test_size,
                'test_file': None,
                'max_depth': max_depth,
                'use_ig_selection': use_ig_selection,
                'ig_threshold': ig_threshold,
            })
