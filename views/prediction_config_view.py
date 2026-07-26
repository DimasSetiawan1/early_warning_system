"""
views/prediction_config_view.py
Tampilan halaman Konfigurasi Prediksi dan sub-konfigurasi per mode.
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import db
from models.file_utils import load_file_from_path
from models.ml_model import load_uci_artifacts
from controllers.prediction_controller import prepare_experiment_run, prepare_primary_run


def show_prediction_config():
    """Halaman konfigurasi prediksi — pilih mode dan file dataset."""
    st.title("⚙️ Konfigurasi Prediksi")

    # Load UCI artifacts
    model_uci, scaler_uci, features_uci = load_uci_artifacts()

    available_files = db.get_uploaded_files()

    if not available_files:
        st.info("💡 **Belum ada file dataset yang tersedia.** Silakan upload file terlebih dahulu melalui menu **Unggah Berkas**.")
        return

    st.markdown("---")
    col_mode, col_file = st.columns(2)

    with col_mode:
        mode = st.selectbox(
            "🛠️ Pilih Mode Analisis",
            options=[
                "Eksperimen (UCI Himpunan Data - Pre-trained Model)",
                "Data Primer (SMK Tunas Teknologi - Train On-the-fly)"
            ]
        )

    with col_file:
        file_options = {}
        for f in available_files:
            label = f"📄 {f['original_filename']} (oleh {f['uploader_name']})"
            file_options[label] = f

        selected_file_label = st.selectbox(
            "📂 Pilih File Himpunan Data",
            options=list(file_options.keys())
        )

    selected_file = file_options.get(selected_file_label)
    if selected_file is None:
        return

    df_raw = load_file_from_path(selected_file['file_path'])

    if df_raw is None:
        st.error("❌ Gagal memuat file dataset. File mungkin rusak atau sudah dihapus.")
        return

    st.success(f"✅ Berhasil memuat file: `{selected_file['original_filename']}` ({df_raw.shape[0]} baris × {df_raw.shape[1]} kolom)")

    if mode == "Eksperimen (UCI Himpunan Data - Pre-trained Model)":
        _config_experiment_mode(df_raw, selected_file)
    else:
        _config_primary_mode(df_raw, selected_file)


# ── Sub-konfigurasi Mode Eksperimen ──────────────────────────────────────────

def _config_experiment_mode(df_raw, selected_file):
    """Konfigurasi mode Eksperimen UCI (pre-trained)."""
    st.markdown("#### Konfigurasi Model (Pre-trained UCI)")
    st.info("Model menggunakan pre-trained weights. Tidak ada parameter yang perlu diubah.")
    if st.button("🚀 Proses Analisis", type="primary", use_container_width=True):
        prepare_experiment_run(df_raw, selected_file)


# ── Sub-konfigurasi Mode Data Primer ─────────────────────────────────────────

def _config_primary_mode(df_raw, selected_file):
    """Konfigurasi mode Data Primer (on-the-fly training)."""
    st.markdown("#### Konfigurasi Pembuatan Model C4.5")

    columns_list = list(df_raw.columns)
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        default_target_idx = 0
        for idx, col in enumerate(columns_list):
            if col.lower() in ['status', 'target', 'label']:
                default_target_idx = idx
                break
        target_col = st.selectbox(
            "Pilih Kolom Target (Y)",
            options=columns_list,
            index=default_target_idx
        )
        default_features = [col for col in columns_list if col != target_col]
        selected_train_features = st.multiselect(
            "Pilih Parameter Fitur (X)",
            options=default_features,
            default=default_features,
            help="Pilih fitur-fitur yang akan digunakan sebagai dasar pembelajaran C4.5."
        )

    with col_cfg2:
        test_size = st.slider("Ukuran Data Testing (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
        max_depth = st.slider("Kedalaman Maksimum Pohon (Max Depth)", min_value=3, max_value=15, value=7)
        use_ig_selection = st.checkbox(
            "Aktifkan Seleksi Fitur otomatis dengan Information Gain", value=False,
            help="Model hanya akan menggunakan fitur yang memiliki Information Gain >= threshold."
        )
        ig_threshold = 0.0
        if use_ig_selection:
            ig_threshold = st.slider("Threshold Information Gain", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

    # Fase 2 — Distribusi Kelas Target
    st.markdown("---")
    st.subheader("📊 Fase 2 — Distribusi Kelas Target")

    y_raw_preview = df_raw[target_col]
    target_counts = y_raw_preview.value_counts()
    target_labels = target_counts.index.tolist()
    target_values = target_counts.values.tolist()
    total_data = sum(target_values)

    color_palette = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    target_colors = color_palette[:len(target_labels)]

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.markdown("##### Distribusi Kelas Target (Asli)")
        fig_dist, ax_dist = plt.subplots(figsize=(7, 5))
        bars = ax_dist.bar(target_labels, target_values, color=target_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, target_values):
            ax_dist.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + total_data * 0.01,
                         f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_dist.set_ylabel('Jumlah Siswa', fontsize=11)
        ax_dist.set_title('Distribusi Kelas Target (Asli)', fontsize=12, fontweight='bold')
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    with dist_col2:
        st.markdown("##### Proporsi Kelas Target")
        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax_pie.pie(
            target_values, labels=target_labels, colors=target_colors,
            autopct='%1.1f%%', startangle=140, pctdistance=0.75,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        ax_pie.axis('equal')
        ax_pie.set_title('Proporsi Kelas Target', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    dist_summary = pd.DataFrame({
        'Kelas': target_labels,
        'Jumlah': target_values,
        'Proporsi (%)': [f"{v / total_data * 100:.1f}%" for v in target_values]
    })
    st.dataframe(dist_summary, use_container_width=True, hide_index=True)

    if st.button("🚀 Proses Analisis", type="primary", use_container_width=True):
        if len(selected_train_features) < 1:
            st.error("Pilih minimal 1 fitur untuk melatih model!")
        else:
            prepare_primary_run(df_raw, selected_file, {
                'target_col': target_col,
                'selected_train_features': selected_train_features,
                'test_size': test_size,
                'max_depth': max_depth,
                'use_ig_selection': use_ig_selection,
                'ig_threshold': ig_threshold,
            })
