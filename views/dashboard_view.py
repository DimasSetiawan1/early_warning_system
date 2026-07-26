"""
views/dashboard_view.py
Tampilan halaman Dasbor Riwayat Prediksi.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import db
from models.file_utils import format_datetime, load_file_from_path


def show_dashboard_riwayat():
    """Dasbor riwayat prediksi dengan visualisasi distribusi kelas."""
    st.title("📊 Dashboard Riwayat Prediksi")

    history = db.get_prediction_history()

    if not history:
        st.info("Belum ada riwayat prediksi. Silakan jalankan analisis melalui menu **Konfigurasi Prediksi**.")
        return

    # Dropdown pilih riwayat
    history_options = {
        f"#{h['id']} — {h['dataset_name']} ({format_datetime(h['run_at'])})": h
        for h in history
    }

    selected_label = st.selectbox(
        "Pilih Riwayat Prediksi untuk Ditampilkan:",
        options=list(history_options.keys()),
        index=0
    )

    selected_h = history_options[selected_label]

    # Parse config_json
    config_data = {}
    if selected_h.get('config_json'):
        try:
            config_data = json.loads(selected_h['config_json'])
        except Exception:
            config_data = {}

    target_col = config_data.get('target', None)

    # Cari file dataset yang sesuai
    uploaded_files = db.get_uploaded_files()
    matched_file = next(
        (f for f in uploaded_files if f['original_filename'] == selected_h['dataset_name']),
        None
    )

    if matched_file is None:
        st.warning(f"File dataset **{selected_h['dataset_name']}** tidak ditemukan di storage.")
    elif target_col is None:
        st.warning("Informasi kolom target tidak tersimpan di riwayat ini.")
    else:
        df_hist = load_file_from_path(matched_file['file_path'])

        if df_hist is None or target_col not in df_hist.columns:
            st.warning("Himpunan Data atau kolom target tidak dapat dimuat.")
        else:
            color_palette = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C']

            # Metrik ringkasan
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Akurasi", f"{selected_h['accuracy']:.2f}%" if selected_h['accuracy'] else '-')
            with m2:
                st.metric("Presisi", f"{selected_h['precision']:.2f}%" if selected_h['precision'] else '-')
            with m3:
                st.metric("Daya Ingat", f"{selected_h['recall']:.2f}%" if selected_h['recall'] else '-')
            with m4:
                st.metric("F1-Score", f"{selected_h['f1_score']:.2f}%" if selected_h['f1_score'] else '-')

            st.markdown("---")

            # Distribusi kelas target dari dataset
            target_counts = df_hist[target_col].value_counts()
            target_labels = [str(l) for l in target_counts.index.tolist()]
            target_values = target_counts.values.tolist()
            total = sum(target_values)
            target_colors = color_palette[:len(target_labels)]

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown(f"##### Distribusi Kelas Target: `{target_col}`")
                fig1, ax1 = plt.subplots(figsize=(7, 5))
                bars = ax1.bar(target_labels, target_values, color=target_colors, edgecolor='white', linewidth=1.5)
                for bar, val in zip(bars, target_values):
                    ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + total * 0.01,
                             f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
                ax1.set_ylabel('Jumlah Siswa', fontsize=11)
                ax1.set_title(f'Distribusi Kelas: {target_col}', fontsize=12, fontweight='bold')
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.tick_params(colors='gray')
                ax1.yaxis.label.set_color('gray')
                plt.tight_layout()
                st.pyplot(fig1, transparent=True)
                plt.close(fig1)

            with chart_col2:
                st.markdown("##### Proporsi Kelas Target")
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                wedges, texts, autotexts = ax2.pie(
                    target_values, labels=target_labels, colors=target_colors,
                    autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                    textprops={'fontsize': 11, 'fontweight': 'bold'}
                )
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                ax2.axis('equal')
                ax2.set_title('Proporsi Kelas Target', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2, transparent=True)
                plt.close(fig2)

            dist_df = pd.DataFrame({
                'Kelas': target_labels,
                'Jumlah': target_values,
                'Proporsi (%)': [f"{v / total * 100:.1f}%" for v in target_values]
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
