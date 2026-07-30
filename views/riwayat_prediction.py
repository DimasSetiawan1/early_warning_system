"""
views/dashboard_view.py
Tampilan halaman Dasbor Riwayat Prediksi & Halaman Utama Publik.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import db
from models.file_utils import format_datetime, load_file_from_path

def render_stat_card(label, value, subtext="", subtext_color="var(--color-text-secondary)"):
    return f"""
    <div style="
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: 8px;
      padding: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      margin-bottom: 24px;
    ">
      <div style="font-size:11px; font-weight:600; color:var(--color-text-secondary);
                  text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">
        {label}
      </div>
      <div style="font-size:40px; font-weight:700; color:var(--color-text-primary);
                  font-feature-settings:'tnum';">
        {value}
      </div>
      <div style="font-size:12px; color:{subtext_color}; margin-top:4px; font-weight: 500;">
        {subtext}
      </div>
    </div>
    """

def show_riwayat_dashboard():
    """Dasbor Riwayat Prediksi untuk pengguna login."""
    st.markdown("<h1 style='font-size: 24px; font-weight: 700; color: var(--color-primary); margin-bottom: 32px;'>Dasbor Riwayat Prediksi</h1>", unsafe_allow_html=True)

    history = db.get_prediction_history()

    if not history:
        st.info("Belum ada file data riwayat prediksi.")
        return

    # Dropdown pilih riwayat
    history_options = {
        f"#{h['id']} — {h['dataset_name']} ({format_datetime(h['run_at'])})": h
        for h in history
    }

    selected_label = st.selectbox(
        "Pilih Data / Riwayat untuk Ditampilkan",
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

    target_col = config_data.get('target', 'Status')

    # Cari file dataset yang sesuai
    uploaded_files = db.get_uploaded_files()
    matched_file = next(
        (f for f in uploaded_files if f['original_filename'] == selected_h['dataset_name']),
        None
    )

    file_path = matched_file['file_path'] if matched_file else f"dataset/{selected_h['dataset_name']}"
    df_hist = load_file_from_path(file_path)

    if df_hist is None or df_hist.empty:
        st.warning("File dataset tidak dapat dimuat.")
        return

    if target_col not in df_hist.columns:
        if 'Status_DO' in df_hist.columns:
            target_col = 'Status_DO'
        elif 'Status' in df_hist.columns:
            target_col = 'Status'

    if target_col not in df_hist.columns:
        st.warning("Target kolom tidak ditemukan pada data ini.")
        return

    # Calculate stats
    total_siswa = len(df_hist)
    non_dropout_count = len(df_hist[df_hist[target_col] == 'Non-Dropout'])
    dropout_count = total_siswa - non_dropout_count
    
    pct_do = (dropout_count / total_siswa * 100) if total_siswa > 0 else 0

    # Draw Stat Cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(render_stat_card("Total Siswa Dievaluasi", f"{total_siswa}", "Total Seluruh Siswa Terdaftar"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_stat_card("Siswa Berisiko (Dropout)", f"{dropout_count}", f"Mewakili {pct_do:.1f}% dari keseluruhan siswa", "var(--color-danger)"), unsafe_allow_html=True)

    # Filter out Non-Dropout data from all visualisations
    df_hist = df_hist[df_hist[target_col] != 'Non-Dropout'].copy()

    # Draw Charts (Gender & Class) if columns exist
    has_gender = 'Gender' in df_hist.columns
    has_kelas = 'Kelas' in df_hist.columns

    if (has_gender or has_kelas) and not df_hist.empty:
        ch1, ch2 = st.columns(2)
        
        palette_mapping = {}
        colors = ['#C0392B', '#E67E22', '#8E44AD', '#2980B9']
        for i, unique_tgt in enumerate(df_hist[target_col].unique()):
            palette_mapping[unique_tgt] = colors[i % len(colors)]

        with ch1:
            if has_gender:
                st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Distribusi Status berdasarkan Gender</p>", unsafe_allow_html=True)
                fig_g, ax_g = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_hist, x='Gender', hue=target_col, palette=palette_mapping, ax=ax_g)
                ax_g.set_xlabel("")
                ax_g.set_ylabel("Jumlah Siswa", fontsize=11, color="gray")
                ax_g.spines['top'].set_visible(False)
                ax_g.spines['right'].set_visible(False)
                ax_g.grid(axis='y', color='#E2E6EA', linestyle='-', linewidth=0.5)
                ax_g.legend(title="", frameon=False, loc="upper right")
                st.pyplot(fig_g, transparent=True)
                plt.close(fig_g)

        with ch2:
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Proporsi Kategori Berisiko</p>", unsafe_allow_html=True)
            status_counts = df_hist[target_col].value_counts()
            status_labels = status_counts.index.tolist()
            status_values = status_counts.values.tolist()
            
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            pie_colors = [palette_mapping[lbl] for lbl in status_labels]
            ax_pie.pie(
                status_values, labels=status_labels, colors=pie_colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            ax_pie.axis('equal')
            plt.tight_layout()
            st.pyplot(fig_pie, transparent=True)
            plt.close(fig_pie)

    if (has_gender or has_kelas) and not df_hist.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px; text-transform: uppercase;'>Tabel Distribusi Siswa Berisiko</p>", unsafe_allow_html=True)
        
        tab_c1, tab_c2 = st.columns(2)
        with tab_c1:
            if has_gender:
                st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Berdasarkan Gender</p>", unsafe_allow_html=True)
                df_gen = pd.crosstab(df_hist['Gender'], df_hist[target_col], margins=True, margins_name="Total")
                st.dataframe(df_gen, use_container_width=True)
        with tab_c2:
            if has_kelas:
                st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Berdasarkan Kelas</p>", unsafe_allow_html=True)
                df_kel = pd.crosstab(df_hist['Kelas'], df_hist[target_col], margins=True, margins_name="Total")
                st.dataframe(df_kel, use_container_width=True)

    st.markdown("<p style='font-size: 12px; color: var(--color-text-secondary); margin-top: 16px;'>Data ini bersumber dari hasil prediksi model C4.5 yang dilatih menggunakan data historis SMK Tunas Teknologi.</p>", unsafe_allow_html=True)
