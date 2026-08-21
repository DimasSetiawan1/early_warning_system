"""
views/riwayat_prediction.py
Tampilan halaman Dasbor Riwayat Prediksi — menampilkan visualisasi dari hasil prediksi yang tersimpan.
"""

import json
import os
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
        st.info("Belum ada riwayat prediksi. Silakan jalankan Konfigurasi Prediksi atau Uji Prediksi terlebih dahulu.")
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

    # ── Muat data prediksi ──
    df_hist = None
    target_col = 'Predicted_Status'
    
    # Prioritas 1: Baca dari file prediksi per-run (disimpan di config_json)
    prediction_file = config_data.get('prediction_file', '')
    if prediction_file and os.path.exists(prediction_file):
        df_hist = pd.read_csv(prediction_file)
    
    # Prioritas 2: Baca dari latest_predictions.csv
    if df_hist is None and os.path.exists('dataset/latest_predictions.csv'):
        df_hist = pd.read_csv('dataset/latest_predictions.csv')
    
    # Prioritas 3: Coba baca dari uploaded_files (cara lama — fallback)
    if df_hist is None:
        uploaded_files = db.get_uploaded_files()
        matched_file = next(
            (f for f in uploaded_files if f['original_filename'] == selected_h['dataset_name']),
            None
        )
        if matched_file:
            file_path = matched_file['file_path']
            df_hist = load_file_from_path(file_path)
            # Pada cara lama, gunakan kolom target asli
            old_target = config_data.get('target', 'Status_DO')
            if old_target in df_hist.columns:
                target_col = old_target
            elif 'Status_DO' in df_hist.columns:
                target_col = 'Status_DO'
            elif 'Status' in df_hist.columns:
                target_col = 'Status'

    if df_hist is None or df_hist.empty:
        st.warning("File data hasil prediksi tidak dapat dimuat. File mungkin telah dihapus.")
        return

    # Pastikan target_col ada
    if target_col not in df_hist.columns:
        # Coba fallback
        for fallback in ['Predicted_Status', 'Status_DO', 'Status']:
            if fallback in df_hist.columns:
                target_col = fallback
                break
    
    if target_col not in df_hist.columns:
        st.warning("Kolom target/prediksi tidak ditemukan pada data ini.")
        return

    # ── Info Ringkasan Riwayat ──
    st.markdown(f"<p style='font-size: 13px; color: var(--color-text-secondary); margin-bottom: 8px;'>Dataset: <strong>{selected_h['dataset_name']}</strong> • Waktu: {format_datetime(selected_h['run_at'])} • Mode: {selected_h['mode_analisis']}</p>", unsafe_allow_html=True)
    
    # Tampilkan metrik jika ada
    if selected_h.get('accuracy', 0) > 0:
        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("Akurasi", f"{selected_h['accuracy']:.2f}%")
        mk2.metric("Presisi", f"{selected_h['precision']:.2f}%")
        mk3.metric("Recall", f"{selected_h['recall']:.2f}%")
        mk4.metric("F1-Score", f"{selected_h['f1_score']:.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALISASI DATA PREDIKSI
    # ═══════════════════════════════════════════════════════════════════════

    # Calculate stats
    total_siswa = len(df_hist)
    non_dropout_count = len(df_hist[df_hist[target_col] == 'Non-Dropout'])
    dropout_count = total_siswa - non_dropout_count
    pct_do = (dropout_count / total_siswa * 100) if total_siswa > 0 else 0

    # Draw Stat Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_stat_card("Total Siswa Dievaluasi", f"{total_siswa}", "Seluruh siswa dalam dataset"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_stat_card("Siswa Berisiko (DO)", f"{dropout_count}", f"Mewakili {pct_do:.1f}% dari keseluruhan", "var(--color-danger)"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_stat_card("Siswa Aman", f"{non_dropout_count}", f"Mewakili {100 - pct_do:.1f}% dari keseluruhan", "#2ECC71"), unsafe_allow_html=True)

    # Filter out Non-Dropout data from all visualisations
    df_risk = df_hist[df_hist[target_col] != 'Non-Dropout'].copy()

    if df_risk.empty:
        st.success("✅ Semua siswa dalam dataset ini terdeteksi Non-Dropout (Aman).")
        return

    # Warna per kategori risiko
    risk_categories = df_risk[target_col].unique().tolist()
    risk_palette = {}
    risk_colors = ['#C0392B', '#E67E22', '#8E44AD', '#2980B9', '#27AE60']
    for i, cat in enumerate(risk_categories):
        risk_palette[cat] = risk_colors[i % len(risk_colors)]

    # ── 1. Distribusi Kategori Risiko ──
    st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Kategori Risiko Dropout</p>", unsafe_allow_html=True)
    
    ch1, ch2 = st.columns(2)
    
    risk_counts = df_risk[target_col].value_counts()
    risk_labels = risk_counts.index.tolist()
    risk_values = risk_counts.values.tolist()
    
    with ch1:
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        bar_colors = [risk_palette.get(l, '#999') for l in risk_labels]
        bars = ax_bar.bar(risk_labels, risk_values, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
        for bar, val in zip(bars, risk_values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                       f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=13)
        ax_bar.set_ylabel('Jumlah Siswa', fontsize=11)
        ax_bar.set_title('Jumlah Siswa per Kategori Risiko', fontsize=12, fontweight='bold')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_bar, transparent=True)
        plt.close(fig_bar)
    
    with ch2:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        pie_colors = [risk_palette.get(l, '#999') for l in risk_labels]
        wedges, texts, autotexts = ax_pie.pie(
            risk_values, labels=risk_labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=140, pctdistance=0.75,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        ax_pie.set_title('Proporsi Kategori Berisiko', fontsize=12, fontweight='bold')
        ax_pie.axis('equal')
        plt.tight_layout()
        st.pyplot(fig_pie, transparent=True)
        plt.close(fig_pie)

    # ── 2. Distribusi per Gender ──
    has_gender = 'Gender' in df_risk.columns
    has_kelas = 'Kelas' in df_risk.columns
    has_angkatan = 'Angkatan' in df_risk.columns

    if has_gender:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Risiko per Gender</p>", unsafe_allow_html=True)
        
        gen_col1, gen_col2 = st.columns(2)
        
        with gen_col1:
            fig_g, ax_g = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_risk, x='Gender', hue=target_col, palette=risk_palette, ax=ax_g)
            ax_g.set_xlabel("")
            ax_g.set_ylabel("Jumlah Siswa", fontsize=11)
            ax_g.set_title("Siswa Berisiko per Gender", fontsize=12, fontweight='bold')
            ax_g.spines['top'].set_visible(False)
            ax_g.spines['right'].set_visible(False)
            ax_g.grid(axis='y', alpha=0.3)
            ax_g.legend(title="", frameon=False, loc="upper right")
            st.pyplot(fig_g, transparent=True)
            plt.close(fig_g)

        with gen_col2:
            st.markdown("<p style='font-size: 13px; font-weight: 600; color: var(--color-text-secondary); margin-bottom: 8px;'>Tabel Crosstab Gender × Status</p>", unsafe_allow_html=True)
            df_gen = pd.crosstab(df_risk['Gender'], df_risk[target_col], margins=True, margins_name="Total")
            st.dataframe(df_gen, use_container_width=True)

    # ── 3. Distribusi per Angkatan ──
    if has_angkatan:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Risiko per Angkatan</p>", unsafe_allow_html=True)
        
        ang_col1, ang_col2 = st.columns(2)
        
        with ang_col1:
            fig_ang, ax_ang = plt.subplots(figsize=(7, 4.5))
            ct_ang = pd.crosstab(df_risk['Angkatan'], df_risk[target_col])
            ct_ang.plot(kind='bar', ax=ax_ang, color=[risk_palette.get(c, '#999') for c in ct_ang.columns],
                       edgecolor='white', linewidth=1)
            ax_ang.set_title('Siswa Berisiko per Angkatan', fontsize=12, fontweight='bold')
            ax_ang.set_xlabel('')
            ax_ang.set_ylabel('Jumlah Siswa', fontsize=11)
            ax_ang.legend(title='', frameon=False, fontsize=9)
            ax_ang.spines['top'].set_visible(False)
            ax_ang.spines['right'].set_visible(False)
            ax_ang.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_ang, transparent=True)
            plt.close(fig_ang)
        
        with ang_col2:
            st.markdown("<p style='font-size: 13px; font-weight: 600; color: var(--color-text-secondary); margin-bottom: 8px;'>Tabel Crosstab Angkatan × Status</p>", unsafe_allow_html=True)
            ct_ang_full = pd.crosstab(df_risk['Angkatan'], df_risk[target_col], margins=True, margins_name='Total')
            st.dataframe(ct_ang_full, use_container_width=True)

    # ── 4. Distribusi per Kelas ──
    if has_kelas:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Risiko per Kelas</p>", unsafe_allow_html=True)
        
        ct_kelas = pd.crosstab(df_risk['Kelas'], df_risk[target_col])
        
        fig_kelas, ax_kelas = plt.subplots(figsize=(10, max(5, len(ct_kelas) * 0.35)))
        ct_kelas.plot(kind='barh', ax=ax_kelas, 
                     color=[risk_palette.get(c, '#999') for c in ct_kelas.columns],
                     edgecolor='white', linewidth=1)
        ax_kelas.set_title('Siswa Berisiko per Kelas', fontsize=12, fontweight='bold')
        ax_kelas.set_xlabel('Jumlah Siswa', fontsize=11)
        ax_kelas.set_ylabel('')
        ax_kelas.legend(title='', frameon=False, fontsize=9)
        ax_kelas.spines['top'].set_visible(False)
        ax_kelas.spines['right'].set_visible(False)
        ax_kelas.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_kelas, transparent=True)
        plt.close(fig_kelas)

    # ── 5. Tabel Detail Siswa Berisiko ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>📋 Daftar Siswa Berisiko Dropout</p>", unsafe_allow_html=True)
    
    display_cols = []
    for col in ['NISN', 'Nama', 'Kelas', 'Angkatan', 'Gender', 'Nilai_Rata_Rata', 
                 'Kehadiran_Persen', 'Panggilan_BK', 'Status_Bantuan_PIP']:
        if col in df_risk.columns:
            display_cols.append(col)
    display_cols.append(target_col)
    
    if 'Probabilitas_Risiko' in df_risk.columns:
        display_cols.append('Probabilitas_Risiko')
    
    df_display = df_risk[display_cols].copy()
    if 'Probabilitas_Risiko' in df_display.columns:
        df_display['Probabilitas_Risiko'] = df_display['Probabilitas_Risiko'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '-')
    df_display = df_display.sort_values(target_col).reset_index(drop=True)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)

    st.markdown("<p style='font-size: 12px; color: var(--color-text-secondary); margin-top: 16px;'>Data ini bersumber dari hasil prediksi model C4.5 yang dilatih menggunakan data historis SMK Tunas Teknologi.</p>", unsafe_allow_html=True)
