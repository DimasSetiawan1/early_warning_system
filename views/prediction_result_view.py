"""
views/prediction_result_view.py
Tampilan halaman Hasil Prediksi — menjalankan model dan menampilkan evaluasi lengkap.
"""

import io
import json as json_lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import time
from sklearn.metrics import roc_curve, auc
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import db
from models.ml_model import (
    load_uci_artifacts,
    preprocess_primary_data,
    apply_information_gain_selection,
    train_c45_model,
    evaluate_model,
    run_cross_validation
)
from models.file_utils import load_file_from_path


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

def show_prediction_results():
    """Entry point halaman Hasil Prediksi."""
    st.markdown("<h1 style='font-size: 24px; font-weight: 700; color: var(--color-primary); margin-bottom: 32px;'>Hasil Analisis Prediksi</h1>", unsafe_allow_html=True)

    if st.button("Kembali ke Dashboard", type="secondary"):
        st.session_state.current_page = 'Dasbor Riwayat'
        st.rerun()

    if 'run_config' not in st.session_state:
        st.warning("Tidak ada data hasil prediksi. Silakan jalankan konfigurasi terlebih dahulu.")
        return

    config = st.session_state.run_config
    is_new_run = not config.get('is_saved', False)

    if config['mode'] == 'Eksperimen':
        _run_experiment_mode(config['df_raw'])
    elif config['mode'] == 'Pre-trained Primer':
        _run_pretrained_primary_mode(config, is_new_run=is_new_run)
    else:
        _run_primary_mode(config, is_new_run=is_new_run)

    if is_new_run:
        st.session_state.run_config['is_saved'] = True


# ── MODE PRE-TRAINED PRIMER (Untuk BK — Uji Prediksi) ─────────────────────────

def _run_pretrained_primary_mode(config: dict, is_new_run: bool = False):
    """Jalankan evaluasi menggunakan model Primer yang tersimpan, lalu tampilkan dashboard visual."""
    st.markdown("<h2 style='font-size: 18px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px; margin-top: 32px;'>Evaluasi Data Menggunakan Model Sistem Tersimpan</h2>", unsafe_allow_html=True)

    test_file = config['test_file']
    dataset_name = config['dataset_name']
    
    from models.ml_model import load_primary_model, preprocess_primary_data
    
    saved_data = load_primary_model()
    if not saved_data:
        st.error("Model Sistem belum dilatih. Harap minta Admin/BK untuk melakukan Konfigurasi Prediksi terlebih dahulu.")
        return
        
    model = saved_data['model']
    scaler = saved_data['scaler']
    sys_config = saved_data['config']
    
    train_features = sys_config['features']
    target_col = sys_config['target']
    
    st.info(f"🚀 Memuat Dataset Uji: **{dataset_name}**")
    df_test_raw = load_file_from_path(test_file['file_path'])
    if df_test_raw is None:
        st.error("Gagal memuat dataset uji. File mungkin rusak atau dihapus.")
        return
        
    # Di Uji Prediksi, target_col TIDAK wajib ada (karena memang mau diprediksi)
    missing_cols = [c for c in train_features if c not in df_test_raw.columns]
    if missing_cols:
        st.error(f"Dataset Anda tidak kompatibel dengan Model Sistem. Kolom fitur yang hilang: {', '.join(missing_cols)}")
        return
        
    with st.spinner("Memproses data dan menjalankan prediksi..."):
        # Jika dataset tidak punya kolom target, kita abaikan label asli
        has_true_labels = target_col in df_test_raw.columns and df_test_raw[target_col].notna().any()
        
        X_test_numeric, y_test, class_names_from_test, categorical_cols = preprocess_primary_data(
            df_test_raw, target_col, train_features, is_training=False
        )
        
        class_names = sys_config.get('class_names', class_names_from_test)
        
        # Transform data
        X_test_scaled = scaler.transform(X_test_numeric)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)
        
        # Prediksi
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Convert prediksi numerik ke label teks
        if class_names:
            label_map = {i: name for i, name in enumerate(class_names)}
        elif len(model.classes_) == 2 and set(model.classes_) == {0, 1}:
            label_map = {0: 'Non-Dropout', 1: 'Dropout'}
        else:
            label_map = {i: str(c) for i, c in enumerate(model.classes_)}
        
        pred_labels = pd.Series(y_pred).map(label_map).values
        
        # Build dataframe hasil
        df_result = df_test_raw.copy()
        df_result['Predicted_Status'] = pred_labels
        
        # Probabilitas tertinggi
        df_result['Probabilitas_Risiko'] = y_pred_proba.max(axis=1)
        
        # Simpan ke latest_predictions.csv (untuk Dashboard Publik)
        import datetime, os
        os.makedirs('dataset', exist_ok=True)
        df_save = df_result.copy()
        df_save['Tested_Dataset'] = dataset_name
        df_save['Tested_At'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_save.to_csv('dataset/latest_predictions.csv', index=False)
        
        # Simpan file per-run (untuk Riwayat Prediksi)
        os.makedirs('dataset/prediction_runs', exist_ok=True)
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_filename = f"prediction_{run_timestamp}.csv"
        run_filepath = f"dataset/prediction_runs/{run_filename}"
        df_save.to_csv(run_filepath, index=False)
    
    # ═══════════════════════════════════════════════════════════════════════
    # DASHBOARD VISUAL HASIL UJI PREDIKSI
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.markdown("<h2 style='font-size: 20px; font-weight: 700; color: var(--color-primary); margin-bottom: 8px;'>📊 Dashboard Hasil Uji Prediksi</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 13px; color: var(--color-text-secondary); margin-bottom: 24px;'>Dataset: <strong>{dataset_name}</strong></p>", unsafe_allow_html=True)
    
    target = 'Predicted_Status'
    total_siswa = len(df_result)
    non_dropout_count = len(df_result[df_result[target] == 'Non-Dropout'])
    dropout_count = total_siswa - non_dropout_count
    pct_do = (dropout_count / total_siswa * 100) if total_siswa > 0 else 0
    
    # ── Stat Cards ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_stat_card(
            "Total Siswa Dievaluasi", f"{total_siswa}",
            "Seluruh siswa dalam dataset uji"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(render_stat_card(
            "Siswa Berisiko (DO)", f"{dropout_count}",
            f"Mewakili {pct_do:.1f}% dari keseluruhan",
            "var(--color-danger)"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(render_stat_card(
            "Siswa Aman", f"{non_dropout_count}",
            f"Mewakili {100 - pct_do:.1f}% dari keseluruhan",
            "#2ECC71"
        ), unsafe_allow_html=True)
    
    # Filter hanya siswa berisiko untuk visualisasi detail
    df_risk = df_result[df_result[target] != 'Non-Dropout'].copy()
    
    # Warna per kategori risiko
    risk_categories = df_risk[target].unique().tolist()
    risk_palette = {}
    risk_colors = ['#C0392B', '#E67E22', '#8E44AD', '#2980B9', '#27AE60']
    for i, cat in enumerate(risk_categories):
        risk_palette[cat] = risk_colors[i % len(risk_colors)]
    
    if not df_risk.empty:
        # ── 1. Distribusi Kategori Risiko DO ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Kategori Risiko Dropout</p>", unsafe_allow_html=True)
        
        ch1, ch2 = st.columns(2)
        
        risk_counts = df_risk[target].value_counts()
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
        
        # ── 2. Distribusi per Angkatan ──
        has_angkatan = 'Angkatan' in df_result.columns
        has_kelas = 'Kelas' in df_result.columns
        has_gender = 'Gender' in df_result.columns
        
        if has_angkatan:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Risiko per Angkatan</p>", unsafe_allow_html=True)
            
            ang_col1, ang_col2 = st.columns(2)
            
            with ang_col1:
                fig_ang, ax_ang = plt.subplots(figsize=(7, 4.5))
                ct_ang = pd.crosstab(df_risk['Angkatan'], df_risk[target])
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
                # Tabel crosstab
                st.markdown("<p style='font-size: 13px; font-weight: 600; color: var(--color-text-secondary); margin-bottom: 8px;'>Tabel Crosstab Angkatan × Status</p>", unsafe_allow_html=True)
                ct_ang_full = pd.crosstab(df_risk['Angkatan'], df_risk[target], margins=True, margins_name='Total')
                st.dataframe(ct_ang_full, use_container_width=True)
        
        # ── 3. Distribusi per Kelas ──
        if has_kelas:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Risiko per Kelas</p>", unsafe_allow_html=True)
            
            ct_kelas = pd.crosstab(df_risk['Kelas'], df_risk[target])
            
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
        
        # ── 4. Distribusi per Gender ──
        if has_gender:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Distribusi Risiko per Gender</p>", unsafe_allow_html=True)
            
            gen_col1, gen_col2 = st.columns(2)
            
            with gen_col1:
                fig_gen, ax_gen = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_risk, x='Gender', hue=target, 
                             palette=risk_palette, ax=ax_gen)
                ax_gen.set_title('Siswa Berisiko per Gender', fontsize=12, fontweight='bold')
                ax_gen.set_xlabel('')
                ax_gen.set_ylabel('Jumlah Siswa', fontsize=11)
                ax_gen.legend(title='', frameon=False, fontsize=9)
                ax_gen.spines['top'].set_visible(False)
                ax_gen.spines['right'].set_visible(False)
                ax_gen.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_gen, transparent=True)
                plt.close(fig_gen)
            
            with gen_col2:
                ct_gen = pd.crosstab(df_risk['Gender'], df_risk[target], margins=True, margins_name='Total')
                st.markdown("<p style='font-size: 13px; font-weight: 600; color: var(--color-text-secondary); margin-bottom: 8px;'>Tabel Crosstab Gender × Status</p>", unsafe_allow_html=True)
                st.dataframe(ct_gen, use_container_width=True)
        
        # ── 6. Tabel Detail Siswa Berisiko ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>📋 Daftar Siswa Berisiko Dropout</p>", unsafe_allow_html=True)
        
        # Pilih kolom yang relevan untuk ditampilkan
        display_cols = []
        for col in ['NISN', 'Nama', 'Kelas', 'Angkatan', 'Gender', 'Nilai_Rata_Rata', 
                     'Kehadiran_Persen', 'Panggilan_BK', 'Status_Bantuan_PIP']:
            if col in df_risk.columns:
                display_cols.append(col)
        display_cols.append(target)
        display_cols.append('Probabilitas_Risiko')
        
        df_display = df_risk[display_cols].copy()
        df_display['Probabilitas_Risiko'] = df_display['Probabilitas_Risiko'].apply(lambda x: f"{x*100:.1f}%")
        df_display = df_display.sort_values(target).reset_index(drop=True)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)
        
        # Download button
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Daftar Siswa Berisiko (CSV)",
            data=csv_data,
            file_name=f"siswa_berisiko_{dataset_name}",
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.success("✅ Semua siswa dalam dataset ini diprediksi **Non-Dropout** (Aman). Tidak ada siswa yang terdeteksi berisiko.")
    
    # ── Evaluasi Model (jika ada label asli) ──
    if has_true_labels:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Metrik Evaluasi Model (Perbandingan dengan Label Asli)", expanded=False):
            metrics = evaluate_model(model, X_test_scaled, y_test, class_names)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Akurasi", f"{metrics['acc']*100:.2f}%")
            m2.metric("Presisi", f"{metrics['prec']*100:.2f}%")
            m3.metric("Recall", f"{metrics['rec']*100:.2f}%")
            m4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
            
            st.text("=== CLASSIFICATION REPORT ===")
            st.text(metrics.get('report', ''))
    
    # ── Simpan History ──
    if is_new_run:
        acc_val = 0.0
        prec_val = 0.0
        rec_val = 0.0
        f1_val = 0.0
        if has_true_labels:
            metrics_save = evaluate_model(model, X_test_scaled, y_test, class_names)
            acc_val = float(metrics_save['acc'] * 100)
            prec_val = float(metrics_save['prec'] * 100)
            rec_val = float(metrics_save['rec'] * 100)
            f1_val = float(metrics_save['f1'] * 100)
        
        db.save_prediction_history(
            run_by=st.session_state.user_id,
            dataset_name=dataset_name,
            mode_analisis='Model Sistem Tersimpan (Uji Prediksi)',
            accuracy=acc_val,
            precision=prec_val,
            recall=rec_val,
            f1_score=f1_val,
            config_json=json_lib.dumps({
                'tested_on': dataset_name, 
                'train_features': train_features,
                'prediction_file': run_filepath,
                'class_names': class_names
            })
        )
        st.toast("Hasil prediksi berhasil disimpan ke riwayat.")


# ── MODE EKSPERIMEN (UCI Pre-trained) ─────────────────────────────────────────

def _run_experiment_mode(df_raw):
    """Tampilkan hasil prediksi menggunakan model pre-trained UCI."""
    model_uci, scaler_uci, features_uci = load_uci_artifacts()

    st.markdown("<h2 style='font-size: 18px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px; margin-top: 32px;'>Evaluasi Data Baru Menggunakan Model Eksperimen UCI</h2>", unsafe_allow_html=True)

    if model_uci is None or scaler_uci is None:
        st.error("Error: File model pre-trained (model_c45_dropout.pkl) atau scaler tidak ditemukan di direktori!")
        return

    st.markdown("<h3 style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Pencocokan Kolom (Parameter Mapping)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; color: var(--color-text-secondary); margin-bottom: 24px;'>Silakan pasangkan 8 parameter yang dibutuhkan model dengan kolom yang ada di file Anda:</p>", unsafe_allow_html=True)

    columns_list = list(df_raw.columns)
    mapped_columns = []
    col_map1, col_map2 = st.columns(2)

    for i, feat in enumerate(features_uci):
        default_idx = 0
        for idx, col in enumerate(columns_list):
            if feat.lower() in col.lower() or col.lower() in feat.lower():
                default_idx = idx
                break
        with col_map1 if i % 2 == 0 else col_map2:
            sel_col = st.selectbox(
                f"Parameter: **{feat}**",
                options=columns_list,
                index=default_idx,
                key=f"map_{i}"
            )
            mapped_columns.append((feat, sel_col))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Kolom Target Aktual (Opsional)</h3>", unsafe_allow_html=True)
    has_target = st.checkbox("File saya memiliki kolom label/target aktual (untuk menghitung Akurasi/Evaluasi)")

    target_col = None
    if has_target:
        default_tgt_idx = 0
        for idx, col in enumerate(columns_list):
            if col.lower() in ['target', 'status', 'status_do', 'label']:
                default_tgt_idx = idx
                break
        target_col = st.selectbox("Pilih Kolom Target Aktual", options=columns_list, index=default_tgt_idx)

    if st.button("Jalankan Prediksi Batch", type="primary"):
        st.markdown("---")
        feature_mapping_dict = {feat: sel_col for feat, sel_col in mapped_columns}
        X_input = df_raw[[feature_mapping_dict[feat] for feat in features_uci]].copy()
        X_input.columns = features_uci

        for col in X_input.columns:
            if X_input[col].isnull().sum() > 0:
                X_input[col].fillna(X_input[col].median(), inplace=True)

        try:
            X_scaled = scaler_uci.transform(X_input)
            y_pred = model_uci.predict(X_scaled)
            y_pred_proba = model_uci.predict_proba(X_scaled)[:, 1] if hasattr(model_uci, "predict_proba") else None

            df_result = df_raw.copy()
            df_result['Hasil_Prediksi_Numerik'] = y_pred
            df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].map({1: 'Dropout', 0: 'Non-Dropout'})
            if y_pred_proba is not None:
                df_result['Probabilitas_Dropout'] = y_pred_proba

            # KPI
            st.markdown("<h3 style='font-size: 18px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Dashboard Evaluasi Hasil Prediksi</h3>", unsafe_allow_html=True)
            total_cnt = len(df_result)
            dropout_cnt = int((y_pred == 1).sum())
            nondropout_cnt = total_cnt - dropout_cnt

            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.markdown(render_stat_card("Total Siswa Dievaluasi", f"{total_cnt}", f"{total_cnt} Orang"), unsafe_allow_html=True)
            with kpi2:
                st.markdown(render_stat_card("Diprediksi Dropout", f"{dropout_cnt}", f"{dropout_cnt / total_cnt * 100:.1f}%", "var(--color-danger)"), unsafe_allow_html=True)
            with kpi3:
                st.markdown(render_stat_card("Diprediksi Non-Dropout", f"{nondropout_cnt}", f"{nondropout_cnt / total_cnt * 100:.1f}%", "var(--color-success)"), unsafe_allow_html=True)

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("##### 📌 Distribusi Hasil Prediksi (Pie Chart)")
                fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
                ax_pie.pie([nondropout_cnt, dropout_cnt], labels=['Non-Dropout', 'Dropout'],
                           colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=140,
                           textprops={'fontsize': 12, 'weight': 'bold'})
                ax_pie.axis('equal')
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            with chart_col2:
                st.markdown("##### 📌 Sebaran Nilai Rata-rata Semester 2 vs Status Prediksi")
                grade_col = feature_mapping_dict.get('Curricular units 2nd sem (grade)')
                if grade_col in df_raw.columns:
                    fig_box, ax_box = plt.subplots(figsize=(7, 5.2))
                    sns.boxplot(data=df_result, x='Hasil_Prediksi', y=grade_col,
                                palette={'Non-Dropout': '#2ECC71', 'Dropout': '#E74C3C'}, ax=ax_box)
                    ax_box.set_title("Nilai Rata-rata Semester 2 vs Hasil Prediksi", fontsize=12, fontweight='bold')
                    ax_box.set_xlabel("Status Prediksi")
                    ax_box.set_ylabel("Nilai (Grade)")
                    st.pyplot(fig_box)
                    plt.close(fig_box)
                else:
                    st.warning("Visualisasi sebaran nilai tidak dapat ditampilkan.")

            # Heatmap Korelasi
            st.markdown("##### 📌 Heatmap Korelasi 8 Parameter Terpilih")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            corr_mat = X_input.corr()
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(corr_mat, mask=mask, annot=True, cmap='RdBu_r', center=0,
                        linewidths=0.5, fmt='.2f', ax=ax_corr, cbar_kws={'label': 'Korelasi'})
            ax_corr.set_title("Matriks Korelasi 8 Fitur Mapped", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_corr)
            plt.close(fig_corr)

            # Evaluasi jika ada target aktual
            if has_target and target_col:
                _show_experiment_evaluation(df_raw, y_pred, y_pred_proba, target_col)

            # Download & data view
            st.markdown("---")
            st.subheader("📋 Hasil Prediksi Lengkap")
            st.dataframe(df_result.head(100))

            towrite = io.BytesIO()
            df_result.to_csv(towrite, index=False, encoding='utf-8')
            towrite.seek(0)
            st.download_button(
                label="📥 Unduh Hasil Prediksi (CSV)",
                data=towrite,
                file_name="hasil_prediksi_eksperimen_uci.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            st.exception(e)


def _show_experiment_evaluation(df_raw, y_pred, y_pred_proba, target_col):
    """Tampilkan evaluasi metrik mode eksperimen terhadap target aktual."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Metrik Evaluasi Model terhadap Target Aktual</h3>", unsafe_allow_html=True)

    y_true_raw = df_raw[target_col]
    unique_true = y_true_raw.unique()
    do_val = None
    for ut in unique_true:
        if str(ut).strip().lower() == 'dropout':
            do_val = ut
            break

    if do_val is not None:
        y_true = y_true_raw.apply(lambda x: 1 if x == do_val else 0).values
    else:
        if 1 in unique_true or '1' in unique_true:
            y_true = y_true_raw.apply(lambda x: 1 if str(x) in ['1', '1.0'] else 0).values
        else:
            y_true = pd.factorize(y_true_raw)[0]

    if len(np.unique(y_true)) < 2:
        st.warning("Kolom target aktual tidak memiliki tepat 2 kelas. Evaluasi dilewati.")
        return

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    avg_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, average=avg_method)
    rec = recall_score(y_true, y_pred, zero_division=0, average=avg_method)
    f1 = f1_score(y_true, y_pred, zero_division=0, average=avg_method)

    met1, met2, met3, met4 = st.columns(4)
    with met1:
        st.metric("Akurasi", f"{acc * 100:.2f}%")
    with met2:
        st.metric("Presisi", f"{prec * 100:.2f}%")
    with met3:
        st.metric("Daya Ingat (Sensitivitas)", f"{rec * 100:.2f}%")
    with met4:
        st.metric("F1-Score", f"{f1 * 100:.2f}%")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Confusion Matrix</p>", unsafe_allow_html=True)
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Non-Dropout', 'Dropout'],
                    yticklabels=['Non-Dropout', 'Dropout'],
                    annot_kws={'size': 14})
        ax_cm.set_xlabel('Prediksi')
        ax_cm.set_ylabel('Aktual')
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with eval_col2:
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>ROC Curve</p>", unsafe_allow_html=True)
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
            ax_roc.plot(fpr, tpr, color='#E24B4A', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#E24B4A')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.info("Kurva ROC tidak dapat diplot karena probabilitas prediksi tidak tersedia.")


# ── MODE DATA PRIMER (Train On-the-fly) ──────────────────────────────────────

def _run_primary_mode(config: dict, is_new_run: bool = False):
    """Tampilkan hasil training dan evaluasi model C4.5 dari data primer."""
    st.markdown("<h2 style='font-size: 18px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px; margin-top: 32px;'>Evaluasi Data Primer SMK Tunas Teknologi (Model Training)</h2>", unsafe_allow_html=True)

    df_raw = config['df_raw']
    target_col = config['target_col']
    selected_train_features = config['selected_train_features']
    test_method = config.get('test_method', 'Split dari Data Latih')
    test_size = config.get('test_size')
    test_file = config.get('test_file')
    max_depth = config['max_depth']
    use_ig_selection = config['use_ig_selection']
    ig_threshold = config['ig_threshold']
    dataset_name = config['dataset_name']

    st.markdown("---")
    with st.spinner("Melatih Model C4.5 (Decision Tree) dan membuat visualisasi..."):

        # ── Preprocessing Train Data ──
        X_numeric, y, class_names, categorical_cols = preprocess_primary_data(
            df_raw, target_col, selected_train_features
        )
        if X_numeric is None:
            return

        if categorical_cols:
            st.info(f"ℹ️ Mengonversi kolom kategorikal: `{', '.join(categorical_cols)}`")

        # ── Handle Separate Test Data ──
        X_test_numeric = None
        y_test_ext = None
        if test_method == "Gunakan Dataset Uji Terpisah" and test_file:
            st.info(f"🚀 Menggunakan Dataset Uji Terpisah: **{test_file['original_filename']}**")
            df_test_raw = load_file_from_path(test_file['file_path'])
            if df_test_raw is None:
                st.error("Gagal memuat dataset uji. File mungkin rusak atau dihapus.")
                return
            
            # Check if required columns exist in test data
            missing_cols = [c for c in selected_train_features + [target_col] if c not in df_test_raw.columns]
            if missing_cols:
                st.error(f"Dataset Uji tidak memiliki kolom yang sama persis dengan Dataset Latih. Kolom yang hilang: {', '.join(missing_cols)}")
                return
                
            X_test_numeric, y_test_ext, _, _ = preprocess_primary_data(
                df_test_raw, target_col, selected_train_features
            )

        # ── Seleksi Fitur (Information Gain) ──
        ig_df = None
        if use_ig_selection:
            try:
                X_numeric, ig_df = apply_information_gain_selection(X_numeric, y, ig_threshold)
                _show_ig_chart(ig_df, ig_threshold, len(selected_train_features))
                
                # Apply same selected features to test set if using separate test dataset
                if X_test_numeric is not None:
                    selected_by_ig = X_numeric.columns.tolist()
                    X_test_numeric = X_test_numeric[selected_by_ig]
            except Exception as e:
                st.error(f"Gagal menghitung Information Gain: {e}")

        # ── Training ──
        start_time = time.time()
        result = train_c45_model(X_numeric, y, test_size, max_depth, X_test_numeric, y_test_ext)
        c45_time = time.time() - start_time
        
        model_c45 = result['model']
        scaler = result['scaler']
        X_test_scaled = result['X_test']
        y_test = result['y_test']

        # ── Evaluasi ──
        metrics = evaluate_model(model_c45, X_test_scaled, y_test, class_names)
        y_pred = metrics['y_pred']
        y_pred_proba = metrics['y_pred_proba']
        is_binary = metrics['is_binary']

        # ── Fase 3 & Fase 4 (Identifikasi Missing Values & Outliers) disembunyikan sesuai permintaan user ──
        pass

        # ── Tampilkan Fase 5: Evaluasi Model ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Fase 5 — Evaluasi Model C4.5</h3>", unsafe_allow_html=True)

        _show_evaluation_metrics(metrics)
        _show_confusion_matrix_and_bar(metrics, class_names)
        _show_roc_curve(y_test, y_pred_proba, class_names, is_binary)
        _show_actual_vs_pred(y_test, y_pred, class_names)

        # ── Pengujian Sistem (System Testing) ──
        # _show_system_testing(X_numeric, y, metrics['acc'], c45_time)

        # ── Cross Validation ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Cross-Validation (10-Fold)</p>", unsafe_allow_html=True)
        cv_scores = run_cross_validation(X_numeric, y, max_depth, is_binary, scaler)
        _show_cv_results(cv_scores)

        # Classification report
        with st.expander("📋 Laporan Klasifikasi Lengkap (Classification Report)"):
            st.text("=== CLASSIFICATION REPORT ===")
            st.text(metrics['report'])

        # ── Hasil — Pohon, Feature Importance, Korelasi ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 16px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Hasil — Model & Prediksi</h3>", unsafe_allow_html=True)

        _show_decision_tree(model_c45, X_numeric, class_names)
        _show_feature_importance_and_corr(model_c45, X_numeric)

        # ── Prediksi Batch ──
        _show_batch_predictions(model_c45, scaler, X_numeric, df_raw, class_names, is_binary)

        # ── Simpan ke Database & Simpan Model Tersimpan (Pre-trained Primer) ──
        if is_new_run:
            config_to_save = {
                'max_depth': max_depth,
                'use_ig_selection': use_ig_selection,
                'ig_threshold': ig_threshold,
                'test_size': test_size,
                'features': selected_train_features,
                'target': target_col,
                'class_names': class_names
            }
            db.save_prediction_history(
                run_by=st.session_state.user_id,
                dataset_name=dataset_name,
                mode_analisis='Data Primer (Train On-the-fly)',
                accuracy=float(metrics['acc'] * 100),
                precision=float(metrics['prec'] * 100),
                recall=float(metrics['rec'] * 100),
                f1_score=float(metrics['f1'] * 100),
                config_json=json_lib.dumps(config_to_save)
            )
            # Simpan model untuk digunakan oleh Guru
            from models.ml_model import save_primary_model
            save_primary_model(model_c45, scaler, config_to_save)
            st.toast("Hasil prediksi berhasil disimpan ke riwayat, dan Model Sistem diperbarui.")


# ── Helper Visualization Functions ───────────────────────────────────────────

def _show_ig_chart(ig_df, ig_threshold, total_features):
    """Tampilkan bar chart Information Gain."""
    st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Seleksi Fitur Information Gain</p>", unsafe_allow_html=True)
    fig_ig, ax_ig = plt.subplots(figsize=(10, max(5, len(ig_df) * 0.35)))
    colors_ig = ['#2ECC71' if v >= ig_threshold else '#E74C3C' for v in ig_df['Information Gain']]
    ax_ig.barh(ig_df['Fitur'][::-1], ig_df['Information Gain'][::-1],
               color=colors_ig[::-1], edgecolor='white', linewidth=0.5)
    ax_ig.axvline(x=ig_threshold, color='red', linestyle='--', linewidth=1,
                  label=f'Threshold ({ig_threshold})')
    ax_ig.set_xlabel('Information Gain')
    ax_ig.set_title('Fase 3 — Information Gain per Fitur', fontsize=12, fontweight='bold')
    ax_ig.legend()
    plt.tight_layout()
    st.pyplot(fig_ig)
    plt.close(fig_ig)

    selected_count = len(ig_df[ig_df['Information Gain'] >= ig_threshold])
    if selected_count == 0:
        st.warning(f"Tidak ada fitur dengan IG >= {ig_threshold}. Menggunakan semua fitur.")
    else:
        st.success(f"Seleksi Fitur: Menggunakan {selected_count} dari {total_features} fitur.")


def _show_evaluation_metrics(metrics: dict):
    """Tampilkan 4 KPI metrik evaluasi."""
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_stat_card("Akurasi", f"{metrics['acc'] * 100:.2f}%", "", "var(--color-text-secondary)"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_stat_card("Presisi", f"{metrics['prec'] * 100:.2f}%", "", "var(--color-text-secondary)"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_stat_card("Daya Ingat", f"{metrics['rec'] * 100:.2f}%", "", "var(--color-text-secondary)"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_stat_card("F1-Score", f"{metrics['f1'] * 100:.2f}%", "", "var(--color-text-secondary)"), unsafe_allow_html=True)


def _show_confusion_matrix_and_bar(metrics: dict, class_names: list):
    """Tampilkan Confusion Matrix dan bar chart ringkasan metrik."""
    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Confusion Matrix</p>", unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 14})
        ax_cm.set_xlabel('Prediksi')
        ax_cm.set_ylabel('Aktual')
        ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with eval_col2:
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Ringkasan Metrik Evaluasi</p>", unsafe_allow_html=True)
        metric_names = ['Akurasi', 'Presisi', 'Daya Ingat', 'Skor-F1']
        metric_values = [metrics['acc'] * 100, metrics['prec'] * 100,
                         metrics['rec'] * 100, metrics['f1'] * 100]
        metric_colors = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C']

        fig_bar, ax_bar = plt.subplots(figsize=(6, 4.5))
        bars_m = ax_bar.bar(metric_names, metric_values, color=metric_colors,
                            edgecolor='white', linewidth=1.5, width=0.6)
        for bar, val in zip(bars_m, metric_values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax_bar.axhline(y=80, color='red', linestyle='--', linewidth=1.2, label='Target minimum (80%)')
        ax_bar.set_ylabel('Nilai (%)', fontsize=11)
        ax_bar.set_ylim(0, max(metric_values) + 15)
        ax_bar.set_title('Ringkasan Metrik Evaluasi', fontsize=12, fontweight='bold')
        ax_bar.legend(loc='upper right', fontsize=9)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)


def _show_roc_curve(y_test, y_pred_proba, class_names: list, is_binary: bool):
    """Tampilkan ROC Curve (binary atau multi-kelas)."""
    st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>ROC Curve</p>", unsafe_allow_html=True)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))

    if is_binary:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color='#E24B4A', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#E24B4A')
    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
        colors_roc = ['#E24B4A', '#3498DB', '#2ECC71', '#F39C12'][:len(class_names)]
        for i, (cls_name, color) in enumerate(zip(class_names, colors_roc)):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc_i = auc(fpr_i, tpr_i)
            ax_roc.plot(fpr_i, tpr_i, color=color, linewidth=2, label=f'{cls_name} (AUC = {roc_auc_i:.4f})')

    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax_roc.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)


def _show_actual_vs_pred(y_test, y_pred, class_names: list):
    """Tampilkan perbandingan distribusi aktual vs prediksi."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px;'>Perbandingan Data Aktual vs Prediksi</p>", unsafe_allow_html=True)
    cmp_col1, cmp_col2 = st.columns(2)

    with cmp_col1:
        st.markdown("<p style='font-size: 13px; font-weight: 600; color: var(--color-text-secondary); margin-bottom: 8px;'>Distribusi Aktual (Data Testing)</p>", unsafe_allow_html=True)
        df_act = pd.DataFrame({'Target': [class_names[i] for i in y_test]})
        df_act = df_act[df_act['Target'] != 'Non-Dropout']
        actual_counts = df_act['Target'].value_counts()
        actual_labels = actual_counts.index.tolist()
        actual_values = actual_counts.values.tolist()
        
        fig_act, ax_act = plt.subplots(figsize=(6, 4.5))
        act_colors = ['#E74C3C', '#F39C12'][:len(actual_labels)]
        bars_a = ax_act.bar(actual_labels, actual_values, color=act_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars_a, actual_values):
            ax_act.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + (max(actual_values) if len(actual_values) > 0 else 1) * 0.01,
                        f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_act.set_ylabel('Jumlah Siswa')
        ax_act.set_title('Distribusi Aktual (Data Testing)', fontsize=11, fontweight='bold')
        ax_act.spines['top'].set_visible(False)
        ax_act.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_act, transparent=True)
        plt.close(fig_act)

    with cmp_col2:
        st.markdown("<p style='font-size: 13px; font-weight: 600; color: var(--color-text-secondary); margin-bottom: 8px;'>Distribusi Prediksi Model C4.5</p>", unsafe_allow_html=True)
        df_prd = pd.DataFrame({'Target': [class_names[i] for i in y_pred]})
        df_prd = df_prd[df_prd['Target'] != 'Non-Dropout']
        pred_counts = df_prd['Target'].value_counts()
        pred_labels = pred_counts.index.tolist()
        pred_values = pred_counts.values.tolist()
        
        fig_prd, ax_prd = plt.subplots(figsize=(6, 4.5))
        prd_colors = ['#E74C3C', '#F39C12'][:len(pred_labels)]
        bars_p = ax_prd.bar(pred_labels, pred_values, color=prd_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars_p, pred_values):
            ax_prd.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + (max(pred_values) if len(pred_values) > 0 else 1) * 0.01,
                        f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_prd.set_ylabel('Jumlah Siswa')
        ax_prd.set_title('Distribusi Prediksi Model C4.5', fontsize=11, fontweight='bold')
        ax_prd.spines['top'].set_visible(False)
        ax_prd.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_prd)
        plt.close(fig_prd)


def _show_cv_results(cv_scores: dict):
    """Tampilkan tabel dan visualisasi hasil 10-Fold Cross Validation."""
    cv_accuracy = cv_scores['accuracy']
    cv_precision = cv_scores['precision']
    cv_recall = cv_scores['recall']
    cv_f1 = cv_scores['f1']

    cv_table = pd.DataFrame({
        'Fold': [f'Fold {i + 1}' for i in range(10)],
        'Akurasi (%)': [f'{v * 100:.2f}' for v in cv_accuracy],
        'Presisi (%)': [f'{v * 100:.2f}' for v in cv_precision],
        'Daya Ingat (%)': [f'{v * 100:.2f}' for v in cv_recall],
        'F1-Score (%)': [f'{v * 100:.2f}' for v in cv_f1],
    })
    avg_row = pd.DataFrame({
        'Fold': ['**Rata-rata**'],
        'Akurasi (%)': [f'{cv_accuracy.mean() * 100:.2f}'],
        'Presisi (%)': [f'{cv_precision.mean() * 100:.2f}'],
        'Daya Ingat (%)': [f'{cv_recall.mean() * 100:.2f}'],
        'F1-Score (%)': [f'{cv_f1.mean() * 100:.2f}'],
    })
    cv_table = pd.concat([cv_table, avg_row], ignore_index=True)
    st.dataframe(cv_table, use_container_width=True, hide_index=True)

    cv_vis_col1, cv_vis_col2 = st.columns(2)
    with cv_vis_col1:
        fig_cv, ax_cv = plt.subplots(figsize=(7, 4.5))
        folds = range(1, 11)
        ax_cv.plot(folds, cv_accuracy * 100, 'o-', color='#3498DB', label='Akurasi', linewidth=2, markersize=6)
        ax_cv.plot(folds, cv_precision * 100, 's-', color='#2ECC71', label='Presisi', linewidth=2, markersize=6)
        ax_cv.plot(folds, cv_recall * 100, '^-', color='#E67E22', label='Daya Ingat', linewidth=2, markersize=6)
        ax_cv.plot(folds, cv_f1 * 100, 'D-', color='#E74C3C', label='Skor-F1', linewidth=2, markersize=6)
        ax_cv.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='Target (80%)')
        ax_cv.set_xlabel('Fold', fontsize=11)
        ax_cv.set_ylabel('Nilai (%)', fontsize=11)
        ax_cv.set_title('Performa per Fold (10-Fold CV)', fontsize=12, fontweight='bold')
        ax_cv.set_xticks(folds)
        ax_cv.legend(fontsize=8, loc='lower left')
        ax_cv.set_ylim(max(0, min(cv_accuracy.min(), cv_recall.min()) * 100 - 10), 105)
        ax_cv.spines['top'].set_visible(False)
        ax_cv.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_cv)
        plt.close(fig_cv)

    with cv_vis_col2:
        st.markdown("###### Ringkasan 10-Fold CV")
        cv_kpi1, cv_kpi2 = st.columns(2)
        with cv_kpi1:
            st.metric("Rata-rata Akurasi", f"{cv_accuracy.mean() * 100:.2f}%", f"± {cv_accuracy.std() * 100:.2f}%")
            st.metric("Rata-rata Presisi", f"{cv_precision.mean() * 100:.2f}%", f"± {cv_precision.std() * 100:.2f}%")
        with cv_kpi2:
            st.metric("Rata-rata Recall", f"{cv_recall.mean() * 100:.2f}%", f"± {cv_recall.std() * 100:.2f}%")
            st.metric("Rata-rata F1-Score", f"{cv_f1.mean() * 100:.2f}%", f"± {cv_f1.std() * 100:.2f}%")


def _show_decision_tree(model_c45, X_numeric, class_names: list):
    """Tampilkan visualisasi pohon keputusan C4.5."""
    st.markdown("##### 📌 Visualisasi Struktur Pohon Keputusan C4.5")
    fig_tree, ax_tree = plt.subplots(figsize=(24, 10), dpi=150)
    plot_tree(
        model_c45,
        feature_names=X_numeric.columns.tolist(),
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=7,
        ax=ax_tree
    )
    st.pyplot(fig_tree)
    plt.close(fig_tree)


def _show_feature_importance_and_corr(model_c45, X_numeric):
    """Tampilkan feature importance dan heatmap korelasi fitur."""
    vis_col3, vis_col4 = st.columns(2)

    with vis_col3:
        st.markdown("##### 📌 Top Fitur Berdasarkan Importance")
        importances = pd.Series(model_c45.feature_importances_, index=X_numeric.columns)
        importances_sorted_asc = importances.sort_values(ascending=True)

        fig_fi, ax_fi = plt.subplots(figsize=(7, max(4.5, len(importances) * 0.35)))
        colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances_sorted_asc)))
        importances_sorted_asc.plot(kind='barh', color=colors_fi, edgecolor='white', linewidth=0.5, ax=ax_fi)
        for i, (val, name) in enumerate(zip(importances_sorted_asc.values, importances_sorted_asc.index)):
            ax_fi.text(val + importances_sorted_asc.max() * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        ax_fi.set_xlabel('Importance Score')
        ax_fi.set_title('Top Fitur Berdasarkan Importance (C4.5)', fontsize=11, fontweight='bold')
        ax_fi.spines['top'].set_visible(False)
        ax_fi.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)

        importances_desc = importances.sort_values(ascending=False)
        fi_table = pd.DataFrame({
            'Rank': range(1, len(importances_desc) + 1),
            'Fitur': importances_desc.index.tolist(),
            'Importance Score': [f'{v:.4f}' for v in importances_desc.values]
        })
        st.dataframe(fi_table, use_container_width=True, hide_index=True)

    with vis_col4:
        st.markdown("##### 📌 Heatmap Korelasi Fitur Terpilih")
        fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
        corr_mat = X_numeric.corr()
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, mask=mask, annot=False, cmap='RdBu_r', center=0, linewidths=0.5, ax=ax_corr)
        ax_corr.set_title("Heatmap Korelasi Fitur Model", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)


def _show_batch_predictions(model_c45, scaler, X_numeric, df_raw, class_names: list, is_binary: bool):
    """Tampilkan prediksi batch seluruh data beserta visualisasi distribusi."""
    st.markdown("---")
    st.subheader("📋 Hasil Prediksi Batch untuk Seluruh Data")

    X_all_scaled = scaler.transform(X_numeric)
    all_preds = model_c45.predict(X_all_scaled)
    all_probs_full = model_c45.predict_proba(X_all_scaled)

    df_result = df_raw.copy()
    df_result['Hasil_Prediksi_Numerik'] = all_preds
    df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].map(
        {i: name for i, name in enumerate(class_names)}
    )
    if df_result['Hasil_Prediksi'].isna().any():
        df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].astype(str)
    if is_binary:
        df_result['Probabilitas_Dropout'] = all_probs_full[:, 1]
    else:
        df_result['Probabilitas_Prediksi'] = all_probs_full.max(axis=1)

    # Ambil data tahun ajaran dari DB
    uploaded_files = db.get_uploaded_files()
    ds_name = st.session_state.run_config.get('dataset_name')
    matched_file = next((f for f in uploaded_files if f['original_filename'] == ds_name), None)
    tahun_ajaran = str(matched_file['uploaded_at'])[:4] if matched_file and matched_file.get('uploaded_at') else "Terbaru"

    df_plot_bp = df_result[df_result['Hasil_Prediksi'] != 'Non-Dropout']
    batch_counts_risk = df_plot_bp['Hasil_Prediksi'].value_counts()
    batch_labels = batch_counts_risk.index.tolist()
    batch_values = batch_counts_risk.values.tolist()
    batch_total_risk = sum(batch_values) if sum(batch_values) > 0 else 1

    batch_colors = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6'][:len(batch_labels)]

    batch_vis1, batch_vis2 = st.columns(2)

    with batch_vis1:
        st.markdown(f"##### 📌 Distribusi Peringatan Dini Tahun Ajaran {tahun_ajaran}")
        fig_bp, ax_bp = plt.subplots(figsize=(7, 5))
        if batch_values:
            bars_bp = ax_bp.bar(batch_labels, batch_values, color=batch_colors, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars_bp, batch_values):
                ax_bp.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + batch_total_risk * 0.01,
                           f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_bp.set_ylabel('Jumlah Siswa', fontsize=11)
        ax_bp.set_title(f'Distribusi Siswa Berisiko Tahun {tahun_ajaran}', fontsize=12, fontweight='bold')
        ax_bp.spines['top'].set_visible(False)
        ax_bp.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_bp, transparent=True)
        plt.close(fig_bp)

    with batch_vis2:
        st.markdown(f"##### 📌 Proporsi Peringatan Dini Tahun Ajaran {tahun_ajaran}")
        fig_bpie, ax_bpie = plt.subplots(figsize=(7, 5))
        if batch_values:
            wedges, texts, autotexts = ax_bpie.pie(
                batch_values, labels=batch_labels, colors=batch_colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
        ax_bpie.axis('equal')
        ax_bpie.set_title(f'Proporsi Siswa Berisiko Tahun {tahun_ajaran}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_bpie, transparent=True)
        plt.close(fig_bpie)

    st.markdown("---")
    st.subheader(f"📦 Boxplot: Sebaran Fitur Numerik Peringatan Dini Tahun Ajaran {tahun_ajaran}")
    st.info("Visualisasi ini membantu melihat perbedaan atau sebaran pada fitur-fitur numerik untuk siswa yang terdeteksi berisiko (Dropout).")
    
    # Gunakan X_numeric yang sudah dienkode agar semua fitur (termasuk kategorial yang dienkode) tampil
    df_box = X_numeric.copy()
    df_box['Hasil_Prediksi'] = df_result['Hasil_Prediksi'].values
    df_box = df_box[df_box['Hasil_Prediksi'] != 'Non-Dropout']
    
    if not df_box.empty:
        features_to_plot_bp = X_numeric.columns.tolist()
        
        if features_to_plot_bp:
            import seaborn as sns
            
            # Buat layout 2 kolom
            box_cols = st.columns(2)
            for idx, selected_feature_bp in enumerate(features_to_plot_bp):
                with box_cols[idx % 2]:
                    fig_box, ax_box = plt.subplots(figsize=(7, 5))
                    sns.boxplot(
                        data=df_box, 
                        x='Hasil_Prediksi', 
                        y=selected_feature_bp, 
                        hue='Hasil_Prediksi',
                        legend=False,
                        palette='Set2',
                        ax=ax_box,
                        showmeans=True,
                        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
                    )
                    ax_box.set_title(f'Sebaran {selected_feature_bp} untuk Siswa Berisiko Tahun {tahun_ajaran}', fontsize=12, fontweight='bold')
                    ax_box.set_xlabel('Kelas Risiko (Prediksi)', fontsize=11)
                    ax_box.set_ylabel(selected_feature_bp, fontsize=11)
                    ax_box.spines['top'].set_visible(False)
                    ax_box.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_box, transparent=True)
                    plt.close(fig_box)
        else:
            st.warning("Tidak ada data / kolom numerik yang dapat divisualisasikan pada kelas berisiko.")
    else:
        st.warning("Tidak ada data dengan kelas berisiko (Dropout) yang terdeteksi.")

    st.dataframe(df_result, use_container_width=True)

    towrite = io.BytesIO()
    df_result.to_csv(towrite, index=False, encoding='utf-8')
    towrite.seek(0)
    st.download_button(
        label="📥 Unduh Hasil Prediksi Lengkap (CSV)",
        data=towrite,
        file_name="hasil_prediksi_data_primer.csv",
        mime="text/csv"
    )

def _show_system_testing(X_numeric, y, c45_accuracy, c45_time):
    """Tampilkan Hasil Pengujian Sistem (Functional, Performance, Comparative)"""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 18px; font-weight: 600; color: var(--color-primary); margin-bottom: 16px;'>2. Pengujian Sistem</h2>", unsafe_allow_html=True)
    
    test_col1, test_col2 = st.columns([1, 1.2])
    
    with test_col1:
        st.markdown("##### a. Functional Validation")
        st.markdown("""
        <div style="background-color: var(--color-surface); padding: 15px; border-radius: 8px; border: 1px solid var(--color-border); margin-bottom: 16px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="color: #2ECC71; font-weight: bold; font-size: 16px; margin-right: 8px;">✓</span> 
                <span style="font-size: 14px;">Ingesti Dataset & Pencocokan Kolom</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="color: #2ECC71; font-weight: bold; font-size: 16px; margin-right: 8px;">✓</span> 
                <span style="font-size: 14px;">Penanganan <i>Missing Values</i> terselesaikan</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="color: #2ECC71; font-weight: bold; font-size: 16px; margin-right: 8px;">✓</span> 
                <span style="font-size: 14px;">Pendeteksian <i>Outliers</i> selesai</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="color: #2ECC71; font-weight: bold; font-size: 16px; margin-right: 8px;">✓</span> 
                <span style="font-size: 14px;">Pembuatan Pohon Keputusan C4.5 Berhasil</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### b. Performance Testing")
        st.markdown(render_stat_card("Waktu Komputasi C4.5", f"{c45_time:.3f} s", "Durasi pelatihan model dan pemrosesan matriks"), unsafe_allow_html=True)

    with test_col2:
        st.markdown("##### c. Comparative Testing (Akurasi)")
        st.markdown("<p style='font-size: 13px; color: var(--color-text-secondary); margin-bottom: 8px;'>Membandingkan kinerja algoritma C4.5 dengan Naive Bayes dan K-Nearest Neighbors (KNN).</p>", unsafe_allow_html=True)
        
        with st.spinner("Memproses Comparative Testing..."):
            # Latih model NB dan KNN dengan cross_val_score untuk evaluasi komparatif cepat
            nb_scores = cross_val_score(GaussianNB(), X_numeric, y, cv=10, scoring='accuracy')
            knn_scores = cross_val_score(KNeighborsClassifier(), X_numeric, y, cv=10, scoring='accuracy')
            
            nb_acc = nb_scores.mean() * 100
            knn_acc = knn_scores.mean() * 100
            
            # C4.5 Akurasi
            c45_acc_val = c45_accuracy * 100
            
            algo_names = ['C4.5 (Pohon Keputusan)', 'Naive Bayes', 'K-Nearest Neighbors']
            algo_acc = [c45_acc_val, nb_acc, knn_acc]
            algo_colors = ['#E74C3C', '#3498DB', '#9B59B6']
            
            fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
            bars = ax_comp.bar(algo_names, algo_acc, color=algo_colors, edgecolor='white', linewidth=1.5)
            
            for bar, acc in zip(bars, algo_acc):
                ax_comp.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax_comp.set_ylabel('Akurasi (%)')
            ax_comp.set_ylim(0, max(algo_acc) + 15)
            ax_comp.spines['top'].set_visible(False)
            ax_comp.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig_comp, transparent=True)
            plt.close(fig_comp)
