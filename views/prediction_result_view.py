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

from sklearn.metrics import roc_curve, auc
from sklearn.tree import plot_tree

import db
from models.ml_model import (
    load_uci_artifacts,
    preprocess_primary_data,
    apply_information_gain_selection,
    train_c45_model,
    evaluate_model,
    run_cross_validation,
)
from models.file_utils import load_file_from_path


def show_prediction_results():
    """Entry point halaman Hasil Prediksi."""
    st.title("📈 Hasil Analisis Prediksi")

    if st.button("⬅️ Kembali ke Dashboard"):
        st.session_state.current_page = 'Dasbor Riwayat'
        st.rerun()

    if 'run_config' not in st.session_state:
        st.warning("Tidak ada data hasil prediksi. Silakan jalankan konfigurasi terlebih dahulu.")
        return

    config = st.session_state.run_config
    is_new_run = not config.get('is_saved', False)

    if config['mode'] == 'Eksperimen':
        _run_experiment_mode(config['df_raw'])
    else:
        _run_primary_mode(config, is_new_run=is_new_run)

    if is_new_run:
        st.session_state.run_config['is_saved'] = True


# ── MODE EKSPERIMEN (UCI Pre-trained) ─────────────────────────────────────────

def _run_experiment_mode(df_raw):
    """Tampilkan hasil prediksi menggunakan model pre-trained UCI."""
    model_uci, scaler_uci, features_uci = load_uci_artifacts()

    st.header("🔬 Evaluasi Data Baru Menggunakan Model Eksperimen UCI")

    if model_uci is None or scaler_uci is None:
        st.error("Error: File model pre-trained (`model_c45_dropout.pkl`) atau scaler tidak ditemukan di direktori!")
        return

    st.subheader("🔗 Pencocokan Kolom (Parameter Mapping)")
    st.write("Silakan pasangkan 8 parameter yang dibutuhkan model dengan kolom yang ada di file Anda:")

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

    st.markdown("---")
    st.subheader("🎯 Kolom Target Aktual (Opsional)")
    has_target = st.checkbox("File saya memiliki kolom label/target aktual (untuk menghitung Akurasi/Evaluasi)")

    target_col = None
    if has_target:
        default_tgt_idx = 0
        for idx, col in enumerate(columns_list):
            if col.lower() in ['target', 'status', 'label']:
                default_tgt_idx = idx
                break
        target_col = st.selectbox("Pilih Kolom Target Aktual", options=columns_list, index=default_tgt_idx)

    if st.button("Jalankan Prediksi Batch 🚀"):
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
            st.subheader("📊 Dashboard Evaluasi Hasil Prediksi")
            total_cnt = len(df_result)
            dropout_cnt = int((y_pred == 1).sum())
            nondropout_cnt = total_cnt - dropout_cnt

            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Total Siswa Dievaluasi", f"{total_cnt} Orang")
            with kpi2:
                st.metric("Diprediksi DROPOUT (Berisiko)", f"{dropout_cnt} Orang",
                          f"{dropout_cnt / total_cnt * 100:.1f}%", delta_color="inverse")
            with kpi3:
                st.metric("Diprediksi NON-DROPOUT", f"{nondropout_cnt} Orang",
                          f"{nondropout_cnt / total_cnt * 100:.1f}%")

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
    st.markdown("---")
    st.subheader("🎯 Metrik Evaluasi Model terhadap Target Aktual")

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
        st.markdown("##### 📌 Confusion Matrix")
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
        st.markdown("##### 📌 ROC Curve")
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
    st.header("📝 Evaluasi Data Primer SMK Tunas Teknologi (Model Training)")

    df_raw = config['df_raw']
    target_col = config['target_col']
    selected_train_features = config['selected_train_features']
    test_size = config['test_size']
    max_depth = config['max_depth']
    use_ig_selection = config['use_ig_selection']
    ig_threshold = config['ig_threshold']
    dataset_name = config['dataset_name']

    st.markdown("---")
    with st.spinner("Melatih Model C4.5 (Decision Tree) dan membuat visualisasi..."):

        # ── Preprocessing ──
        X_numeric, y, class_names, categorical_cols = preprocess_primary_data(
            df_raw, target_col, selected_train_features
        )
        if X_numeric is None:
            return

        if categorical_cols:
            st.info(f"ℹ️ Mengonversi kolom kategorikal: `{', '.join(categorical_cols)}`")

        # ── Seleksi Fitur (Information Gain) ──
        ig_df = None
        if use_ig_selection:
            try:
                X_numeric, ig_df = apply_information_gain_selection(X_numeric, y, ig_threshold)
                _show_ig_chart(ig_df, ig_threshold, len(selected_train_features))
            except Exception as e:
                st.error(f"Gagal menghitung Information Gain: {e}")

        # ── Training ──
        result = train_c45_model(X_numeric, y, test_size, max_depth)
        model_c45 = result['model']
        scaler = result['scaler']
        X_test_scaled = result['X_test']
        y_test = result['y_test']

        # ── Evaluasi ──
        metrics = evaluate_model(model_c45, X_test_scaled, y_test, class_names)
        y_pred = metrics['y_pred']
        y_pred_proba = metrics['y_pred_proba']
        is_binary = metrics['is_binary']

        # ── Tampilkan Fase 5: Evaluasi Model ──
        st.markdown("---")
        st.header("📊 Fase 5 — Evaluasi Model C4.5")

        _show_evaluation_metrics(metrics)
        _show_confusion_matrix_and_bar(metrics, class_names)
        _show_roc_curve(y_test, y_pred_proba, class_names, is_binary)
        _show_actual_vs_pred(y_test, y_pred, class_names)

        # ── Cross Validation ──
        st.markdown("---")
        st.markdown("##### 📌 Cross-Validation (10-Fold)")
        cv_scores = run_cross_validation(X_numeric, y, max_depth, is_binary, scaler)
        _show_cv_results(cv_scores)

        # Classification report
        with st.expander("📋 Laporan Klasifikasi Lengkap (Classification Report)"):
            st.text("=== CLASSIFICATION REPORT ===")
            st.text(metrics['report'])

        # ── Hasil — Pohon, Feature Importance, Korelasi ──
        st.markdown("---")
        st.header("🌳 Hasil — Model & Prediksi")

        _show_decision_tree(model_c45, X_numeric, class_names)
        _show_feature_importance_and_corr(model_c45, X_numeric)

        # ── Prediksi Batch ──
        _show_batch_predictions(model_c45, scaler, X_numeric, df_raw, class_names, is_binary)

        # ── Simpan ke Database ──
        if is_new_run:
            db.save_prediction_history(
                run_by=st.session_state.user_id,
                dataset_name=dataset_name,
                mode_analisis='Data Primer (Train On-the-fly)',
                accuracy=float(metrics['acc'] * 100),
                precision=float(metrics['prec'] * 100),
                recall=float(metrics['rec'] * 100),
                f1_score=float(metrics['f1'] * 100),
                config_json=json_lib.dumps({
                    'max_depth': max_depth,
                    'use_ig_selection': use_ig_selection,
                    'ig_threshold': ig_threshold,
                    'test_size': test_size,
                    'features': selected_train_features,
                    'target': target_col
                })
            )
            st.toast("✅ Hasil prediksi berhasil disimpan ke riwayat!")


# ── Helper Visualization Functions ───────────────────────────────────────────

def _show_ig_chart(ig_df, ig_threshold, total_features):
    """Tampilkan bar chart Information Gain."""
    st.markdown("##### 🔍 Seleksi Fitur Information Gain")
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
        st.warning(f"⚠️ Tidak ada fitur dengan IG >= {ig_threshold}. Menggunakan semua fitur.")
    else:
        st.success(f"🎯 Seleksi Fitur: Menggunakan {selected_count} dari {total_features} fitur.")


def _show_evaluation_metrics(metrics: dict):
    """Tampilkan 4 KPI metrik evaluasi."""
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Akurasi", f"{metrics['acc'] * 100:.2f}%")
    with m2:
        st.metric("Presisi", f"{metrics['prec'] * 100:.2f}%")
    with m3:
        st.metric("Daya Ingat", f"{metrics['rec'] * 100:.2f}%")
    with m4:
        st.metric("F1-Score", f"{metrics['f1'] * 100:.2f}%")


def _show_confusion_matrix_and_bar(metrics: dict, class_names: list):
    """Tampilkan Confusion Matrix dan bar chart ringkasan metrik."""
    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.markdown("##### 📌 Confusion Matrix")
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
        st.markdown("##### 📌 Ringkasan Metrik Evaluasi")
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
    st.markdown("##### 📌 ROC Curve")
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
    st.markdown("##### 📌 Perbandingan Data Aktual vs Prediksi")
    cmp_col1, cmp_col2 = st.columns(2)

    with cmp_col1:
        st.markdown("###### Distribusi Aktual (Data Testing)")
        actual_counts = pd.Series(y_test).value_counts().sort_index()
        actual_labels = [class_names[i] for i in actual_counts.index]
        actual_values = actual_counts.values
        fig_act, ax_act = plt.subplots(figsize=(6, 4.5))
        act_colors = ['#2ECC71', '#E74C3C'][:len(actual_labels)]
        bars_a = ax_act.bar(actual_labels, actual_values, color=act_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars_a, actual_values):
            ax_act.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(actual_values) * 0.01,
                        f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_act.set_ylabel('Jumlah Siswa')
        ax_act.set_title('Distribusi Aktual (Data Testing)', fontsize=11, fontweight='bold')
        ax_act.spines['top'].set_visible(False)
        ax_act.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_act)
        plt.close(fig_act)

    with cmp_col2:
        st.markdown("###### Distribusi Prediksi Model C4.5")
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        pred_labels = [class_names[i] for i in pred_counts.index]
        pred_values = pred_counts.values
        fig_prd, ax_prd = plt.subplots(figsize=(6, 4.5))
        prd_colors = ['#2ECC71', '#E74C3C'][:len(pred_labels)]
        bars_p = ax_prd.bar(pred_labels, pred_values, color=prd_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars_p, pred_values):
            ax_prd.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(pred_values) * 0.01,
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
    if is_binary:
        df_result['Probabilitas_Dropout'] = all_probs_full[:, 1]
    else:
        df_result['Probabilitas_Prediksi'] = all_probs_full.max(axis=1)

    batch_total = len(df_result)
    batch_counts = df_result['Hasil_Prediksi'].value_counts()

    batch_cols = st.columns(len(class_names) + 1)
    with batch_cols[0]:
        st.metric("Total Data", f"{batch_total} Siswa")
    for i, name in enumerate(class_names):
        cnt = batch_counts.get(name, 0)
        with batch_cols[i + 1]:
            pct = cnt / batch_total * 100 if batch_total > 0 else 0
            st.metric(f"Prediksi {name}", f"{cnt} Siswa", f"{pct:.1f}%")

    batch_labels = batch_counts.index.tolist()
    batch_values = batch_counts.values.tolist()
    batch_colors = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12'][:len(batch_labels)]

    batch_vis1, batch_vis2 = st.columns(2)

    with batch_vis1:
        st.markdown("##### Distribusi Hasil Prediksi (Seluruh Data)")
        fig_bp, ax_bp = plt.subplots(figsize=(7, 5))
        bars_bp = ax_bp.bar(batch_labels, batch_values, color=batch_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars_bp, batch_values):
            ax_bp.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + batch_total * 0.01,
                       f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_bp.set_ylabel('Jumlah Siswa', fontsize=11)
        ax_bp.set_title('Distribusi Hasil Prediksi (Seluruh Data)', fontsize=12, fontweight='bold')
        ax_bp.spines['top'].set_visible(False)
        ax_bp.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_bp)
        plt.close(fig_bp)

    with batch_vis2:
        st.markdown("##### Proporsi Hasil Prediksi")
        fig_bpie, ax_bpie = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax_bpie.pie(
            batch_values, labels=batch_labels, colors=batch_colors,
            autopct='%1.1f%%', startangle=140, pctdistance=0.75,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        ax_bpie.axis('equal')
        ax_bpie.set_title('Proporsi Hasil Prediksi', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_bpie)
        plt.close(fig_bpie)

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
