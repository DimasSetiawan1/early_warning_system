import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import io

# Set Page Config
st.set_page_config(page_title="Dashboard Prediksi Mahasiswa/Siswa Dropout", page_icon="🎓", layout="wide")

# Styling Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# Helper function to load uploaded file
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            # Read first line to detect separator
            bytes_data = uploaded_file.read(1024)
            uploaded_file.seek(0)
            text_data = bytes_data.decode('utf-8', errors='ignore')
            sep = ';' if ';' in text_data else ','
            df = pd.read_csv(uploaded_file, sep=sep)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# Helper to load model artifacts (UCI)
@st.cache_resource
def load_uci_artifacts():
    try:
        model = pickle.load(open('model_c45_dropout.pkl', 'rb'))
        scaler = pickle.load(open('scaler_dropout.pkl', 'rb'))
        features = pickle.load(open('selected_features.pkl', 'rb'))
        return model, scaler, features
    except Exception as e:
        return None, None, None

# Main Title
st.title("🎓 Dashboard Evaluasi & Prediksi Siswa/Mahasiswa Dropout")
st.markdown("""
Aplikasi ini diimplementasikan mengikuti metodologi **CRISP-DM** menggunakan algoritma **C4.5 (Decision Tree)**.
Aplikasi mendukung dua jenis data: **Data Eksperimen (UCI Dataset)** dan **Data Primer (SMK Tunas Teknologi)** .
""")

# Sidebar
st.sidebar.header("🛠️ Pengaturan Dashboard")

# Selection Mode
mode = st.sidebar.selectbox(
    "Pilih Mode Analisis",
    options=[
        "Eksperimen (UCI Dataset - Pre-trained Model)",
        "Data Primer (SMK Tunas Teknologi - Train On-the-fly)"
    ]
)

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Unggah File Dataset (CSV / Excel)",
    type=["csv", "xlsx"],
    help="Unggah file berisi data siswa/mahasiswa untuk dianalisis dan diprediksi."
)

# Load UCI pre-trained assets if in Mode 1
model_uci, scaler_uci, features_uci = load_uci_artifacts()

if uploaded_file is None:
    # ── INTRO SCREEN (No file uploaded yet) ───────────────────────────────────
    st.info("💡 **Petunjuk Penggunaan:** Silakan unggah file dataset Anda (.csv atau .xlsx) melalui panel samping (sidebar) untuk memulai analisis.")
    
    col_desc1, col_desc2 = st.columns(2)
    
    with col_desc1:
        st.subheader("📊 Mode Eksperimen (UCI Dataset)")
        st.write("""
        Mode ini menggunakan model pohon keputusan C4.5 yang **sudah dilatih** sebelumnya pada dataset sekunder UCI (*Predict Students' Dropout and Academic Success*).
        
        * **Jumlah Fitur Penting:** 8 fitur pilihan (termasuk nilai akademik semester 1 & 2, biaya SPP, dll).
        * **Output:** Prediksi batch untuk data baru, visualisasi sebaran prediksi, dan visualisasi statistik.
        """)
        
        # Load sample info
        if features_uci:
            st.markdown("**8 Parameter yang wajib dicocokkan:**")
            for i, f in enumerate(features_uci, 1):
                st.markdown(f"{i}. `{f}`")
                
    with col_desc2:
        st.subheader("📝 Mode Data Primer (SMK Tunas Teknologi)")
        st.write("""
        Mode ini dirancang khusus untuk dataset lokal yang memiliki **14 parameter** lokal (kategori Kehadiran, Sosial, dan Ekonomi).
        
        * **Proses Dinamis (On-the-fly):** Model akan dilatih langsung dari data yang Anda unggah.
        * **Kustomisasi Parameter:** Anda bisa memilih fitur mana saja yang ingin disertakan/dihilangkan.
        * **Evaluasi Lengkap:** Menghasilkan akurasi, presisi, recall, F1-score, Confusion Matrix, ROC-AUC, tingkat kepentingan fitur, dan gambar pohon keputusan interaktif.
        """)
        st.markdown("**14 Fitur Data Primer (lokal):**")
        st.markdown("""
        1. Kehadiran Semester 1 & 2
        2. Nilai Rata-rata Semester 1 & 2
        3. Jumlah Pelanggaran
        4. Pendidikan Ayah & Ibu
        5. Status Keluarga & Jumlah Saudara
        6. Jarak Rumah (km)
        7. Status SPP & Penerima Beasiswa
        8. Penghasilan Ortu & Tanggungan Keluarga
        """)

else:
    # ── PARSE UPLOADED FILE ──────────────────────────────────────────────────
    df_raw = load_uploaded_file(uploaded_file)
    
    if df_raw is not None:
        st.success(f"✅ Berhasil memuat file: `{uploaded_file.name}` ({df_raw.shape[0]} baris × {df_raw.shape[1]} kolom)")
        
        # Display raw data preview
        with st.expander("👁️ Lihat Preview Data Unggahan"):
            st.dataframe(df_raw.head(10))
            
        # ── MODE 1: EKSPERIMEN (UCI PRE-TRAINED) ──────────────────────────────
        if mode == "Eksperimen (UCI Dataset - Pre-trained Model)":
            st.header("🔬 Evaluasi Data Baru Menggunakan Model Eksperimen UCI")
            
            if model_uci is None or scaler_uci is None:
                st.error("Error: File model pre-trained (`model_c45_dropout.pkl`) atau scaler tidak ditemukan di direktori!")
            else:
                st.subheader("🔗 Pencocokan Kolom (Parameter Mapping)")
                st.write("Silakan pasangkan 8 parameter yang dibutuhkan model dengan kolom yang ada di file Anda:")
                
                # Auto-matching helper
                columns_list = list(df_raw.columns)
                mapped_columns = []
                
                col_map1, col_map2 = st.columns(2)
                
                # We will map each of the 8 features
                for i, feat in enumerate(features_uci):
                    # Try to find a exact or partial match
                    default_idx = 0
                    for idx, col in enumerate(columns_list):
                        if feat.lower() in col.lower() or col.lower() in feat.lower():
                            default_idx = idx
                            break
                    
                    # Distribute across two columns
                    with col_map1 if i % 2 == 0 else col_map2:
                        sel_col = st.selectbox(
                            f"Parameter: **{feat}**",
                            options=columns_list,
                            index=default_idx,
                            key=f"map_{i}"
                        )
                        mapped_columns.append((feat, sel_col))
                
                # Optional target mapping to compute metrics
                st.markdown("---")
                st.subheader("🎯 Kolom Target Aktual (Opsional)")
                has_target = st.checkbox("File saya memiliki kolom label/target aktual (untuk menghitung Akurasi/Evaluasi)")
                
                target_col = None
                if has_target:
                    # Look for default target
                    default_tgt_idx = 0
                    for idx, col in enumerate(columns_list):
                        if col.lower() in ['target', 'status', 'label']:
                            default_tgt_idx = idx
                            break
                    target_col = st.selectbox(
                        "Pilih Kolom Target Aktual",
                        options=columns_list,
                        index=default_tgt_idx
                    )
                
                # Start Prediction Button
                if st.button("Jalankan Prediksi Batch 🚀"):
                    st.markdown("---")
                    
                    # Prepare Data for prediction
                    feature_mapping_dict = {feat: sel_col for feat, sel_col in mapped_columns}
                    X_input = df_raw[[feature_mapping_dict[feat] for feat in features_uci]].copy()
                    
                    # Rename columns to match model expectations
                    X_input.columns = features_uci
                    
                    # Impute missing values if any
                    for col in X_input.columns:
                        if X_input[col].isnull().sum() > 0:
                            X_input[col].fillna(X_input[col].median(), inplace=True)
                    
                    try:
                        # Scaling
                        X_scaled = scaler_uci.transform(X_input)
                        
                        # Predict
                        y_pred = model_uci.predict(X_scaled)
                        y_pred_proba = model_uci.predict_proba(X_scaled)[:, 1] if hasattr(model_uci, "predict_proba") else None
                        
                        # Add predictions back to the original df
                        df_result = df_raw.copy()
                        df_result['Hasil_Prediksi_Numerik'] = y_pred
                        df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].map({1: 'Dropout', 0: 'Non-Dropout'})
                        if y_pred_proba is not None:
                            df_result['Probabilitas_Dropout'] = y_pred_proba
                        
                        # --- DISPLAY DASHBOARD ---
                        st.subheader("📊 Dashboard Evaluasi Hasil Prediksi")
                        
                        # KPI Cards
                        kpi1, kpi2, kpi3 = st.columns(3)
                        total_cnt = len(df_result)
                        dropout_cnt = int((y_pred == 1).sum())
                        nondropout_cnt = total_cnt - dropout_cnt
                        
                        with kpi1:
                            st.metric("Total Siswa Dievaluasi", f"{total_cnt} Orang")
                        with kpi2:
                            st.metric("Diprediksi DROPOUT (Berisiko)", f"{dropout_cnt} Orang", f"{dropout_cnt/total_cnt*100:.1f}%", delta_color="inverse")
                        with kpi3:
                            st.metric("Diprediksi NON-DROPOUT", f"{nondropout_cnt} Orang", f"{nondropout_cnt/total_cnt*100:.1f}%")
                            
                        # Layout for charts
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            st.markdown("##### 📌 Distribusi Hasil Prediksi (Pie Chart)")
                            fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
                            counts = [nondropout_cnt, dropout_cnt]
                            labels = ['Non-Dropout', 'Dropout']
                            colors = ['#2ECC71', '#E74C3C']
                            ax_pie.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, 
                                       textprops={'fontsize': 12, 'weight': 'bold'})
                            ax_pie.axis('equal')
                            st.pyplot(fig_pie)
                            plt.close(fig_pie)
                            
                        with chart_col2:
                            st.markdown("##### 📌 Sebaran Nilai Rata-rata Semester 2 vs Status Prediksi")
                            # Try to plot a boxplot if we can find Curricular units 2nd sem (grade)
                            grade_col = feature_mapping_dict.get('Curricular units 2nd sem (grade)')
                            if grade_col in df_raw.columns:
                                fig_box, ax_box = plt.subplots(figsize=(7, 5.2))
                                sns.boxplot(data=df_result, x='Hasil_Prediksi', y=grade_col, palette={'Non-Dropout': '#2ECC71', 'Dropout': '#E74C3C'}, ax=ax_box)
                                ax_box.set_title("Nilai Rata-rata Semester 2 vs Hasil Prediksi", fontsize=12, fontweight='bold')
                                ax_box.set_xlabel("Status Prediksi")
                                ax_box.set_ylabel("Nilai (Grade)")
                                st.pyplot(fig_box)
                                plt.close(fig_box)
                            else:
                                st.warning("Visualisasi sebaran nilai tidak dapat ditampilkan karena kolom 'Curricular units 2nd sem (grade)' tidak ditemukan/dipetakan.")
                                
                        # Heatmap Korelasi Mapped Features
                        st.markdown("##### 📌 Heatmap Korelasi 8 Parameter Terpilih")
                        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                        corr_mat = X_input.corr()
                        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
                        sns.heatmap(corr_mat, mask=mask, annot=True, cmap='RdBu_r', center=0, linewidths=0.5, fmt='.2f', ax=ax_corr, cbar_kws={'label': 'Korelasi'})
                        ax_corr.set_title("Matriks Korelasi 8 Fitur Mapped", fontsize=12, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig_corr)
                        plt.close(fig_corr)
                        
                        # Evaluation section if has_target is checked
                        if has_target and target_col:
                            st.markdown("---")
                            st.subheader("🎯 Metrik Evaluasi Model terhadap Target Aktual")
                            
                            # Clean target data
                            y_true_raw = df_raw[target_col]
                            # Try mapping
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
                                    # Fallback
                                    y_true = pd.factorize(y_true_raw)[0]
                                    # Ensure 1 matches dropout or whatever is secondary
                                    if len(np.unique(y_true)) > 1:
                                        pass
                            
                            if len(np.unique(y_true)) == 2:
                                acc = accuracy_score(y_true, y_pred)
                                prec = precision_score(y_true, y_pred, zero_division=0)
                                rec = recall_score(y_true, y_pred, zero_division=0)
                                f1 = f1_score(y_true, y_pred, zero_division=0)
                                
                                # Show Metrics
                                met1, met2, met3, met4 = st.columns(4)
                                with met1:
                                    st.metric("Akurasi", f"{acc*100:.2f}%")
                                with met2:
                                    st.metric("Presisi", f"{prec*100:.2f}%")
                                with met3:
                                    st.metric("Recall (Sensitivitas)", f"{rec*100:.2f}%")
                                with met4:
                                    st.metric("F1-Score", f"{f1*100:.2f}%")
                                    
                                # Plot CM and ROC
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
                            else:
                                st.warning("Kolom target aktual tidak memiliki tepat 2 kelas (binary) setelah dibaca. Evaluasi dilewati.")
                        
                        # --- DOWNLOAD & DATA VIEW ---
                        st.markdown("---")
                        st.subheader("📋 Hasil Prediksi Lengkap")
                        st.dataframe(df_result.head(100))
                        
                        # Export data
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
                        
        # ── MODE 2: DATA PRIMER (TRAIN ON-THE-FLY) ───────────────────────────
        elif mode == "Data Primer (SMK Tunas Teknologi - Train On-the-fly)":
            st.header("📝 Evaluasi Data Primer SMK Tunas Teknologi (Model Training)")
            
            columns_list = list(df_raw.columns)
            
            # Form configuration for training
            st.subheader("⚙️ Konfigurasi Pembuatan Model C4.5")
            
            col_cfg1, col_cfg2 = st.columns(2)
            
            with col_cfg1:
                # 1. Select Target Column
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
                
                # Remove target from default feature list
                default_features = [col for col in columns_list if col != target_col]
                
                # 2. Select Features
                selected_train_features = st.multiselect(
                    "Pilih Parameter Fitur (X)",
                    options=default_features,
                    default=default_features,
                    help="Pilih fitur-fitur yang akan digunakan sebagai dasar pembelajaran C4.5."
                )
                
            with col_cfg2:
                # 3. Model Hyperparameters
                test_size = st.slider("Ukuran Data Testing (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
                max_depth = st.slider("Kedalaman Maksimum Pohon (Max Depth)", min_value=3, max_value=15, value=7)
                
                # 4. Feature Selection using Information Gain
                use_ig_selection = st.checkbox("Aktifkan Seleksi Fitur otomatis dengan Information Gain", value=False,
                                                help="Model hanya akan menggunakan fitur yang memiliki Information Gain >= threshold.")
                ig_threshold = 0.0
                if use_ig_selection:
                    ig_threshold = st.slider("Threshold Information Gain", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
            
            # Start Training Button
            if st.button("Latih Model C4.5 & Tampilkan Dashboard 🚀"):
                if len(selected_train_features) < 1:
                    st.error("Pilih minimal 1 fitur untuk melatih model!")
                else:
                    st.markdown("---")
                    with st.spinner("Melatih Model C4.5 (Decision Tree) dan membuat visualisasi..."):
                        
                        # Copy data
                        df_model = df_raw.copy()
                        
                        # --- PREPROCESSING ---
                        # 1. Handle Missing Values
                        # Impute: numeric with median, categorical with mode
                        for col in selected_train_features:
                            if df_model[col].isnull().sum() > 0:
                                if df_model[col].dtype in ['int64', 'float64']:
                                    df_model[col].fillna(df_model[col].median(), inplace=True)
                                else:
                                    df_model[col].fillna(df_model[col].mode()[0], inplace=True)
                                    
                        # 2. Target Binarization
                        y_raw = df_model[target_col]
                        unique_targets = y_raw.unique()
                        
                        # Look for 'Dropout' in target classes
                        dropout_val = None
                        for val in unique_targets:
                            if str(val).strip().lower() == 'dropout':
                                dropout_val = val
                                break
                        
                        if dropout_val is not None:
                            y = y_raw.apply(lambda x: 1 if x == dropout_val else 0).values
                            class_names = ['Non-Dropout', 'Dropout']
                        else:
                            # Map numeric 1 to dropout
                            if 1 in unique_targets or '1' in unique_targets:
                                y = y_raw.apply(lambda x: 1 if str(x) in ['1', '1.0'] else 0).values
                                class_names = ['Non-Dropout', 'Dropout']
                            else:
                                # Fallback factorize
                                sorted_unique = sorted(list(unique_targets))
                                y = y_raw.map({sorted_unique[0]: 0, sorted_unique[1]: 1}).values
                                class_names = [str(sorted_unique[0]), str(sorted_unique[1])]
                                
                        X_all = df_model[selected_train_features]
                        
                        # Convert any categorical features in X to numerical codes (Factorize/OneHot)
                        # C4.5 in sklearn Decision Tree only takes numeric inputs
                        X_numeric = X_all.copy()
                        categorical_cols = []
                        for col in X_numeric.columns:
                            if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
                                X_numeric[col] = pd.factorize(X_numeric[col])[0]
                                categorical_cols.append(col)
                                
                        if len(categorical_cols) > 0:
                            st.info(f"ℹ️ Mengonversi kolom kategorikal berikut menjadi kode numerik untuk pelatihan: `{', '.join(categorical_cols)}`")
                            
                        # --- FEATURE SELECTION (INFORMATION GAIN) ---
                        features_to_use = selected_train_features.copy()
                        
                        if use_ig_selection:
                            try:
                                ig_scores = mutual_info_classif(X_numeric, y, random_state=42)
                                ig_df = pd.DataFrame({
                                    'Fitur': X_numeric.columns,
                                    'Information Gain': ig_scores
                                }).sort_values('Information Gain', ascending=False).reset_index(drop=True)
                                
                                # Filter features by threshold
                                selected_by_ig = ig_df[ig_df['Information Gain'] >= ig_threshold]['Fitur'].tolist()
                                
                                st.markdown("##### 🔍 Seleksi Fitur Information Gain")
                                
                                # Show plot of Information Gain
                                fig_ig, ax_ig = plt.subplots(figsize=(10, max(5, len(ig_df) * 0.35)))
                                colors_ig = ['#2ECC71' if v >= ig_threshold else '#E74C3C' for v in ig_df['Information Gain']]
                                ax_ig.barh(ig_df['Fitur'][::-1], ig_df['Information Gain'][::-1], color=colors_ig[::-1], edgecolor='white', linewidth=0.5)
                                ax_ig.axvline(x=ig_threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({ig_threshold})')
                                ax_ig.set_xlabel('Information Gain')
                                ax_ig.set_title('Fase 3 — Information Gain per Fitur', fontsize=12, fontweight='bold')
                                ax_ig.legend()
                                plt.tight_layout()
                                st.pyplot(fig_ig)
                                plt.close(fig_ig)
                                
                                if len(selected_by_ig) == 0:
                                    st.warning(f"⚠️ Tidak ada fitur dengan Information Gain >= {ig_threshold}. Seleksi fitur dilewati, menggunakan seluruh parameter yang dipilih.")
                                else:
                                    features_to_use = selected_by_ig
                                    X_numeric = X_numeric[features_to_use]
                                    st.success(f"🎯 Seleksi Fitur selesai: Menggunakan {len(features_to_use)} fitur dari {len(selected_train_features)} fitur yang di-upload.")
                                    
                            except Exception as e:
                                st.error(f"Gagal menghitung Information Gain: {e}")
                                
                        # --- TRAIN TEST SPLIT & SCALING ---
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_numeric, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_numeric.columns, index=X_train.index)
                        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_numeric.columns, index=X_test.index)
                        
                        # --- TRAINING C4.5 ---
                        model_c45 = DecisionTreeClassifier(
                            criterion='entropy',
                            max_depth=max_depth,
                            random_state=42
                        )
                        model_c45.fit(X_train_scaled, y_train)
                        
                        # Predict
                        y_pred = model_c45.predict(X_test_scaled)
                        y_pred_proba = model_c45.predict_proba(X_test_scaled)[:, 1]
                        
                        # --- EVALUATION METRICS ---
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        st.subheader("📊 Dashboard Evaluasi Model C4.5")
                        
                        # Metrics Row
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Akurasi Model", f"{acc*100:.2f}%")
                        with m2:
                            st.metric("Presisi Model", f"{prec*100:.2f}%")
                        with m3:
                            st.metric("Recall (Sensitivitas)", f"{rec*100:.2f}%")
                        with m4:
                            st.metric("F1-Score", f"{f1*100:.2f}%")
                            
                        # Visualizations Row 1
                        vis_col1, vis_col2 = st.columns(2)
                        
                        with vis_col1:
                            st.markdown("##### 📌 Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                        xticklabels=class_names,
                                        yticklabels=class_names,
                                        annot_kws={'size': 14})
                            ax_cm.set_xlabel('Prediksi')
                            ax_cm.set_ylabel('Aktual')
                            st.pyplot(fig_cm)
                            plt.close(fig_cm)
                            
                        with vis_col2:
                            st.markdown("##### 📌 ROC Curve")
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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
                            
                        # Visualizations Row 2: Tree Plot (CRITICAL!)
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
                        
                        # Feature Importance & Correlation
                        vis_col3, vis_col4 = st.columns(2)
                        
                        with vis_col3:
                            st.markdown("##### 📌 Tingkat Kepentingan Fitur (Feature Importance)")
                            importances = pd.Series(model_c45.feature_importances_, index=X_numeric.columns)
                            importances = importances.sort_values(ascending=True)
                            
                            fig_fi, ax_fi = plt.subplots(figsize=(7, max(4.5, len(importances)*0.35)))
                            colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
                            importances.plot(kind='barh', color=colors_fi, edgecolor='white', linewidth=0.5, ax=ax_fi)
                            ax_fi.set_xlabel('Importance Score')
                            ax_fi.set_title('Feature Importance (C4.5)')
                            plt.tight_layout()
                            st.pyplot(fig_fi)
                            plt.close(fig_fi)
                            
                        with vis_col4:
                            st.markdown("##### 📌 Heatmap Korelasi Fitur Terpilih")
                            fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
                            corr_mat = X_numeric.corr()
                            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
                            sns.heatmap(corr_mat, mask=mask, annot=False, cmap='RdBu_r', center=0, linewidths=0.5, ax=ax_corr)
                            ax_corr.set_title("Heatmap Korelasi Fitur Model")
                            plt.tight_layout()
                            st.pyplot(fig_corr)
                            plt.close(fig_corr)
                            
                        # Extra: Show classification report as text
                        with st.expander("📋 Laporan Klasifikasi Lengkap (Classification Report)"):
                            st.text("=== CLASSIFICATION REPORT ===")
                            st.text(classification_report(y_test, y_pred, target_names=class_names))
                            
                        # --- BATCH PREDICTIONS FOR ENTIRE FILE ---
                        st.markdown("---")
                        st.subheader("📋 Hasil Prediksi Batch untuk Seluruh Data")
                        
                        # Predict entire dataset
                        X_all_scaled = scaler.transform(X_numeric)
                        all_preds = model_c45.predict(X_all_scaled)
                        all_probs = model_c45.predict_proba(X_all_scaled)[:, 1]
                        
                        df_result = df_raw.copy()
                        df_result['Hasil_Prediksi_Numerik'] = all_preds
                        df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].map({1: 'Dropout', 0: 'Non-Dropout'})
                        df_result['Probabilitas_Dropout'] = all_probs
                        
                        st.dataframe(df_result.head(100))
                        
                        # Export data
                        towrite = io.BytesIO()
                        df_result.to_csv(towrite, index=False, encoding='utf-8')
                        towrite.seek(0)
                        
                        st.download_button(
                            label="📥 Unduh Hasil Prediksi Lengkap (CSV)",
                            data=towrite,
                            file_name="hasil_prediksi_data_primer.csv",
                            mime="text/csv"
                        )
