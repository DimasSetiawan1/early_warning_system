import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import sys

def run_preprocessing_demo(filepath):
    print("="*70)
    print("DEMONSTRASI PIPELINE PREPROCESSING DATASET")
    print("="*70)

    # Membaca dataset
    try:
        df = pd.read_csv(filepath)
        print(f"Berhasil memuat {filepath} ({df.shape[0]} baris, {df.shape[1]} kolom)")
    except FileNotFoundError:
        print(f"Error: File {filepath} tidak ditemukan.")
        return

    # Menyimulasikan missing value agar proses imputasi terlihat bekerja
    # Cari 2 kolom fitur pertama (bukan target) untuk disisipkan missing value
    fitur_cols = df.columns[:-1]
    if len(fitur_cols) >= 2:
        col1 = fitur_cols[0]
        col2 = fitur_cols[1]
        df.loc[0, col1] = np.nan
        df.loc[2, col2] = np.nan
        print("\n--- STATUS AWAL (Dengan Simulasi Missing Value) ---")
        print(df[[col1, col2, df.columns[-1]]].head(4))
    else:
        print("\n--- STATUS AWAL ---")
        print(df.head(4))

    print("\n[1] PENANGANAN MISSING VALUE (Imputasi)")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f" - Mengisi {col} (Numerik) dengan Median: {median_val}")
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f" - Mengisi {col} (Kategorik) dengan Modus: {mode_val}")
    
    print("\n[1B] STATISTIK DESKRIPTIF FITUR")
    # Menghitung statistik dasar untuk fitur numerik
    target_col_temp = df.columns[-1]
    numerics = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col_temp], errors='ignore')
    if not numerics.empty:
        stats_df = pd.DataFrame({
            'Mean': numerics.mean(),
            'Median': numerics.median(),
            'Std': numerics.std(),
            'Min': numerics.min(),
            'Max': numerics.max()
        })
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(stats_df.to_string())
    else:
        print(" - Tidak ada fitur numerik untuk ditampilkan statistiknya.")

    print("\n[2] PENGKODEAN TARGET (Target Encoding)")
    target_col = df.columns[-1]
    print(f" - Kolom target '{target_col}' sebelum pengkodean:")
    print(df[target_col].value_counts())
    
    # Pengkodean Label untuk target (Mendukung multi-kelas)
    df['Target_Encoded'] = pd.factorize(df[target_col])[0]
    print("\n - Kolom target setelah pengkodean numerik:")
    print(df['Target_Encoded'].value_counts())

    print("\n[3] LABEL ENCODING FITUR")
    # Mengubah fitur bertipe string/object menjadi kode numerik
    categorical_cols = []
    features = df.drop(columns=[target_col, 'Target_Encoded'])
    
    for col in features.columns:
        if features[col].dtype == 'object' or features[col].dtype.name == 'category':
            features[col] = pd.factorize(features[col])[0]
            categorical_cols.append(col)
    
    if len(categorical_cols) > 0:
        print(f" - Fitur yang di-encode: {', '.join(categorical_cols)}")
    else:
        print(" - Tidak ada fitur kategorikal string yang perlu di-encode (semua sudah numerik).")

    print("\n[4] SELEKSI FITUR (Information Gain)")
    # Menghitung nilai Information Gain antara fitur dan target
    X = features
    y = df['Target_Encoded']
    
    if len(y.unique()) > 1:
        ig_scores = mutual_info_classif(X, y, random_state=42)
        ig_df = pd.DataFrame({
            'Fitur': X.columns,
            'Information_Gain': ig_scores
        }).sort_values('Information_Gain', ascending=False).reset_index(drop=True)
        
        print(" - Hasil Perhitungan Information Gain:")
        print(ig_df.to_string())
    else:
        print(" - Dilewati karena target hanya memiliki 1 kelas.")

    print("\n[5] PEMBAGIAN DATA LATIH & UJI (80% Train, 20% Test)")
    if len(y.unique()) > 1:
        # Membagi data menggunakan stratifikasi pada kolom target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f" - Jumlah Data Latih (Train) : {X_train.shape[0]} baris")
        print(f" - Jumlah Data Uji (Test)  : {X_test.shape[0]} baris")
        print(" - Proporsi Target pada Data Latih:")
        print(y_train.value_counts(normalize=True).round(3))
    else:
        print(" - Gagal melakukan pembagian data karena kelas target tidak cukup.")
        return

    print("\n[6] NORMALISASI (StandardScaler)")
    # Z-Score Normalization
    scaler = StandardScaler()
    
    # Fit dan Transform HANYA pada data latih
    X_train_scaled = scaler.fit_transform(X_train)
    # Hanya Transform pada data uji (untuk mencegah data leakage)
    X_test_scaled = scaler.transform(X_test)
    
    # Konversi kembali ke dataframe untuk kemudahan melihat hasil
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    
    print(" - 3 Baris Pertama Data Latih SEBELUM Normalisasi:")
    print(X_train.head(3).round(1))
    print("\n - 3 Baris Pertama Data Latih SESUDAH Normalisasi (Mean=0, Std=1):")
    print(X_train_scaled_df.head(3).round(3))

    print("\n" + "="*70)
    print("Tahapan Preprocessing Selesai.")
    print("="*70)

if __name__ == "__main__":
    filepath = 'data_siswa_final.csv' if len(sys.argv) == 1 else sys.argv[1]
    run_preprocessing_demo(filepath)
