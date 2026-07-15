import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

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
    # (Karena data_dummy_siswa.csv bawaannya tidak ada missing value)
    df.loc[0, 'Kehadiran_Semester_1'] = np.nan
    df.loc[2, 'Pendidikan_Ayah'] = np.nan

    print("\n--- STATUS AWAL (Dengan Simulasi Missing Value) ---")
    print(df[['Kehadiran_Semester_1', 'Pendidikan_Ayah', 'Status']].head(4))

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
    
    print("\n[2] BINARISASI TARGET")
    target_col = 'Status'
    print(f" - Kolom target '{target_col}' sebelum binarisasi:")
    print(df[target_col].value_counts())
    
    # Binarisasi: Dropout = 1, Non-Dropout = 0
    df['Status_Biner'] = df[target_col].apply(lambda x: 1 if str(x).strip().lower() == 'dropout' else 0)
    print("\n - Kolom target setelah binarisasi (Dropout=1, Non-Dropout=0):")
    print(df['Status_Biner'].value_counts())

    print("\n[3] LABEL ENCODING")
    # Mengubah fitur bertipe string/object menjadi kode numerik
    categorical_cols = []
    features = df.drop(columns=[target_col, 'Status_Biner'])
    
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
    y = df['Status_Biner']
    
    ig_scores = mutual_info_classif(X, y, random_state=42)
    ig_df = pd.DataFrame({
        'Fitur': X.columns,
        'Information_Gain': ig_scores
    }).sort_values('Information_Gain', ascending=False).reset_index(drop=True)
    
    print(" - Hasil Perhitungan Information Gain:")
    print(ig_df.to_string())

    print("\n[5] TRAIN-TEST SPLIT (80% Train, 20% Test)")
    # Membagi data menggunakan stratifikasi pada kolom target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" - Jumlah Data Latih (Train) : {X_train.shape[0]} baris")
    print(f" - Jumlah Data Uji (Test)  : {X_test.shape[0]} baris")
    print(" - Proporsi Target pada Data Latih:")
    print(y_train.value_counts(normalize=True).round(3))

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
    run_preprocessing_demo('data_dummy_siswa.csv')
