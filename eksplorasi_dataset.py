import pandas as pd
import numpy as np

def run_exploration(filepath):
    print("="*60)
    print(f"Eksplorasi Data Awal (EDA) - {filepath}")
    print("="*60)

    # 1. Membaca dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} tidak ditemukan.")
        return

    # 2. Info Umum
    print("\n[1] INFO UMUM DATASET")
    print(f"Jumlah Baris (Siswa) : {df.shape[0]}")
    print(f"Jumlah Kolom (Fitur) : {df.shape[1]}")
    
    # 3. Missing Values
    print("\n[2] IDENTIFIKASI MISSING VALUES")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("Tidak ada data kosong (missing values).")
    else:
        print("Terdapat data kosong pada kolom:")
        print(missing[missing > 0])

    # 4. Statistik Deskriptif (Sesuai Gambar Request)
    print("\n[3] STATISTIK DESKRIPTIF (Mean, Median, Std, Min, Max)")
    # Mengambil hanya kolom numerik (int64, float64)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Menghitung metrik secara spesifik
    stats = pd.DataFrame(index=numeric_cols)
    stats['Mean'] = df[numeric_cols].mean()
    stats['Median'] = df[numeric_cols].median()
    stats['Std'] = df[numeric_cols].std()
    stats['Min'] = df[numeric_cols].min()
    stats['Max'] = df[numeric_cols].max()
    
    # Menampilkan dengan perataan dan pembulatan
    print(stats.round(2).to_string())

    # 5. Distribusi Target
    print("\n[4] DISTRIBUSI KELAS TARGET")
    if 'Status' in df.columns:
        counts = df['Status'].value_counts()
        props = df['Status'].value_counts(normalize=True) * 100
        
        target_df = pd.DataFrame({
            'Jumlah': counts,
            'Persentase (%)': props.round(2)
        })
        print(target_df.to_string())
    else:
        print("Kolom 'Status' tidak ditemukan.")

    print("\n" + "="*60)

if __name__ == "__main__":
    # Menjalankan eksplorasi untuk dataset primer
    run_exploration('data_dummy_siswa.csv')
