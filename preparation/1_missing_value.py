import pandas as pd
import os

print("--- Tahap 1: Penanganan Missing Value ---")
# Membaca dataset dari folder utama
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_siswa_final.csv'))
print("Sebelum Penanganan Missing Value:")
print(df.isnull().sum())

# Iterasi setiap kolom dalam dataset
for col in df.columns:
    if df[col].isnull().sum() > 0: # Cek jika ada data yang kosong (NaN)
        # Jika tipe data numerik, isi missing value dengan Median
        if df[col].dtype in ['int64', 'float64']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        # Jika tipe data kategorikal (teks), isi dengan Modus
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

print("\nSetelah Penanganan Missing Value:")
print(df.isnull().sum())
