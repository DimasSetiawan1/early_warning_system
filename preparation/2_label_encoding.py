import pandas as pd
import os

print("--- Tahap 2: Label Encoding ---")
# Membaca dataset dari folder utama
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_siswa_final.csv'))

# Mengubah fitur bertipe string/kategorikal menjadi angka
fitur = df.drop(columns=['Status']) # Pisahkan dari kolom target
print("Sebelum Label Encoding (Tipe Data):")
print(fitur.dtypes)

for col in fitur.columns:
    # Jika kolom berisi teks (object/category)
    if fitur[col].dtype == 'object' or fitur[col].dtype.name == 'category':
        print(f"\nMeng-encode kolom: {col}")
        # Ubah menjadi kode numerik (0, 1, 2, dst)
        fitur[col] = pd.factorize(fitur[col])[0]

print("\nSetelah Label Encoding (Tipe Data):")
print(fitur.dtypes)
