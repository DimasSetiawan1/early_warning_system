import pandas as pd
import os

print("--- Tahap 3: Binarisasi (Pengkodean) Target ---")
# Membaca dataset dari folder utama
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_siswa_final.csv'))

# Memetakan target multi-kelas ('DO-Nilai', 'DO-Masalah') menjadi angka numerik
target_col = 'Status'
y_raw = df[target_col]

print("Sebelum Binarisasi/Pengkodean Target:")
print(y_raw.value_counts())

# Ekstrak nilai unik secara terurut (alfabetis)
sorted_unique = sorted([str(v) for v in y_raw.unique()])

# Buat pemetaan (misal: 'DO-Masalah' -> 0, 'DO-Nilai' -> 1)
label_map = {val: idx for idx, val in enumerate(sorted_unique)}
print(f"\nPemetaan: {label_map}")

y = y_raw.astype(str).map(label_map).values

print("\nSetelah Binarisasi/Pengkodean Target:")
print(pd.Series(y).value_counts())
