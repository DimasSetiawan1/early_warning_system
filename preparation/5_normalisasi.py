import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Tahap 5: Normalisasi (Z-Score) ---")
# Menyiapkan data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_siswa_final.csv'))
X = df.drop(columns=['Status', 'NISN']) # Menghilangkan NISN agar tidak mengacaukan contoh
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3 Baris Data Latih SEBELUM Normalisasi:")
print(X_train.head(3).round(2))

# Menggunakan metode Z-Score (Mean=0, Std=1)
scaler = StandardScaler()

# Melatih scaler HANYA pada data latih untuk mencegah kebocoran data
X_train_scaled = scaler.fit_transform(X_train)

# Menerapkan pola yang sama pada data uji
X_test_scaled = scaler.transform(X_test)

# Menampilkan hasil
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
print("\n3 Baris Data Latih SETELAH Normalisasi (Mean mendekati 0, Std mendekati 1):")
print(X_train_scaled_df.head(3).round(2))
