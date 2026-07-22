import pandas as pd
import os
from sklearn.model_selection import train_test_split

print("--- Tahap 4: Train-Test Split ---")
# Membaca dataset dari folder utama
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_siswa_final.csv'))

# Penyiapan dasar sebelum di-split
X = df.drop(columns=['Status']) # Variabel Independen (Fitur)
sorted_unique = sorted([str(v) for v in df['Status'].unique()])
label_map = {val: idx for idx, val in enumerate(sorted_unique)}
y = df['Status'].astype(str).map(label_map).values # Variabel Dependen (Target)

# Membagi data latih (80%) dan data uji (20%) secara proporsional (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Total Dataset Awal: {len(X)} baris")
print(f"Total Data Latih (Train 80%): {len(X_train)} baris")
print(f"Total Data Uji (Test 20%): {len(X_test)} baris")
