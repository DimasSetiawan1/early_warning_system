import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

print("--- Tahap 6: Feature Selection (Information Gain) ---")
# Menyiapkan data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_siswa_final.csv'))
X = df.drop(columns=['Status'])
label_map = {val: idx for idx, val in enumerate(sorted([str(v) for v in df['Status'].unique()]))}
y = df['Status'].astype(str).map(label_map).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = StandardScaler().fit_transform(X_train)

# Menghitung Information Gain untuk setiap fitur
ig_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)

# Memasukkan hasil ke dalam tabel dan mengurutkan dari nilai terbesar
ig_df = pd.DataFrame({
    'Fitur': X.columns,
    'Information_Gain': ig_scores
}).sort_values('Information_Gain', ascending=False)

print("\nHasil Perhitungan Information Gain:")
print(ig_df)

# Mengambil fitur-fitur yang lolos threshold (ambang batas)
ambang_batas = 0.05
fitur_terpilih = ig_df[ig_df['Information_Gain'] >= ambang_batas]['Fitur'].tolist()
print(f"\nFitur yang terpilih (Information Gain >= {ambang_batas}):")
print(fitur_terpilih)
