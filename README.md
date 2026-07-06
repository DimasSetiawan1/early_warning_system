# 🎓 Early Warning System — Prediksi Siswa Dropout Sekolah

Aplikasi web berbasis **Streamlit** untuk memprediksi risiko siswa putus sekolah (dropout) menggunakan algoritma **C4.5 (Decision Tree)** dengan pendekatan metodologi **CRISP-DM**. Sistem ini dilengkapi dengan fitur autentikasi multi-role, manajemen file dataset, dan dashboard analitik interaktif.

---

## 📋 Daftar Isi

1. [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
2. [Struktur Project](#-struktur-project)
3. [Arsitektur Sistem & Role](#-arsitektur-sistem--role)
4. [Dataset & Fitur](#-dataset--fitur)
5. [Implementasi Preprocessing Data](#-implementasi-preprocessing-data)
6. [Implementasi Algoritma C4.5](#-implementasi-algoritma-c45)
7. [Evaluasi Model](#-evaluasi-model)
8. [Cara Menjalankan](#-cara-menjalankan)

---

## 🛠 Teknologi yang Digunakan

| Komponen | Teknologi | Versi / Keterangan |
|----------|-----------|---------------------|
| Bahasa Pemrograman | Python | 3.x |
| Framework Web | Streamlit | Dashboard interaktif |
| Database | SQLite3 | Built-in Python, file `early_warning.db` |
| Machine Learning | Scikit-learn | `DecisionTreeClassifier` (C4.5) |
| Data Processing | Pandas, NumPy | Manipulasi dataframe & komputasi numerik |
| Visualisasi | Matplotlib, Seaborn | Grafik, heatmap, pohon keputusan |
| Hashing Password | hashlib (SHA-256) | Built-in Python |
| File Handling | uuid, os, io | Built-in Python |

### Library Python (`requirements.txt`)

```
streamlit
pandas
scikit-learn
numpy
matplotlib
seaborn
openpyxl
```

> **Catatan:** `sqlite3`, `hashlib`, `uuid`, `os`, `io` sudah bawaan Python sehingga tidak perlu diinstal terpisah.

---

## 📁 Struktur Project

```
early_warning_system/
├── app.py                      # Aplikasi utama Streamlit
├── db.py                       # Modul database SQLite (koneksi, CRUD, seed)
├── requirements.txt            # Daftar dependensi Python
├── penjabaran_fitur_dataset.md # Dokumentasi lengkap 14 fitur dataset
│
├── model_c45_dropout.pkl       # Model C4.5 pre-trained (UCI Dataset)
├── scaler_dropout.pkl          # StandardScaler pre-trained (UCI Dataset)
├── selected_features.pkl       # Daftar 8 fitur terpilih (UCI Dataset)
│
├── dataset/
│   └── data_dummy_siswa.csv    # Dataset dummy 501 baris, 14 fitur + 1 target
│
├── uploads/                    # Direktori penyimpanan file yang diupload user
│   └── .gitkeep
│
├── early_warning.db            # Database SQLite (auto-generated saat pertama run)
└── inspect_nb.py               # Script utilitas untuk inspeksi notebook
```

---

## 🔐 Arsitektur Sistem & Role

Sistem menggunakan **autentikasi berbasis session** dengan 3 role pengguna:

| Role | Hak Akses |
|------|-----------|
| **Admin** | Akses penuh: upload file, hapus semua file, manajemen user (CRUD), lihat semua hasil prediksi |
| **BK** | Hanya melihat hasil prediksi dari **semua** file yang diupload oleh Wali Kelas |
| **Wali Kelas** | Upload file, hapus file **miliknya sendiri**, melihat hasil prediksi **file miliknya sendiri** |

### Skema Database SQLite

#### Tabel `users`

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | INTEGER (PK, AI) | ID unik user |
| `username` | TEXT (UNIQUE) | Username login |
| `password` | TEXT | Password yang di-hash SHA-256 |
| `nama_lengkap` | TEXT | Nama lengkap user |
| `role` | TEXT | `Admin` / `BK` / `Wali Kelas` |
| `created_at` | TIMESTAMP | Waktu pembuatan akun |

#### Tabel `uploaded_files`

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | INTEGER (PK, AI) | ID unik file |
| `filename` | TEXT | Nama file unik di server (UUID prefix) |
| `original_filename` | TEXT | Nama file asli yang diupload |
| `file_path` | TEXT | Path absolut file di disk |
| `uploaded_by` | INTEGER (FK) | ID user yang mengupload |
| `uploaded_at` | TIMESTAMP | Waktu upload |
| `file_size` | INTEGER | Ukuran file dalam bytes |
| `description` | TEXT | Deskripsi opsional |

### Akun Default (Seed Data)

| Username | Password | Role | Nama Lengkap |
|----------|----------|------|--------------|
| `admin` | `admin123` | Admin | Administrator |
| `bk` | `admin123` | BK | Guru BK |
| `walikelas` | `admin123` | Wali Kelas | Wali Kelas 1 |

---

## 📊 Dataset & Fitur

### Deskripsi Dataset

Dataset terdiri dari **14 fitur** yang dibagi ke dalam 3 kategori utama, plus 1 kolom target:

| No | Nama Fitur | Kategori | Tipe Data | Range/Nilai |
|----|------------|----------|-----------|-------------|
| 1 | `Kehadiran_Semester_1` | Kehadiran | Numerik (int) | 50 – 100 (%) |
| 2 | `Kehadiran_Semester_2` | Kehadiran | Numerik (int) | 50 – 100 (%) |
| 3 | `Nilai_Rata_Semester_1` | Kehadiran | Numerik (float) | 40.0 – 95.0 |
| 4 | `Nilai_Rata_Semester_2` | Kehadiran | Numerik (float) | 40.0 – 95.0 |
| 5 | `Jumlah_Pelanggaran` | Kehadiran | Numerik (int) | 0 – 20 |
| 6 | `Pendidikan_Ayah` | Sosial | Kategorik ordinal (int) | 1–5 (SD s.d. S1+) |
| 7 | `Pendidikan_Ibu` | Sosial | Kategorik ordinal (int) | 1–5 (SD s.d. S1+) |
| 8 | `Status_Keluarga` | Sosial | Kategorik nominal (int) | 1–3 (Lengkap/Single parent/Wali) |
| 9 | `Jumlah_Saudara` | Sosial | Numerik (int) | 0 – 5 |
| 10 | `Jarak_Rumah_km` | Sosial | Numerik (float) | 0.5 – 30.0 km |
| 11 | `Status_SPP` | Ekonomi | Biner (int) | 0 = Menunggak, 1 = Lunas |
| 12 | `Penerima_Beasiswa` | Ekonomi | Biner (int) | 0 = Tidak, 1 = Ya |
| 13 | `Penghasilan_Ortu` | Ekonomi | Kategorik ordinal (int) | 1–4 (<1jt s.d. >5jt) |
| 14 | `Tanggungan_Keluarga` | Ekonomi | Numerik (int) | 1 – 7 |
| — | `Status` | **Target** | Kategorik | `Dropout` / `Non-Dropout` |

### Dua Mode Analisis

| Mode | Sumber Data | Proses |
|------|-------------|--------|
| **Eksperimen (UCI Dataset)** | Dataset sekunder UCI (*Predict Students' Dropout and Academic Success*) | Menggunakan model **pre-trained** (`model_c45_dropout.pkl`) dengan 8 fitur terpilih |
| **Data Primer (SMK Tunas Teknologi)** | Dataset lokal 14 fitur konteks SMA/SMK Indonesia | Model dilatih **on-the-fly** dari data yang diupload |

---

## 🔧 Implementasi Preprocessing Data

### 1. Pembacaan File (Data Ingestion)

File dataset didukung dalam 2 format:
- **CSV** — dengan deteksi separator otomatis (`,` atau `;`)
- **Excel (.xlsx)** — menggunakan library `openpyxl`

```python
# Deteksi separator CSV secara otomatis
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    first_line = f.readline()
sep = ';' if ';' in first_line else ','
df = pd.read_csv(file_path, sep=sep)
```

### 2. Penanganan Missing Values (Handling Missing Values)

Sistem menggunakan **strategi imputasi berdasarkan tipe data**:

| Tipe Data | Teknik Imputasi | Justifikasi |
|-----------|----------------|-------------|
| **Numerik** (`int64`, `float64`) | **Median** | Median lebih robust terhadap outlier dibanding mean. Tidak terpengaruh oleh nilai ekstrem. |
| **Kategorikal** (`object`, `category`) | **Modus** (nilai paling sering) | Modus mempertahankan distribusi kategori yang dominan tanpa memperkenalkan nilai baru. |

```python
# Implementasi dalam kode (app.py, baris 874-880)
for col in selected_train_features:
    if df_model[col].isnull().sum() > 0:
        if df_model[col].dtype in ['int64', 'float64']:
            df_model[col].fillna(df_model[col].median(), inplace=True)  # Median untuk numerik
        else:
            df_model[col].fillna(df_model[col].mode()[0], inplace=True)  # Modus untuk kategorikal
```

> **Catatan:** Imputasi dilakukan **per kolom** sebelum proses training, memastikan tidak ada nilai kosong yang masuk ke model.

### 3. Encoding / Transformasi Data Kategorikal

Fitur bertipe kategorikal (string/object) dikonversi ke **kode numerik** menggunakan **Label Encoding** (`pd.factorize`):

| Teknik | Implementasi | Justifikasi |
|--------|-------------|-------------|
| **Label Encoding** (`pd.factorize`) | Setiap kategori unik di-assign integer (0, 1, 2, ...) | Decision Tree tidak memerlukan One-Hot Encoding karena melakukan splitting berbasis threshold, bukan jarak. Label Encoding sudah cukup dan lebih efisien. |

```python
# Implementasi (app.py, baris 906-912)
X_numeric = X_all.copy()
categorical_cols = []
for col in X_numeric.columns:
    if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
        X_numeric[col] = pd.factorize(X_numeric[col])[0]
        categorical_cols.append(col)
```

### 4. Binarisasi Target (Target Encoding)

Kolom target dikonversi ke format **biner** (0/1) dengan logika multi-level:

| Prioritas | Kondisi | Mapping |
|-----------|---------|---------|
| 1 | Ditemukan nilai `"Dropout"` (case-insensitive) | `Dropout` → 1, lainnya → 0 |
| 2 | Ditemukan nilai numerik `1` | `1` → 1 (Dropout), `0` → 0 |
| 3 | Fallback | `pd.factorize` otomatis → kelas pertama = 0, kedua = 1 |

```python
# Implementasi (app.py, baris 882-902)
dropout_val = None
for val in unique_targets:
    if str(val).strip().lower() == 'dropout':
        dropout_val = val
        break

if dropout_val is not None:
    y = y_raw.apply(lambda x: 1 if x == dropout_val else 0).values
    class_names = ['Non-Dropout', 'Dropout']
```

### 5. Normalisasi / Standarisasi Data (Feature Scaling)

| Teknik | Library | Formula | Justifikasi |
|--------|---------|---------|-------------|
| **StandardScaler** (Z-Score Normalization) | `sklearn.preprocessing.StandardScaler` | `z = (x - μ) / σ` | Mengubah distribusi fitur agar memiliki mean = 0 dan std = 1. Meskipun Decision Tree secara teori tidak memerlukan scaling, standarisasi membantu konsistensi interpretasi dan komparabilitas antar fitur. |

```python
# Implementasi (app.py, baris 958-963)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform pada data training
X_test_scaled = scaler.transform(X_test)          # transform saja pada data testing
```

> **Penting:** `fit_transform()` hanya diterapkan pada **data training** untuk menghindari data leakage. Data testing di-transform menggunakan parameter (mean, std) dari data training.

### 6. Pembagian Data (Train-Test Split)

| Parameter | Nilai Default | Konfigurasi |
|-----------|--------------|-------------|
| **Rasio Testing** | 20% | Dapat diatur user via slider (10% – 50%) |
| **Stratifikasi** | `stratify=y` | Mempertahankan proporsi kelas Dropout/Non-Dropout di kedua split |
| **Random State** | 42 | Menjamin reproduktibilitas hasil |

```python
# Implementasi (app.py, baris 953-956)
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=test_size, random_state=42, stratify=y
)
```

### 7. Seleksi Fitur (Feature Selection) — Information Gain

| Teknik | Library | Aktivasi |
|--------|---------|----------|
| **Mutual Information Classification** (Information Gain) | `sklearn.feature_selection.mutual_info_classif` | Opsional, diaktifkan via checkbox |

Proses seleksi fitur:
1. Hitung **Information Gain** untuk setiap fitur terhadap target
2. Bandingkan dengan **threshold** yang ditentukan user (default: 0.05)
3. Fitur dengan IG ≥ threshold **dipertahankan**, sisanya dieliminasi
4. Divisualisasikan dalam bentuk **horizontal bar chart** (hijau = lolos, merah = tereliminasi)

```python
# Implementasi (app.py, baris 920-948)
ig_scores = mutual_info_classif(X_numeric, y, random_state=42)
ig_df = pd.DataFrame({
    'Fitur': X_numeric.columns,
    'Information Gain': ig_scores
}).sort_values('Information Gain', ascending=False)

selected_by_ig = ig_df[ig_df['Information Gain'] >= ig_threshold]['Fitur'].tolist()
```

---

## 🌳 Implementasi Algoritma C4.5

### Konfigurasi Model

Algoritma C4.5 diimplementasikan menggunakan `DecisionTreeClassifier` dari scikit-learn dengan parameter:

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `criterion` | `'entropy'` | Menggunakan **Information Gain** (Entropy-based) sebagai kriteria splitting — ciri khas C4.5 |
| `max_depth` | 7 (default, adjustable 3–15) | Kedalaman maksimum pohon untuk mencegah overfitting |
| `random_state` | 42 | Reproduktibilitas |

```python
# Implementasi (app.py, baris 966-971)
model_c45 = DecisionTreeClassifier(
    criterion='entropy',    # C4.5 menggunakan entropy/information gain
    max_depth=max_depth,    # Pruning: membatasi kedalaman pohon
    random_state=42
)
model_c45.fit(X_train_scaled, y_train)
```

> **Mengapa `criterion='entropy'`?**  
> Algoritma C4.5 menggunakan **Information Gain** (berbasis Entropy) untuk memilih atribut splitting terbaik di setiap node. Ini berbeda dengan CART yang menggunakan Gini Impurity. Rumus Entropy:
>
> **Entropy(S) = -Σ pᵢ × log₂(pᵢ)**
>
> Di mana pᵢ adalah proporsi sampel untuk kelas i.

### Pembentukan Pohon Keputusan

Pohon keputusan divisualisasikan menggunakan `sklearn.tree.plot_tree`:

```python
# Implementasi (app.py, baris 1025-1039)
plot_tree(
    model_c45,
    feature_names=X_numeric.columns.tolist(),
    class_names=class_names,
    filled=True,       # Warna node berdasarkan kelas dominan
    rounded=True,      # Node dengan sudut membulat
    proportion=True,   # Tampilkan proporsi, bukan count absolut
    fontsize=7,
    ax=ax_tree
)
```

### Feature Importance (Atribut Paling Berpengaruh)

Setelah model dilatih, tingkat kepentingan setiap fitur dihitung berdasarkan **total reduction of entropy** yang dihasilkan oleh fitur tersebut di seluruh node pohon:

```python
# Implementasi (app.py, baris 1044-1056)
importances = pd.Series(model_c45.feature_importances_, index=X_numeric.columns)
importances = importances.sort_values(ascending=True)
```

Divisualisasikan dalam **horizontal bar chart** dengan warna gradien RdYlGn.

---

## 📈 Evaluasi Model

### Metrik Evaluasi

| Metrik | Formula | Keterangan |
|--------|---------|------------|
| **Akurasi** | `(TP + TN) / Total` | Persentase prediksi benar secara keseluruhan |
| **Presisi** | `TP / (TP + FP)` | Dari yang diprediksi Dropout, berapa yang benar-benar Dropout |
| **Recall (Sensitivitas)** | `TP / (TP + FN)` | Dari yang benar-benar Dropout, berapa yang berhasil terdeteksi |
| **F1-Score** | `2 × (Presisi × Recall) / (Presisi + Recall)` | Harmonic mean dari Presisi dan Recall |

```python
# Implementasi (app.py, baris 976-980)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
```

### Visualisasi Evaluasi

| Visualisasi | Library | Keterangan |
|-------------|---------|------------|
| **Confusion Matrix** | Seaborn (`heatmap`) | Matriks 2×2 menunjukkan distribusi TP, TN, FP, FN |
| **ROC Curve** | Matplotlib | Kurva trade-off antara True Positive Rate vs False Positive Rate |
| **AUC (Area Under Curve)** | `sklearn.metrics.auc` | Nilai 0–1, semakin mendekati 1 semakin baik |
| **Pohon Keputusan** | `sklearn.tree.plot_tree` | Visualisasi struktur pohon lengkap |
| **Feature Importance** | Matplotlib (`barh`) | Bar chart horizontal menunjukkan kontribusi setiap fitur |
| **Heatmap Korelasi** | Seaborn (`heatmap`) | Matriks korelasi antar fitur (mask segitiga atas) |
| **Pie Chart** | Matplotlib | Distribusi prediksi Dropout vs Non-Dropout |
| **Box Plot** | Seaborn (`boxplot`) | Sebaran nilai per status prediksi |
| **Classification Report** | scikit-learn | Laporan lengkap per kelas (precision, recall, f1, support) |

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Akses di Browser

Buka `http://localhost:8501` dan login menggunakan salah satu akun default:

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | Admin |
| `bk` | `admin123` | BK |
| `walikelas` | `admin123` | Wali Kelas |

---

## 📝 Ringkasan Pipeline Preprocessing & Modeling

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE LENGKAP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA INGESTION                                              │
│     └─ Baca CSV/Excel → Deteksi separator otomatis              │
│                                                                 │
│  2. PREPROCESSING                                               │
│     ├─ Missing Values  → Median (numerik) / Modus (kategorikal) │
│     ├─ Encoding        → Label Encoding (pd.factorize)          │
│     └─ Target Encoding → Binarisasi (Dropout=1, Non-Dropout=0)  │
│                                                                 │
│  3. FEATURE SELECTION (Opsional)                                │
│     └─ Information Gain (mutual_info_classif) ≥ threshold       │
│                                                                 │
│  4. TRAIN-TEST SPLIT                                            │
│     └─ Stratified split (default 80:20, random_state=42)        │
│                                                                 │
│  5. FEATURE SCALING                                             │
│     └─ StandardScaler (Z-Score Normalization)                   │
│                                                                 │
│  6. MODEL TRAINING                                              │
│     └─ DecisionTreeClassifier(criterion='entropy', max_depth=7) │
│                                                                 │
│  7. EVALUASI                                                    │
│     ├─ Akurasi, Presisi, Recall, F1-Score                       │
│     ├─ Confusion Matrix, ROC Curve, AUC                         │
│     ├─ Feature Importance                                       │
│     └─ Classification Report                                    │
│                                                                 │
│  8. PREDIKSI BATCH                                              │
│     └─ Prediksi seluruh data → Export CSV                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
