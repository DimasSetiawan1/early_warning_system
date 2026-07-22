# 🎓 Sistem Peringatan Dini — Prediksi Siswa Putus Sekolah

Aplikasi web berbasis **Streamlit** untuk memprediksi risiko siswa putus sekolah menggunakan algoritma **C4.5 (Pohon Keputusan)** dengan pendekatan metodologi **CRISP-DM**. Sistem ini dilengkapi dengan fitur autentikasi banyak-peran, manajemen berkas himpunan data, dan dasbor analitik interaktif.

---

## 📋 Daftar Isi

1. [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
2. [Struktur Proyek](#-struktur-proyek)
3. [Arsitektur Sistem & Peran](#-arsitektur-sistem--peran)
4. [Himpunan Data & Fitur](#-himpunan-data--fitur)
5. [Implementasi Pra-pemrosesan Data](#-implementasi-pra-pemrosesan-data)
6. [Implementasi Algoritma C4.5](#-implementasi-algoritma-c45)
7. [Evaluasi Model](#-evaluasi-model)
8. [Cara Menjalankan](#-cara-menjalankan)

---

## 🛠 Teknologi yang Digunakan

| Komponen | Teknologi | Versi / Keterangan |
|----------|-----------|---------------------|
| Bahasa Pemrograman | Python | 3.x |
| Kerangka Kerja Web | Streamlit | Dasbor interaktif |
| Basis Data | SQLite3 | Bawaan Python, berkas `early_warning.db` |
| Pembelajaran Mesin | Scikit-learn | `DecisionTreeClassifier` (C4.5) |
| Pemrosesan Data | Pandas, NumPy | Manipulasi kerangka data & komputasi numerik |
| Visualisasi | Matplotlib, Seaborn | Grafik, peta panas, pohon keputusan |
| Enkripsi Kata Sandi | hashlib (SHA-256) | Bawaan Python |
| Penanganan Berkas | uuid, os, io | Bawaan Python |

### Pustaka Python (`requirements.txt`)

```
streamlit
pandas
scikit-learn
numpy
matplotlib
seaborn
openpyxl
```

> **Catatan:** `sqlite3`, `hashlib`, `uuid`, `os`, `io` sudah menjadi bawaan Python sehingga tidak perlu dipasang terpisah.

---

## 📁 Struktur Proyek

```
early_warning_system/
├── app.py                      # Aplikasi utama Streamlit
├── db.py                       # Modul basis data SQLite (koneksi, operasi, data awal)
├── requirements.txt            # Daftar dependensi Python
├── penjabaran_fitur_dataset.md # Dokumentasi lengkap 14 fitur himpunan data
│
├── model_c45_dropout.pkl       # Model C4.5 pra-latih (Himpunan Data UCI)
├── scaler_dropout.pkl          # Pembuat Skala Standar pra-latih (Himpunan Data UCI)
├── selected_features.pkl       # Daftar 8 fitur terpilih (Himpunan Data UCI)
│
├── dataset/
│   └── data_dummy_siswa.csv    # Data tiruan 501 baris, 14 fitur + 1 target
│
├── uploads/                    # Direktori penyimpanan berkas yang diunggah pengguna
│   └── .gitkeep
│
├── early_warning.db            # Basis data SQLite (otomatis dibuat saat dijalankan)
└── inspect_nb.py               # Skrip utilitas untuk inspeksi catatan
```

---

## 🔐 Arsitektur Sistem & Peran

Sistem menggunakan **autentikasi berbasis sesi** dengan 2 peran pengguna:

| Peran | Hak Akses |
|------|-----------|
| **BK** | Akses penuh: unggah berkas, hapus semua berkas, manajemen pengguna, lihat semua hasil prediksi, dan jalankan konfigurasi |
| **Guru** | Unggah berkas, hapus berkas **miliknya sendiri**, dan melihat riwayat prediksi di dasbor |

### Skema Basis Data SQLite

#### Tabel `users` (Pengguna)

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | INTEGER (PK, AI) | ID unik pengguna |
| `username` | TEXT (UNIQUE) | Nama pengguna untuk masuk |
| `password` | TEXT | Kata sandi yang dienkripsi SHA-256 |
| `nama_lengkap` | TEXT | Nama lengkap pengguna |
| `role` | TEXT | `BK` / `Guru` |
| `created_at` | TIMESTAMP | Waktu pembuatan akun |

#### Tabel `uploaded_files` (Berkas Unggahan)

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | INTEGER (PK, AI) | ID unik berkas |
| `filename` | TEXT | Nama berkas unik di peladen (Awalan UUID) |
| `original_filename` | TEXT | Nama berkas asli yang diunggah |
| `file_path` | TEXT | Jalur absolut berkas di penyimpanan |
| `uploaded_by` | INTEGER (FK) | ID pengguna yang mengunggah |
| `uploaded_at` | TIMESTAMP | Waktu unggah |
| `file_size` | INTEGER | Ukuran berkas dalam bita |
| `description` | TEXT | Deskripsi opsional |

### Akun Bawaan (Data Awal)

| Nama Pengguna | Kata Sandi | Peran | Nama Lengkap |
|----------|----------|------|--------------|
| `bk` | `admin123` | BK | Guru BK |
| `guru` | `admin123` | Guru | Guru Mata Pelajaran |

---

## 📊 Himpunan Data & Fitur

### Deskripsi Himpunan Data

Himpunan data terdiri dari **14 fitur** yang dibagi ke dalam 3 kategori utama, ditambah 1 kolom target:

| No | Nama Fitur | Kategori | Tipe Data | Rentang/Nilai |
|----|------------|----------|-----------|-------------|
| 1 | `Kehadiran_Semester_1` | Kehadiran | Numerik (int) | 50 – 100 (%) |
| 2 | `Kehadiran_Semester_2` | Kehadiran | Numerik (int) | 50 – 100 (%) |
| 3 | `Nilai_Rata_Semester_1` | Kehadiran | Numerik (float) | 40.0 – 95.0 |
| 4 | `Nilai_Rata_Semester_2` | Kehadiran | Numerik (float) | 40.0 – 95.0 |
| 5 | `Jumlah_Pelanggaran` | Kehadiran | Numerik (int) | 0 – 20 |
| 6 | `Pendidikan_Ayah` | Sosial | Kategorik ordinal (int) | 1–5 (SD s.d. S1+) |
| 7 | `Pendidikan_Ibu` | Sosial | Kategorik ordinal (int) | 1–5 (SD s.d. S1+) |
| 8 | `Status_Keluarga` | Sosial | Kategorik nominal (int) | 1–3 (Lengkap/Orang tua tunggal/Wali) |
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
| **Eksperimen (Himpunan Data UCI)** | Himpunan data sekunder UCI (*Prediksi Siswa Putus Sekolah dan Kesuksesan Akademik*) | Menggunakan model **pra-latih** (`model_c45_dropout.pkl`) dengan 8 fitur terpilih |
| **Data Primer (SMK Tunas Teknologi)** | Data lokal 14 fitur konteks SMA/SMK Indonesia | Model dilatih **secara langsung** dari data yang diunggah |

---

## 🔧 Implementasi Pra-pemrosesan Data

### 1. Pemasukan Data

Berkas himpunan data didukung dalam 2 format:
- **CSV** — dengan deteksi pemisah otomatis (`,` atau `;`)
- **Excel (.xlsx)** — menggunakan pustaka `openpyxl`

```python
# Deteksi pemisah CSV secara otomatis
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    first_line = f.readline()
sep = ';' if ';' in first_line else ','
df = pd.read_csv(file_path, sep=sep)
```

### 2. Penanganan Nilai Kosong

Sistem menggunakan **strategi pengisian berdasarkan tipe data**:

| Tipe Data | Teknik Pengisian | Justifikasi |
|-----------|----------------|-------------|
| **Numerik** (`int64`, `float64`) | **Nilai Tengah (Median)** | Nilai tengah lebih kebal terhadap pencilan dibanding rata-rata. Tidak terpengaruh oleh nilai ekstrem. |
| **Kategorikal** (`object`, `category`) | **Modus** (nilai paling sering) | Modus mempertahankan distribusi kategori yang dominan tanpa memperkenalkan nilai baru. |

```python
# Implementasi dalam kode (app.py, baris 874-880)
for col in selected_train_features:
    if df_model[col].isnull().sum() > 0:
        if df_model[col].dtype in ['int64', 'float64']:
            df_model[col].fillna(df_model[col].median(), inplace=True)  # Nilai tengah untuk numerik
        else:
            df_model[col].fillna(df_model[col].mode()[0], inplace=True)  # Modus untuk kategorikal
```

> **Catatan:** Pengisian dilakukan **per kolom** sebelum proses pelatihan, memastikan tidak ada nilai kosong yang masuk ke model.

### 3. Pengkodean / Transformasi Data Kategorikal

Fitur bertipe kategorikal (untaian/objek) dikonversi ke **kode numerik** menggunakan **Pengkodean Label** (`pd.factorize`):

| Teknik | Implementasi | Justifikasi |
|--------|-------------|-------------|
| **Pengkodean Label** (`pd.factorize`) | Setiap kategori unik diberi bilangan bulat (0, 1, 2, ...) | Pohon Keputusan tidak memerlukan Pengkodean Satu-Aktif (One-Hot) karena melakukan pemisahan berbasis ambang batas, bukan jarak. Pengkodean Label sudah cukup dan lebih efisien. |

```python
# Implementasi (app.py, baris 906-912)
X_numeric = X_all.copy()
categorical_cols = []
for col in X_numeric.columns:
    if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
        X_numeric[col] = pd.factorize(X_numeric[col])[0]
        categorical_cols.append(col)
```

### 4. Pengkodean Target 

Kolom target dikonversi ke format **biner** (0/1) dengan logika banyak tingkat:

| Prioritas | Kondisi | Pemetaan |
|-----------|---------|---------|
| 1 | Ditemukan nilai `"Dropout"` (tidak peka huruf kapital) | `Dropout` → 1, lainnya → 0 |
| 2 | Ditemukan nilai numerik `1` | `1` → 1 (Dropout), `0` → 0 |
| 3 | Cadangan | `pd.factorize` otomatis → kelas pertama = 0, kedua = 1 |

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

### 5. Normalisasi / Standarisasi Data (Penskalaan Fitur)

| Teknik | Pustaka | Rumus | Justifikasi |
|--------|---------|---------|-------------|
| **Pembuat Skala Standar** (Normalisasi Z-Score) | `sklearn.preprocessing.StandardScaler` | `z = (x - μ) / σ` | Mengubah distribusi fitur agar memiliki rata-rata = 0 dan simpangan baku = 1. Meskipun Pohon Keputusan secara teori tidak memerlukan penskalaan, standarisasi membantu konsistensi interpretasi dan perbandingan antar fitur. |

```python
# Implementasi (app.py, baris 958-963)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # pas + ubah pada data latih
X_test_scaled = scaler.transform(X_test)          # ubah saja pada data uji
```

> **Penting:** `fit_transform()` hanya diterapkan pada **data latih** untuk menghindari kebocoran data. Data uji diubah menggunakan parameter (rata-rata, simpangan baku) dari data latih.

### 6. Pembagian Data Latih dan Uji

| Parameter | Nilai Bawaan | Konfigurasi |
|-----------|--------------|-------------|
| **Rasio Pengujian** | 20% | Dapat diatur pengguna melalui penggeser (10% – 50%) |
| **Stratifikasi** | `stratify=y` | Mempertahankan proporsi kelas Dropout/Non-Dropout di kedua bagian |
| **Kondisi Acak** | 42 | Menjamin reproduktibilitas hasil |

```python
# Implementasi (app.py, baris 953-956)
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=test_size, random_state=42, stratify=y
)
```

### 7. Seleksi Fitur — Perolehan Informasi (Information Gain)

| Teknik | Pustaka | Aktivasi |
|--------|---------|----------|
| **Klasifikasi Informasi Timbal Balik** (Perolehan Informasi) | `sklearn.feature_selection.mutual_info_classif` | Opsional, diaktifkan via kotak centang |

Proses seleksi fitur:
1. Hitung **Perolehan Informasi** untuk setiap fitur terhadap target
2. Bandingkan dengan **ambang batas** yang ditentukan pengguna (bawaan: 0.05)
3. Fitur dengan nilai ≥ ambang batas **dipertahankan**, sisanya dieliminasi
4. Divisualisasikan dalam bentuk **grafik batang horizontal** (hijau = lolos, merah = tereliminasi)

```python
# Implementasi (app.py, baris 920-948)
ig_scores = mutual_info_classif(X_numeric, y, random_state=42)
ig_df = pd.DataFrame({
    'Fitur': X_numeric.columns,
    'Perolehan Informasi': ig_scores
}).sort_values('Perolehan Informasi', ascending=False)

selected_by_ig = ig_df[ig_df['Perolehan Informasi'] >= ig_threshold]['Fitur'].tolist()
```

---

## 🌳 Implementasi Algoritma C4.5

### Konfigurasi Model

Algoritma C4.5 diimplementasikan menggunakan penggolong Pohon Keputusan dari scikit-learn dengan parameter:

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `kriteria` | `'entropi'` | Menggunakan **Perolehan Informasi** (Berbasis entropi) sebagai kriteria pemisahan — ciri khas C4.5 |
| `kedalaman_maksimal` | 7 (bawaan, dapat diatur 3–15) | Kedalaman maksimum pohon untuk mencegah *overfitting* |
| `kondisi_acak` | 42 | Reproduktibilitas |

```python
# Implementasi (app.py, baris 966-971)
model_c45 = DecisionTreeClassifier(
    criterion='entropy',    # C4.5 menggunakan entropi/perolehan informasi
    max_depth=max_depth,    # Pemangkasan: membatasi kedalaman pohon
    random_state=42
)
model_c45.fit(X_train_scaled, y_train)
```

> **Mengapa `criterion='entropy'`?**  
> Algoritma C4.5 menggunakan **Perolehan Informasi** (berbasis Entropi) untuk memilih atribut pemisahan terbaik di setiap simpul. Ini berbeda dengan algoritma CART yang menggunakan Impuritas Gini. Rumus Entropi:
>
> **Entropi(S) = -Σ pᵢ × log₂(pᵢ)**
>
> Di mana pᵢ adalah proporsi sampel untuk kelas i.

### Pembentukan Pohon Keputusan

Pohon keputusan divisualisasikan menggunakan alat visualisasi dari scikit-learn:

```python
# Implementasi (app.py, baris 1025-1039)
plot_tree(
    model_c45,
    feature_names=X_numeric.columns.tolist(),
    class_names=class_names,
    filled=True,       # Warna simpul berdasarkan kelas dominan
    rounded=True,      # Simpul dengan sudut membulat
    proportion=True,   # Tampilkan proporsi, bukan hitungan mutlak
    fontsize=7,
    ax=ax_tree
)
```

### Tingkat Kepentingan Fitur (Atribut Paling Berpengaruh)

Setelah model dilatih, tingkat kepentingan setiap fitur dihitung berdasarkan **total pengurangan entropi** yang dihasilkan oleh fitur tersebut di seluruh simpul pohon:

```python
# Implementasi (app.py, baris 1044-1056)
importances = pd.Series(model_c45.feature_importances_, index=X_numeric.columns)
importances = importances.sort_values(ascending=True)
```

Divisualisasikan dalam **grafik batang horizontal** dengan warna gradasi.

---

## 📈 Evaluasi Model

### Metrik Evaluasi

| Metrik | Rumus | Keterangan |
|--------|---------|------------|
| **Akurasi** | `(TP + TN) / Total` | Persentase prediksi benar secara keseluruhan |
| **Presisi** | `TP / (TP + FP)` | Dari yang diprediksi Putus Sekolah, berapa yang benar-benar Putus Sekolah |
| **Daya Ingat (Sensitivitas)** | `TP / (TP + FN)` | Dari yang benar-benar Putus Sekolah, berapa yang berhasil terdeteksi |
| **Skor-F1** | `2 × (Presisi × Daya Ingat) / (Presisi + Daya Ingat)` | Rata-rata harmonik dari Presisi dan Daya Ingat |

```python
# Implementasi (app.py, baris 976-980)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
```

### Visualisasi Evaluasi

| Visualisasi | Pustaka | Keterangan |
|-------------|---------|------------|
| **Matriks Kebingungan** | Seaborn | Matriks 2×2 menunjukkan distribusi TP, TN, FP, FN |
| **Kurva ROC** | Matplotlib | Kurva pertukaran antara Tingkat Positif Benar dan Tingkat Positif Salah |
| **AUC (Luas Di Bawah Kurva)** | `sklearn.metrics.auc` | Nilai 0–1, semakin mendekati 1 semakin baik |
| **Pohon Keputusan** | `sklearn.tree` | Visualisasi struktur pohon lengkap |
| **Tingkat Kepentingan Fitur** | Matplotlib | Grafik batang horizontal menunjukkan kontribusi setiap fitur |
| **Peta Panas Korelasi** | Seaborn | Matriks korelasi antar fitur |
| **Diagram Lingkaran** | Matplotlib | Distribusi prediksi Putus Sekolah vs Tidak Putus Sekolah |
| **Diagram Kotak** | Seaborn | Sebaran nilai per status prediksi |
| **Laporan Klasifikasi** | scikit-learn | Laporan lengkap per kelas (presisi, daya ingat, skor-f1, dukungan) |

---

## 🚀 Cara Menjalankan

### 1. Pasang Dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Akses di Peramban

Buka `http://localhost:8501` dan masuk menggunakan salah satu akun bawaan:

| Nama Pengguna | Kata Sandi | Peran |
|----------|----------|------|
| `bk` | `admin123` | BK |
| `guru` | `admin123` | Guru |

---

## 📝 Ringkasan Alur Pra-pemrosesan & Pemodelan

```text
┌─────────────────────────────────────────────────────────────────┐
│                    ALUR KERJA LENGKAP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PEMASUKAN DATA                                              │
│     └─ Baca CSV/Excel → Deteksi pemisah otomatis                │
│                                                                 │
│  2. PRA-PEMROSESAN                                              │
│     ├─ Nilai Kosong    → Nilai Tengah (numerik) / Modus (kata)  │
│     ├─ Pengkodean      → Pengkodean Label (pd.factorize)        │
│     └─ Pengkodean Target → Binarisasi (Dropout=1, Non=0)        │
│                                                                 │
│  3. SELEKSI FITUR (Opsional)                                    │
│     └─ Perolehan Informasi (mutual_info_classif) ≥ ambang       │
│                                                                 │
│  4. PEMBAGIAN DATA LATIH-UJI                                    │
│     └─ Pemisahan stratifikasi (bawaan 80:20, kondisi_acak=42)   │
│                                                                 │
│  5. PENSKALAAN FITUR                                            │
│     └─ Pembuat Skala Standar (Normalisasi Z-Score)              │
│                                                                 │
│  6. PELATIHAN MODEL                                             │
│     └─ DecisionTreeClassifier(kriteria='entropi', maksimal=7)   │
│                                                                 │
│  7. EVALUASI                                                    │
│     ├─ Akurasi, Presisi, Daya Ingat, Skor-F1                    │
│     ├─ Matriks Kebingungan, Kurva ROC, AUC                      │
│     ├─ Tingkat Kepentingan Fitur                                │
│     └─ Laporan Klasifikasi                                      │
│                                                                 │
│  8. PREDIKSI SERENTAK                                           │
│     └─ Prediksi seluruh data → Ekspor CSV                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
