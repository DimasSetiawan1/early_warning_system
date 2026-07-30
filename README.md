# 🎓 Sistem Peringatan Dini — Prediksi Siswa Putus Sekolah

Aplikasi web berbasis **Streamlit** untuk memprediksi risiko siswa putus sekolah menggunakan algoritma **C4.5 (Pohon Keputusan)** dengan pendekatan metodologi **CRISP-DM**. Sistem ini dilengkapi dengan fitur autentikasi banyak-peran, manajemen berkas himpunan data, dan dasbor analitik interaktif.

---

## 📋 Daftar Isi

1. [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
2. [Struktur Proyek](#-struktur-proyek)
3. [Arsitektur MVC](#-arsitektur-mvc)
4. [Arsitektur Sistem & Peran](#-arsitektur-sistem--peran)
5. [Himpunan Data & Fitur](#-himpunan-data--fitur)
6. [Implementasi Pra-pemrosesan Data](#-implementasi-pra-pemrosesan-data)
7. [Implementasi Algoritma C4.5](#-implementasi-algoritma-c45)
8. [Evaluasi Model](#-evaluasi-model)
9. [Cara Menjalankan](#-cara-menjalankan)

---

## 🛠 Teknologi yang Digunakan

| Komponen            | Teknologi           | Versi / Keterangan                           |
| ------------------- | ------------------- | -------------------------------------------- |
| Bahasa Pemrograman  | Python              | 3.x                                          |
| Kerangka Kerja Web  | Streamlit           | Dasbor interaktif                            |
| Basis Data          | SQLite3             | Bawaan Python, berkas `early_warning.db`     |
| Pembelajaran Mesin  | Scikit-learn        | `DecisionTreeClassifier` (C4.5)              |
| Pemrosesan Data     | Pandas, NumPy       | Manipulasi kerangka data & komputasi numerik |
| Visualisasi         | Matplotlib, Seaborn | Grafik, peta panas, pohon keputusan          |
| Enkripsi Kata Sandi | hashlib (SHA-256)   | Bawaan Python                                |
| Penanganan Berkas   | uuid, os, io        | Bawaan Python                                |

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

Proyek menggunakan pola **MVC (Model-View-Controller)** untuk memisahkan tanggung jawab setiap lapisan kode.

```
early_warning_system/
├── app.py                          # Entry point (router utama, ~60 baris)
├── db.py                           # Modul basis data SQLite (koneksi, CRUD, data awal)
├── requirements.txt                # Daftar dependensi Python
├── penjabaran_fitur_dataset.md     # Dokumentasi lengkap fitur himpunan data
│
├── model_c45_dropout.pkl           # Model C4.5 pra-latih (Himpunan Data UCI)
├── scaler_dropout.pkl              # Pembuat Skala Standar pra-latih (UCI)
├── selected_features.pkl           # Daftar 8 fitur terpilih (UCI)
│
├── models/                         # ── LAYER MODEL ──
│   ├── __init__.py
│   ├── file_utils.py               # Utilitas file: baca CSV/Excel, format ukuran & waktu
│   └── ml_model.py                 # Logika ML: load artifacts, preprocessing, C4.5, evaluasi
│
├── controllers/                    # ── LAYER CONTROLLER ──
│   ├── __init__.py
│   ├── auth_controller.py          # Login, logout, init session state
│   ├── file_controller.py          # Upload, ambil daftar, hapus file
│   ├── prediction_controller.py    # Siapkan konfigurasi prediksi, simpan riwayat
│   └── user_controller.py          # CRUD pengguna (wrapper db.py)
│
├── views/                          # ── LAYER VIEW ──
│   ├── __init__.py
│   ├── login_view.py               # Halaman login
│   ├── sidebar_view.py             # Sidebar navigasi berdasarkan peran
│   ├── dashboard_view.py           # Dasbor riwayat prediksi
│   ├── upload_view.py              # Halaman unggah berkas
│   ├── file_management_view.py     # Halaman manajemen berkas
│   ├── user_management_view.py     # Halaman manajemen pengguna (BK only)
│   ├── prediction_config_view.py   # Konfigurasi mode & parameter prediksi
│   └── prediction_result_view.py   # Hasil & visualisasi prediksi lengkap
│
├── dataset/
│   ├── data_siswa.csv              # Data awal siswa
│   └── fix/
│       └── data_siswa.csv          # Himpunan data final siap pakai
│
├── uploads/                        # Direktori penyimpanan berkas yang diunggah pengguna
│   └── .gitkeep
│
└── early_warning.db                # Basis data SQLite (otomatis dibuat saat dijalankan)
```

---

## 🏗️ Arsitektur MVC

Aplikasi dibangun menggunakan pola **Model-View-Controller (MVC)** untuk memisahkan tanggung jawab kode agar mudah dikembangkan dan dipelihara.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         app.py (Entry Point)                         │
│              Routing halaman berdasarkan session_state               │
└────────────┬─────────────────┬────────────────────────┬─────────────┘
             │                 │                        │
      ┌──────▼──────┐  ┌───────▼───────┐      ┌────────▼────────┐
      │  CONTROLLER │  │     VIEW      │      │      MODEL      │
      │ controllers/│  │    views/     │      │    models/      │
      │             │  │               │      │                 │
      │ auth_       │  │ login_view    │      │ file_utils.py   │
      │ controller  │  │ sidebar_view  │      │ (baca file,     │
      │             │  │ dashboard_    │      │  format data)   │
      │ file_       │  │ view          │      │                 │
      │ controller  │  │ upload_view   │      │ ml_model.py     │
      │             │  │ file_mgmt_    │      │ (preprocessing, │
      │ prediction_ │  │ view          │      │  C4.5 training, │
      │ controller  │  │ user_mgmt_    │      │  evaluasi,      │
      │             │  │ view          │      │  cross-val)     │
      │ user_       │  │ prediction_   │      │                 │
      │ controller  │  │ config_view   │      └────────┬────────┘
      └──────┬──────┘  │ prediction_   │               │
             │          │ result_view   │               │
             │          └───────────────┘               │
             │                                          │
      ┌──────▼──────────────────────────────────────────▼──┐
      │                      db.py                          │
      │         SQLite: users, uploaded_files,              │
      │                 prediction_history                  │
      └─────────────────────────────────────────────────────┘
```

### Tanggung Jawab Setiap Layer

| Layer           | Folder         | Tanggung Jawab                                                       |
| --------------- | -------------- | -------------------------------------------------------------------- |
| **Entry Point** | `app.py`       | Inisialisasi, konfigurasi Streamlit, routing halaman                 |
| **Model**       | `models/`      | Logika ML (preprocessing, training, evaluasi) dan utilitas file      |
| **Controller**  | `controllers/` | Logika bisnis: auth, manajemen file, konfigurasi prediksi, CRUD user |
| **View**        | `views/`       | Semua tampilan Streamlit (`st.*`), tidak mengandung logika bisnis    |
| **Data Layer**  | `db.py`        | Koneksi SQLite, CRUD tabel, inisialisasi & seed data awal            |

### Alur Data Antar Layer

```
Pengguna → View → Controller → Model / db.py → Controller → View → Pengguna
```

**Contoh alur Login:**

1. `views/login_view.py` — menampilkan form dan menerima input username/password
2. `controllers/auth_controller.py` — memanggil `db.authenticate_user()`
3. `db.py` — query SQLite, return dict user
4. `auth_controller.py` — set `session_state` jika berhasil
5. `login_view.py` — memanggil `st.rerun()` untuk redirect

**Contoh alur Prediksi Data Primer:**

1. `views/prediction_config_view.py` — tampilkan UI konfigurasi
2. `controllers/prediction_controller.py` — simpan konfigurasi ke `session_state`, redirect
3. `views/prediction_result_view.py` — baca konfigurasi, panggil model
4. `models/ml_model.py` — preprocessing → training C4.5 → evaluasi → return hasil
5. `prediction_result_view.py` — tampilkan semua visualisasi
6. `db.py` — simpan riwayat prediksi ke database

---

## 🔐 Arsitektur Sistem & Peran

Sistem menggunakan **autentikasi berbasis sesi** dengan 2 peran pengguna:

| Peran    | Hak Akses                                                                                |
| -------- | ---------------------------------------------------------------------------------------- |
| **BK**   | Akses penuh: Konfigurasi Prediksi, Dasbor Riwayat Hasil Prediksi, dan Manajemen Pengguna |
| **Guru** | Unggah Berkas dan Manajemen Berkas (hanya berkas miliknya sendiri)                       |

### Skema Basis Data SQLite

#### Tabel `users` (Pengguna)

| Kolom          | Tipe             | Keterangan                         |
| -------------- | ---------------- | ---------------------------------- |
| `id`           | INTEGER (PK, AI) | ID unik pengguna                   |
| `username`     | TEXT (UNIQUE)    | Nama pengguna untuk masuk          |
| `password`     | TEXT             | Kata sandi yang dienkripsi SHA-256 |
| `nama_lengkap` | TEXT             | Nama lengkap pengguna              |
| `role`         | TEXT             | `BK` / `Guru`                      |
| `created_at`   | TIMESTAMP        | Waktu pembuatan akun               |

#### Tabel `uploaded_files` (Berkas Unggahan)

| Kolom               | Tipe             | Keterangan                                |
| ------------------- | ---------------- | ----------------------------------------- |
| `id`                | INTEGER (PK, AI) | ID unik berkas                            |
| `filename`          | TEXT             | Nama berkas unik di peladen (Awalan UUID) |
| `original_filename` | TEXT             | Nama berkas asli yang diunggah            |
| `file_path`         | TEXT             | Jalur absolut berkas di penyimpanan       |
| `uploaded_by`       | INTEGER (FK)     | ID pengguna yang mengunggah               |
| `uploaded_at`       | TIMESTAMP        | Waktu unggah                              |
| `file_size`         | INTEGER          | Ukuran berkas dalam bita                  |
| `description`       | TEXT             | Deskripsi opsional                        |

### Akun Bawaan (Data Awal)

| Nama Pengguna | Kata Sandi | Peran | Nama Lengkap        |
| ------------- | ---------- | ----- | ------------------- |
| `bk`          | `admin123` | BK    | Guru BK             |
| `guru`        | `admin123` | Guru  | Guru Mata Pelajaran |

---

## 📊 Himpunan Data & Fitur

### Deskripsi Himpunan Data (`fix/data_siswa.csv`)

Himpunan data utama yang digunakan terdiri dari **10 kolom** (identitas, demografi, dan metrik prediksi) dengan fitur pendorong utama serta **1 kolom target**:

| No  | Nama Fitur           | Kategori     | Tipe Data             | Deskripsi / Rentang Nilai                                 |
| --- | -------------------- | ------------ | --------------------- | --------------------------------------------------------- |
| 1   | `NISN`               | Identitas    | Teks/Numerik          | Nomor Induk Siswa Nasional (Pengenal unik)                |
| 2   | `Nama`               | Identitas    | Teks                  | Nama lengkap siswa                                        |
| 3   | `Kelas`              | Identitas    | Teks                  | Kelas siswa                                               |
| 4   | `Angkatan`           | Identitas    | Teks/Numerik          | Tahun angkatan siswa                                      |
| 5   | `Gender`             | Demografi    | Kategorik             | Jenis kelamin siswa                                       |
| 6   | `Nilai_Rata_Rata`    | Akademik     | Numerik (float)       | Nilai Rata-rata Kumulatif Siswa                           |
| 7   | `Kehadiran_Persen`   | Kehadiran    | Numerik (float)       | Persentase Kehadiran Siswa                                |
| 8   | `Panggilan_BK`       | Kedisiplinan | Numerik (int)         | Frekuensi / Poin Pelanggaran & Panggilan BK               |
| 9   | `Status_bantuan_PIP` | Ekonomi      | Kategorik (Ya/Tidak)  | Status penerimaan bantuan PIP                             |
| 10  | `Status_DO`          | **Target**   | Kategorik Multi-Kelas | Status Siswa (`Non-Dropout`, `DO-Akademik`, `DO-Masalah`) |

### Dua Mode Analisis

| Mode                                  | Sumber Data                                                                         | Proses                                                                            |
| ------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Eksperimen (Himpunan Data UCI)**    | Himpunan data sekunder UCI (_Prediksi Siswa Putus Sekolah dan Kesuksesan Akademik_) | Menggunakan model **pra-latih** (`model_c45_dropout.pkl`) dengan 8 fitur terpilih |
| **Data Primer (SMK Tunas Teknologi)** | Data lokal dengan fitur sesuai konteks SMA/SMK Indonesia (`fix/data_siswa.csv`)     | Model dilatih **secara langsung** dari data yang diunggah                         |

---

## 🔧 Implementasi Pra-pemrosesan Data

### Alur Aktivitas Pra-pemrosesan (Activity Diagram Flow)

Untuk keperluan pembuatan _Activity Diagram_, berikut adalah alur sekuensial (berurutan) yang persis terjadi di dalam sistem (`app.py` & `preprocessing_dataset.py`):

1. **Menerima Dataset (Input Data)**: Sistem menerima dataset mentah dari unggahan pengguna (berformat CSV/Excel).
2. **Pembersihan Missing Value**:
   - Sistem memisahkan pengecekan per kolom.
   - Jika kolom numerik memiliki data kosong, diisi dengan **Median**.
   - Jika kolom kategorikal memiliki data kosong, diisi dengan **Modus**.
3. **Pembersihan Target NaN**: Sistem mendeteksi kolom target (Status). Jika ada baris dengan target yang kosong (NaN), baris tersebut akan dihapus secara keseluruhan (drop).
4. **Target Encoding (Binarisasi/Multiklasifikasi)**: Sistem memetakan nilai teks pada kolom target menjadi numerik murni secara otomatis (misal: `DO-Masalah` menjadi `0`, `DO-Nilai` menjadi `1`).
5. **Label Encoding (Fitur)**: Sistem memindai semua kolom fitur. Kolom bertipe teks/kategorikal diubah menjadi angka menggunakan fungsi `factorize` (misal: "Laki-laki" -> 0, "Perempuan" -> 1).
6. **Feature Selection (Information Gain)** (Opsional jika diaktifkan):
   - Sistem menghitung nilai _Entropy_ dan _Information Gain_ dari setiap fitur.
   - Fitur dengan skor di bawah ambang batas (threshold) dibuang, menyisakan fitur-fitur yang paling relevan.
7. **Train-Test Split**: Sistem memecah himpunan data menjadi **Data Latih (80%)** dan **Data Uji (20%)** menggunakan proporsi terstruktur (_stratified_).
8. **Normalisasi Data (Z-Score)**:
   - Sistem melakukan _scaling_ menggunakan _Standard Scaler_ agar rentang data seimbang (Mean=0, Std=1).
   - Skala dihitung (fit) hanya pada Data Latih, lalu diterapkan (transform) pada Data Latih dan Data Uji.
9. **Data Siap (Output Data)**: Himpunan data selesai diproses dan siap dimasukkan ke dalam model C4.5.

---

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

| Tipe Data                              | Teknik Pengisian                | Justifikasi                                                                                           |
| -------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Numerik** (`int64`, `float64`)       | **Nilai Tengah (Median)**       | Nilai tengah lebih kebal terhadap pencilan dibanding rata-rata. Tidak terpengaruh oleh nilai ekstrem. |
| **Kategorikal** (`object`, `category`) | **Modus** (nilai paling sering) | Modus mempertahankan distribusi kategori yang dominan tanpa memperkenalkan nilai baru.                |

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

| Teknik                                | Implementasi                                              | Justifikasi                                                                                                                                                                     |
| ------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
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

Kolom target dikonversi ke format numerik murni dengan mendukung format **Biner** maupun **Multi-Kelas** (misal: `DO-Masalah`, `DO-Nilai`, `Non-Dropout`):

| Prioritas | Kondisi Target                                     | Pemetaan                                          |
| --------- | -------------------------------------------------- | ------------------------------------------------- |
| 1         | Ditemukan label `"Dropout"`                        | `Dropout` → 1, Lainnya → 0 (Biner)                |
| 2         | Target berupa biner `0` dan `1`                    | `1` → 1, `0` → 0                                  |
| 3         | Target Multi-Kelas (`DO-Masalah`, `DO-Nilai`, dll) | Pemetaan urut otomatis (`label_map`) → 0, 1, 2... |

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

| Teknik                                          | Pustaka                                | Rumus             | Justifikasi                                                                                                                                                                                                                   |
| ----------------------------------------------- | -------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pembuat Skala Standar** (Normalisasi Z-Score) | `sklearn.preprocessing.StandardScaler` | `z = (x - μ) / σ` | Mengubah distribusi fitur agar memiliki rata-rata = 0 dan simpangan baku = 1. Meskipun Pohon Keputusan secara teori tidak memerlukan penskalaan, standarisasi membantu konsistensi interpretasi dan perbandingan antar fitur. |

```python
# Implementasi (app.py, baris 958-963)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # pas + ubah pada data latih
X_test_scaled = scaler.transform(X_test)          # ubah saja pada data uji
```

> **Penting:** `fit_transform()` hanya diterapkan pada **data latih** untuk menghindari kebocoran data. Data uji diubah menggunakan parameter (rata-rata, simpangan baku) dari data latih.

### 6. Pembagian Data Latih dan Uji

| Parameter           | Nilai Bawaan | Konfigurasi                                                       |
| ------------------- | ------------ | ----------------------------------------------------------------- |
| **Rasio Pengujian** | 20%          | Dapat diatur pengguna melalui penggeser (10% – 50%)               |
| **Stratifikasi**    | `stratify=y` | Mempertahankan proporsi kelas Dropout/Non-Dropout di kedua bagian |
| **Kondisi Acak**    | 42           | Menjamin reproduktibilitas hasil                                  |

```python
# Implementasi (app.py, baris 953-956)
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=test_size, random_state=42, stratify=y
)
```

### 7. Seleksi Fitur — Perolehan Informasi (Information Gain)

| Teknik                                                       | Pustaka                                         | Aktivasi                               |
| ------------------------------------------------------------ | ----------------------------------------------- | -------------------------------------- |
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

| Parameter            | Nilai                         | Keterangan                                                                                         |
| -------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------- |
| `kriteria`           | `'entropi'`                   | Menggunakan **Perolehan Informasi** (Berbasis entropi) sebagai kriteria pemisahan — ciri khas C4.5 |
| `kedalaman_maksimal` | 7 (bawaan, dapat diatur 3–15) | Kedalaman maksimum pohon untuk mencegah _overfitting_                                              |
| `kondisi_acak`       | 42                            | Reproduktibilitas                                                                                  |

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

| Metrik                        | Rumus                                                 | Keterangan                                                                |
| ----------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------- |
| **Akurasi**                   | `(TP + TN) / Total`                                   | Persentase prediksi benar secara keseluruhan                              |
| **Presisi**                   | `TP / (TP + FP)`                                      | Dari yang diprediksi Putus Sekolah, berapa yang benar-benar Putus Sekolah |
| **Daya Ingat (Sensitivitas)** | `TP / (TP + FN)`                                      | Dari yang benar-benar Putus Sekolah, berapa yang berhasil terdeteksi      |
| **Skor-F1**                   | `2 × (Presisi × Daya Ingat) / (Presisi + Daya Ingat)` | Rata-rata harmonik dari Presisi dan Daya Ingat                            |

```python
# Implementasi (app.py, baris 976-980)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
```

### Visualisasi Evaluasi

| Visualisasi                   | Pustaka               | Keterangan                                                              |
| ----------------------------- | --------------------- | ----------------------------------------------------------------------- |
| **Matriks Kebingungan**       | Seaborn               | Matriks menunjukkan distribusi TP, TN, FP, FN per kelas                 |
| **Kurva ROC**                 | Matplotlib            | Kurva pertukaran antara Tingkat Positif Benar dan Tingkat Positif Salah |
| **AUC (Luas Di Bawah Kurva)** | `sklearn.metrics.auc` | Nilai 0–1, semakin mendekati 1 semakin baik                             |
| **Pohon Keputusan**           | `sklearn.tree`        | Visualisasi struktur pohon lengkap                                      |
| **Tingkat Kepentingan Fitur** | Matplotlib            | Grafik batang horizontal menunjukkan kontribusi setiap fitur            |
| **Peta Panas Korelasi**       | Seaborn               | Matriks korelasi antar fitur                                            |
| **Diagram Lingkaran**         | Matplotlib            | Proporsi Siswa Berisiko (Dropout)                                       |
| **Grafik Batang**             | Matplotlib            | Distribusi Siswa Berisiko (Dropout)                                     |
| **Diagram Kotak (Boxplot)**   | Seaborn               | Sebaran nilai seluruh fitur prediksi untuk Siswa Berisiko               |
| **Laporan Klasifikasi**       | scikit-learn          | Laporan lengkap per kelas (presisi, daya ingat, skor-f1, dukungan)      |
| **Tabel Missing Values**      | Pandas/Streamlit      | Rekapitulasi nilai kosong (NaN) beserta persentasenya                   |
| **Tabel Outliers (IQR)**      | Pandas/Streamlit      | Rekapitulasi batas IQR dan jumlah pencilan data                         |

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
| ------------- | ---------- | ----- |
| `bk`          | `admin123` | BK    |
| `guru`        | `admin123` | Guru  |

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
