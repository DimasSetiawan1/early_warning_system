# DESIGN SPECIFICATION

## Sistem Peringatan Dini — Prediksi Risiko Siswa Putus Sekolah

## SMK Tunas Teknologi

---

## 1. Design Brief

**Produk**: Aplikasi web Streamlit untuk prediksi risiko dropout siswa SMK  
**Audiens**: Guru (Wali Kelas) dan Guru BK — pengguna non-teknis di lingkungan sekolah  
**Tujuan utama**: Menyajikan data prediksi yang serius, dapat dipercaya, dan mudah dibaca tanpa distraksi visual

---

## 2. Token System

### 2.1 Warna

| Token                    | Hex       | Peran                                     |
| ------------------------ | --------- | ----------------------------------------- |
| `--color-bg`             | `#F8F9FB` | Background utama halaman                  |
| `--color-surface`        | `#FFFFFF` | Card, panel, tabel                        |
| `--color-border`         | `#E2E6EA` | Garis pemisah, border card                |
| `--color-primary`        | `#1B3A6B` | Aksi utama, heading, sidebar aktif        |
| `--color-primary-light`  | `#EBF0F9` | Hover state, badge latar, highlight baris |
| `--color-accent`         | `#2563EB` | Tombol primer, link, indikator aktif      |
| `--color-danger`         | `#C0392B` | Label Dropout, badge risiko tinggi        |
| `--color-danger-light`   | `#FDECEA` | Background badge Dropout                  |
| `--color-success`        | `#1A7A4A` | Label Non-Dropout, badge aman             |
| `--color-success-light`  | `#E8F5EE` | Background badge Non-Dropout              |
| `--color-text-primary`   | `#111827` | Teks utama                                |
| `--color-text-secondary` | `#6B7280` | Label, caption, metadata                  |
| `--color-text-muted`     | `#9CA3AF` | Placeholder, teks nonaktif                |

**Prinsip warna**: Navy gelap (`#1B3A6B`) sebagai warna institusional yang mencerminkan otoritas dan kepercayaan. Merah (`#C0392B`) digunakan eksklusif untuk status dropout — tidak digunakan sebagai dekorasi. Seluruh palet bersifat netral dan tidak menggunakan warna hangat yang berkesan informal.

### 2.2 Tipografi

| Peran                     | Typeface         | Weight | Ukuran  |
| ------------------------- | ---------------- | ------ | ------- |
| Display / Heading halaman | `Inter`          | 700    | 24–32px |
| Heading kartu / seksi     | `Inter`          | 600    | 16–20px |
| Body teks                 | `Inter`          | 400    | 14px    |
| Label, caption, metadata  | `Inter`          | 400    | 12px    |
| Data numerik / statistik  | `Inter`          | 700    | 28–40px |
| Kode / data teknis        | `JetBrains Mono` | 400    | 13px    |

**Catatan**: Gunakan `font-feature-settings: "tnum"` pada semua angka dalam tabel agar kolom numerik rata sempurna.

### 2.3 Layout

- **Sidebar lebar**: 260px, fixed
- **Content area**: fluid, max-width 1200px, padding 24px
- **Card padding**: 24px
- **Border radius card**: 8px
- **Border radius tombol**: 6px
- **Gutter antar card**: 16px
- **Spacing vertikal antar seksi**: 32px

### 2.4 Elemen Signature

Satu elemen unik yang menjadi identitas visual aplikasi ini: **tabel distribusi dengan baris yang dapat di-highlight berdasarkan status risiko** — baris Dropout diberi latar `#FDECEA` dengan border kiri 3px solid `#C0392B`, sedangkan Non-Dropout tetap putih bersih. Tidak ada ikon, tidak ada warna solid penuh, hanya aksentuasi tipis yang cukup untuk membedakan tanpa berteriak.

---

## 3. Komponen Global

### 3.1 Sidebar Navigasi

```
+---------------------------+
|  LOGO / NAMA SISTEM       |
|  [teks, tidak ada ikon]   |
+---------------------------+
|                           |
|  Dasbor Publik            |
|                           |
|  -- (setelah Masuk) --    |
|                           |
|  Manajemen File      [G]  |
|  Konfigurasi Prediksi [B] |
|  Riwayat Prediksi    [G/B]|
|                           |
+---------------------------+
|  Masuk / Keluar           |
|  [nama pengguna aktif]    |
+---------------------------+
```

- `[G]` = hanya Guru, `[B]` = hanya BK, `[G/B]` = keduanya
- Menu yang tidak bisa diakses disembunyikan, bukan di-grey-out
- Item aktif: latar `--color-primary-light`, teks `--color-primary`, border kiri 3px solid `--color-accent`
- Tidak ada ikon, navigasi murni teks

### 3.2 Card

```
+-----------------------------------------------+
|  Label seksi (12px, --color-text-secondary)    |
|                                                |
|  Konten utama                                  |
|                                                |
+-----------------------------------------------+
```

- Background: `--color-surface`
- Border: 1px solid `--color-border`
- Box shadow: `0 1px 3px rgba(0,0,0,0.06)`
- Tidak ada drop shadow tebal

### 3.3 Badge Status

```
[ Non-Dropout ]   --> bg: #E8F5EE, teks: #1A7A4A, border: 1px solid #A7D7BC
[ Dropout     ]   --> bg: #FDECEA, teks: #C0392B, border: 1px solid #F0B8B3
[ DO-Masalah  ]   --> sama dengan Dropout
[ DO-Akademik ]   --> sama dengan Dropout
```

- Font size: 11px, font weight: 500
- Padding: 2px 8px
- Border radius: 4px
- Semua huruf kapital pertama saja, bukan ALL CAPS

### 3.4 Tombol

| Tipe     | Style                                                               |
| -------- | ------------------------------------------------------------------- |
| Primer   | bg `--color-accent`, teks putih, hover darken 10%                   |
| Sekunder | bg transparan, border `--color-border`, teks `--color-text-primary` |
| Bahaya   | bg `--color-danger`, teks putih — hanya untuk hapus/Keluar          |

- Padding: 8px 16px
- Font weight: 500
- Tidak ada shadow pada tombol

### 3.5 Tabel Data

- Header: bg `#F1F4F8`, teks `--color-text-secondary`, font weight 600, font size 12px, huruf uppercase
- Baris: alternating putih dan `#FAFBFC`
- Baris Dropout: bg `#FDECEA`, border-left 3px solid `#C0392B`
- Hover baris: bg `--color-primary-light`
- Border antar baris: 1px solid `--color-border`
- Tidak ada border vertikal antar kolom

---

## 4. Halaman

---

### Halaman 1 — Dasbor Publik (Tanpa Masuk)

**Akses**: Semua pengunjung, tidak perlu Masuk  
**Tujuan**: Menampilkan gambaran umum distribusi status dropout siswa secara transparan

#### Layout

```
+--SIDEBAR--+------------------CONTENT AREA------------------+
|           |                                                  |
| Dasbor    |  DISTRIBUSI STATUS SISWA                        |
| Publik    |  Data Prediksi Risiko Putus Sekolah             |
|           |  SMK Tunas Teknologi                            |
| [Masuk]   |                                                  |
|           |  +------------+  +------------+  +------------+ |
|           |  | Total      |  | Non-Dropout|  | Dropout    | |
|           |  | Siswa      |  |            |  |            | |
|           |  | 573        |  | 516 (90%)  |  | 57 (10%)   | |
|           |  +------------+  +------------+  +------------+ |
|           |                                                  |
|           |  +---------------------+  +------------------+  |
|           |  | Distribusi Gender   |  | Distribusi Kelas |  |
|           |  | Bar chart           |  | Bar chart        |  |
|           |  +---------------------+  +------------------+  |
|           |                                                  |
|           |  TABEL DETAIL SISWA                             |
|           |  Filter: [ Semua Status ] [ Semua Gender ]      |
|           |                                                  |
|           |  Nama | Gender | Kelas | Kehadiran | Nilai | Status |
|           |  ------------------------------------------------|
|           |  ...  | ...    | ...   | ...       | ...   | badge  |
|           |                                                  |
+--SIDEBAR--+--------------------------------------------------+
```

#### Spesifikasi Komponen

**Stat Card (3 buah, grid 3 kolom)**

- Total Siswa: angka `573`, label "Total Siswa Terdaftar"
- Non-Dropout: angka `516`, persentase `90.1%` di bawahnya (12px, success color)
- Dropout: angka `57`, persentase `9.9%` di bawahnya (12px, danger color)
- Angka menggunakan font size 40px, weight 700
- Label di atas angka: 11px, `--color-text-secondary`, uppercase

**Bar Chart Distribusi Gender**

- Sumbu X: label gender (Laki-laki / Perempuan)
- Sumbu Y: jumlah siswa
- Warna batang: `--color-primary` untuk Non-Dropout, `--color-danger` untuk Dropout
- Grouped bar, tidak stacked
- Tidak ada gridline vertikal, gridline horizontal tipis `#E2E6EA`
- Judul chart: "Distribusi Status berdasarkan Gender"

**Bar Chart Distribusi Kelas**

- Sumbu X: nama kelas
- Sumbu Y: jumlah siswa
- Warna sama dengan chart gender
- Judul chart: "Distribusi Status berdasarkan Kelas"

**Filter Tabel**

- Dropdown "Semua Status" → opsi: Semua, Non-Dropout, DO-Masalah, DO-Akademik
- Dropdown "Semua Gender" → opsi: Semua, Laki-laki, Perempuan
- Input teks "Cari nama siswa..." — live filter
- Ketiga filter sejajar horizontal, di atas tabel

**Tabel Detail Siswa**

- Kolom: No, Nama, Gender, Kelas, Angkatan, Kehadiran (%), Nilai Rata-rata, Panggilan BK, Status Beasiswa PIP, Status
- Kolom Status menggunakan badge komponen
- Baris Dropout memiliki highlight sesuai aturan komponen tabel
- Pagination: 20 baris per halaman
- Baris dapat diurutkan dengan klik header kolom (ascending/descending)

**Catatan UX**: Dasbor ini bersifat read-only. Tidak ada tombol aksi apapun. Di bagian bawah tabel terdapat teks kecil: "Data ini bersumber dari hasil prediksi model C4.5 yang dilatih menggunakan data historis SMK Tunas Teknologi."

---

### Halaman 2 — Masuk

**Akses**: Semua pengunjung  
**Tujuan**: Autentikasi untuk Guru dan BK

#### Layout

```
+--SIDEBAR--+------------------CONTENT AREA------------------+
|           |                                                  |
|           |          +------------------------------+        |
|           |          |                              |        |
|           |          |  Masuk ke Sistem             |        |
|           |          |  Sistem Peringatan Dini      |        |
|           |          |  SMK Tunas Teknologi         |        |
|           |          |                              |        |
|           |          |  Username                    |        |
|           |          |  [________________________] |        |
|           |          |                              |        |
|           |          |  Kata Sandi                  |        |
|           |          |  [________________________] |        |
|           |          |                              |        |
|           |          |  [      Masuk      ]         |        |
|           |          |                              |        |
|           |          +------------------------------+        |
|           |                                                  |
+--SIDEBAR--+--------------------------------------------------+
```

#### Spesifikasi Komponen

**Form card**

- Lebar: 400px, center horizontal
- Padding: 40px
- Border: 1px solid `--color-border`
- Box shadow: `0 4px 12px rgba(0,0,0,0.08)`

**Heading**

- "Masuk ke Sistem" — 20px, weight 700, `--color-text-primary`
- "Sistem Peringatan Dini · SMK Tunas Teknologi" — 13px, `--color-text-secondary`
- Margin bawah heading: 32px

**Input field**

- Label di atas input, 12px, weight 500
- Input height: 40px, border: 1px solid `--color-border`
- Focus border: `--color-accent`
- Placeholder teks: warna `--color-text-muted`

**Tombol Masuk**

- Full width, tinggi 40px, style primer
- Teks: "Masuk"

**Pesan error**

- Teks 13px, warna `--color-danger`
- Muncul di antara field terakhir dan tombol
- Contoh: "Username atau kata sandi tidak sesuai."

**Tidak ada**: link registrasi, pilihan "ingat saya", atau teks promosi apapun.

---

### Halaman 3 — Manajemen File (Guru)

**Akses**: Hanya Guru (setelah Masuk)  
**Tujuan**: Upload dan kelola file dataset siswa

#### Layout

```
+--SIDEBAR--+------------------CONTENT AREA------------------+
|           |                                                  |
| Manajemen |  MANAJEMEN FILE                                 |
| File  [*] |  Kelola dataset siswa untuk diproses prediksi   |
|           |                                                  |
| Riwayat   |  +---------------------------------------------+ |
|           |  |  Upload File Baru                           | |
| [Keluar]  |  |                                             | |
|           |  |  [  Pilih File CSV atau Excel  ]            | |
|           |  |  Deskripsi (opsional):                      | |
|           |  |  [________________________________]         | |
|           |  |  [   Upload   ]                             | |
|           |  +---------------------------------------------+ |
|           |                                                  |
|           |  FILE SAYA                                      |
|           |                                                  |
|           |  Nama File | Ukuran | Tanggal Upload | Aksi     |
|           |  --------------------------------------------- |
|           |  file.csv  | 24 KB  | 12 Jul 2025    | [Hapus]  |
|           |  ...                                            |
|           |                                                  |
+--SIDEBAR--+--------------------------------------------------+
```

#### Spesifikasi Komponen

**Upload area**

- Bukan drag-and-drop dengan animasi — cukup tombol "Pilih File" standar Streamlit
- Teks di bawah tombol: "Format yang didukung: CSV, XLSX. Maksimum 10MB."
- Font size 12px, `--color-text-secondary`

**Tabel File**

- Kolom: No, Nama File Asli, Ukuran, Deskripsi, Tanggal Upload, Aksi
- Kolom Aksi: tombol "Hapus" style bahaya, ukuran kecil (font 12px)
- Konfirmasi hapus: muncul teks konfirmasi di bawah baris sebelum benar-benar dihapus
- Jika belum ada file: tampilkan teks "Belum ada file yang diunggah." di tengah area tabel

**Batasan akses**

- Guru hanya melihat file miliknya sendiri
- Tidak ada tombol atau navigasi ke halaman Konfigurasi Prediksi

---

### Halaman 4 — Konfigurasi Prediksi (BK)

**Akses**: Hanya BK (setelah Masuk)  
**Tujuan**: Memilih file dan mengatur parameter model sebelum menjalankan prediksi

#### Layout

```
+--SIDEBAR--+------------------CONTENT AREA------------------+
|           |                                                  |
| Dasbor    |  KONFIGURASI PREDIKSI                          |
| Manajemen |  Atur parameter model dan pilih dataset         |
| Konfigur* |                                                  |
| Riwayat   |  LANGKAH 1 — Pilih Dataset                     |
| Kelola    |  +---------------------------------------------+ |
| Pengguna  |  |  File yang tersedia:                        | |
|           |  |  ( ) file_guru1.csv — 12 Jul 2025          | |
| [Keluar]  |  |  ( ) file_guru2.xlsx — 10 Jul 2025         | |
|           |  +---------------------------------------------+ |
|           |                                                  |
|           |  LANGKAH 2 — Parameter Model                   |
|           |  +---------------------------------------------+ |
|           |  |  Rasio Data Uji:  [ 20% ]                   | |
|           |  |  Kedalaman Pohon: [ 7   ]  (3 – 15)         | |
|           |  |  Feature Selection: [ ] Aktifkan            | |
|           |  |    Ambang Information Gain: [ 0.05 ]        | |
|           |  +---------------------------------------------+ |
|           |                                                  |
|           |  [ Jalankan Prediksi ]                          |
|           |                                                  |
+--SIDEBAR--+--------------------------------------------------+
```

#### Spesifikasi Komponen

**Struktur langkah**

- Dua seksi berlabel "LANGKAH 1" dan "LANGKAH 2" — label 11px, uppercase, `--color-text-secondary`
- Tidak menggunakan stepper visual beranimasi, cukup label teks yang jelas

**Pemilihan file**

- Radio button list menampilkan semua file dari semua guru
- Setiap item: nama file asli + nama guru pengunggah + tanggal upload
- Format: `nama_file.csv — diunggah oleh Guru A — 12 Jul 2025`

**Parameter model**

- Rasio Data Uji: slider 10%–50%, default 20%
- Kedalaman Pohon: number input, min 3, max 15, default 7
- Feature Selection: checkbox; jika dicentang, muncul slider Ambang Information Gain (0.01–0.20, default 0.05)

**Tombol Jalankan**

- Style primer, lebar penuh
- Disabled jika tidak ada file yang dipilih
- Setelah diklik: tampilkan spinner dengan teks "Model sedang dilatih..."

---

### Halaman 5 — Riwayat Prediksi (BK dan Guru)

**Akses**: BK dan Guru (setelah Masuk)  
**Tujuan**: Melihat riwayat prediksi yang pernah dijalankan beserta hasilnya

#### Layout

```
+--SIDEBAR--+------------------CONTENT AREA------------------+
|           |                                                  |
| Riwayat[*]|  RIWAYAT PREDIKSI                              |
|           |                                                  |
|           |  +------------+  +------------+  +------------+ |
|           |  | Total      |  | Terakhir   |  | Akurasi    | |
|           |  | Prediksi   |  | Dijalankan |  | Terakhir   | |
|           |  | 8          |  | 29 Jul '25 |  | 87.3%      | |
|           |  +------------+  +------------+  +------------+ |
|           |                                                  |
|           |  DAFTAR RIWAYAT                                 |
|           |                                                  |
|           |  Tgl | File | Akurasi | F1 | Split | Oleh | Aksi|
|           |  ------------------------------------------------|
|           |  ...  ...    ...       ...  ...     ...   [Lihat]|
|           |                                                  |
|           |  -- detail prediksi (expand saat klik Lihat) -- |
|           |                                                  |
|           |  Confusion Matrix | ROC Curve | Feature Import. |
|           |  Daftar Siswa Berisiko Dropout                  |
|           |  [Ekspor CSV]                                   |
|           |                                                  |
+--SIDEBAR--+--------------------------------------------------+
```

#### Spesifikasi Komponen

**Stat card ringkasan (3 kolom)**

- Total Prediksi, Terakhir Dijalankan, Akurasi Prediksi Terakhir
- Sama seperti stat card di Dasbor Publik

**Tabel riwayat**

- Kolom: No, Tanggal, Nama File, Akurasi, F1-Score, Split Data, Kedalaman, Dijalankan Oleh, Aksi
- Kolom Aksi: tombol "Lihat Detail" style sekunder, ukuran kecil
- BK melihat seluruh riwayat dari semua pengguna
- Guru hanya melihat riwayat prediksi dari file miliknya

**Panel detail (muncul setelah klik "Lihat Detail")**

- Muncul di bawah baris yang diklik, bukan di halaman baru
- Berisi:
  - Metrik: Akurasi, Presisi, Recall, F1-Score dalam 4 stat card kecil
  - Tab navigasi: "Confusion Matrix" / "ROC Curve" / "Feature Importance" / "Siswa Berisiko"
  - Konten tab aktif ditampilkan di bawah tab
- Tab "Siswa Berisiko": tabel nama siswa + nilai fitur + badge status prediksi + tombol "Ekspor CSV"

**Tombol Ekspor CSV**

- Style sekunder
- Teks: "Unduh Hasil Prediksi (.csv)"
- Ditempatkan di kanan atas tabel siswa berisiko

---

## 5. Aturan Implementasi Streamlit

### Warna via CSS Injection

```python
st.markdown("""
<style>
:root {
  --color-bg: #F8F9FB;
  --color-surface: #FFFFFF;
  --color-border: #E2E6EA;
  --color-primary: #1B3A6B;
  --color-primary-light: #EBF0F9;
  --color-accent: #2563EB;
  --color-danger: #C0392B;
  --color-danger-light: #FDECEA;
  --color-success: #1A7A4A;
  --color-success-light: #E8F5EE;
  --color-text-primary: #111827;
  --color-text-secondary: #6B7280;
  --color-text-muted: #9CA3AF;
}
.stApp { background-color: var(--color-bg); }
</style>
""", unsafe_allow_html=True)
```

### Sidebar

```python
with st.sidebar:
    st.markdown("### Sistem Peringatan Dini")
    st.markdown("SMK Tunas Teknologi")
    st.divider()
    # navigasi berdasarkan role dan status Masuk
```

### Stat Card via HTML

```python
st.markdown(f"""
<div style="
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
">
  <div style="font-size:11px; font-weight:600; color:var(--color-text-secondary);
              text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">
    Total Siswa
  </div>
  <div style="font-size:40px; font-weight:700; color:var(--color-text-primary);
              font-feature-settings:'tnum';">
    573
  </div>
</div>
""", unsafe_allow_html=True)
```

### Badge Status

```python
def render_badge(status: str) -> str:
    if status == "Non-Dropout":
        return (
            '<span style="background:#E8F5EE; color:#1A7A4A; border:1px solid #A7D7BC; '
            'font-size:11px; font-weight:500; padding:2px 8px; border-radius:4px;">'
            'Non-Dropout</span>'
        )
    else:
        return (
            '<span style="background:#FDECEA; color:#C0392B; border:1px solid #F0B8B3; '
            'font-size:11px; font-weight:500; padding:2px 8px; border-radius:4px;">'
            f'{status}</span>'
        )
```

### Baris Tabel Dropout (Highlight)

```python
def highlight_dropout(row):
    if row["Status_DO"] != "Non-Dropout":
        return ["background-color: #FDECEA; border-left: 3px solid #C0392B"] * len(row)
    return [""] * len(row)

st.dataframe(df.style.apply(highlight_dropout, axis=1))
```

---

## 6. Aturan Desain Global

- Tidak ada emoji di seluruh antarmuka
- Tidak ada ikon (Font Awesome, Material Icons, dsb)
- Semua heading menggunakan sentence case, bukan ALL CAPS kecuali label seksi 11px
- Tidak ada animasi atau transisi — antarmuka statis dan responsif
- Pesan kosong selalu deskriptif: bukan "Tidak ada data" tapi "Belum ada file yang diunggah. Unggah file CSV atau Excel melalui halaman ini."
- Semua teks error bersifat spesifik: sebutkan apa yang salah dan apa yang harus dilakukan
- Konsistensi label: gunakan istilah yang sama di seluruh halaman — "Dropout" bukan berganti-ganti dengan "Putus Sekolah" di UI
