# 📊 Keterangan Dataset Primer (Data Dummy Siswa)

Dokumen ini menjabarkan spesifikasi **Dataset Primer** yang digunakan dalam penelitian/skripsi ini. Dataset ini disimpan dalam file `data_dummy_siswa.csv` dan digunakan pada mode **Train On-the-fly** (Pelatihan Model Langsung) di dalam aplikasi *Early Warning System*.

- **Nama File:** `data_dummy_siswa.csv`
- **Tipe Dataset:** Primer (Konteks SMA/SMK Indonesia)
- **Jumlah Baris:** 500+ record siswa
- **Jumlah Kolom:** 14 Fitur + 1 Target (Status)

Dataset primer ini dirancang untuk menangkap faktor-faktor lokal yang memengaruhi risiko siswa putus sekolah (dropout) di Indonesia, yang terbagi dalam 3 kategori utama: **Kehadiran & Akademik**, **Sosial**, dan **Ekonomi**.

---

## 1. Kategori Kehadiran & Akademik
Kategori ini berfokus pada tingkat kedisiplinan dan performa belajar siswa di sekolah.

| Nama Fitur | Tipe Data | Range/Nilai | Deskripsi & Relevansi |
|------------|-----------|-------------|-----------------------|
| **`Kehadiran_Semester_1`** | Numerik (Int) | 50 – 100 (%) | Persentase kehadiran semester 1. Kehadiran rendah di awal adalah indikator awal risiko. |
| **`Kehadiran_Semester_2`** | Numerik (Int) | 50 – 100 (%) | Persentase kehadiran semester 2. Penurunan drastis dari semester 1 ke 2 adalah sinyal kuat risiko putus sekolah. |
| **`Nilai_Rata_Semester_1`** | Numerik (Float)| 40.0 – 95.0 | Rata-rata nilai rapor semester 1. Nilai di bawah KKM menunjukkan kesulitan akademik. |
| **`Nilai_Rata_Semester_2`** | Numerik (Float)| 40.0 – 95.0 | Rata-rata nilai rapor semester 2. Membantu melihat tren akademik siswa antar semester. |
| **`Jumlah_Pelanggaran`** | Numerik (Int) | 0 – 20 | Total catatan pelanggaran tata tertib di BK. Frekuensi tinggi mencerminkan kurangnya motivasi dan kepatuhan. |

---

## 2. Kategori Sosial (Latar Belakang)
Kategori ini berfokus pada lingkungan rumah dan keluarga yang sangat memengaruhi psikologis siswa.

| Nama Fitur | Tipe Data | Range/Nilai | Deskripsi & Relevansi |
|------------|-----------|-------------|-----------------------|
| **`Pendidikan_Ayah`** | Kategorik | 1 (SD) – 5 (S1+) | Tingkat pendidikan ayah. Berpengaruh pada tingkat dukungan akademik yang bisa diberikan di rumah. |
| **`Pendidikan_Ibu`** | Kategorik | 1 (SD) – 5 (S1+) | Tingkat pendidikan ibu. Seringkali berkorelasi lebih kuat dengan motivasi belajar anak sehari-hari. |
| **`Status_Keluarga`** | Kategorik | 1=Lengkap, 2=Single Parent, 3=Wali | Keluarga tidak lengkap berpotensi memberi tekanan psikologis lebih tinggi pada siswa. |
| **`Jumlah_Saudara`** | Numerik (Int) | 0 – 5 | Jumlah saudara kandung serumah. Memengaruhi pembagian perhatian orang tua dan biaya. |
| **`Jarak_Rumah_km`** | Numerik (Float)| 0.5 – 30.0 km | Jarak ke sekolah. Semakin jauh, semakin tinggi risiko sering terlambat atau bolos karena kendala transportasi. |

---

## 3. Kategori Ekonomi
Kategori ini berfokus pada kemampuan finansial keluarga untuk menopang kebutuhan pendidikan siswa.

| Nama Fitur | Tipe Data | Range/Nilai | Deskripsi & Relevansi |
|------------|-----------|-------------|-----------------------|
| **`Status_SPP`** | Biner (Int) | 0=Menunggak, 1=Lunas | Indikator paling langsung dari kesulitan ekonomi. Tunggakan SPP adalah penyebab umum putus sekolah. |
| **`Penerima_Beasiswa`**| Biner (Int) | 0=Tidak, 1=Ya | Siswa yang mendapat PIP/KIP memiliki "jaring pengaman" finansial yang menekan risiko dropout. |
| **`Penghasilan_Ortu`** | Kategorik | 1 (< Rp1 Juta) – 4 (> Rp5 Juta) | Estimasi rentang gaji gabungan orang tua per bulan. |
| **`Tanggungan_Keluarga`**| Numerik (Int) | 1 – 7 | Jumlah orang yang harus dihidupi dari penghasilan orang tua. Semakin banyak, beban ekonomi semakin berat. |

---

## 4. Target / Label
Kolom terakhir yang menjadi tujuan prediksi algoritma **C4.5 (Decision Tree)**.

| Nama Fitur | Tipe Data | Range/Nilai | Deskripsi |
|------------|-----------|-------------|-----------|
| **`Status`** | Kategorik (Teks) | Dropout / Non-Dropout | Label akhir hasil evaluasi siswa. `Dropout` berarti siswa putus sekolah, `Non-Dropout` berarti siswa bertahan/lulus. Kolom inilah yang diprediksi oleh sistem. |
