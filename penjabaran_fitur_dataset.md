# Penjabaran Fitur Dataset Prediksi Siswa Putus Sekolah

Dataset ini terdiri dari **14 fitur** yang dibagi ke dalam 3 kategori utama: Kehadiran, Sosial, dan Ekonomi, serta 1 kolom label/target. Fitur-fitur ini dikontekskan untuk lingkungan SMA/SMK di Indonesia dengan mengacu pada struktur dataset UCI *Predict Students' Dropout and Academic Success* (Dataset ID: 697).

---

## 1. Kategori Kehadiran

### 1.1 Kehadiran_Semester_1
- **Tipe data:** Numerik (integer)
- **Range:** 50 – 100 (dalam persen %)
- **Deskripsi:** Persentase kehadiran siswa selama semester 1. Dihitung dari jumlah hari hadir dibagi total hari efektif dikali 100.
- **Relevansi:** Kehadiran rendah di semester awal merupakan sinyal awal risiko dropout.
- **Mapping ke UCI:** Curricular units 1st sem (evaluations)

### 1.2 Kehadiran_Semester_2
- **Tipe data:** Numerik (integer)
- **Range:** 50 – 100 (dalam persen %)
- **Deskripsi:** Persentase kehadiran siswa selama semester 2. Tren penurunan dari semester 1 ke semester 2 menjadi indikator kuat risiko putus sekolah.
- **Relevansi:** Penurunan kehadiran antar semester mengindikasikan peningkatan risiko.
- **Mapping ke UCI:** Curricular units 2nd sem (evaluations)

### 1.3 Nilai_Rata_Semester_1
- **Tipe data:** Numerik (float)
- **Range:** 40.0 – 95.0
- **Deskripsi:** Nilai rata-rata seluruh mata pelajaran pada semester 1. Skala 0–100 sesuai sistem penilaian SMA/SMK Indonesia. KKM umumnya 70.
- **Relevansi:** Nilai di bawah KKM (< 70) berkorelasi dengan risiko tidak naik kelas dan dropout.
- **Mapping ke UCI:** Curricular units 1st sem (grade)

### 1.4 Nilai_Rata_Semester_2
- **Tipe data:** Numerik (float)
- **Range:** 40.0 – 95.0
- **Deskripsi:** Nilai rata-rata seluruh mata pelajaran pada semester 2. Dikombinasikan dengan semester 1 untuk melihat tren akademik siswa.
- **Relevansi:** Nilai yang konsisten rendah di dua semester berturut-turut memperkuat prediksi dropout.
- **Mapping ke UCI:** Curricular units 2nd sem (grade)

### 1.5 Jumlah_Pelanggaran
- **Tipe data:** Numerik (integer)
- **Range:** 0 – 20
- **Deskripsi:** Total catatan pelanggaran tata tertib sekolah selama satu tahun ajaran, seperti terlambat, tidak membawa atribut, atau pelanggaran lainnya yang tercatat di BK.
- **Relevansi:** Frekuensi pelanggaran tinggi mencerminkan ketidakpatuhan dan rendahnya motivasi belajar.
- **Mapping ke UCI:** Tidak ada padanan langsung — fitur tambahan konteks lokal.

---

## 2. Kategori Sosial

### 2.1 Pendidikan_Ayah
- **Tipe data:** Kategorik (ordinal, integer)
- **Nilai:** 1 = SD, 2 = SMP, 3 = SMA/SMK, 4 = D3/D4, 5 = S1 ke atas
- **Deskripsi:** Tingkat pendidikan terakhir ayah/wali laki-laki siswa.
- **Relevansi:** Orang tua dengan pendidikan rendah cenderung kurang memahami pentingnya pendidikan, sehingga dukungan akademik ke anak lebih terbatas.
- **Mapping ke UCI:** Father's qualification

### 2.2 Pendidikan_Ibu
- **Tipe data:** Kategorik (ordinal, integer)
- **Nilai:** 1 = SD, 2 = SMP, 3 = SMA/SMK, 4 = D3/D4, 5 = S1 ke atas
- **Deskripsi:** Tingkat pendidikan terakhir ibu/wali perempuan siswa.
- **Relevansi:** Pendidikan ibu secara khusus berkorelasi kuat dengan motivasi dan prestasi anak dalam berbagai penelitian Educational Data Mining (EDM).
- **Mapping ke UCI:** Mother's qualification

### 2.3 Status_Keluarga
- **Tipe data:** Kategorik (nominal, integer)
- **Nilai:** 1 = Keluarga lengkap, 2 = Single parent, 3 = Tinggal dengan wali
- **Deskripsi:** Kondisi struktur keluarga tempat siswa tinggal dan diasuh.
- **Relevansi:** Siswa dari keluarga tidak lengkap cenderung memiliki tekanan psikologis lebih tinggi yang berdampak pada motivasi belajar.
- **Mapping ke UCI:** Marital status (dikontekskan ke keluarga siswa, bukan orang tua)

### 2.4 Jumlah_Saudara
- **Tipe data:** Numerik (integer)
- **Range:** 0 – 5
- **Deskripsi:** Jumlah saudara kandung siswa yang tinggal dalam satu rumah tangga.
- **Relevansi:** Jumlah saudara yang banyak berkaitan dengan pembagian perhatian orang tua dan potensi tekanan ekonomi keluarga.
- **Mapping ke UCI:** Tidak ada padanan langsung — fitur tambahan konteks lokal.

### 2.5 Jarak_Rumah_km
- **Tipe data:** Numerik (float)
- **Range:** 0.5 – 30.0 km
- **Deskripsi:** Jarak tempuh dari rumah siswa ke sekolah dalam satuan kilometer.
- **Relevansi:** Jarak jauh berkorelasi dengan tingkat keterlambatan dan ketidakhadiran, terutama bagi siswa yang tidak memiliki kendaraan atau biaya transportasi yang terbatas.
- **Mapping ke UCI:** Tidak ada padanan langsung — fitur tambahan konteks lokal.

---

## 3. Kategori Ekonomi

### 3.1 Status_SPP
- **Tipe data:** Kategorik (biner, integer)
- **Nilai:** 0 = Menunggak, 1 = Lunas
- **Deskripsi:** Status pembayaran SPP (Sumbangan Pembinaan Pendidikan) siswa pada tahun ajaran berjalan.
- **Relevansi:** Tunggakan SPP adalah indikator langsung kesulitan ekonomi keluarga dan sering menjadi penyebab utama dropout.
- **Mapping ke UCI:** Tuition fees up to date

### 3.2 Penerima_Beasiswa
- **Tipe data:** Kategorik (biner, integer)
- **Nilai:** 0 = Tidak, 1 = Ya
- **Deskripsi:** Status apakah siswa terdaftar sebagai penerima beasiswa (PIP, KIP, beasiswa sekolah, atau beasiswa lainnya).
- **Relevansi:** Siswa penerima beasiswa memiliki perlindungan finansial yang mengurangi risiko dropout akibat faktor ekonomi.
- **Mapping ke UCI:** Scholarship holder

### 3.3 Penghasilan_Ortu
- **Tipe data:** Kategorik (ordinal, integer)
- **Nilai:** 1 = < Rp1 juta, 2 = Rp1–3 juta, 3 = Rp3–5 juta, 4 = > Rp5 juta
- **Deskripsi:** Estimasi penghasilan gabungan orang tua per bulan dalam rentang kategori.
- **Relevansi:** Penghasilan rendah secara langsung membatasi kemampuan keluarga membiayai pendidikan dan kebutuhan sekolah siswa.
- **Mapping ke UCI:** Mother's occupation + Father's occupation (digabung menjadi satu fitur penghasilan)

### 3.4 Tanggungan_Keluarga
- **Tipe data:** Numerik (integer)
- **Range:** 1 – 7
- **Deskripsi:** Jumlah anggota keluarga yang menjadi tanggungan finansial orang tua, termasuk siswa itu sendiri.
- **Relevansi:** Semakin banyak tanggungan dengan penghasilan tetap, semakin besar tekanan ekonomi yang berpotensi mendorong siswa keluar dari sekolah untuk bekerja.
- **Mapping ke UCI:** Tidak ada padanan langsung — fitur tambahan konteks lokal.

---

## 4. Label / Target

### 4.1 Status
- **Tipe data:** Kategorik (nominal, string)
- **Nilai:** Dropout / Non-Dropout
- **Deskripsi:** Label hasil klasifikasi yang menjadi target prediksi model C4.5. Nilai ini merepresentasikan apakah siswa teridentifikasi berisiko putus sekolah atau tidak.
- **Mapping ke UCI:** Target (Dropout / Graduate / Enrolled → disederhanakan menjadi biner)

---

## Ringkasan Fitur

| No | Nama Fitur | Kategori | Tipe Data | Mapping UCI |
|----|------------|----------|-----------|-------------|
| 1 | Kehadiran_Semester_1 | Kehadiran | Numerik | Curricular units 1st sem (evaluations) |
| 2 | Kehadiran_Semester_2 | Kehadiran | Numerik | Curricular units 2nd sem (evaluations) |
| 3 | Nilai_Rata_Semester_1 | Kehadiran | Numerik | Curricular units 1st sem (grade) |
| 4 | Nilai_Rata_Semester_2 | Kehadiran | Numerik | Curricular units 2nd sem (grade) |
| 5 | Jumlah_Pelanggaran | Kehadiran | Numerik | Konteks lokal |
| 6 | Pendidikan_Ayah | Sosial | Kategorik | Father's qualification |
| 7 | Pendidikan_Ibu | Sosial | Kategorik | Mother's qualification |
| 8 | Status_Keluarga | Sosial | Kategorik | Marital status |
| 9 | Jumlah_Saudara | Sosial | Numerik | Konteks lokal |
| 10 | Jarak_Rumah_km | Sosial | Numerik | Konteks lokal |
| 11 | Status_SPP | Ekonomi | Biner | Tuition fees up to date |
| 12 | Penerima_Beasiswa | Ekonomi | Biner | Scholarship holder |
| 13 | Penghasilan_Ortu | Ekonomi | Kategorik | Mother's + Father's occupation |
| 14 | Tanggungan_Keluarga | Ekonomi | Numerik | Konteks lokal |
| — | Status | Target | Kategorik | Target (biner) |

> **Catatan:** 7 dari 14 fitur memiliki padanan langsung dengan dataset UCI, sedangkan 4 fitur merupakan tambahan yang disesuaikan dengan konteks pendidikan SMA/SMK di Indonesia. Hal ini menjadi salah satu kontribusi penelitian dalam mengadaptasi model prediksi dropout ke lingkungan pendidikan lokal.
