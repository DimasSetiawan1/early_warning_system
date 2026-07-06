import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import io
import os
import uuid
from datetime import datetime

# Import database module
import db

# ── INISIALISASI ──────────────────────────────────────────────────────────────
# Inisialisasi database saat aplikasi pertama kali dijalankan
db.init_db()

# Set Page Config
st.set_page_config(
    page_title="Dashboard Prediksi Mahasiswa/Siswa Dropout",
    page_icon="🎓",
    layout="wide"
)

# Styling Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'nama_lengkap' not in st.session_state:
    st.session_state.nama_lengkap = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard Prediksi'


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def load_file_from_path(file_path):
    """Load dataframe dari path file yang tersimpan."""
    try:
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
            sep = ';' if ';' in first_line else ','
            df = pd.read_csv(file_path, sep=sep)
        else:
            df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None


@st.cache_resource
def load_uci_artifacts():
    """Load model artifacts (UCI)."""
    try:
        model = pickle.load(open('model_c45_dropout.pkl', 'rb'))
        scaler = pickle.load(open('scaler_dropout.pkl', 'rb'))
        features = pickle.load(open('selected_features.pkl', 'rb'))
        return model, scaler, features
    except Exception as e:
        return None, None, None


def format_file_size(size_bytes):
    """Format ukuran file ke format yang readable."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_datetime(dt_str):
    """Format datetime string ke format Indonesia."""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d %B %Y, %H:%M WIB")
    except Exception:
        return dt_str


# ── HALAMAN LOGIN ─────────────────────────────────────────────────────────────

def show_login_page():
    """Tampilkan halaman login."""
    # Center the login form
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("---")
        st.markdown(
            "<h1 style='text-align: center;'>🎓 Early Warning System</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h4 style='text-align: center; color: gray;'>Dashboard Prediksi Siswa Dropout</h4>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: gray;'>Silakan login untuk mengakses dashboard</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="Masukkan username")
            password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password")
            submitted = st.form_submit_button("🔑 Login", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("⚠️ Username dan password harus diisi!")
                else:
                    user = db.authenticate_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user['id']
                        st.session_state.username = user['username']
                        st.session_state.role = user['role']
                        st.session_state.nama_lengkap = user['nama_lengkap']
                        st.session_state.current_page = 'Dashboard Prediksi'
                        st.rerun()
                    else:
                        st.error("❌ Username atau password salah!")

        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; font-size: 0.8em; color: gray;'>"
            "Akun Default:<br>"
            "Admin: <code>admin</code> / <code>admin123</code><br>"
            "BK: <code>bk</code> / <code>admin123</code><br>"
            "Wali Kelas: <code>walikelas</code> / <code>admin123</code>"
            "</p>",
            unsafe_allow_html=True
        )


# ── SIDEBAR NAVIGATION ───────────────────────────────────────────────────────

def show_sidebar():
    """Tampilkan sidebar dengan navigasi berdasarkan role."""
    with st.sidebar:
        st.markdown(f"### 👋 Selamat Datang!")
        st.markdown(f"**{st.session_state.nama_lengkap}**")
        st.markdown(f"🏷️ Role: `{st.session_state.role}`")
        st.markdown("---")

        st.markdown("### 📋 Menu Navigasi")

        role = st.session_state.role

        # Menu items based on role
        if role == 'Admin':
            menu_items = [
                '📊 Dashboard Prediksi',
                '📤 Upload File',
                '📁 Manajemen File',
                '👥 Manajemen User'
            ]
        elif role == 'Wali Kelas':
            menu_items = [
                '📊 Dashboard Prediksi',
                '📤 Upload File',
                '📁 Manajemen File'
            ]
        elif role == 'BK':
            menu_items = [
                '📊 Dashboard Prediksi'
            ]
        else:
            menu_items = []

        for item in menu_items:
            # Strip emoji prefix for page key
            page_key = item.split(' ', 1)[1] if ' ' in item else item
            if st.button(item, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            for key in ['logged_in', 'user_id', 'username', 'role', 'nama_lengkap', 'current_page']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ── HALAMAN UPLOAD FILE ──────────────────────────────────────────────────────

def show_upload_page():
    """Halaman upload file dataset."""
    st.header("📤 Upload File Dataset")
    st.markdown("Upload file dataset siswa dalam format **CSV** atau **Excel (.xlsx)** untuk digunakan dalam analisis prediksi.")

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Pilih File Dataset",
            type=["csv", "xlsx"],
            help="File berisi data siswa/mahasiswa untuk diprediksi."
        )
        description = st.text_area(
            "Deskripsi File (Opsional)",
            placeholder="Contoh: Data siswa kelas XII IPA semester genap 2024/2025",
            max_chars=500
        )
        submitted = st.form_submit_button("📤 Upload File", use_container_width=True)

        if submitted and uploaded_file is not None:
            # Generate unique filename to avoid conflicts
            file_ext = os.path.splitext(uploaded_file.name)[1]
            unique_filename = f"{uuid.uuid4().hex[:12]}_{uploaded_file.name}"
            file_path = os.path.join(db.UPLOAD_DIR, unique_filename)

            # Simpan file ke disk
            os.makedirs(db.UPLOAD_DIR, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Simpan record ke database
            file_id = db.save_uploaded_file(
                filename=unique_filename,
                original_filename=uploaded_file.name,
                file_path=file_path,
                uploaded_by=st.session_state.user_id,
                file_size=uploaded_file.size,
                description=description if description else None
            )

            st.success(f"✅ File **{uploaded_file.name}** berhasil diupload! (ID: {file_id})")
            st.rerun()

        elif submitted and uploaded_file is None:
            st.warning("⚠️ Pilih file terlebih dahulu!")

    # Tampilkan daftar file yang sudah diupload oleh user ini
    st.markdown("---")
    st.subheader("📁 File Yang Sudah Anda Upload")

    if st.session_state.role == 'Admin':
        my_files = db.get_uploaded_files()  # Admin lihat semua
    else:
        my_files = db.get_uploaded_files(uploaded_by=st.session_state.user_id)

    if not my_files:
        st.info("Belum ada file yang diupload.")
    else:
        for f in my_files:
            with st.expander(f"📄 {f['original_filename']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Nama File:** {f['original_filename']}")
                    st.markdown(f"**Ukuran:** {format_file_size(f['file_size'])}")
                    st.markdown(f"**Deskripsi:** {f['description'] or '-'}")
                with col2:
                    st.markdown(f"**Diupload oleh:** {f['uploader_name']} (`{f['uploader_username']}`)")
                    st.markdown(f"**Tanggal Upload:** {format_datetime(f['uploaded_at'])}")
                    st.markdown(f"**ID File:** {f['id']}")


# ── HALAMAN MANAJEMEN FILE ───────────────────────────────────────────────────

def show_file_management_page():
    """Halaman manajemen file — tampilkan semua file, hapus sesuai permission."""
    st.header("📁 Manajemen File Dataset")

    role = st.session_state.role
    user_id = st.session_state.user_id

    # Admin: lihat semua | Wali Kelas: hanya miliknya
    if role == 'Admin':
        files = db.get_uploaded_files()
        st.markdown("Menampilkan **semua** file yang terupload (mode Admin).")
    else:
        files = db.get_uploaded_files(uploaded_by=user_id)
        st.markdown("Menampilkan file yang **Anda upload**.")

    if not files:
        st.info("Belum ada file yang terupload.")
        return

    # Tampilkan tabel ringkasan
    table_data = []
    for f in files:
        table_data.append({
            'ID': f['id'],
            'Nama File': f['original_filename'],
            'Ukuran': format_file_size(f['file_size']),
            'Diupload Oleh': f['uploader_name'],
            'Role': f['uploader_role'],
            'Tanggal Upload': format_datetime(f['uploaded_at']),
            'Deskripsi': f['description'] or '-'
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🗑️ Hapus File")

    # Build list of deletable files
    deletable_files = []
    for f in files:
        if role == 'Admin' or f['uploaded_by'] == user_id:
            deletable_files.append(f)

    if not deletable_files:
        st.info("Tidak ada file yang dapat Anda hapus.")
        return

    # Select file to delete
    file_options = {f"[ID:{f['id']}] {f['original_filename']} (oleh {f['uploader_name']})": f['id'] for f in deletable_files}
    selected_file_label = st.selectbox("Pilih file yang akan dihapus:", list(file_options.keys()))

    if selected_file_label:
        selected_file_id = file_options[selected_file_label]
        col_del1, col_del2 = st.columns([1, 3])
        with col_del1:
            if st.button("🗑️ Hapus File", type="primary"):
                success = db.delete_uploaded_file(selected_file_id)
                if success:
                    st.success("✅ File berhasil dihapus!")
                    st.rerun()
                else:
                    st.error("❌ Gagal menghapus file.")


# ── HALAMAN MANAJEMEN USER (ADMIN ONLY) ──────────────────────────────────────

def show_user_management_page():
    """Halaman manajemen user — hanya untuk Admin."""
    st.header("👥 Manajemen User")

    if st.session_state.role != 'Admin':
        st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
        return

    # Tampilkan daftar user
    users = db.get_all_users()

    st.subheader("📋 Daftar User Terdaftar")
    if users:
        table_data = []
        for u in users:
            table_data.append({
                'ID': u['id'],
                'Username': u['username'],
                'Nama Lengkap': u['nama_lengkap'],
                'Role': u['role'],
                'Tanggal Dibuat': format_datetime(u['created_at']) if u['created_at'] else '-'
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Tambah User Baru ──
    st.markdown("---")
    st.subheader("➕ Tambah User Baru")

    with st.form("add_user_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Username", placeholder="Masukkan username baru")
            new_password = st.text_input("Password", type="password", placeholder="Masukkan password")
        with col2:
            new_nama = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap")
            new_role = st.selectbox("Role", options=['Admin', 'BK', 'Wali Kelas'])

        if st.form_submit_button("➕ Tambah User", use_container_width=True):
            if not new_username or not new_password or not new_nama:
                st.error("⚠️ Semua field harus diisi!")
            else:
                success = db.create_user(new_username, new_password, new_nama, new_role)
                if success:
                    st.success(f"✅ User **{new_username}** ({new_role}) berhasil ditambahkan!")
                    st.rerun()
                else:
                    st.error(f"❌ Username **{new_username}** sudah digunakan!")

    # ── Edit User ──
    st.markdown("---")
    st.subheader("✏️ Edit User")

    editable_users = [u for u in users if u['id'] != st.session_state.user_id]
    if not editable_users:
        st.info("Tidak ada user lain yang bisa diedit.")
    else:
        user_options = {f"[{u['id']}] {u['username']} - {u['nama_lengkap']} ({u['role']})": u['id'] for u in editable_users}
        selected_user_label = st.selectbox("Pilih User:", list(user_options.keys()), key="edit_user_select")

        if selected_user_label:
            selected_user_id = user_options[selected_user_label]
            selected_user = db.get_user_by_id(selected_user_id)

            if selected_user:
                with st.form("edit_user_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        edit_nama = st.text_input("Nama Lengkap", value=selected_user['nama_lengkap'])
                        edit_role = st.selectbox(
                            "Role",
                            options=['Admin', 'BK', 'Wali Kelas'],
                            index=['Admin', 'BK', 'Wali Kelas'].index(selected_user['role'])
                        )
                    with col2:
                        edit_password = st.text_input(
                            "Password Baru (kosongkan jika tidak diubah)",
                            type="password",
                            placeholder="Biarkan kosong jika tidak ingin mengubah"
                        )

                    if st.form_submit_button("💾 Simpan Perubahan", use_container_width=True):
                        db.update_user(
                            selected_user_id,
                            nama_lengkap=edit_nama,
                            role=edit_role,
                            password=edit_password if edit_password else None
                        )
                        st.success("✅ Data user berhasil diperbarui!")
                        st.rerun()

    # ── Hapus User ──
    st.markdown("---")
    st.subheader("🗑️ Hapus User")

    deletable_users = [u for u in users if u['id'] != st.session_state.user_id]
    if not deletable_users:
        st.info("Tidak ada user yang bisa dihapus.")
    else:
        del_user_options = {f"[{u['id']}] {u['username']} - {u['nama_lengkap']} ({u['role']})": u['id'] for u in deletable_users}
        selected_del_label = st.selectbox("Pilih User yang akan dihapus:", list(del_user_options.keys()), key="del_user_select")

        if selected_del_label:
            selected_del_id = del_user_options[selected_del_label]
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1:
                if st.button("🗑️ Hapus User", type="primary"):
                    db.delete_user(selected_del_id)
                    st.success("✅ User berhasil dihapus!")
                    st.rerun()


# ── HALAMAN DASHBOARD PREDIKSI ───────────────────────────────────────────────

def show_prediction_dashboard():
    """Halaman utama dashboard prediksi — dengan pemilihan file dari database."""
    st.title("🎓 Dashboard Prediksi Siswa Dropout Sekolah")
    st.markdown("""
    Aplikasi ini diimplementasikan mengikuti metodologi **CRISP-DM** menggunakan algoritma **C4.5 (Decision Tree)**.
    Aplikasi mendukung dua jenis data: **Data Eksperimen (UCI Dataset)** dan **Data Primer (SMK Tunas Teknologi)** .
    """)

    role = st.session_state.role
    user_id = st.session_state.user_id

    # ── Mode Selection ──
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Pengaturan Dashboard")

    # Wali Kelas default ke Data Primer, role lain default ke Eksperimen
    default_mode_index = 1 if role == 'Wali Kelas' else 0

    mode = st.sidebar.selectbox(
        "Pilih Mode Analisis",
        options=[
            "Eksperimen (UCI Dataset - Pre-trained Model)",
            "Data Primer (SMK Tunas Teknologi - Train On-the-fly)"
        ],
        index=default_mode_index
    )

    # ── File Selection dari database ──
    st.sidebar.markdown("---")
    st.sidebar.header("📂 Pilih File Dataset")

    # Ambil file berdasarkan role
    if role == 'Admin':
        available_files = db.get_uploaded_files()
    elif role == 'BK':
        # BK bisa lihat semua hasil dari semua Wali Kelas
        available_files = db.get_uploaded_files()
    elif role == 'Wali Kelas':
        # Wali Kelas hanya lihat file miliknya
        available_files = db.get_uploaded_files(uploaded_by=user_id)
    else:
        available_files = []

    # Load UCI pre-trained assets
    model_uci, scaler_uci, features_uci = load_uci_artifacts()

    if not available_files:
        st.info("💡 **Belum ada file dataset yang tersedia.** Silakan upload file terlebih dahulu melalui menu **Upload File**.")

        # Tampilkan intro
        _show_intro_screen(features_uci)
        return

    # Buat opsi file di sidebar
    file_options = {}
    for f in available_files:
        label = f"📄 {f['original_filename']} (oleh {f['uploader_name']})"
        file_options[label] = f

    selected_file_label = st.sidebar.selectbox(
        "Pilih file untuk dianalisis:",
        options=list(file_options.keys())
    )

    selected_file = file_options[selected_file_label] if selected_file_label else None

    if selected_file is None:
        st.info("💡 Pilih file dataset dari sidebar untuk memulai analisis.")
        _show_intro_screen(features_uci)
        return

    # ── Display file info ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### ℹ️ Info File Terpilih")
    st.sidebar.markdown(f"**File:** {selected_file['original_filename']}")
    st.sidebar.markdown(f"**Oleh:** {selected_file['uploader_name']}")
    st.sidebar.markdown(f"**Upload:** {format_datetime(selected_file['uploaded_at'])}")
    if selected_file['description']:
        st.sidebar.markdown(f"**Deskripsi:** {selected_file['description']}")

    # ── Load File ──
    df_raw = load_file_from_path(selected_file['file_path'])

    if df_raw is None:
        st.error("❌ Gagal memuat file dataset. File mungkin rusak atau sudah dihapus.")
        return

    st.success(f"✅ Berhasil memuat file: `{selected_file['original_filename']}` ({df_raw.shape[0]} baris × {df_raw.shape[1]} kolom)")

    # Display raw data preview
    with st.expander("👁️ Lihat Preview Data"):
        st.dataframe(df_raw.head(10))

    # ── PROSES PREDIKSI ──────────────────────────────────────────────────────
    if mode == "Eksperimen (UCI Dataset - Pre-trained Model)":
        _run_experiment_mode(df_raw, model_uci, scaler_uci, features_uci)
    elif mode == "Data Primer (SMK Tunas Teknologi - Train On-the-fly)":
        _run_primary_mode(df_raw)


def _show_intro_screen(features_uci):
    """Tampilkan layar intro saat belum ada file yang dipilih."""
    col_desc1, col_desc2 = st.columns(2)

    with col_desc1:
        st.subheader("📊 Mode Eksperimen (UCI Dataset)")
        st.write("""
        Mode ini menggunakan model pohon keputusan C4.5 yang **sudah dilatih** sebelumnya pada dataset sekunder UCI (*Predict Students' Dropout and Academic Success*).
        
        * **Jumlah Fitur Penting:** 8 fitur pilihan (termasuk nilai akademik semester 1 & 2, biaya SPP, dll).
        * **Output:** Prediksi batch untuk data baru, visualisasi sebaran prediksi, dan visualisasi statistik.
        """)

        if features_uci:
            st.markdown("**8 Parameter yang wajib dicocokkan:**")
            for i, f in enumerate(features_uci, 1):
                st.markdown(f"{i}. `{f}`")

    with col_desc2:
        st.subheader("📝 Mode Data Primer (SMK Tunas Teknologi)")
        st.write("""
        Mode ini dirancang khusus untuk dataset lokal yang memiliki **14 parameter** lokal (kategori Kehadiran, Sosial, dan Ekonomi).
        
        * **Proses Dinamis (On-the-fly):** Model akan dilatih langsung dari data yang Anda unggah.
        * **Kustomisasi Parameter:** Anda bisa memilih fitur mana saja yang ingin disertakan/dihilangkan.
        * **Evaluasi Lengkap:** Menghasilkan akurasi, presisi, recall, F1-score, Confusion Matrix, ROC-AUC, tingkat kepentingan fitur, dan gambar pohon keputusan interaktif.
        """)
        st.markdown("**14 Fitur Data Primer (lokal):**")
        st.markdown("""
        1. Kehadiran Semester 1 & 2
        2. Nilai Rata-rata Semester 1 & 2
        3. Jumlah Pelanggaran
        4. Pendidikan Ayah & Ibu
        5. Status Keluarga & Jumlah Saudara
        6. Jarak Rumah (km)
        7. Status SPP & Penerima Beasiswa
        8. Penghasilan Ortu & Tanggungan Keluarga
        """)


def _run_experiment_mode(df_raw, model_uci, scaler_uci, features_uci):
    """Mode 1: Eksperimen (UCI Pre-trained)."""
    st.header("🔬 Evaluasi Data Baru Menggunakan Model Eksperimen UCI")

    if model_uci is None or scaler_uci is None:
        st.error("Error: File model pre-trained (`model_c45_dropout.pkl`) atau scaler tidak ditemukan di direktori!")
        return

    st.subheader("🔗 Pencocokan Kolom (Parameter Mapping)")
    st.write("Silakan pasangkan 8 parameter yang dibutuhkan model dengan kolom yang ada di file Anda:")

    columns_list = list(df_raw.columns)
    mapped_columns = []

    col_map1, col_map2 = st.columns(2)

    for i, feat in enumerate(features_uci):
        default_idx = 0
        for idx, col in enumerate(columns_list):
            if feat.lower() in col.lower() or col.lower() in feat.lower():
                default_idx = idx
                break

        with col_map1 if i % 2 == 0 else col_map2:
            sel_col = st.selectbox(
                f"Parameter: **{feat}**",
                options=columns_list,
                index=default_idx,
                key=f"map_{i}"
            )
            mapped_columns.append((feat, sel_col))

    # Optional target mapping
    st.markdown("---")
    st.subheader("🎯 Kolom Target Aktual (Opsional)")
    has_target = st.checkbox("File saya memiliki kolom label/target aktual (untuk menghitung Akurasi/Evaluasi)")

    target_col = None
    if has_target:
        default_tgt_idx = 0
        for idx, col in enumerate(columns_list):
            if col.lower() in ['target', 'status', 'label']:
                default_tgt_idx = idx
                break
        target_col = st.selectbox(
            "Pilih Kolom Target Aktual",
            options=columns_list,
            index=default_tgt_idx
        )

    # Start Prediction
    if st.button("Jalankan Prediksi Batch 🚀"):
        st.markdown("---")

        feature_mapping_dict = {feat: sel_col for feat, sel_col in mapped_columns}
        X_input = df_raw[[feature_mapping_dict[feat] for feat in features_uci]].copy()
        X_input.columns = features_uci

        for col in X_input.columns:
            if X_input[col].isnull().sum() > 0:
                X_input[col].fillna(X_input[col].median(), inplace=True)

        try:
            X_scaled = scaler_uci.transform(X_input)
            y_pred = model_uci.predict(X_scaled)
            y_pred_proba = model_uci.predict_proba(X_scaled)[:, 1] if hasattr(model_uci, "predict_proba") else None

            df_result = df_raw.copy()
            df_result['Hasil_Prediksi_Numerik'] = y_pred
            df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].map({1: 'Dropout', 0: 'Non-Dropout'})
            if y_pred_proba is not None:
                df_result['Probabilitas_Dropout'] = y_pred_proba

            # --- DISPLAY DASHBOARD ---
            st.subheader("📊 Dashboard Evaluasi Hasil Prediksi")

            kpi1, kpi2, kpi3 = st.columns(3)
            total_cnt = len(df_result)
            dropout_cnt = int((y_pred == 1).sum())
            nondropout_cnt = total_cnt - dropout_cnt

            with kpi1:
                st.metric("Total Siswa Dievaluasi", f"{total_cnt} Orang")
            with kpi2:
                st.metric("Diprediksi DROPOUT (Berisiko)", f"{dropout_cnt} Orang", f"{dropout_cnt/total_cnt*100:.1f}%", delta_color="inverse")
            with kpi3:
                st.metric("Diprediksi NON-DROPOUT", f"{nondropout_cnt} Orang", f"{nondropout_cnt/total_cnt*100:.1f}%")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("##### 📌 Distribusi Hasil Prediksi (Pie Chart)")
                fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
                counts = [nondropout_cnt, dropout_cnt]
                labels = ['Non-Dropout', 'Dropout']
                colors = ['#2ECC71', '#E74C3C']
                ax_pie.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
                           textprops={'fontsize': 12, 'weight': 'bold'})
                ax_pie.axis('equal')
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            with chart_col2:
                st.markdown("##### 📌 Sebaran Nilai Rata-rata Semester 2 vs Status Prediksi")
                grade_col = feature_mapping_dict.get('Curricular units 2nd sem (grade)')
                if grade_col in df_raw.columns:
                    fig_box, ax_box = plt.subplots(figsize=(7, 5.2))
                    sns.boxplot(data=df_result, x='Hasil_Prediksi', y=grade_col, palette={'Non-Dropout': '#2ECC71', 'Dropout': '#E74C3C'}, ax=ax_box)
                    ax_box.set_title("Nilai Rata-rata Semester 2 vs Hasil Prediksi", fontsize=12, fontweight='bold')
                    ax_box.set_xlabel("Status Prediksi")
                    ax_box.set_ylabel("Nilai (Grade)")
                    st.pyplot(fig_box)
                    plt.close(fig_box)
                else:
                    st.warning("Visualisasi sebaran nilai tidak dapat ditampilkan karena kolom 'Curricular units 2nd sem (grade)' tidak ditemukan/dipetakan.")

            # Heatmap Korelasi
            st.markdown("##### 📌 Heatmap Korelasi 8 Parameter Terpilih")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            corr_mat = X_input.corr()
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(corr_mat, mask=mask, annot=True, cmap='RdBu_r', center=0, linewidths=0.5, fmt='.2f', ax=ax_corr, cbar_kws={'label': 'Korelasi'})
            ax_corr.set_title("Matriks Korelasi 8 Fitur Mapped", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_corr)
            plt.close(fig_corr)

            # Evaluation section
            if has_target and target_col:
                st.markdown("---")
                st.subheader("🎯 Metrik Evaluasi Model terhadap Target Aktual")

                y_true_raw = df_raw[target_col]
                unique_true = y_true_raw.unique()
                do_val = None
                for ut in unique_true:
                    if str(ut).strip().lower() == 'dropout':
                        do_val = ut
                        break

                if do_val is not None:
                    y_true = y_true_raw.apply(lambda x: 1 if x == do_val else 0).values
                else:
                    if 1 in unique_true or '1' in unique_true:
                        y_true = y_true_raw.apply(lambda x: 1 if str(x) in ['1', '1.0'] else 0).values
                    else:
                        y_true = pd.factorize(y_true_raw)[0]
                        if len(np.unique(y_true)) > 1:
                            pass

                if len(np.unique(y_true)) == 2:
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)

                    met1, met2, met3, met4 = st.columns(4)
                    with met1:
                        st.metric("Akurasi", f"{acc*100:.2f}%")
                    with met2:
                        st.metric("Presisi", f"{prec*100:.2f}%")
                    with met3:
                        st.metric("Recall (Sensitivitas)", f"{rec*100:.2f}%")
                    with met4:
                        st.metric("F1-Score", f"{f1*100:.2f}%")

                    eval_col1, eval_col2 = st.columns(2)
                    with eval_col1:
                        st.markdown("##### 📌 Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                    xticklabels=['Non-Dropout', 'Dropout'],
                                    yticklabels=['Non-Dropout', 'Dropout'],
                                    annot_kws={'size': 14})
                        ax_cm.set_xlabel('Prediksi')
                        ax_cm.set_ylabel('Aktual')
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                    with eval_col2:
                        st.markdown("##### 📌 ROC Curve")
                        if y_pred_proba is not None:
                            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                            roc_auc = auc(fpr, tpr)

                            fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
                            ax_roc.plot(fpr, tpr, color='#E24B4A', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                            ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
                            ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#E24B4A')
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.legend(loc='lower right')
                            st.pyplot(fig_roc)
                            plt.close(fig_roc)
                        else:
                            st.info("Kurva ROC tidak dapat diplot karena probabilitas prediksi tidak tersedia.")
                else:
                    st.warning("Kolom target aktual tidak memiliki tepat 2 kelas (binary) setelah dibaca. Evaluasi dilewati.")

            # --- DOWNLOAD & DATA VIEW ---
            st.markdown("---")
            st.subheader("📋 Hasil Prediksi Lengkap")
            st.dataframe(df_result.head(100))

            towrite = io.BytesIO()
            df_result.to_csv(towrite, index=False, encoding='utf-8')
            towrite.seek(0)

            st.download_button(
                label="📥 Unduh Hasil Prediksi (CSV)",
                data=towrite,
                file_name="hasil_prediksi_eksperimen_uci.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            st.exception(e)


def _run_primary_mode(df_raw):
    """Mode 2: Data Primer (Train On-the-fly)."""
    st.header("📝 Evaluasi Data Primer SMK Tunas Teknologi (Model Training)")

    columns_list = list(df_raw.columns)

    st.subheader("⚙️ Konfigurasi Pembuatan Model C4.5")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        default_target_idx = 0
        for idx, col in enumerate(columns_list):
            if col.lower() in ['status', 'target', 'label']:
                default_target_idx = idx
                break
        target_col = st.selectbox(
            "Pilih Kolom Target (Y)",
            options=columns_list,
            index=default_target_idx
        )

        default_features = [col for col in columns_list if col != target_col]

        selected_train_features = st.multiselect(
            "Pilih Parameter Fitur (X)",
            options=default_features,
            default=default_features,
            help="Pilih fitur-fitur yang akan digunakan sebagai dasar pembelajaran C4.5."
        )

    with col_cfg2:
        test_size = st.slider("Ukuran Data Testing (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
        max_depth = st.slider("Kedalaman Maksimum Pohon (Max Depth)", min_value=3, max_value=15, value=7)

        use_ig_selection = st.checkbox("Aktifkan Seleksi Fitur otomatis dengan Information Gain", value=False,
                                        help="Model hanya akan menggunakan fitur yang memiliki Information Gain >= threshold.")
        ig_threshold = 0.0
        if use_ig_selection:
            ig_threshold = st.slider("Threshold Information Gain", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

    # Start Training
    if st.button("Latih Model C4.5 & Tampilkan Dashboard 🚀"):
        if len(selected_train_features) < 1:
            st.error("Pilih minimal 1 fitur untuk melatih model!")
        else:
            st.markdown("---")
            with st.spinner("Melatih Model C4.5 (Decision Tree) dan membuat visualisasi..."):

                df_model = df_raw.copy()

                # --- PREPROCESSING ---
                for col in selected_train_features:
                    if df_model[col].isnull().sum() > 0:
                        if df_model[col].dtype in ['int64', 'float64']:
                            df_model[col].fillna(df_model[col].median(), inplace=True)
                        else:
                            df_model[col].fillna(df_model[col].mode()[0], inplace=True)

                # Target Binarization
                y_raw = df_model[target_col]
                unique_targets = y_raw.unique()

                dropout_val = None
                for val in unique_targets:
                    if str(val).strip().lower() == 'dropout':
                        dropout_val = val
                        break

                if dropout_val is not None:
                    y = y_raw.apply(lambda x: 1 if x == dropout_val else 0).values
                    class_names = ['Non-Dropout', 'Dropout']
                else:
                    if 1 in unique_targets or '1' in unique_targets:
                        y = y_raw.apply(lambda x: 1 if str(x) in ['1', '1.0'] else 0).values
                        class_names = ['Non-Dropout', 'Dropout']
                    else:
                        sorted_unique = sorted(list(unique_targets))
                        y = y_raw.map({sorted_unique[0]: 0, sorted_unique[1]: 1}).values
                        class_names = [str(sorted_unique[0]), str(sorted_unique[1])]

                X_all = df_model[selected_train_features]

                # Convert categorical to numerical
                X_numeric = X_all.copy()
                categorical_cols = []
                for col in X_numeric.columns:
                    if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
                        X_numeric[col] = pd.factorize(X_numeric[col])[0]
                        categorical_cols.append(col)

                if len(categorical_cols) > 0:
                    st.info(f"ℹ️ Mengonversi kolom kategorikal berikut menjadi kode numerik untuk pelatihan: `{', '.join(categorical_cols)}`")

                # --- FEATURE SELECTION (INFORMATION GAIN) ---
                features_to_use = selected_train_features.copy()

                if use_ig_selection:
                    try:
                        ig_scores = mutual_info_classif(X_numeric, y, random_state=42)
                        ig_df = pd.DataFrame({
                            'Fitur': X_numeric.columns,
                            'Information Gain': ig_scores
                        }).sort_values('Information Gain', ascending=False).reset_index(drop=True)

                        selected_by_ig = ig_df[ig_df['Information Gain'] >= ig_threshold]['Fitur'].tolist()

                        st.markdown("##### 🔍 Seleksi Fitur Information Gain")

                        fig_ig, ax_ig = plt.subplots(figsize=(10, max(5, len(ig_df) * 0.35)))
                        colors_ig = ['#2ECC71' if v >= ig_threshold else '#E74C3C' for v in ig_df['Information Gain']]
                        ax_ig.barh(ig_df['Fitur'][::-1], ig_df['Information Gain'][::-1], color=colors_ig[::-1], edgecolor='white', linewidth=0.5)
                        ax_ig.axvline(x=ig_threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({ig_threshold})')
                        ax_ig.set_xlabel('Information Gain')
                        ax_ig.set_title('Fase 3 — Information Gain per Fitur', fontsize=12, fontweight='bold')
                        ax_ig.legend()
                        plt.tight_layout()
                        st.pyplot(fig_ig)
                        plt.close(fig_ig)

                        if len(selected_by_ig) == 0:
                            st.warning(f"⚠️ Tidak ada fitur dengan Information Gain >= {ig_threshold}. Seleksi fitur dilewati, menggunakan seluruh parameter yang dipilih.")
                        else:
                            features_to_use = selected_by_ig
                            X_numeric = X_numeric[features_to_use]
                            st.success(f"🎯 Seleksi Fitur selesai: Menggunakan {len(features_to_use)} fitur dari {len(selected_train_features)} fitur yang di-upload.")

                    except Exception as e:
                        st.error(f"Gagal menghitung Information Gain: {e}")

                # --- TRAIN TEST SPLIT & SCALING ---
                X_train, X_test, y_train, y_test = train_test_split(
                    X_numeric, y, test_size=test_size, random_state=42, stratify=y
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_numeric.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_numeric.columns, index=X_test.index)

                # --- TRAINING C4.5 ---
                model_c45 = DecisionTreeClassifier(
                    criterion='entropy',
                    max_depth=max_depth,
                    random_state=42
                )
                model_c45.fit(X_train_scaled, y_train)

                y_pred = model_c45.predict(X_test_scaled)
                y_pred_proba = model_c45.predict_proba(X_test_scaled)[:, 1]

                # --- EVALUATION METRICS ---
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                st.subheader("📊 Dashboard Evaluasi Model C4.5")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Akurasi Model", f"{acc*100:.2f}%")
                with m2:
                    st.metric("Presisi Model", f"{prec*100:.2f}%")
                with m3:
                    st.metric("Recall (Sensitivitas)", f"{rec*100:.2f}%")
                with m4:
                    st.metric("F1-Score", f"{f1*100:.2f}%")

                # Visualizations Row 1
                vis_col1, vis_col2 = st.columns(2)

                with vis_col1:
                    st.markdown("##### 📌 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=class_names,
                                yticklabels=class_names,
                                annot_kws={'size': 14})
                    ax_cm.set_xlabel('Prediksi')
                    ax_cm.set_ylabel('Aktual')
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                with vis_col2:
                    st.markdown("##### 📌 ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)

                    fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
                    ax_roc.plot(fpr, tpr, color='#E24B4A', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#E24B4A')
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.legend(loc='lower right')
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)

                # Tree visualization
                st.markdown("##### 📌 Visualisasi Struktur Pohon Keputusan C4.5")
                fig_tree, ax_tree = plt.subplots(figsize=(24, 10), dpi=150)
                plot_tree(
                    model_c45,
                    feature_names=X_numeric.columns.tolist(),
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    proportion=True,
                    fontsize=7,
                    ax=ax_tree
                )
                st.pyplot(fig_tree)
                plt.close(fig_tree)

                # Feature Importance & Correlation
                vis_col3, vis_col4 = st.columns(2)

                with vis_col3:
                    st.markdown("##### 📌 Tingkat Kepentingan Fitur (Feature Importance)")
                    importances = pd.Series(model_c45.feature_importances_, index=X_numeric.columns)
                    importances = importances.sort_values(ascending=True)

                    fig_fi, ax_fi = plt.subplots(figsize=(7, max(4.5, len(importances)*0.35)))
                    colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
                    importances.plot(kind='barh', color=colors_fi, edgecolor='white', linewidth=0.5, ax=ax_fi)
                    ax_fi.set_xlabel('Importance Score')
                    ax_fi.set_title('Feature Importance (C4.5)')
                    plt.tight_layout()
                    st.pyplot(fig_fi)
                    plt.close(fig_fi)

                with vis_col4:
                    st.markdown("##### 📌 Heatmap Korelasi Fitur Terpilih")
                    fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
                    corr_mat = X_numeric.corr()
                    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
                    sns.heatmap(corr_mat, mask=mask, annot=False, cmap='RdBu_r', center=0, linewidths=0.5, ax=ax_corr)
                    ax_corr.set_title("Heatmap Korelasi Fitur Model")
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)

                # Classification report
                with st.expander("📋 Laporan Klasifikasi Lengkap (Classification Report)"):
                    st.text("=== CLASSIFICATION REPORT ===")
                    st.text(classification_report(y_test, y_pred, target_names=class_names))

                # --- BATCH PREDICTIONS ---
                st.markdown("---")
                st.subheader("📋 Hasil Prediksi Batch untuk Seluruh Data")

                X_all_scaled = scaler.transform(X_numeric)
                all_preds = model_c45.predict(X_all_scaled)
                all_probs = model_c45.predict_proba(X_all_scaled)[:, 1]

                df_result = df_raw.copy()
                df_result['Hasil_Prediksi_Numerik'] = all_preds
                df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].map({1: 'Dropout', 0: 'Non-Dropout'})
                df_result['Probabilitas_Dropout'] = all_probs

                st.dataframe(df_result.head(100))

                towrite = io.BytesIO()
                df_result.to_csv(towrite, index=False, encoding='utf-8')
                towrite.seek(0)

                st.download_button(
                    label="📥 Unduh Hasil Prediksi Lengkap (CSV)",
                    data=towrite,
                    file_name="hasil_prediksi_data_primer.csv",
                    mime="text/csv"
                )


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN APP FLOW ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.logged_in:
    show_login_page()
else:
    show_sidebar()

    page = st.session_state.current_page
    role = st.session_state.role

    if page == 'Dashboard Prediksi':
        show_prediction_dashboard()
    elif page == 'Upload File':
        if role in ['Admin', 'Wali Kelas']:
            show_upload_page()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    elif page == 'Manajemen File':
        if role in ['Admin', 'Wali Kelas']:
            show_file_management_page()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    elif page == 'Manajemen User':
        if role == 'Admin':
            show_user_management_page()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    else:
        show_prediction_dashboard()
