import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True

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
    st.session_state.current_page = 'Dashboard Riwayat'


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
                '📊 Dashboard Riwayat',
                '⚙️ Konfigurasi Prediksi',
                '📤 Upload File',
                '📁 Manajemen File',
                '👥 Manajemen User'
            ]
        elif role == 'BK':
            menu_items = [
                '📊 Dashboard Riwayat',
                '⚙️ Konfigurasi Prediksi',
                '📤 Upload File',
                '📁 Manajemen File'
            ]
        elif role == 'Wali Kelas':
            menu_items = [
                '📊 Dashboard Riwayat',
                '📤 Upload File',
                '📁 Manajemen File'
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


# ── HALAMAN DASHBOARD RIWAYAT ───────────────────────────────────────────────

def show_dashboard_riwayat():
    st.title("📊 Dashboard Riwayat Prediksi")
    # st.markdown("Ringkasan dan histori prediksi yang pernah dilakukan.")

    # ── RIWAYAT PREDIKSI TABLE ─────────────────────────────────────────────────
    # st.subheader("📋 Riwayat Prediksi")
    history = db.get_prediction_history()

    if not history:
        st.info("Belum ada riwayat prediksi. Silakan jalankan analisis melalui menu **Konfigurasi Prediksi**.")
    else:
        # hist_data = []
        # for h in history:
        #     hist_data.append({
        #         'ID': h['id'],
        #         'Tanggal': format_datetime(h['run_at']),
        #         'Oleh': h.get('run_by_name', '-'),
        #         'Dataset': h['dataset_name'],
        #         'Mode': h['mode_analisis'],
        #         'Akurasi': f"{h['accuracy']:.2f}%" if h['accuracy'] else '-',
        #         'F1-Score': f"{h['f1_score']:.2f}%" if h['f1_score'] else '-'
        #     })
        # st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)

        # ── DROPDOWN PILIH RIWAYAT UNTUK VISUALISASI ───────────────────────────
        # st.markdown("---")
        # st.subheader("📈 Distribusi Hasil Prediksi")

        # Buat label pilihan dari riwayat
        history_options = {
            f"#{h['id']} — {h['dataset_name']} ({format_datetime(h['run_at'])})": h
            for h in history
        }

        selected_label = st.selectbox(
            "Pilih Riwayat Prediksi untuk Ditampilkan:",
            options=list(history_options.keys()),
            index=0
        )

        selected_h = history_options[selected_label]

        # Parse config_json dari riwayat yang dipilih
        import json
        config_data = {}
        if selected_h.get('config_json'):
            try:
                config_data = json.loads(selected_h['config_json'])
            except Exception:
                config_data = {}

        target_col = config_data.get('target', None)
        features = config_data.get('features', [])

        # Load file dataset yang sesuai
        uploaded_files = db.get_uploaded_files()
        matched_file = next(
            (f for f in uploaded_files if f['original_filename'] == selected_h['dataset_name']),
            None
        )

        if matched_file is None:
            st.warning(f"File dataset **{selected_h['dataset_name']}** tidak ditemukan di storage. Distribusi tidak dapat ditampilkan.")
        elif target_col is None:
            st.warning("Informasi kolom target tidak tersimpan di riwayat ini.")
        else:
            df_hist = load_file_from_path(matched_file['file_path'])

            if df_hist is None or target_col not in df_hist.columns:
                st.warning("Dataset atau kolom target tidak dapat dimuat.")
            else:
                color_palette = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C']

                # Metrik ringkasan
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Akurasi", f"{selected_h['accuracy']:.2f}%" if selected_h['accuracy'] else '-')
                with m2:
                    st.metric("Presisi", f"{selected_h['precision']:.2f}%" if selected_h['precision'] else '-')
                with m3:
                    st.metric("Recall", f"{selected_h['recall']:.2f}%" if selected_h['recall'] else '-')
                with m4:
                    st.metric("F1-Score", f"{selected_h['f1_score']:.2f}%" if selected_h['f1_score'] else '-')

                st.markdown("---")

                # Distribusi kelas target dari dataset
                target_counts = df_hist[target_col].value_counts()
                target_labels = [str(l) for l in target_counts.index.tolist()]
                target_values = target_counts.values.tolist()
                total = sum(target_values)
                target_colors = color_palette[:len(target_labels)]

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown(f"##### Distribusi Kelas Target: `{target_col}`")
                    fig1, ax1 = plt.subplots(figsize=(7, 5))
                    bars = ax1.bar(target_labels, target_values, color=target_colors, edgecolor='white', linewidth=1.5)
                    for bar, val in zip(bars, target_values):
                        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + total*0.01,
                                 f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
                    ax1.set_ylabel('Jumlah Siswa', fontsize=11)
                    ax1.set_title(f'Distribusi Kelas: {target_col}', fontsize=12, fontweight='bold')
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    ax1.tick_params(colors='gray')
                    ax1.yaxis.label.set_color('gray')
                    plt.tight_layout()
                    st.pyplot(fig1, transparent=True)
                    plt.close(fig1)

                with chart_col2:
                    st.markdown("##### Proporsi Kelas Target")
                    fig2, ax2 = plt.subplots(figsize=(7, 5))
                    wedges, texts, autotexts = ax2.pie(
                        target_values, labels=target_labels, colors=target_colors,
                        autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                        textprops={'fontsize': 11, 'fontweight': 'bold'}
                    )
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                    ax2.axis('equal')
                    ax2.set_title('Proporsi Kelas Target', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig2, transparent=True)
                    plt.close(fig2)

                # Tabel ringkasan distribusi
                dist_df = pd.DataFrame({
                    'Kelas': target_labels,
                    'Jumlah': target_values,
                    'Proporsi (%)': [f"{v/total*100:.1f}%" for v in target_values]
                })
                st.dataframe(dist_df, use_container_width=True, hide_index=True)



def show_prediction_config():
    st.title("⚙️ Konfigurasi Prediksi")
    
    role = st.session_state.role
    user_id = st.session_state.user_id

    # Load UCI pre-trained assets
    model_uci, scaler_uci, features_uci = load_uci_artifacts()

    # Ambil file berdasarkan role
    if role == 'Admin':
        available_files = db.get_uploaded_files()
    elif role == 'BK':
        available_files = db.get_uploaded_files()
    else:
        available_files = []

    if not available_files:
        st.info("💡 **Belum ada file dataset yang tersedia.** Silakan upload file terlebih dahulu melalui menu **Upload File**.")
        return

    st.markdown("---")
    col_mode, col_file = st.columns(2)

    with col_mode:
        mode = st.selectbox(
            "🛠️ Pilih Mode Analisis",
            options=[
                "Eksperimen (UCI Dataset - Pre-trained Model)",
                "Data Primer (SMK Tunas Teknologi - Train On-the-fly)"
            ]
        )

    with col_file:
        file_options = {}
        for f in available_files:
            label = f"📄 {f['original_filename']} (oleh {f['uploader_name']})"
            file_options[label] = f

        selected_file_label = st.selectbox(
            "📂 Pilih File Dataset",
            options=list(file_options.keys())
        )

    selected_file = file_options[selected_file_label] if selected_file_label else None

    if selected_file is None:
        return

    df_raw = load_file_from_path(selected_file['file_path'])

    if df_raw is None:
        st.error("❌ Gagal memuat file dataset. File mungkin rusak atau sudah dihapus.")
        return

    st.success(f"✅ Berhasil memuat file: `{selected_file['original_filename']}` ({df_raw.shape[0]} baris × {df_raw.shape[1]} kolom)")

    if mode == "Eksperimen (UCI Dataset - Pre-trained Model)":
        _config_experiment_mode(df_raw, selected_file)
    else:
        _config_primary_mode(df_raw, selected_file)


def _config_experiment_mode(df_raw, selected_file):
    st.markdown("#### Konfigurasi Model (Pre-trained UCI)")
    st.info("Model menggunakan pre-trained weights. Tidak ada parameter yang perlu diubah.")
    if st.button("🚀 Proses Analisis", type="primary", use_container_width=True):
        st.session_state.run_config = {
            'mode': 'Eksperimen',
            'df_raw': df_raw,
            'dataset_name': selected_file['original_filename']
        }
        st.session_state.current_page = 'Hasil Prediksi'
        st.rerun()

def _config_primary_mode(df_raw, selected_file):
    st.markdown("#### Konfigurasi Pembuatan Model C4.5")

    columns_list = list(df_raw.columns)

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

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — DISTRIBUSI KELAS TARGET (ditampilkan sebelum training)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 Fase 2 — Distribusi Kelas Target")

    y_raw_preview = df_raw[target_col]
    target_counts = y_raw_preview.value_counts()
    target_labels = target_counts.index.tolist()
    target_values = target_counts.values.tolist()
    total_data = sum(target_values)

    color_palette = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    target_colors = color_palette[:len(target_labels)]

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.markdown("##### Distribusi Kelas Target (Asli)")
        fig_dist, ax_dist = plt.subplots(figsize=(7, 5))
        bars = ax_dist.bar(target_labels, target_values, color=target_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, target_values):
            ax_dist.text(bar.get_x() + bar.get_width()/2., bar.get_height() + total_data*0.01,
                        f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax_dist.set_ylabel('Jumlah Siswa', fontsize=11)
        ax_dist.set_title('Distribusi Kelas Target (Asli)', fontsize=12, fontweight='bold')
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    with dist_col2:
        st.markdown("##### Proporsi Kelas Target")
        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax_pie.pie(
            target_values, labels=target_labels, colors=target_colors,
            autopct='%1.1f%%', startangle=140, pctdistance=0.75,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        ax_pie.axis('equal')
        ax_pie.set_title('Proporsi Kelas Target', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    dist_summary = pd.DataFrame({
        'Kelas': target_labels,
        'Jumlah': target_values,
        'Proporsi (%)': [f"{v/total_data*100:.1f}%" for v in target_values]
    })
    st.dataframe(dist_summary, use_container_width=True, hide_index=True)

    if st.button("🚀 Proses Analisis", type="primary", use_container_width=True):
        if len(selected_train_features) < 1:
            st.error("Pilih minimal 1 fitur untuk melatih model!")
        else:
            st.session_state.run_config = {
                'mode': 'Primer',
                'df_raw': df_raw,
                'dataset_name': selected_file['original_filename'],
                'target_col': target_col,
                'selected_train_features': selected_train_features,
                'test_size': test_size,
                'max_depth': max_depth,
                'use_ig_selection': use_ig_selection,
                'ig_threshold': ig_threshold
            }
            st.session_state.current_page = 'Hasil Prediksi'
            st.rerun()

# ── HALAMAN HASIL PREDIKSI ───────────────────────────────────────────────

def show_prediction_results():
    st.title("📈 Hasil Analisis Prediksi")
    
    if st.button("⬅️ Kembali ke Dashboard"):
        st.session_state.current_page = 'Dashboard Riwayat'
        st.rerun()
        
    if 'run_config' not in st.session_state:
        st.warning("Tidak ada data hasil prediksi. Silakan jalankan konfigurasi terlebih dahulu.")
        return
        
    config = st.session_state.run_config
    
    # Run the model logic based on mode
    # Note: DB saving is handled inside the run mode functions to get accuracy metrics.
    # To prevent duplicate saving on reruns, we pass a flag to the function.
    is_new_run = not config.get('is_saved', False)
    
    if config['mode'] == 'Eksperimen':
        _run_experiment_mode(config['df_raw'])
    else:
        _run_primary_mode(config, is_new_run=is_new_run)
        
    if is_new_run:
        st.session_state.run_config['is_saved'] = True



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


def _run_primary_mode(config, is_new_run=False):
    """Mode 2: Data Primer (Train On-the-fly). Menerima config dict dari halaman konfigurasi."""
    st.header("📝 Evaluasi Data Primer SMK Tunas Teknologi (Model Training)")

    # Ekstrak konfigurasi dari config dict
    df_raw = config['df_raw']
    target_col = config['target_col']
    selected_train_features = config['selected_train_features']
    test_size = config['test_size']
    max_depth = config['max_depth']
    use_ig_selection = config['use_ig_selection']
    ig_threshold = config['ig_threshold']
    dataset_name = config['dataset_name']

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

        # ══════════════════════════════════════════════════════════════
        # BAGIAN 1: EVALUASI MODEL
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("📊 Fase 5 — Evaluasi Model C4.5")

        # KPI Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Akurasi", f"{acc*100:.2f}%")
        with m2:
            st.metric("Presisi", f"{prec*100:.2f}%")
        with m3:
            st.metric("Recall", f"{rec*100:.2f}%")
        with m4:
            st.metric("F1-Score", f"{f1*100:.2f}%")

        # Confusion Matrix + Ringkasan Metrik Bar Chart
        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            st.markdown("##### 📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=class_names,
                        yticklabels=class_names,
                        annot_kws={'size': 14})
            ax_cm.set_xlabel('Prediksi')
            ax_cm.set_ylabel('Aktual')
            ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with eval_col2:
            st.markdown("##### 📌 Ringkasan Metrik Evaluasi")
            metric_names = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
            metric_values = [acc*100, prec*100, rec*100, f1*100]
            metric_colors = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C']

            fig_bar, ax_bar = plt.subplots(figsize=(6, 4.5))
            bars_m = ax_bar.bar(metric_names, metric_values, color=metric_colors,
                               edgecolor='white', linewidth=1.5, width=0.6)
            # Tambah label di atas bar
            for bar, val in zip(bars_m, metric_values):
                ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
            # Garis target minimum
            ax_bar.axhline(y=80, color='red', linestyle='--', linewidth=1.2, label='Target minimum (80%)')
            ax_bar.set_ylabel('Nilai (%)', fontsize=11)
            ax_bar.set_ylim(0, max(metric_values) + 15)
            ax_bar.set_title('Ringkasan Metrik Evaluasi', fontsize=12, fontweight='bold')
            ax_bar.legend(loc='upper right', fontsize=9)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        # ROC Curve
        st.markdown("##### 📌 ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color='#E24B4A', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#E24B4A')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax_roc.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

        # ── Distribusi Hasil Prediksi vs Aktual ──
        st.markdown("##### 📌 Perbandingan Data Aktual vs Prediksi")
        cmp_col1, cmp_col2 = st.columns(2)

        with cmp_col1:
            st.markdown("###### Distribusi Aktual (Data Testing)")
            actual_counts = pd.Series(y_test).value_counts().sort_index()
            actual_labels = [class_names[i] for i in actual_counts.index]
            actual_values = actual_counts.values

            fig_act, ax_act = plt.subplots(figsize=(6, 4.5))
            act_colors = ['#2ECC71', '#E74C3C'][:len(actual_labels)]
            bars_a = ax_act.bar(actual_labels, actual_values, color=act_colors, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars_a, actual_values):
                ax_act.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(actual_values)*0.01,
                           f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            ax_act.set_ylabel('Jumlah Siswa')
            ax_act.set_title('Distribusi Aktual (Data Testing)', fontsize=11, fontweight='bold')
            ax_act.spines['top'].set_visible(False)
            ax_act.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_act)
            plt.close(fig_act)

        with cmp_col2:
            st.markdown("###### Distribusi Prediksi Model C4.5")
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            pred_labels = [class_names[i] for i in pred_counts.index]
            pred_values = pred_counts.values

            fig_prd, ax_prd = plt.subplots(figsize=(6, 4.5))
            prd_colors = ['#2ECC71', '#E74C3C'][:len(pred_labels)]
            bars_p = ax_prd.bar(pred_labels, pred_values, color=prd_colors, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars_p, pred_values):
                ax_prd.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(pred_values)*0.01,
                           f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            ax_prd.set_ylabel('Jumlah Siswa')
            ax_prd.set_title('Distribusi Prediksi Model C4.5', fontsize=11, fontweight='bold')
            ax_prd.spines['top'].set_visible(False)
            ax_prd.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_prd)
            plt.close(fig_prd)

        # ── CROSS-VALIDATION (10-Fold) ──
        st.markdown("---")
        st.markdown("##### 📌 Cross-Validation (10-Fold)")

        cv_model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
        X_all_scaled_cv = scaler.fit_transform(X_numeric)

        cv_accuracy = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='accuracy')
        cv_precision = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='precision')
        cv_recall = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='recall')
        cv_f1 = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='f1')

        # Tabel hasil per fold
        cv_table = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(10)],
            'Akurasi (%)': [f'{v*100:.2f}' for v in cv_accuracy],
            'Presisi (%)': [f'{v*100:.2f}' for v in cv_precision],
            'Recall (%)': [f'{v*100:.2f}' for v in cv_recall],
            'F1-Score (%)': [f'{v*100:.2f}' for v in cv_f1],
        })
        # Tambah baris rata-rata
        avg_row = pd.DataFrame({
            'Fold': ['**Rata-rata**'],
            'Akurasi (%)': [f'{cv_accuracy.mean()*100:.2f}'],
            'Presisi (%)': [f'{cv_precision.mean()*100:.2f}'],
            'Recall (%)': [f'{cv_recall.mean()*100:.2f}'],
            'F1-Score (%)': [f'{cv_f1.mean()*100:.2f}'],
        })
        cv_table = pd.concat([cv_table, avg_row], ignore_index=True)
        st.dataframe(cv_table, use_container_width=True, hide_index=True)

        # Visualisasi CV - line chart per fold
        cv_vis_col1, cv_vis_col2 = st.columns(2)
        with cv_vis_col1:
            fig_cv, ax_cv = plt.subplots(figsize=(7, 4.5))
            folds = range(1, 11)
            ax_cv.plot(folds, cv_accuracy*100, 'o-', color='#3498DB', label='Akurasi', linewidth=2, markersize=6)
            ax_cv.plot(folds, cv_precision*100, 's-', color='#2ECC71', label='Presisi', linewidth=2, markersize=6)
            ax_cv.plot(folds, cv_recall*100, '^-', color='#E67E22', label='Recall', linewidth=2, markersize=6)
            ax_cv.plot(folds, cv_f1*100, 'D-', color='#E74C3C', label='F1-Score', linewidth=2, markersize=6)
            ax_cv.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='Target (80%)')
            ax_cv.set_xlabel('Fold', fontsize=11)
            ax_cv.set_ylabel('Nilai (%)', fontsize=11)
            ax_cv.set_title('Performa per Fold (10-Fold CV)', fontsize=12, fontweight='bold')
            ax_cv.set_xticks(folds)
            ax_cv.legend(fontsize=8, loc='lower left')
            ax_cv.set_ylim(max(0, min(cv_accuracy.min(), cv_recall.min())*100 - 10), 105)
            ax_cv.spines['top'].set_visible(False)
            ax_cv.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_cv)
            plt.close(fig_cv)

        with cv_vis_col2:
            st.markdown("###### Ringkasan 10-Fold CV")
            cv_kpi1, cv_kpi2 = st.columns(2)
            with cv_kpi1:
                st.metric("Rata-rata Akurasi", f"{cv_accuracy.mean()*100:.2f}%", f"± {cv_accuracy.std()*100:.2f}%")
                st.metric("Rata-rata Presisi", f"{cv_precision.mean()*100:.2f}%", f"± {cv_precision.std()*100:.2f}%")
            with cv_kpi2:
                st.metric("Rata-rata Recall", f"{cv_recall.mean()*100:.2f}%", f"± {cv_recall.std()*100:.2f}%")
                st.metric("Rata-rata F1-Score", f"{cv_f1.mean()*100:.2f}%", f"± {cv_f1.std()*100:.2f}%")

        # Classification report
        with st.expander("📋 Laporan Klasifikasi Lengkap (Classification Report)"):
            st.text("=== CLASSIFICATION REPORT ===")
            st.text(classification_report(y_test, y_pred, target_names=class_names))

        # ══════════════════════════════════════════════════════════════
        # BAGIAN 2: HASIL (Pohon, Feature Importance, Korelasi, Prediksi)
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("🌳 Hasil — Model & Prediksi")

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

        # Top Feature Importance & Heatmap Korelasi
        vis_col3, vis_col4 = st.columns(2)

        with vis_col3:
            st.markdown("##### 📌 Top Fitur Berdasarkan Importance")
            importances = pd.Series(model_c45.feature_importances_, index=X_numeric.columns)
            importances = importances.sort_values(ascending=True)

            fig_fi, ax_fi = plt.subplots(figsize=(7, max(4.5, len(importances)*0.35)))
            colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
            importances.plot(kind='barh', color=colors_fi, edgecolor='white', linewidth=0.5, ax=ax_fi)
            # Label di ujung bar
            for i, (val, name) in enumerate(zip(importances.values, importances.index)):
                ax_fi.text(val + importances.max()*0.01, i, f'{val:.4f}', va='center', fontsize=9)
            ax_fi.set_xlabel('Importance Score')
            ax_fi.set_title('Top Fitur Berdasarkan Importance (C4.5)', fontsize=11, fontweight='bold')
            ax_fi.spines['top'].set_visible(False)
            ax_fi.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)

            # Tabel Feature Importance
            fi_table = pd.DataFrame({
                'Rank': range(1, len(importances)+1),
                'Fitur': importances.sort_values(ascending=False).index.tolist(),
                'Importance Score': [f'{v:.4f}' for v in importances.sort_values(ascending=False).values]
            })
            st.dataframe(fi_table, use_container_width=True, hide_index=True)

        with vis_col4:
            st.markdown("##### 📌 Heatmap Korelasi Fitur Terpilih")
            fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
            corr_mat = X_numeric.corr()
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(corr_mat, mask=mask, annot=False, cmap='RdBu_r', center=0, linewidths=0.5, ax=ax_corr)
            ax_corr.set_title("Heatmap Korelasi Fitur Model", fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_corr)
            plt.close(fig_corr)

        # --- BATCH PREDICTIONS ---
        st.markdown("---")
        st.subheader("📋 Hasil Prediksi Batch untuk Seluruh Data")

        X_all_scaled = scaler.transform(X_numeric)
        all_preds = model_c45.predict(X_all_scaled)
        all_probs = model_c45.predict_proba(X_all_scaled)[:, 1]

        df_result = df_raw.copy()
        df_result['Hasil_Prediksi_Numerik'] = all_preds
        df_result['Hasil_Prediksi'] = df_result['Hasil_Prediksi_Numerik'].map(
            {i: name for i, name in enumerate(class_names)}
        )
        df_result['Probabilitas_Dropout'] = all_probs

        # KPI ringkasan prediksi batch
        batch_total = len(df_result)
        batch_counts = df_result['Hasil_Prediksi'].value_counts()

        batch_cols = st.columns(len(class_names) + 1)
        with batch_cols[0]:
            st.metric("Total Data", f"{batch_total} Siswa")
        for i, name in enumerate(class_names):
            cnt = batch_counts.get(name, 0)
            with batch_cols[i+1]:
                pct = cnt/batch_total*100 if batch_total > 0 else 0
                st.metric(f"Prediksi {name}", f"{cnt} Siswa", f"{pct:.1f}%")

        # Visualisasi distribusi prediksi batch
        batch_labels = batch_counts.index.tolist()
        batch_values = batch_counts.values.tolist()
        batch_colors = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12'][:len(batch_labels)]

        batch_vis1, batch_vis2 = st.columns(2)

        with batch_vis1:
            st.markdown("##### Distribusi Hasil Prediksi (Seluruh Data)")
            fig_bp, ax_bp = plt.subplots(figsize=(7, 5))
            bars_bp = ax_bp.bar(batch_labels, batch_values, color=batch_colors, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars_bp, batch_values):
                ax_bp.text(bar.get_x() + bar.get_width()/2., bar.get_height() + batch_total*0.01,
                          f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            ax_bp.set_ylabel('Jumlah Siswa', fontsize=11)
            ax_bp.set_title('Distribusi Hasil Prediksi (Seluruh Data)', fontsize=12, fontweight='bold')
            ax_bp.spines['top'].set_visible(False)
            ax_bp.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_bp)
            plt.close(fig_bp)

        with batch_vis2:
            st.markdown("##### Proporsi Hasil Prediksi")
            fig_bpie, ax_bpie = plt.subplots(figsize=(7, 5))
            wedges, texts, autotexts = ax_bpie.pie(
                batch_values, labels=batch_labels, colors=batch_colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
            ax_bpie.axis('equal')
            ax_bpie.set_title('Proporsi Hasil Prediksi', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_bpie)
            plt.close(fig_bpie)

        st.dataframe(df_result, use_container_width=True)

        towrite = io.BytesIO()
        df_result.to_csv(towrite, index=False, encoding='utf-8')
        towrite.seek(0)

        st.download_button(
            label="📥 Unduh Hasil Prediksi Lengkap (CSV)",
            data=towrite,
            file_name="hasil_prediksi_data_primer.csv",
            mime="text/csv"
        )

        # ── SIMPAN KE DATABASE ──
        if is_new_run:
            import json as json_lib
            db.save_prediction_history(
                run_by=st.session_state.user_id,
                dataset_name=dataset_name,
                mode_analisis='Data Primer (Train On-the-fly)',
                accuracy=float(acc * 100),
                precision=float(prec * 100),
                recall=float(rec * 100),
                f1_score=float(f1 * 100),
                config_json=json_lib.dumps({
                    'max_depth': max_depth,
                    'use_ig_selection': use_ig_selection,
                    'ig_threshold': ig_threshold,
                    'test_size': test_size,
                    'features': selected_train_features,
                    'target': target_col
                })
            )
            st.toast("✅ Hasil prediksi berhasil disimpan ke riwayat!")

# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN APP FLOW ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.logged_in:
    show_login_page()
else:
    show_sidebar()

    page = st.session_state.current_page
    role = st.session_state.role

    if page == 'Dashboard Riwayat':
        show_dashboard_riwayat()
    elif page == 'Konfigurasi Prediksi':
        if role in ['Admin', 'BK']:
            show_prediction_config()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    elif page == 'Hasil Prediksi':
        show_prediction_results()
    elif page == 'Upload File':
        if role in ['Admin', 'Wali Kelas', 'BK']:
            show_upload_page()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    elif page == 'Manajemen File':
        if role in ['Admin', 'Wali Kelas', 'BK']:
            show_file_management_page()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    elif page == 'Manajemen User':
        if role == 'Admin':
            show_user_management_page()
        else:
            st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
    else:
        show_dashboard_riwayat()
