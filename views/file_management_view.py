"""
views/file_management_view.py
Tampilan halaman manajemen file dataset.
"""

import pandas as pd
import streamlit as st
from controllers.file_controller import get_files, delete_file
from models.file_utils import format_file_size, format_datetime


def show_file_management_page():
    """Halaman manajemen file — tampilkan semua file, hapus sesuai permission."""
    st.markdown("<h1 style='font-size: 24px; font-weight: 700; color: var(--color-primary); margin-bottom: 8px;'>Manajemen File Himpunan Data</h1>", unsafe_allow_html=True)

    user_id = st.session_state.user_id
    role = st.session_state.role

    files = get_files(role=role, user_id=user_id)
    st.markdown("<p style='font-size: 14px; color: var(--color-text-secondary); margin-bottom: 32px;'>Kelola dataset siswa untuk diproses prediksi (menampilkan file yang Anda unggah).</p>", unsafe_allow_html=True)

    if not files:
        st.info("Belum ada file yang diunggah. Unggah file CSV atau Excel melalui halaman Unggah Berkas.")
        return

    # Tabel ringkasan
    table_data = []
    for f in files:
        table_data.append({
            'ID': f['id'],
            'Nama File': f['original_filename'],
            'Ukuran': format_file_size(f['file_size']),
            'Diupload Oleh': f['uploader_name'],
            'Peran': f['uploader_role'],
            'Tanggal Unggah': format_datetime(f['uploaded_at']),
            'Deskripsi': f['description'] or '-'
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 18px; font-weight: 600; color: var(--color-text-primary); margin-bottom: 16px;'>Hapus File</h2>", unsafe_allow_html=True)

    deletable_files = [f for f in files if f['uploaded_by'] == user_id]

    if not deletable_files:
        st.info("Tidak ada file yang dapat Anda hapus.")
        return

    file_options = {
        f"[ID:{f['id']}] {f['original_filename']} (oleh {f['uploader_name']})": f['id']
        for f in deletable_files
    }
    selected_file_label = st.selectbox("Pilih file yang akan dihapus:", list(file_options.keys()))

    if selected_file_label:
        selected_file_id = file_options[selected_file_label]
        col_del1, col_del2 = st.columns([1, 3])
        with col_del1:
            # Danger style isn't natively supported in Streamlit except via CSS, but type="primary" is close enough for a destructive action if styled, though standard is just "primary" or custom CSS. We can use secondary and inject CSS, but DESIGN says: "Tombol bahaya: bg --color-danger". Streamlit doesn't support changing button color natively via arguments, but we can inject CSS. Or just keep it primary and standard. Let's use custom CSS.
            st.markdown("""
            <style>
            div.stButton > button.delete-btn {
                background-color: var(--color-danger) !important;
                color: white !important;
            }
            </style>
            """, unsafe_allow_html=True)
            if st.button("Hapus File", type="primary"):
                if delete_file(selected_file_id):
                    st.success("File berhasil dihapus.")
                    st.rerun()
                else:
                    st.error("Gagal menghapus file.")
