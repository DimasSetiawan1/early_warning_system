"""
views/upload_view.py
Tampilan halaman unggah berkas dataset.
"""

import streamlit as st
from controllers.file_controller import upload_file, get_files
from models.file_utils import format_file_size, format_datetime


def show_upload_page():
    """Halaman unggah berkas himpunan data."""
    st.header("📤 Unggah Berkas Himpunan Data")
    st.markdown("Unggah file dataset siswa dalam format **CSV** atau **Excel (.xlsx)** untuk digunakan dalam analisis prediksi.")

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Pilih File Himpunan Data",
            type=["csv", "xlsx"],
            help="File berisi data siswa/mahasiswa untuk diprediksi."
        )
        description = st.text_area(
            "Deskripsi File (Opsional)",
            placeholder="Contoh: Data siswa kelas XII IPA semester genap 2024/2025",
            max_chars=500
        )
        submitted = st.form_submit_button("📤 Unggah Berkas", use_container_width=True)

        if submitted and uploaded_file is not None:
            file_id = upload_file(uploaded_file, description, st.session_state.user_id)
            if file_id:
                st.success(f"✅ File **{uploaded_file.name}** berhasil diupload! (ID: {file_id})")
                st.rerun()
        elif submitted and uploaded_file is None:
            st.warning("⚠️ Pilih file terlebih dahulu!")

    # Daftar file yang sudah diunggah
    st.markdown("---")
    st.subheader("📁 File Yang Sudah Anda Unggah")

    my_files = get_files(role=st.session_state.role, user_id=st.session_state.user_id)

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
                    st.markdown(f"**Diunggah oleh:** {f['uploader_name']} (`{f['uploader_username']}`)")
                    st.markdown(f"**Tanggal Unggah:** {format_datetime(f['uploaded_at'])}")
                    st.markdown(f"**ID File:** {f['id']}")
