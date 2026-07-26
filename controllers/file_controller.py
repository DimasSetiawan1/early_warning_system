"""
controllers/file_controller.py
Logika bisnis manajemen file dataset: upload, ambil daftar, hapus.
"""

import os
import uuid
import streamlit as st
import db


def upload_file(uploaded_file, description: str, user_id: int):
    """
    Simpan file yang diunggah ke disk dan catat ke database.
    Returns: file_id (int) atau None jika gagal.
    """
    try:
        unique_filename = f"{uuid.uuid4().hex[:12]}_{uploaded_file.name}"
        file_path = os.path.join(db.UPLOAD_DIR, unique_filename)

        os.makedirs(db.UPLOAD_DIR, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        file_id = db.save_uploaded_file(
            filename=unique_filename,
            original_filename=uploaded_file.name,
            file_path=file_path,
            uploaded_by=user_id,
            file_size=uploaded_file.size,
            description=description if description else None
        )
        return file_id
    except Exception as e:
        st.error(f"Gagal mengunggah file: {e}")
        return None


def get_files(role: str = None, user_id: int = None):
    """
    Ambil daftar file sesuai hak akses:
    - BK: semua file (uploaded_by=None)
    - Guru: hanya file miliknya
    """
    if role == 'BK':
        return db.get_uploaded_files()
    return db.get_uploaded_files(uploaded_by=user_id)


def delete_file(file_id: int) -> bool:
    """
    Hapus file dari disk dan database.
    Returns True jika berhasil.
    """
    return db.delete_uploaded_file(file_id)
