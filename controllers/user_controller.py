"""
controllers/user_controller.py
Logika bisnis manajemen pengguna (hanya untuk peran BK).
"""

import db


def get_all_users() -> list:
    """Ambil semua pengguna dari database."""
    return db.get_all_users()


def get_user_by_id(user_id: int) -> dict:
    """Ambil data satu pengguna berdasarkan ID."""
    return db.get_user_by_id(user_id)


def add_user(username: str, password: str, nama_lengkap: str, role: str) -> bool:
    """
    Tambah pengguna baru.
    Returns True jika berhasil, False jika username sudah ada.
    """
    return db.create_user(username, password, nama_lengkap, role)


def edit_user(user_id: int, nama_lengkap: str = None,
              role: str = None, password: str = None) -> bool:
    """Update data pengguna. Hanya field yang diberikan yang diperbarui."""
    return db.update_user(user_id, nama_lengkap=nama_lengkap,
                          role=role, password=password)


def remove_user(user_id: int) -> bool:
    """Hapus pengguna berdasarkan ID. Returns True jika berhasil."""
    return db.delete_user(user_id)
