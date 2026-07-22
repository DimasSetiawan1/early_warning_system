"""
db.py - Module Database SQLite untuk Early Warning System
Menangani koneksi database, pembuatan tabel, seed data, dan fungsi CRUD.
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'early_warning.db')
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')


def get_connection():
    """Mendapatkan koneksi ke database SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password: str) -> str:
    """Hash password menggunakan SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def init_db():
    """Inisialisasi database: buat tabel dan seed data default."""
    # Pastikan folder uploads ada
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    # Buat tabel users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            nama_lengkap TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('BK', 'Guru')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Buat tabel uploaded_files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_by INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            description TEXT,
            FOREIGN KEY (uploaded_by) REFERENCES users(id)
        )
    ''')

    # Buat tabel prediction_history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_by INTEGER NOT NULL,
            dataset_name TEXT NOT NULL,
            mode_analisis TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            config_json TEXT,
            run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_by) REFERENCES users(id)
        )
    ''')

    conn.commit()

    # Seed default users jika belum ada
    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    count = cursor.fetchone()['cnt']

    if count == 0:
        default_users = [
            ('bk', hash_password('admin123'), 'Guru BK', 'BK'),
            ('guru', hash_password('admin123'), 'Guru Mata Pelajaran', 'Guru'),
        ]
        cursor.executemany(
            "INSERT INTO users (username, password, nama_lengkap, role) VALUES (?, ?, ?, ?)",
            default_users
        )
        conn.commit()

    conn.close()


# ── USER CRUD ──────────────────────────────────────────────────────────────────

def authenticate_user(username: str, password: str):
    """Verifikasi login. Return dict user jika berhasil, None jika gagal."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, hash_password(password))
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None


def create_user(username: str, password: str, nama_lengkap: str, role: str):
    """Tambah user baru. Return True jika berhasil, False jika username sudah ada."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password, nama_lengkap, role) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), nama_lengkap, role)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def get_all_users():
    """Ambil semua user. Return list of dict."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, nama_lengkap, role, created_at FROM users ORDER BY id")
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users


def get_user_by_id(user_id: int):
    """Ambil user berdasarkan ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, nama_lengkap, role, created_at FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None


def update_user(user_id: int, nama_lengkap: str = None, role: str = None, password: str = None):
    """Update data user. Hanya field yang diberikan yang akan diupdate."""
    conn = get_connection()
    cursor = conn.cursor()

    updates = []
    params = []

    if nama_lengkap is not None:
        updates.append("nama_lengkap = ?")
        params.append(nama_lengkap)
    if role is not None:
        updates.append("role = ?")
        params.append(role)
    if password is not None:
        updates.append("password = ?")
        params.append(hash_password(password))

    if not updates:
        conn.close()
        return False

    params.append(user_id)
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, params)
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def delete_user(user_id: int):
    """Hapus user berdasarkan ID. Return True jika berhasil."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


# ── FILE CRUD ──────────────────────────────────────────────────────────────────

def save_uploaded_file(filename: str, original_filename: str, file_path: str,
                       uploaded_by: int, file_size: int, description: str = None):
    """Simpan record file yang diupload ke database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO uploaded_files 
           (filename, original_filename, file_path, uploaded_by, file_size, description) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (filename, original_filename, file_path, uploaded_by, file_size, description)
    )
    conn.commit()
    file_id = cursor.lastrowid
    conn.close()
    return file_id


def get_uploaded_files(uploaded_by: int = None):
    """
    Ambil semua file yang diupload beserta info uploader.
    Jika uploaded_by diberikan, hanya ambil file milik user tersebut.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT uf.*, u.nama_lengkap as uploader_name, u.username as uploader_username, u.role as uploader_role
        FROM uploaded_files uf
        JOIN users u ON uf.uploaded_by = u.id
    """
    params = []

    if uploaded_by is not None:
        query += " WHERE uf.uploaded_by = ?"
        params.append(uploaded_by)

    query += " ORDER BY uf.uploaded_at DESC"
    cursor.execute(query, params)
    files = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return files


def get_file_by_id(file_id: int):
    """Ambil 1 file berdasarkan ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT uf.*, u.nama_lengkap as uploader_name, u.username as uploader_username
           FROM uploaded_files uf
           JOIN users u ON uf.uploaded_by = u.id
           WHERE uf.id = ?""",
        (file_id,)
    )
    file_record = cursor.fetchone()
    conn.close()
    if file_record:
        return dict(file_record)
    return None


def delete_uploaded_file(file_id: int):
    """Hapus record file dari database dan file fisik dari disk."""
    file_record = get_file_by_id(file_id)
    if file_record is None:
        return False

    # Hapus file fisik
    if os.path.exists(file_record['file_path']):
        os.remove(file_record['file_path'])

    # Hapus record dari database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


# ── PREDICTION HISTORY ─────────────────────────────────────────────────────────

def save_prediction_history(run_by: int, dataset_name: str, mode_analisis: str, accuracy: float, precision: float, recall: float, f1_score: float, config_json: str):
    """Simpan riwayat prediksi ke database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO prediction_history 
           (run_by, dataset_name, mode_analisis, accuracy, precision, recall, f1_score, config_json) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_by, dataset_name, mode_analisis, accuracy, precision, recall, f1_score, config_json)
    )
    conn.commit()
    hist_id = cursor.lastrowid
    conn.close()
    return hist_id


def get_prediction_history():
    """
    Ambil semua riwayat prediksi beserta info user.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT ph.*, u.nama_lengkap as run_by_name, u.role as run_by_role
        FROM prediction_history ph
        JOIN users u ON ph.run_by = u.id
        ORDER BY ph.run_at DESC
    """
    cursor.execute(query)
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history

