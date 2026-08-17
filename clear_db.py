import sqlite3

def clear_database():
    conn = sqlite3.connect('early_warning.db')
    cur = conn.cursor()

    # Hapus semua riwayat prediksi
    cur.execute("DELETE FROM prediction_history")
    print(f"Berhasil menghapus {cur.rowcount} data riwayat prediksi.")

    # Hapus semua file yang diupload (di database)
    cur.execute("DELETE FROM uploaded_files")
    print(f"Berhasil menghapus {cur.rowcount} data file unggahan.")

    # Reset auto-increment counter supaya ID kembali mulai dari 1
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('prediction_history', 'uploaded_files')")

    conn.commit()
    conn.close()
    print("Database berhasil dibersihkan! (Tabel users/pengguna TIDAK dihapus).")

if __name__ == "__main__":
    clear_database()
