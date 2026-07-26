"""
views/user_management_view.py
Tampilan halaman manajemen pengguna (hanya untuk peran BK).
"""

import pandas as pd
import streamlit as st
from controllers.user_controller import (
    get_all_users, get_user_by_id, add_user, edit_user, remove_user
)
from models.file_utils import format_datetime


def show_user_management_page():
    """Halaman manajemen user — hanya untuk BK."""
    st.header("👥 Manajemen User")

    if st.session_state.role != 'BK':
        st.error("⛔ Anda tidak memiliki akses ke halaman ini.")
        return

    users = get_all_users()

    # ── Daftar User ──
    st.subheader("📋 Daftar User Terdaftar")
    if users:
        table_data = []
        for u in users:
            table_data.append({
                'ID': u['id'],
                'Nama Pengguna': u['username'],
                'Nama Lengkap': u['nama_lengkap'],
                'Peran': u['role'],
                'Tanggal Dibuat': format_datetime(u['created_at']) if u['created_at'] else '-'
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Tambah User Baru ──
    st.markdown("---")
    st.subheader("➕ Tambah User Baru")

    with st.form("add_user_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Nama Pengguna", placeholder="Masukkan nama pengguna baru")
            new_password = st.text_input("Kata Sandi", type="password", placeholder="Masukkan kata sandi")
        with col2:
            new_nama = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap")
            new_role = st.selectbox("Peran", options=['BK', 'Guru'])

        if st.form_submit_button("➕ Tambah User", use_container_width=True):
            if not new_username or not new_password or not new_nama:
                st.error("⚠️ Semua field harus diisi!")
            else:
                if add_user(new_username, new_password, new_nama, new_role):
                    st.success(f"✅ User **{new_username}** ({new_role}) berhasil ditambahkan!")
                    st.rerun()
                else:
                    st.error(f"❌ Nama Pengguna **{new_username}** sudah digunakan!")

    # ── Edit User ──
    st.markdown("---")
    st.subheader("✏️ Edit User")

    editable_users = [u for u in users if u['id'] != st.session_state.user_id]
    if not editable_users:
        st.info("Tidak ada user lain yang bisa diedit.")
    else:
        user_options = {
            f"[{u['id']}] {u['username']} - {u['nama_lengkap']} ({u['role']})": u['id']
            for u in editable_users
        }
        selected_user_label = st.selectbox("Pilih User:", list(user_options.keys()), key="edit_user_select")

        if selected_user_label:
            selected_user_id = user_options[selected_user_label]
            selected_user = get_user_by_id(selected_user_id)

            if selected_user:
                with st.form("edit_user_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        edit_nama = st.text_input("Nama Lengkap", value=selected_user['nama_lengkap'])
                        edit_role = st.selectbox(
                            "Peran",
                            options=['BK', 'Guru'],
                            index=['BK', 'Guru'].index(selected_user['role'])
                            if selected_user['role'] in ['BK', 'Guru'] else 0
                        )
                    with col2:
                        edit_password = st.text_input(
                            "Kata Sandi Baru (kosongkan jika tidak diubah)",
                            type="password",
                            placeholder="Biarkan kosong jika tidak ingin mengubah"
                        )

                    if st.form_submit_button("💾 Simpan Perubahan", use_container_width=True):
                        edit_user(
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
        del_user_options = {
            f"[{u['id']}] {u['username']} - {u['nama_lengkap']} ({u['role']})": u['id']
            for u in deletable_users
        }
        selected_del_label = st.selectbox(
            "Pilih User yang akan dihapus:", list(del_user_options.keys()), key="del_user_select"
        )

        if selected_del_label:
            selected_del_id = del_user_options[selected_del_label]
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1:
                if st.button("🗑️ Hapus User", type="primary"):
                    remove_user(selected_del_id)
                    st.success("✅ User berhasil dihapus!")
                    st.rerun()
