"""
models/file_utils.py
Utilitas untuk membaca file dataset dan format tampilan.
"""

import pandas as pd
import streamlit as st
from datetime import datetime


def load_file_from_path(file_path: str):
    """Load dataframe dari path file yang tersimpan (CSV atau Excel)."""
    try:
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
            sep = ';' if ';' in first_line else ','
            df = pd.read_csv(file_path, sep=sep)
        else:
            df = pd.read_excel(file_path)

        if df is not None and not df.empty:
            # Isi kolom identitas yang kosong dengan nilai dummy agar baris data tidak terbuang
            df = df.reset_index(drop=True)
            if 'NISN' in df.columns:
                df['NISN'] = df['NISN'].astype(str).replace('nan', None).fillna(pd.Series([str(i) for i in range(len(df))]))
            if 'NIS' in df.columns:
                df['NIS'] = df['NIS'].astype(str).replace('nan', None).fillna(pd.Series([ str(i) for i in range(len(df))]))
            if 'Nama' in df.columns:
                df['Nama'] = df['Nama'].astype(str).replace('nan', None).fillna(pd.Series([ str(i) for i in range(len(df))]))

            # Auto derive Status column if missing
            if 'Status' not in df.columns and 'Status_DO' not in df.columns:
                import numpy as np
                cond_do = pd.Series(False, index=df.index)
                if 'Kehadiran_Persen' in df.columns:
                    cond_do = cond_do | (df['Kehadiran_Persen'] < 85)
                if 'Sikap_Jumlah_Panggilan_BK' in df.columns:
                    cond_do = cond_do | (df['Sikap_Jumlah_Panggilan_BK'] >= 2)
                if 'Nilai_Rata_Rata' in df.columns:
                    cond_do = cond_do | (df['Nilai_Rata_Rata'] < 75)
                df['Status_DO'] = np.where(cond_do, 'Dropout', 'Non-Dropout')

        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """Format ukuran file ke format yang mudah dibaca (B / KB / MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_datetime(dt_str: str) -> str:
    """Format datetime string ke format Indonesia (dd Bulan yyyy, HH:MM WIB)."""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d %B %Y, %H:%M WIB")
    except Exception:
        return dt_str
