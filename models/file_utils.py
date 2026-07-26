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
