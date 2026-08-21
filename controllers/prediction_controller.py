"""
controllers/prediction_controller.py
Logika bisnis konfigurasi dan penyimpanan hasil prediksi.
"""

import json
import streamlit as st
import db


def prepare_experiment_run(df_raw, selected_file: dict):
    """
    Simpan konfigurasi mode Eksperimen ke session_state dan arahkan ke halaman hasil.
    """
    st.session_state.run_config = {
        'mode': 'Eksperimen',
        'df_raw': df_raw,
        'dataset_name': selected_file['original_filename'],
        'is_saved': False,
    }
    st.session_state.current_page = 'Hasil Prediksi'
    st.rerun()


def prepare_primary_run(df_raw, selected_file: dict, config_params: dict):
    """
    Simpan konfigurasi mode Data Primer ke session_state dan arahkan ke halaman hasil.
    config_params harus berisi: target_col, selected_train_features, test_size,
                                max_depth, use_ig_selection, ig_threshold
    """
    st.session_state.run_config = {
        'mode': 'Primer',
        'df_raw': df_raw,
        'dataset_name': selected_file['original_filename'],
        'is_saved': False,
        **config_params,
    }
    st.session_state.current_page = 'Hasil Prediksi'
    st.rerun()


def prepare_pretrained_primary_run(selected_file: dict):
    """
    Simpan konfigurasi mode Pre-trained Primer (Model Sistem) ke session_state.
    """
    st.session_state.run_config = {
        'mode': 'Pre-trained Primer',
        'test_file': selected_file,
        'dataset_name': selected_file['original_filename'],
        'is_saved': False,
    }
    st.session_state.current_page = 'Hasil Prediksi'
    st.rerun()


def save_prediction_history(user_id: int, dataset_name: str, metrics: dict,
                             config_params: dict):
    """
    Simpan riwayat prediksi ke database.
    metrics: dict dengan kunci acc, prec, rec, f1 (nilai 0–100).
    config_params: dict konfigurasi yang akan diserialisasi sebagai JSON.
    """
    db.save_prediction_history(
        run_by=user_id,
        dataset_name=dataset_name,
        mode_analisis=config_params.get('mode_analisis', 'Unknown'),
        accuracy=metrics['acc'],
        precision=metrics['prec'],
        recall=metrics['rec'],
        f1_score=metrics['f1'],
        config_json=json.dumps(config_params.get('config_detail', {}))
    )
