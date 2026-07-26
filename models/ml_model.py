"""
models/ml_model.py
Logika Machine Learning: load artifacts UCI, preprocessing, training C4.5, evaluasi.
"""

import pickle
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


@st.cache_resource
def load_uci_artifacts():
    """Load model artifacts pre-trained (UCI): model, scaler, fitur terpilih."""
    try:
        model = pickle.load(open('model_c45_dropout.pkl', 'rb'))
        scaler = pickle.load(open('scaler_dropout.pkl', 'rb'))
        features = pickle.load(open('selected_features.pkl', 'rb'))
        return model, scaler, features
    except Exception:
        return None, None, None


def preprocess_primary_data(df_raw: pd.DataFrame, target_col: str, selected_features: list):
    """
    Preprocessing untuk mode Data Primer:
    - Isi missing value (median untuk numerik, modus untuk kategorikal)
    - Drop baris dengan target NaN
    - Encode target (biner atau multi-kelas)
    - Encode fitur kategorikal dengan factorize

    Returns:
        X_numeric (DataFrame), y (ndarray), class_names (list), categorical_cols (list)
        atau None jika gagal.
    """
    df_model = df_raw.copy()

    # Isi missing value pada fitur
    for col in selected_features:
        if df_model[col].isnull().sum() > 0:
            if df_model[col].dtype in ['int64', 'float64']:
                df_model[col].fillna(df_model[col].median(), inplace=True)
            else:
                df_model[col].fillna(df_model[col].mode()[0], inplace=True)

    # Hapus baris dengan target NaN
    y_raw = df_model[target_col].copy()
    nan_mask = y_raw.isna()
    if nan_mask.sum() > 0:
        st.warning(f"⚠️ Ditemukan {nan_mask.sum()} baris dengan target kosong (NaN). Baris-baris tersebut akan dihapus.")
        df_model = df_model[~nan_mask].reset_index(drop=True)
        y_raw = df_model[target_col].copy()

    unique_targets = y_raw.unique()

    # Encode target
    dropout_val = None
    for val in unique_targets:
        if str(val).strip().lower() == 'dropout':
            dropout_val = val
            break

    if dropout_val is not None:
        y = y_raw.apply(lambda x: 1 if x == dropout_val else 0).values
        class_names = ['Non-Dropout', 'Dropout']
    elif set(str(v) for v in unique_targets) <= {'0', '1', '0.0', '1.0'}:
        y = y_raw.apply(lambda x: int(float(x))).values
        class_names = ['Non-Dropout', 'Dropout']
    else:
        sorted_unique = sorted([str(v) for v in unique_targets])
        label_map = {val: idx for idx, val in enumerate(sorted_unique)}
        y = y_raw.astype(str).map(label_map).values
        class_names = sorted_unique
        if pd.isna(y).any():
            st.error("❌ Gagal mengkodekan kolom target. Pastikan kolom target tidak memiliki nilai yang tidak terduga.")
            return None, None, None, None

    # Encode fitur kategorikal
    X_all = df_model[selected_features]
    X_numeric = X_all.copy()
    categorical_cols = []
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
            X_numeric[col] = pd.factorize(X_numeric[col])[0]
            categorical_cols.append(col)

    return X_numeric, y, class_names, categorical_cols


def apply_information_gain_selection(X_numeric: pd.DataFrame, y: np.ndarray,
                                      ig_threshold: float):
    """
    Seleksi fitur berbasis Information Gain.
    Returns: (X_numeric_filtered, ig_df) — ig_df berisi skor semua fitur.
    """
    ig_scores = mutual_info_classif(X_numeric, y, random_state=42)
    ig_df = pd.DataFrame({
        'Fitur': X_numeric.columns,
        'Information Gain': ig_scores
    }).sort_values('Information Gain', ascending=False).reset_index(drop=True)

    selected_by_ig = ig_df[ig_df['Information Gain'] >= ig_threshold]['Fitur'].tolist()

    if len(selected_by_ig) == 0:
        return X_numeric, ig_df  # Kembalikan semua jika tidak ada yang lolos

    return X_numeric[selected_by_ig], ig_df


def train_c45_model(X_numeric: pd.DataFrame, y: np.ndarray,
                    test_size: float, max_depth: int):
    """
    Train C4.5 model (DecisionTreeClassifier dengan criterion='entropy').
    Returns dict berisi: model, scaler, split data, metrik evaluasi.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_numeric.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_numeric.columns, index=X_test.index)

    model_c45 = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model_c45.fit(X_train_scaled, y_train)

    return {
        'model': model_c45,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
    }


def evaluate_model(model, X_test, y_test, class_names: list):
    """
    Hitung semua metrik evaluasi: akurasi, presisi, recall, F1.
    Returns dict metrik dan prediksi.
    """
    is_binary = len(class_names) == 2
    avg_method = 'binary' if is_binary else 'weighted'

    y_pred = model.predict(X_test)
    y_pred_proba = (model.predict_proba(X_test)[:, 1]
                    if is_binary
                    else model.predict_proba(X_test))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0, average=avg_method)
    rec = recall_score(y_test, y_pred, zero_division=0, average=avg_method)
    f1 = f1_score(y_test, y_pred, zero_division=0, average=avg_method)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    return {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm,
        'report': report,
        'is_binary': is_binary,
    }


def run_cross_validation(X_numeric: pd.DataFrame, y: np.ndarray,
                         max_depth: int, is_binary: bool, scaler):
    """
    Jalankan 10-Fold Cross Validation.
    Returns dict cv scores: accuracy, precision, recall, f1.
    """
    from sklearn.metrics import make_scorer

    cv_model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    X_all_scaled_cv = scaler.fit_transform(X_numeric)

    cv_accuracy = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='accuracy')

    if is_binary:
        cv_precision = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='precision')
        cv_recall = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='recall')
        cv_f1 = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10, scoring='f1')
    else:
        cv_precision = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10,
                                       scoring=make_scorer(precision_score, average='weighted', zero_division=0))
        cv_recall = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10,
                                    scoring=make_scorer(recall_score, average='weighted', zero_division=0))
        cv_f1 = cross_val_score(cv_model, X_all_scaled_cv, y, cv=10,
                                scoring=make_scorer(f1_score, average='weighted', zero_division=0))

    return {
        'accuracy': cv_accuracy,
        'precision': cv_precision,
        'recall': cv_recall,
        'f1': cv_f1,
    }
