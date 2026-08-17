

 Lampiran 1: Pembersihan & Pra-pemrosesan Data (*Data Preprocessing*)

python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col='Status'):
    # 1. Penanganan Missing Value (Imputasi Median & Modus)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # 2. Pembersihan & Pengkodean Target (Target Encoding)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    y_raw = df[target_col].astype(str)
    sorted_unique = sorted(list(y_raw.unique()))
    label_map = {val: idx for idx, val in enumerate(sorted_unique)}
    y = y_raw.map(label_map).values

    # 3. Label Encoding untuk Fitur Kategorikal
    X = df.drop(columns=[target_col]).copy()
    for col in X.columns:
        if X[col].dtype in ['object', 'category']:
            X[col] = pd.factorize(X[col])[0]
            
    return X, y, sorted_unique


---

 Lampiran 2: Seleksi Fitur (*Information Gain*)

python
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

def select_features_ig(X, y, threshold=0.05):
    # Perhitungan Information Gain berbasis Entropy (Mutual Information)
    ig_scores = mutual_info_classif(X, y, random_state=42)
    
    ig_df = pd.DataFrame({
        'Fitur': X.columns,
        'Information_Gain': ig_scores
    }).sort_values('Information_Gain', ascending=False)
    
    # Filter fitur berdasarkan ambang batas (threshold)
    selected_features = ig_df[ig_df['Information_Gain'] >= threshold]['Fitur'].tolist()
    return X[selected_features], ig_df


---

 Lampiran 3: Pembagian Data & Pelatihan Algoritma C4.5

python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. Pembagian Data Latih (80%) dan Uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Standarisasi Fitur (Z-Score Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Inisialisasi & Pelatihan Pohon Keputusan C4.5 (Entropy Criterion)
model_c45 = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)
model_c45.fit(X_train_scaled, y_train)


---

 Lampiran 4: Evaluasi Kinerja Model & 10-Fold Cross Validation

python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# 1. Prediksi Data Uji
y_pred = model_c45.predict(X_test_scaled)
avg_type = 'binary' if len(set(y)) == 2 else 'weighted'

# 2. Perhitungan Metrik Evaluasi
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
rec = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)

# 3. Evaluasi 10-Fold Cross-Validation
cv_scores = cross_val_score(model_c45, X_train_scaled, y_train, cv=10, scoring='accuracy')
mean_cv_acc = cv_scores.mean()

