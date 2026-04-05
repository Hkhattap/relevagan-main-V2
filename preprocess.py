##  4️⃣ preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, label_col="Label", zero_day_attack_name="Bot"):
    """
// Prepare data and fix label casing (Benign vs BENIGN).
"""
    df = df.copy()

    # 1. // Data cleaning (handle infinite and missing values)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 2. // Normalize Label column casing (Critical step).
    # // Convert all to UPPERCASE and trim extra spaces.
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    
    #// Convert target Zero-Day to UPPERCASE for successful comparison.
    zero_day_upper = str(zero_day_attack_name).strip().upper()

    # 3. // Encode labels: 0 for normal and 1 for attack.
    # // Standardize all "Benign" variations to 0.                                                                 دلوقتى "BENIGN" أو "Benign" أو "benign" كلهم هيتحولوا لـ 0
    df['Is_Attack'] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)

    # 4.// Split data for Zero-Day testing scenario.
    # // Use all data for training except "Bot" samples.
    train_df = df[df[label_col] != zero_day_upper]
    
    # // Test on full dataset to evaluate model detection of unseen "Bot" attacks.
    test_df = df.copy()

    # 5. // Filter numerical data and remove labels.                            اختيار الأعمدة الرقمية فقط واستبعاد الـ Labels
    #// Drop non-numerical columns (like Timestamp) to avoid Scaler errors.
    features = df.select_dtypes(include=["float64", "int64"]).columns
    features = [f for f in features if f not in ['Is_Attack', label_col, 'Timestamp']]

    X_train_raw = train_df[features]
    y_train = train_df['Is_Attack']
    
    X_test_raw = test_df[features]
    y_test = test_df['Is_Attack']

    # 6. // Apply Min-Max scaling to all features to the range [0, 1].
    scaler = MinMaxScaler()
    
    # Fit Scaler on Training data only
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # // Apply (Transform) the Scaler to the Test data.
    X_test_scaled = scaler.transform(X_test_raw)

    print(f"[INFO] Zero-Day Attack isolated: {zero_day_upper}")
    print(f"[INFO] Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    print(f"[INFO] Classes in Train: {np.unique(y_train)}")

    return X_train_scaled, y_train.values, X_test_scaled, y_test.values, scaler
