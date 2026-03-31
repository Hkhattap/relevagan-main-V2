## 🔥 4️⃣ preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, label_col="Label", zero_day_attack_name="Bot"):
    """
    تجهيز البيانات وحل مشكلة حالة الأحرف (Benign vs BENIGN).
    """
    df = df.copy()

    # 1. تنظيف البيانات (التعامل مع القيم اللانهائية والمفقودة)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 2. توحيد حالة الأحرف في عمود الـ Label (أهم خطوة)
    # بنحول كل حاجة لـ UPPERCASE وبنمسح المسافات الزايدة
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    
    # تحويل الـ Zero-Day المستهدف لـ UPPERCASE برضه عشان المقارنة تنجح
    zero_day_upper = str(zero_day_attack_name).strip().upper()

    # 3. تحويل الـ Label لأرقام (0 للطبيعي، 1 للهجوم)
    # دلوقتى "BENIGN" أو "Benign" أو "benign" كلهم هيتحولوا لـ 0
    df['Is_Attack'] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)

    # 4. تقسيم البيانات لنظام الـ Zero-Day
    # التدريب يكون على البيانات اللي مش "Bot" (الطبيعي + باقي الهجمات)
    train_df = df[df[label_col] != zero_day_upper]
    
    # الاختبار يكون على الداتا كلها (عشان نقيم قدرة الموديل على كشف الـ Bot المستخبي)
    test_df = df.copy()

    # 5. اختيار الأعمدة الرقمية فقط واستبعاد الـ Labels
    # بنشيل أي عمود غير رقمي (زي الـ Timestamp) عشان الـ Scaler ميزعلش
    features = df.select_dtypes(include=["float64", "int64"]).columns
    features = [f for f in features if f not in ['Is_Attack', label_col, 'Timestamp']]

    X_train_raw = train_df[features]
    y_train = train_df['Is_Attack']
    
    X_test_raw = test_df[features]
    y_test = test_df['Is_Attack']

    # 6. عمل Scaling (تحجيم البيانات بين 0 و 1)
    scaler = MinMaxScaler()
    
    # الـ Scaler بيتعلم من الـ Train فقط (مهم جداً للبحث العلمي)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # وبيطبق على الـ Test
    X_test_scaled = scaler.transform(X_test_raw)

    print(f"[INFO] Zero-Day Attack isolated: {zero_day_upper}")
    print(f"[INFO] Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    print(f"[INFO] Classes in Train: {np.unique(y_train)}")

    return X_train_scaled, y_train.values, X_test_scaled, y_test.values, scaler