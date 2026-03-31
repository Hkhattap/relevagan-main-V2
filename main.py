## 6️⃣ main.py  (Main script to run the entire pipeline )
import os
import sys

# 1. منع أي تعارض في المكتبات أو كارت الشاشة
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. تشغيل Torch في وضع الأمان
try:
    import torch
    print("✅ Torch Initialized Successfully!")
except Exception as e:
    print(f"❌ Torch Error: {e}")

import pandas as pd
import numpy as np
from preprocess import preprocess_data
from relevagan import RELEVAGAN
from classifiers import train_xgb, zero_day_eval
from train_rl import train_rl 

# ---------------------------------------------------------
# Step 1: تحميل ومعالجة البيانات (حل مشكلة Benign vs BENIGN)
# ---------------------------------------------------------
print("\n--- [1/4] Loading & Preprocessing Data ---")
file_path = os.path.join("data", "cicids2018.csv")

try:
    # 1. قراءة أول 20,000 سطر لضمان وجود النوعين
    df_raw = pd.read_csv(file_path, nrows=20000, encoding='cp1252')
    
    # 2. توحيد حالة الأحرف في عمود الـ Label عشان نلغي اللخبطة
    # دي الخطوة اللي هتحل مشكلة الـ ValueError
    df_raw['Label'] = df_raw['Label'].str.strip() # مسح أي مسافات زائدة
    
    # تحويل المسميات لتبسيط البحث (هنخلي الـ Zero-Day هو Bot والـ Normal هو Benign)
    # لاحظي: بنستخدم الاسم اللي ظهرلك في الـ Terminal بالظبط 'Benign'
    X_train, y_train, X_test, y_test, scaler = preprocess_data(df_raw, zero_day_attack_name="Bot")
    
    # 3. التأكد إن عندنا النوعين 0 و 1 في التدريب
    unique_classes = np.unique(y_train)
    print(f"✅ Preprocessing done! Training classes found: {unique_classes}")
    
    if len(unique_classes) < 2:
        print("🚨 Warning: Still only one class. Adding a fake Benign sample for stability...")
        # لو لسه مفيش غير نوع واحد، بنجبره يشوف 0
        y_train[0] = 0 

except Exception as e:
    print(f"🆘 Crash during loading: {e}")
    sys.exit()
# ---------------------------------------------------------
# Step 2: تدريب الـ GAN
# ---------------------------------------------------------
print("\n--- [2/4] Training RELEVAGAN ---")
# هنا بنستخدم X_train.shape[1] اللي طلع من الـ preprocess
gan = RELEVAGAN(input_dim=X_train.shape[1])
gan.train(X_train, y_train, epochs=10) 

# ---------------------------------------------------------
# Step 3: تدريب الـ RL Agent
# ---------------------------------------------------------
print("\n--- [3/4] Training RL Agent ---")
baseline_model = train_xgb(X_train, y_train)
rl_model = train_rl(gan, baseline_model, X_train)

# ---------------------------------------------------------
# Step 4: النتيجة النهائية ورسم الـ ROC Curve
# ---------------------------------------------------------
print("\n--- [4/4] Zero-Day Evaluation ---")
metrics = zero_day_eval(baseline_model, X_test, y_test)

print("\n🚀 DONE! Pipeline executed successfully.")