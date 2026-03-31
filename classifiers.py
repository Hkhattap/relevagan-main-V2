## 📁 4️⃣ classifiers.py (XGBoost + ROC)
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def train_xgb(X_train, y_train):
    """
    تدريب كلاسيفاير XGBoost. 
    استخدمنا أوزان تلقائية للتعامل مع عدم توازن البيانات (Imbalance) الشائع في هجمات الشبكات.
    """
    print("[INFO] Training XGBoost Classifier...")
    
    # نستخدم scale_pos_weight لو البيانات فيها عدد قليل من الهجمات مقارنة بالطبيعي
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    return model

def zero_day_eval(model, X_test, y_test):
    """
    تقييم الموديل على هجمات الـ Zero-Day (الهجمات التي لم تظهر في التدريب).
    """
    print("\n" + "="*30)
    print("      ZERO-DAY EVALUATION      ")
    print("="*30)

    # 1. التنبؤات
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # 2. طباعة التقارير الأساسية
    print("\n[Classification Report]:")
    print(classification_report(y_test, preds))
    
    auc_score = roc_auc_score(y_test, probs)
    print(f"AUC Score: {auc_score:.4f}")

    # 3. رسم المصفوفة المحيرة (Confusion Matrix)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Zero-Day Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 4. رسم منحنى الـ ROC Curve (مهم جداً للرسالة)
    plot_roc_curve(y_test, probs)

    return {"auc": auc_score, "report": classification_report(y_test, preds, output_dict=True)}

def plot_roc_curve(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, probs):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()