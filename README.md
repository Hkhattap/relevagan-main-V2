# 🛡️ Adaptive Hybrid IDS: GANs, DRL, and XAI Framework
### Next-Gen Intrusion Detection for Zero-Day & Adversarial Threats (2022-2025)

---

## 📌 Project Overview
As IoT and Cloud infrastructures expand, traditional IDS struggle against sophisticated **Zero-Day** and **Adversarial attacks**. This repository presents a **Two-Layer Adaptive Framework** that integrates **Generative Adversarial Networks (GANs)**, **Deep Reinforcement Learning (DRL)**, and **Explainable AI (XAI)** to create a resilient and interpretable security engine.

This project is an advanced evolution of the **RELEVAGAN** architecture, optimized for modern cyber-threat landscapes.

---

## 🏗️ Architecture: A Two-Layer Defense

### Layer 1: Adversarial Intelligence (GAN + DRL)
* **The Generator:** Synthesizes realistic network traffic to overcome dataset imbalance.
* **The DRL Agent:** Acts as an "Intelligent Attacker," learning optimal evasion strategies and simulating evolving botnet behaviors.

### Layer 2: Robust Detection (Hardened Classifier)
* **Hybrid Engines:** Utilizes **CNNs, LSTMs, and Transformers** to analyze spatial and sequential traffic patterns.
* **Adversarial Training:** Integrates crafted perturbations into the training cycle, making the classifier immune to evasion attempts.

---

## 🚀 Key Features
* **Interpretability (XAI):** Integrated **SHAP & LIME** to provide transparency in threat classification.
* **Multi-Dataset Benchmarking:** Validated on **CICIDS2017, NSL-KDD, UNSW-NB15, and N-BaIoT.**
* **Zero-Day Resilience:** Specifically engineered to detect unseen attack patterns.

---

## 🛠️ Technical Stack & Requirements
* **Languages:** Python 3.x
* **Frameworks:** TensorFlow 2.13, Keras 2.13, Gymnasium, Stable-Baselines3.
* **Tools:** NumPy, Pandas, Scikit-learn, XGBoost, SHAP.

### Installation
```bash
pip install -r requirements.txt
