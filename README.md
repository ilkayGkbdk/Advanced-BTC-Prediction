# 🚀 Bitcoin Price Prediction - Multi-Model Project

Advanced Bitcoin fiyat tahmin projesi - 5 farklı makine öğrenmesi ve deep learning modeli

---

## 🎯 Hızlı Başlangıç (1 Dakika!)

### ✅ Anaconda ile Kurulum (ÖNERİLEN - En Kolay)

**1. Anaconda Prompt açın**

```bash
# Proje dizinine gidin
cd "C:\xampp\htdocs\Advanced-BTC-Prediction-main - Kopya"

# Sadece 4 eksik paketi yükleyin (30 saniye)
pip install yfinance lightgbm xgboost ta

# Jupyter başlatın
jupyter notebook
```

**İşte bu kadar!** Herhangi bir `main.ipynb` dosyasını açıp çalıştırın.

**Anaconda'da zaten var:** NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, Jupyter

---

### 🔧 Alternatif: Python Virtual Environment

<details>
<summary>Anaconda yoksa bu yöntemi kullanın (tıklayın)</summary>

**Windows (CMD):**

```cmd
# 1. Virtual environment oluşturun
python -m venv .venv

# 2. Aktifleştirin
.venv\Scripts\activate.bat

# 3. Paketleri yükleyin
pip install -r requirements.txt

# 4. Jupyter başlatın
jupyter notebook
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

</details>

---

## 📊 Modeller

### 1️⃣ LightGBM (Gradient Boosting)

- 📂 `22040101024_ÖmerAvcı_LightGBM/main.ipynb`
- ⚡ Hızlı eğitim (~5-10 dakika)
- 🎯 Feature importance analizi

### 2️⃣ XGBoost (Extreme Gradient Boosting)

- 📂 `22040101047_İlkayGökbudak_XGBoost/main.ipynb`
- 🔥 Güçlü performans
- 📊 Learning curve analizi

### 3️⃣ PyTorch GRU (Gated Recurrent Unit)

- 📂 `22040101038_BerkantŞimşek_PyTorchGRU/main.ipynb`
- 🧠 Deep Learning
- 🔄 Sequence modeling

### 4️⃣ PyTorch LSTM (Long Short-Term Memory)

- 📂 `22040101112_BarchınoyKodırova_PyTorchLSTM/main.ipynb`
- 🎯 Stacked LSTM katmanları
- 📉 Early stopping

### 5️⃣ Hybrid Model (Gelişmiş Ensemble)

- 📂 `Hybrid-BTC-Prediction_*/main_pipeline.ipynb`
- 🎲 Monte Carlo simülasyon
- 🔄 Walk-Forward validation

---

## ⚙️ Sistem Gereksinimleri

- **Python:** 3.8+ (Anaconda önerilir)
- **Paketler:** requirements.txt (otomatik yüklenir)
- **İnternet:** İlk çalıştırmada veri indirmek için
- **Disk:** ~500 MB (modeller ve veriler için)

**Detaylı kurulum:** [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) dosyasına bakın.

---

## 📦 Proje İçeriği

Tüm modeller tamamen çalışır durumda ve sıfırdan eğitilebilir:

- ✅ **5 farklı model** (LightGBM, XGBoost, GRU, LSTM, Hybrid)
- ✅ **Otomatik veri indirme** (yfinance)
- ✅ **requirements.txt** (tek komutla kurulum)
- ✅ **Test scripti** (test_kurulum.py)
- ✅ **Detaylı dokümantasyon** (SETUP_INSTRUCTIONS.md)

---

## 🎓 Proje Özellikleri

- 📊 Bitcoin fiyat tahmini (LSTM, GRU, LightGBM, XGBoost)
- 🔄 Time series analysis
- 📈 Feature engineering (lag features, teknik göstergeler)
- 🎯 Hyperparameter optimization
- 📉 Model evaluation (RMSE, MAE, MAPE, R²)
- 📊 Görselleştirme (matplotlib, seaborn)

---

## 👥 Katkıda Bulunanlar

- **22040101024** - Ömer Avcı (LightGBM + Hybrid)
- **22040101038** - Berkant Şimşek (PyTorch GRU)
- **22040101047** - İlkay Gökbudak (XGBoost)
- **22040101112** - Barchınoy Kodırova (PyTorch LSTM + Hybrid)
- **RAM:** 4GB+ (önerilir)
- **İnternet:** Veri indirmek için gerekli

---

**🚀 Başarılı Tahminler!**
