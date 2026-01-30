# 🚀 Proje Kurulum Kılavuzu

## ⚡ Hızlı Başlangıç (Anaconda - ÖNERİLEN)

### Neden Anaconda?

- ✅ NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch **zaten kurulu**
- ✅ Sadece 4 paket yüklersiniz (30 saniye)
- ✅ Uyumluluk sorunu yok

---

## 📋 Kurulum Adımları (3 Adım)

### 1️⃣ Anaconda Prompt Açın

**Windows Başlat** → "Anaconda Prompt" yazın → **Enter**

### 2️⃣ Proje Klasörüne Gidin

```bash
cd "C:\proje\klasörü\yolu"
```

_(Kendi proje yolunuzu yazın)_

### 3️⃣ Eksik Paketleri Yükleyin

```bash
pip install yfinance lightgbm xgboost ta
```

⏱️ **30 saniye** - Tamamdır!

---

## 🎯 Projeyi Çalıştırma

### 4️⃣ Jupyter Başlatın

```bash
jupyter notebook
```

### 5️⃣ Herhangi Bir Modeli Çalıştırın

**Tarayıcıda Jupyter açılacak:**

1. Bir model klasörünü açın (örn: `22040101024_ÖmerAvcı_LightGBM`)
2. `main.ipynb` dosyasına tıklayın
3. **Cell** → **Run All** (veya her hücrede **Shift+Enter**)

**✅ İşte bu kadar! Tüm modeller çalışacak.**

---

## 🔧 Test (Opsiyonel)

Kurulumu test etmek isterseniz:

```bash
python test_kurulum.py
```

---

## 📦 Çalışan Modeller

Tüm modeller sıfırdan çalışır (internet gerekli):

| Model            | Klasör                                       | Süre      |
| ---------------- | -------------------------------------------- | --------- |
| **LightGBM**     | `22040101024_ÖmerAvcı_LightGBM/`             | ~5-10 dk  |
| **XGBoost**      | `22040101047_İlkayGökbudak_XGBoost/`         | ~10-15 dk |
| **PyTorch GRU**  | `22040101038_BerkantŞimşek_PyTorchGRU/`      | ~15-20 dk |
| **PyTorch LSTM** | `22040101112_BarchınoyKodırova_PyTorchLSTM/` | ~20-30 dk |
| **Hybrid**       | `Hybrid-BTC-Prediction_*/`                   | ~30-45 dk |

Her model:

- ✅ Otomatik Bitcoin verisi indirir (yfinance)
- ✅ Modeli sıfırdan eğitir
- ✅ Tahminler yapar
- ✅ Grafikler ve CSV oluşturur

---

## ⚠️ Sorun Giderme

### "ModuleNotFoundError" Hatası

```bash
# Paketleri yeniden yükleyin
pip install yfinance lightgbm xgboost ta
```

### PowerShell Hatası (Windows)

Anaconda Prompt yerine **CMD** veya **Anaconda Prompt** kullanın.

### Jupyter Açılmıyor

```bash
pip install --upgrade jupyter notebook
jupyter notebook
```

---

## 🔄 Alternatif: Python (Virtual Environment)

<details>
<summary><b>Anaconda yoksa tıklayın</b></summary>

### Windows (CMD):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
jupyter notebook
```

### Linux/Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

**Not:** Bu yöntem daha uzun sürer (~3-5 dakika)

</details>

---

## 💡 İpuçları

- **İlk Test:** LightGBM ile başlayın (en hızlı)
- **GPU:** Varsa PyTorch otomatik kullanır
- **İnternet:** İlk çalıştırmada veri indirir
- **Kaydetme:** Modeller ve sonuçlar otomatik kaydedilir

---

**🎉 Başarılı Tahminler!**
