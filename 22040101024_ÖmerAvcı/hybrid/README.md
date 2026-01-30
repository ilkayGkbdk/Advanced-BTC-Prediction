# 🚀 Hybrid Bitcoin Price Prediction Pipeline

Profesyonel bir Bitcoin fiyat tahmini projesi. LightGBM ve LSTM modellerini karşılaştırarak 30 günlük tahmin yapar.

## 📋 Özellikler

- ✅ **Çoklu Veri Kaynağı**: OHLCV, makroekonomik veriler, teknik indikatörler, sentiment
- ✅ **Log Returns**: Fiyat gürültüsünü azaltır
- ✅ **TimeSeriesSplit**: Veri sızıntısını engeller
- ✅ **LightGBM & LSTM**: İki güçlü model karşılaştırması
- ✅ **Recursive Forecasting**: 30 günlük özyinelemeli tahmin
- ✅ **Feature Importance**: Hangi faktör daha etkili?
- ✅ **Modüler Mimari**: DRY prensibi, temiz kod

## 🏗️ Proje Yapısı

```
Hybrid-BTC-Prediction/
│
├── src/                          # Modüler kod
│   ├── data_loader.py           # Veri toplama (Yahoo, FRED, etc.)
│   ├── feature_engineering.py   # Özellik mühendisliği
│   ├── preprocessing.py         # Ön işleme
│   ├── models.py                # LightGBM & LSTM modelleri
│   ├── forecasting.py           # Özyinelemeli tahmin
│   └── visualization.py         # Görselleştirme
│
├── data/                         # Veri dosyaları
├── outputs/                      # Grafikler, tahminler, modeller
├── main_pipeline.ipynb          # Ana notebook (tüm pipeline)
├── requirements.txt             # Gerekli kütüphaneler
└── README.md                    # Bu dosya
```

## 🚀 Kurulum

### 1. Gerekli Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

### 2. Ana Pipeline'ı Çalıştır

Jupyter Notebook ile `main_pipeline.ipynb` dosyasını aç ve sırayla çalıştır.

## 📊 Veri Kaynakları

| Kaynak            | Açıklama                 | Frekans              |
| ----------------- | ------------------------ | -------------------- |
| **BTC-USD**       | Bitcoin fiyatı (OHLCV)   | Günlük               |
| **S&P 500**       | Hisse senedi korelasyonu | Günlük               |
| **DXY**           | Dolar Endeksi            | Günlük               |
| **10Y Treasury**  | ABD Tahvil Faizleri      | Günlük               |
| **Fear & Greed**  | Sentiment Index          | Günlük (placeholder) |
| **Google Trends** | Arama hacmi              | Günlük (placeholder) |

⚠️ **Önemli:** Hafta sonu problemi için forward fill (`ffill`) kullanılır!

## 🔧 Feature Engineering

### 1. Teknik İndikatörler

- RSI (14)
- MACD
- Bollinger Bands
- ATR (14)
- VWAP
- EMA (50, 200)

### 2. Lag Features

- 30 günlük geçmiş fiyat bilgisi
- Volume lag'leri (3, 7, 14 gün)

### 3. Rolling Statistics

- Moving Averages (7, 14, 30 gün)
- Rolling Volatility
- Rolling Returns

### 4. Cyclical Encoding

- Haftanın günü (Sin/Cos)
- Ayın günü (Sin/Cos)
- Ay (Sin/Cos)

### 5. Momentum Features

- ROC (Rate of Change)
- Price Position

### 6. Volume Features

- Volume Moving Averages
- Volume Ratio

### 7. Macro Interactions

- SPX-BTC korelasyonu
- DXY-BTC ters korelasyon
- Faiz değişimi

## 🤖 Modeller

### LightGBM

**Avantajlar:**

- ⚡ Hızlı eğitim
- 📊 Feature importance
- 🎯 Missing value handling

**Parametreler:**

```python
{
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8
}
```

### LSTM (PyTorch)

**Avantajlar:**

- 📈 Sequence learning
- 🔄 Uzun vadeli bağımlılıklar
- 🌊 Temporal patterns

**Mimari:**

```python
Input → LSTM(128) → LSTM(128) → Dense(1)
Dropout: 0.2
Sequence Length: 60
```

## 🔮 Recursive Forecasting

30 günlük tahmin için özyinelemeli yöntem:

```
1. t+1'i tahmin et
2. Bu tahmini "gerçekleşmiş" gibi ekle
3. t+2'yi tahmin et
4. 30 adım tekrarla
```

⚠️ **Risk:** Hata birikimi! İlk günlerdeki hatalar sonraki günleri etkiler.

## 📈 Metrikler

| Metrik   | Açıklama                       |
| -------- | ------------------------------ |
| **RMSE** | Root Mean Squared Error        |
| **MAE**  | Mean Absolute Error            |
| **R²**   | Determination Coefficient      |
| **MAPE** | Mean Absolute Percentage Error |

## 🎨 Görselleştirmeler

Pipeline otomatik olarak şu grafikleri oluşturur:

1. ✅ Actual vs Predicted (Test Set)
2. ✅ Feature Importance (LightGBM)
3. ✅ LSTM Loss Curve
4. ✅ 30 Günlük Tahminler
5. ✅ Güven Aralıkları
6. ✅ Model Karşılaştırması
7. ✅ Residual Analysis

## 📝 Kullanım Örneği

```python
# 1. Veri yükle
from src.data_loader import DataLoader
loader = DataLoader()
raw_data = loader.merge_all_data()

# 2. Feature engineering
from src.feature_engineering import FeatureEngineer
engineer = FeatureEngineer(raw_data)
featured_data = engineer.create_all_features(n_lags=30)

# 3. Preprocessing
from src.preprocessing import FullPipeline
pipeline = FullPipeline(featured_data)
lgb_data = pipeline.run_lightgbm_pipeline()

# 4. Model eğitimi
from src.models import LightGBMModel
model = LightGBMModel()
model.train(lgb_data['X_train'], lgb_data['y_train'])

# 5. Forecast
from src.forecasting import RecursiveForecaster
forecaster = RecursiveForecaster(model, lgb_data['preprocessor'], lgb_data['feature_names'])
forecast = forecaster.forecast_lightgbm(X_last, n_steps=30, last_price=50000)
```

## ⚠️ Önemli Notlar

### Veri Sızıntısı Önleme

- ✅ TimeSeriesSplit kullanılır (gelecek verisi eğitimde yok)
- ✅ Scaler sadece train verisi ile fit edilir
- ✅ Feature'lar geçmiş verilerden hesaplanır

### Hafta Sonu Problemi

- Bitcoin 7/24 işlem görür
- Makro veriler (SPX, Tahviller) hafta sonları kapalı
- Çözüm: `method='ffill'` (forward fill)

### Missing Values

```python
# Önce forward fill
df = df.fillna(method='ffill')

# Sonra backward fill (başlangıç NaN'ları için)
df = df.fillna(method='bfill')
```

## 📦 Çıktılar

Pipeline sonunda `outputs/` klasöründe:

- 📊 **Grafikler** (.png): Tüm görselleştirmeler
- 📈 **Tahminler** (.csv): 30 günlük fiyat tahminleri
- 🤖 **Modeller** (.pkl, .pth): Eğitilmiş modeller

## 🎯 Sonuçlar

Pipeline tamamlandığında:

1. ✅ Test set performans metrikleri
2. ✅ Feature importance analizi (hangi faktör daha önemli?)
3. ✅ 30 günlük fiyat tahminleri (LightGBM & LSTM)
4. ✅ Model karşılaştırması
5. ✅ Profesyonel grafikler

## 🔍 Gelecek Geliştirmeler

- [ ] Gerçek Fear & Greed Index API entegrasyonu
- [ ] Google Trends API
- [ ] Ensemble methods (LightGBM + LSTM)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Model confidence intervals (Monte Carlo)
- [ ] Real-time prediction API

## 📚 Kaynaklar

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [PyTorch LSTM Tutorial](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [TA Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [Yahoo Finance API](https://github.com/ranaroussi/yfinance)

## 👨‍💻 Geliştirici Notları

### Modüler Tasarım (DRY Prensibi)

Her modül tek bir sorumluluğa sahip:

- `data_loader`: Sadece veri toplama
- `feature_engineering`: Sadece feature oluşturma
- `preprocessing`: Sadece ön işleme
- `models`: Sadece model tanımları
- `forecasting`: Sadece tahmin
- `visualization`: Sadece görselleştirme

### Kod Kalitesi

- ✅ Docstring'ler (her fonksiyon açıklamalı)
- ✅ Type hints (kod okunabilirliği)
- ✅ Error handling (try-except blokları)
- ✅ Logging (işlem adımları takip edilebilir)

## 📞 İletişim

Sorular veya öneriler için issue açabilirsiniz.

---

**⚠️ DİKKAT:** Bu proje eğitim amaçlıdır. Finansal tavsiye değildir. Yatırım kararları tamamen sizin sorumluluğunuzdadır.
