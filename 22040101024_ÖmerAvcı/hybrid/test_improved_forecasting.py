"""
Test Improved Forecasting
=========================
Yeni geliştirilen recursive forecasting sistemini test eder.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Module'leri import et
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import FullPipeline
from src.models import LightGBMModel
from src.forecasting import RecursiveForecaster

print("="*70)
print("🧪 IMPROVED FORECASTING TEST")
print("="*70)

# 1. Veri yükle
print("\n📊 Veri yükleniyor...")
loader = DataLoader()
raw_data = loader.merge_all_data()

# 2. Feature engineering
print("\n🔧 Feature engineering...")
engineer = FeatureEngineer(raw_data)
featured_data = engineer.create_all_features(n_lags=30)

# 3. Preprocessing
print("\n📋 Preprocessing...")
pipeline = FullPipeline(featured_data)
lgb_data = pipeline.run_lightgbm_pipeline()

# 4. Model eğitimi (Quick training)
print("\n🤖 Model eğitimi...")
lgb_model = LightGBMModel(params={
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 200,  # Hızlı test için azaltıldı
    'verbose': -1
})
lgb_model.train(lgb_data['X_train'], lgb_data['y_train'])

# 5. Geçmiş returns hesapla (quantile için)
print("\n📈 Geçmiş returns analizi...")
y_train_returns = lgb_data['y_train']
print(f"   Min return: {y_train_returns.min():.4f}")
print(f"   Max return: {y_train_returns.max():.4f}")
print(f"   Std: {y_train_returns.std():.4f}")
print(f"   5% quantile: {np.percentile(y_train_returns, 5):.4f}")
print(f"   95% quantile: {np.percentile(y_train_returns, 95):.4f}")

# 6. Son tarih bul
last_date = featured_data['Date'].iloc[-1]
print(f"\n📅 Son tarih: {last_date}")

# 7. IMPROVED FORECASTING
print("\n🔮 IMPROVED FORECASTING BAŞLIYOR...")
X_last = lgb_data['X_test'][-1]

# Son fiyatı featured_data'dan al
last_price = featured_data['Close'].iloc[-1]
print(f"💰 Son Fiyat: ${last_price:,.2f}")

# Forecaster oluştur (improved version)
forecaster = RecursiveForecaster(
    model=lgb_model,
    preprocessor=lgb_data['preprocessor'],
    feature_names=lgb_data['feature_names'],
    historical_returns=y_train_returns,  # Quantile için
    last_date=last_date  # Cyclical encoding için
)

# 30 günlük tahmin
forecast_result = forecaster.forecast_lightgbm(
    X_last=X_last,
    n_steps=30,
    last_price=last_price
)

# 8. Sonuçları analiz et
print("\n"+"="*70)
print("📊 SONUÇLAR")
print("="*70)

log_returns = forecast_result['log_returns']
prices = forecast_result['prices']

print(f"\nLog Returns:")
print(f"   Min: {log_returns.min():.4f}")
print(f"   Max: {log_returns.max():.4f}")
print(f"   Mean: {log_returns.mean():.4f}")
print(f"   Std: {log_returns.std():.4f}")
print(f"   Unique değerler: {len(np.unique(log_returns))}/30")

print(f"\nFiyatlar:")
print(f"   Başlangıç: ${prices[0]:,.2f}")
print(f"   Son: ${prices[-1]:,.2f}")
print(f"   Değişim: {((prices[-1] / prices[0]) - 1) * 100:.2f}%")
print(f"   Min: ${prices.min():,.2f}")
print(f"   Max: ${prices.max():,.2f}")

# İlk ve son 5 gün
print(f"\n📋 İlk 5 Gün:")
for i in range(min(5, len(prices))):
    print(f"   Gün {i+1}: ${prices[i]:,.2f} (log_return: {log_returns[i]:.4f})")

print(f"\n📋 Son 5 Gün:")
for i in range(max(0, len(prices)-5), len(prices)):
    print(f"   Gün {i+1}: ${prices[i]:,.2f} (log_return: {log_returns[i]:.4f})")

print("\n✅ Test tamamlandı!")
print("="*70)
