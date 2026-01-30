"""
QUICK START GUIDE
=================
Bitcoin Price Prediction Pipeline'ı hızlıca başlatma rehberi.

Bu dosyayı çalıştırarak tüm pipeline'ı otomatik olarak test edebilirsiniz.
"""

import sys
sys.path.append('src')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from preprocessing import FullPipeline
from models import LightGBMModel
from forecasting import RecursiveForecaster, create_future_dates
import pandas as pd


def quick_test():
    """
    Pipeline'ın tüm bileşenlerini test eder.
    """
    print("\n" + "="*70)
    print("🚀 HYBRID BITCOIN PRICE PREDICTION - QUICK TEST")
    print("="*70 + "\n")
    
    # 1. VERİ YÜKLEME
    print("📊 ADIM 1: Veri Yükleme...")
    print("-" * 70)
    loader = DataLoader()
    raw_data = loader.merge_all_data()
    print(f"✅ {len(raw_data)} gün veri yüklendi\n")
    
    # 2. FEATURE ENGINEERING
    print("🔧 ADIM 2: Feature Engineering...")
    print("-" * 70)
    engineer = FeatureEngineer(raw_data)
    featured_data = engineer.create_all_features(n_lags=30)
    feature_names = engineer.get_feature_names()
    print(f"✅ {len(feature_names)} özellik oluşturuldu\n")
    
    # 3. PREPROCESSING
    print("📋 ADIM 3: Preprocessing...")
    print("-" * 70)
    pipeline = FullPipeline(featured_data)
    lgb_data = pipeline.run_lightgbm_pipeline(test_size=0.2)
    print(f"✅ Train: {lgb_data['X_train'].shape}, Test: {lgb_data['X_test'].shape}\n")
    
    # 4. MODEL EĞİTİMİ
    print("🤖 ADIM 4: Model Eğitimi (LightGBM)...")
    print("-" * 70)
    lgb_model = LightGBMModel()
    lgb_model.train(
        lgb_data['X_train'], 
        lgb_data['y_train'],
        feature_names=lgb_data['feature_names']
    )
    print("✅ Model eğitimi tamamlandı\n")
    
    # 5. DEĞERLENDIRME
    print("📊 ADIM 5: Model Değerlendirme...")
    print("-" * 70)
    metrics = lgb_model.evaluate(lgb_data['X_test'], lgb_data['y_test'])
    print()
    
    # 6. FEATURE IMPORTANCE
    print("🎯 ADIM 6: Feature Importance (Top 10)...")
    print("-" * 70)
    importance = lgb_model.get_feature_importance(top_n=10)
    print(importance)
    print()
    
    # 7. FORECASTING (5 günlük test)
    print("🔮 ADIM 7: 5 Günlük Tahmin Testi...")
    print("-" * 70)
    forecaster = RecursiveForecaster(
        model=lgb_model,
        preprocessor=lgb_data['preprocessor'],
        feature_names=lgb_data['feature_names']
    )
    
    last_price = lgb_data['original_df']['Close'].iloc[-1]
    X_last = lgb_data['X_test'][-1]
    
    forecast = forecaster.forecast_lightgbm(
        X_last=X_last,
        n_steps=5,
        last_price=last_price
    )
    
    print(f"\n📈 Tahmin Sonuçları:")
    print(f"   Mevcut Fiyat: ${last_price:,.2f}")
    print(f"   5 Gün Sonra: ${forecast['prices'][-1]:,.2f}")
    print(f"   Değişim: {((forecast['prices'][-1] - last_price) / last_price * 100):.2f}%")
    
    # 8. SONUÇ
    print("\n" + "="*70)
    print("✅ TÜM TESTLER BAŞARIYLA TAMAMLANDI!")
    print("="*70)
    print("\n📁 Tam pipeline için main_pipeline.ipynb dosyasını çalıştırın.")
    print("📚 Detaylı bilgi için README.md dosyasına bakın.\n")
    
    return {
        'data': featured_data,
        'model': lgb_model,
        'metrics': metrics,
        'forecast': forecast
    }


if __name__ == "__main__":
    try:
        results = quick_test()
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        print("\nLütfen requirements.txt dosyasındaki kütüphanelerin kurulu olduğundan emin olun:")
        print("   pip install -r requirements.txt")
