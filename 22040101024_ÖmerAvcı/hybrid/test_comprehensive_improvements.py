"""
Comprehensive Improved System Test
===================================
Tüm iyileştirmeleri test eder ve karşılaştırmalı sonuçlar verir.

YENİ ÖZELLİKLER:
1. Monte Carlo Simülasyonu (1000 senaryo)
2. Walk-Forward Validation (zamansal doğrulama)
3. Gerçek Fear & Greed Index (Alternative.me API)
4. Support/Resistance Levels (likidite analizi)

KARŞILAŞTIRMA:
- Eski sistem (tek tahmin)
- Yeni sistem (Monte Carlo + sentiment)
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Module'leri import et
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import FullPipeline
from src.models import LightGBMModel
from src.forecasting import RecursiveForecaster
from src.walk_forward_validation import WalkForwardValidator
from src.sentiment_api import SentimentAggregator

print("="*80)
print("🚀 COMPREHENSIVE IMPROVED SYSTEM TEST")
print("="*80)
print("""
Bu test şunları değerlendirir:
1. ✅ Monte Carlo Simülasyonu ile olasılık bantları
2. ✅ Walk-Forward Validation ile zamansal performans
3. ✅ Gerçek Sentiment verileri (Fear & Greed API)
4. ✅ Support/Resistance seviyelerinin model etkisi
5. ✅ Eski vs Yeni sistem karşılaştırması
""")
print("="*80)

# =============================================================================
# ADIM 1: VERİ YÜKLEME VE İYİLEŞTİRME
# =============================================================================

print("\n" + "="*80)
print("📊 STEP 1: DATA LOADING & SENTIMENT INTEGRATION")
print("="*80)

# Fix: yfinance API sorunu - mevcut cached veriyi kullan
print("\n⚠️ yfinance API şu anda çalışmıyor, cached veri kullanılıyor...")
try:
    featured_data = pd.read_csv('data/featured_data.csv')
    featured_data['Date'] = pd.to_datetime(featured_data['Date'])
    print(f"✅ Cached data loaded: {featured_data.shape}")
    print(f"   Date range: {featured_data['Date'].min()} to {featured_data['Date'].max()}")
    
    # Sentiment API ile güncelle (varsa)
    try:
        print("\n🌐 Trying to update with real-time sentiment data...")
        sentiment_agg = SentimentAggregator()
        sentiment_agg.print_current_sentiment()
        print("   Note: Sentiment data is separate from cached price data")
    except Exception as e:
        print(f"   ⚠️ Sentiment API update skipped: {e}")
    
    # Skip to feature engineering stage
    print("\n✅ Proceeding with cached featured data (already includes features)")
    
except FileNotFoundError:
    print("❌ Cached data not found. Trying live data load...")
    loader = DataLoader(start_date='2021-01-01', end_date='2024-12-31')
    raw_data = loader.merge_all_data()

    print(f"\n✅ Raw data loaded: {raw_data.shape}")

    # Sentiment API ile gerçek Fear & Greed verisini ekle
    sentiment_agg = SentimentAggregator()

    try:
        print("\n🌐 Fetching real Fear & Greed Index from Alternative.me API...")
        raw_data = sentiment_agg.create_sentiment_features(raw_data)
        print("✅ Real sentiment data integrated!")
        
        # Güncel sentiment göster
        sentiment_agg.print_current_sentiment()
        
    except Exception as e:
        print(f"⚠️ Sentiment API failed: {e}")
        print("   Continuing with existing Fear_Greed column...")

    # =============================================================================
    # ADIM 2: FEATURE ENGINEERING (İYİLEŞTİRİLMİŞ)
    # =============================================================================

    print("\n" + "="*80)
    print("🔧 STEP 2: IMPROVED FEATURE ENGINEERING")
    print("="*80)

    engineer = FeatureEngineer(raw_data)
    featured_data = engineer.create_all_features(n_lags=30)

    print(f"\n✅ Featured data shape: {featured_data.shape}")
    print(f"   Features count: {featured_data.shape[1]}")

    # Yeni eklenen feature'ları listele
    new_features = [col for col in featured_data.columns 
                    if any(x in col for x in ['Pivot', 'R1', 'S1', 'Liquidity', 'FG_'])]
    print(f"\n🆕 NEW FEATURES ({len(new_features)}):")
    for feat in new_features[:20]:  # İlk 20'sini göster
        print(f"   - {feat}")
    if len(new_features) > 20:
        print(f"   ... and {len(new_features) - 20} more")

# =============================================================================
# ADIM 3: PREPROCESSING
# =============================================================================

print("\n" + "="*80)
print("📋 STEP 3: PREPROCESSING")
print("="*80)

pipeline = FullPipeline(featured_data)
lgb_data = pipeline.run_lightgbm_pipeline()

print(f"\n✅ Train set: {lgb_data['X_train'].shape}")
print(f"✅ Test set: {lgb_data['X_test'].shape}")

# =============================================================================
# ADIM 4: MODEL EĞİTİMİ
# =============================================================================

print("\n" + "="*80)
print("🤖 STEP 4: MODEL TRAINING")
print("="*80)

lgb_model = LightGBMModel(params={
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
})

lgb_model.train(
    lgb_data['X_train'], 
    lgb_data['y_train'],
    X_val=lgb_data['X_test'],
    y_val=lgb_data['y_test'],
    feature_names=lgb_data['feature_names'],
    early_stopping_rounds=50
)

# Model performansı
metrics = lgb_model.evaluate(lgb_data['X_test'], lgb_data['y_test'])
print(f"\n📊 Test Set Performance:")
print(f"   RMSE: {metrics['RMSE']:.6f}")
print(f"   MAE:  {metrics['MAE']:.6f}")
print(f"   R²:   {metrics['R2']:.4f}")

# Feature importance (yeni feature'lar ne kadar önemli?)
print(f"\n🎯 TOP 20 Most Important Features:")
print(lgb_model.feature_importance_df.head(20).to_string(index=False))

# =============================================================================
# ADIM 5: WALK-FORWARD VALIDATION
# =============================================================================

print("\n" + "="*80)
print("🔄 STEP 5: WALK-FORWARD VALIDATION")
print("="*80)
print("Testing model's temporal consistency...")

# Validation için veri hazırla
X_full = lgb_data['X_train']
y_full = lgb_data['y_train']

# Get dates from featured_data (first 716 rows which correspond to train set)
train_size = X_full.shape[0]
dates_full = featured_data['Date'].iloc[:train_size]

# Date kolonu ekle (geçici)
X_full_with_date = pd.DataFrame(X_full, columns=lgb_data['feature_names'])
X_full_with_date['Date'] = dates_full.values

# Walk-Forward Validator
wf_validator = WalkForwardValidator(
    train_window_months=12,  # 12 ay eğitim
    test_window_months=1,    # 1 ay test
    step_months=1,           # 1 ay kaydırma
    min_train_size=200
)

# Fold'ları oluştur
folds = wf_validator.create_folds(X_full_with_date, date_column='Date')

# Validation çalıştır (ilk 5 fold - hız için)
if len(folds) > 5:
    print(f"\n⚡ Testing first 5 folds (out of {len(folds)}) for speed...")
    test_folds = folds[:5]
else:
    test_folds = folds

# Date kolonunu kaldır
X_full_clean = X_full_with_date.drop('Date', axis=1)

wf_results = wf_validator.validate(
    model_class=LightGBMModel,
    X=X_full_clean,
    y=y_full,
    folds=test_folds,
    model_params=lgb_model.params,
    feature_names=lgb_data['feature_names'],
    verbose=False
)

# Regime change tespiti
regime_changes = wf_validator.detect_regime_changes(threshold=0.3)

# =============================================================================
# ADIM 6: ESKİ SİSTEM (TEK TAHMİN)
# =============================================================================

print("\n" + "="*80)
print("📈 STEP 6A: OLD SYSTEM - Single Forecast")
print("="*80)

X_last = lgb_data['X_test'][-1]
last_price = featured_data['Close'].iloc[-1]
last_date = featured_data['Date'].iloc[-1]

print(f"💰 Last Price: ${last_price:,.2f}")
print(f"📅 Last Date: {last_date}")

# Forecaster oluştur
forecaster = RecursiveForecaster(
    model=lgb_model,
    preprocessor=lgb_data['preprocessor'],
    feature_names=lgb_data['feature_names'],
    historical_returns=lgb_data['y_train'],
    last_date=last_date
)

# Tek tahmin (eski sistem)
print("\n🔮 Running single forecast (30 days)...")
old_forecast = forecaster.forecast_lightgbm(
    X_last=X_last,
    n_steps=30,
    last_price=last_price
)

print(f"\n📊 OLD SYSTEM Results:")
print(f"   Final Price: ${old_forecast['prices'][-1]:,.2f}")
print(f"   Price Change: {((old_forecast['prices'][-1] / last_price) - 1) * 100:.2f}%")
print(f"   Avg Log Return: {old_forecast['log_returns'].mean():.6f}")

# =============================================================================
# ADIM 7: YENİ SİSTEM (MONTE CARLO)
# =============================================================================

print("\n" + "="*80)
print("🎲 STEP 6B: NEW SYSTEM - Monte Carlo Simulation")
print("="*80)

print("""
Monte Carlo ile:
- 1000 farklı senaryo
- Her senaryoda farklı rastgele gürültü
- Sonuç: Olasılık bantları (5%, 25%, 50%, 75%, 95%)
""")

# Monte Carlo (yeni sistem)
mc_forecast = forecaster.forecast_monte_carlo(
    X_last=X_last,
    n_steps=30,
    last_price=last_price,
    n_simulations=1000,
    confidence_levels=[0.05, 0.25, 0.50, 0.75, 0.95]
)

# =============================================================================
# ADIM 8: KARŞILAŞTIRMA VE ANALİZ
# =============================================================================

print("\n" + "="*80)
print("📊 STEP 7: COMPREHENSIVE COMPARISON")
print("="*80)

print(f"\n{'='*80}")
print(f"{'SYSTEM COMPARISON':^80}")
print(f"{'='*80}")

print(f"\n{'Metric':<40} {'Old System':<20} {'New System (MC)':<20}")
print(f"{'-'*80}")

# Başlangıç fiyatı
print(f"{'Starting Price':<40} ${last_price:>18,.2f}  ${last_price:>18,.2f}")

# Final fiyat
old_final = old_forecast['prices'][-1]
new_final = mc_forecast['statistics']['final_price_median']
print(f"{'Final Price (Day 30)':<40} ${old_final:>18,.2f}  ${new_final:>18,.2f}")

# Değişim
old_change = ((old_final / last_price) - 1) * 100
new_change = ((new_final / last_price) - 1) * 100
print(f"{'Price Change %':<40} {old_change:>18.2f}%  {new_change:>18.2f}%")

# Volatilite
old_vol = np.std(old_forecast['log_returns'])
new_vol = mc_forecast['statistics']['volatility_mean']
print(f"{'Volatility (log returns)':<40} {old_vol:>19.6f}  {new_vol:>19.6f}")

# Güven aralıkları (sadece yeni sistem)
print(f"\n{'='*80}")
print(f"{'CONFIDENCE INTERVALS (New System Only)':^80}")
print(f"{'='*80}")

for percentile, values in mc_forecast['percentiles'].items():
    final_price = values[-1]
    change_pct = ((final_price / last_price) - 1) * 100
    print(f"{percentile:<10} ${final_price:>15,.2f}  ({change_pct:>+7.2f}%)")

# Walk-Forward sonuçları
print(f"\n{'='*80}")
print(f"{'WALK-FORWARD VALIDATION SUMMARY':^80}")
print(f"{'='*80}")

wf_summary = wf_results['summary']
print(f"\nAverage Performance across {wf_summary['n_folds']} time periods:")
print(f"   RMSE: {wf_summary['avg_rmse']:.6f} (±{wf_summary['std_rmse']:.6f})")
print(f"   R²:   {wf_summary['avg_r2']:.4f} (±{wf_summary['std_r2']:.4f})")
print(f"   Consistency Score: {wf_summary['consistency_score']:.4f}")

if wf_summary['consistency_score'] > 0.7:
    print("\n✅ Model is CONSISTENT across different market regimes")
else:
    print("\n⚠️ Model shows variability - consider regime-specific models")

# =============================================================================
# ADIM 9: TAVSİYELER
# =============================================================================

print("\n" + "="*80)
print("💡 RECOMMENDATIONS")
print("="*80)

print("""
GERÇEKÇİLİK DEĞERLENDİRMESİ:

1. R² Skoru Analizi:
""")

if metrics['R2'] > 0.95:
    print("   ⚠️ R² > 0.95: Muhtemel DATA LEAKAGE!")
    print("   → Close_Lag_1'e aşırı bağımlılık olabilir")
    print("   → Feature ablation study önerilir")
elif metrics['R2'] > 0.7:
    print("   ✅ R² iyi ama makul seviyede")
    print("   → Sağlıklı bir model")
elif metrics['R2'] > 0.3:
    print("   ⚠️ R² orta seviyede")
    print("   → Daha fazla feature engineering gerekebilir")
else:
    print("   ❌ R² düşük")
    print("   → Model veriyi öğrenememiş")

print(f"""
2. Walk-Forward Consistency:
   Score: {wf_summary['consistency_score']:.2f}
""")

if wf_summary['consistency_score'] > 0.7:
    print("   ✅ Model zamana karşı tutarlı")
else:
    print("   ⚠️ Model regime change'lere hassas")
    print("   → Regime-switching model düşünülebilir")

print(f"""
3. Monte Carlo Sonuçları:
   Median vs Mean: ${new_final:,.2f} vs ${mc_forecast['statistics']['final_price_mean']:,.2f}
""")

if abs(new_final - mc_forecast['statistics']['final_price_mean']) / new_final < 0.05:
    print("   ✅ Distribüsyon simetrik (sağlıklı)")
else:
    print("   ⚠️ Distribüsyon çarpık (skewed)")
    print("   → Fat-tail riski olabilir")

print(f"""
4. Volatilite:
   Model Volatility: {new_vol:.4f}
   Historical Volatility: {np.std(lgb_data['y_train']):.4f}
""")

if new_vol / np.std(lgb_data['y_train']) < 1.5:
    print("   ✅ Tahmin volatilitesi makul")
else:
    print("   ⚠️ Tahmin çok volatil")
    print("   → Noise'ı azaltmayı düşünün")

print("""

SONRAKİ ADIMLAR:
─────────────────
1. ✅ Feature Ablation Study
   → Hangi feature'lar gerçekten önemli?
   → Close_Lag_1 olmadan model nasıl perform eder?

2. ✅ Regime-Specific Models
   → Bull/Bear piyasalar için ayrı modeller
   → Fear/Greed bazlı model seçimi

3. ✅ Ensemble Methods
   → LightGBM + LSTM + XGBoost kombinasyonu
   → Voting veya stacking

4. ✅ Real-time Backtesting
   → Gerçek trading simülasyonu
   → Transaction costs dahil

5. ✅ Risk Management
   → Position sizing
   → Stop-loss/take-profit seviyeleri
""")

print("\n" + "="*80)
print("✅ COMPREHENSIVE TEST COMPLETED!")
print("="*80)
