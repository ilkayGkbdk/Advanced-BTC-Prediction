"""
IMPROVED HYBRID BTC PREDICTION PIPELINE
==========================================
Yeni sistemleri outputs/ klasörüne kaydeden entegre pipeline.

OZELLIKLER:
- Monte Carlo Simulation (1000 senaryo)
- Walk-Forward Validation (temporal consistency)
- Real Sentiment API (Fear & Greed Index)
- Support/Resistance Levels (liquidity zones)
- Tüm sonuçlar outputs/ klasörüne kaydedilir
"""

import sys
import os

# UTF-8 encoding zorla
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.append('src')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from preprocessing import FullPipeline
from models import LightGBMModel
from forecasting import RecursiveForecaster
from sentiment_api import SentimentAggregator
from walk_forward_validation import WalkForwardValidator

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Output klasörünü oluştur
os.makedirs('outputs', exist_ok=True)

print("="*80)
print("🚀 IMPROVED HYBRID BTC PREDICTION PIPELINE")
print("="*80)
print("\nÖZELLİKLER:")
print("✅ Monte Carlo Simulation (1000 scenarios)")
print("✅ Walk-Forward Validation (5 folds)")
print("✅ Real-time Fear & Greed Index")
print("✅ Support/Resistance Levels")
print("✅ All results saved to outputs/")
print("="*80)

# =============================================================================
# 1. VERİ YÜKLEME
# =============================================================================
print("\n📊 STEP 1: DATA LOADING")
print("-" * 80)

# Cached data kullan (yfinance API problemi varsa)
try:
    data = pd.read_csv('data/featured_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    print(f"✅ Cached data loaded: {data.shape}")
    print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
except FileNotFoundError:
    print("⚠️ Cached data not found, loading fresh data...")
    loader = DataLoader(start_date='2021-01-01', end_date='2024-12-31')
    raw_data = loader.merge_all_data()
    
    # Sentiment API ekle
    sentiment_agg = SentimentAggregator()
    raw_data = sentiment_agg.create_sentiment_features(raw_data)
    
    # Feature engineering
    engineer = FeatureEngineer(raw_data)
    data = engineer.create_all_features(n_lags=30)
    
    # Cache'e kaydet
    data.to_csv('data/featured_data.csv', index=False)
    print(f"✅ Data loaded and cached: {data.shape}")

# Güncel sentiment göster
print("\n🌡️ CURRENT MARKET SENTIMENT:")
sentiment_agg = SentimentAggregator()
try:
    sentiment_agg.print_current_sentiment()
except:
    print("⚠️ Could not fetch current sentiment")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n📋 STEP 2: PREPROCESSING")
print("-" * 80)

pipeline = FullPipeline(featured_df=data, target_col='Close')
lgb_data = pipeline.run_lightgbm_pipeline(test_size=0.2, scaler_type='minmax')

print(f"✅ Train set: {lgb_data['X_train'].shape}")
print(f"✅ Test set: {lgb_data['X_test'].shape}")

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================
print("\n🤖 STEP 3: MODEL TRAINING")
print("-" * 80)

lgb_model = LightGBMModel()
lgb_model.train(
    lgb_data['X_train'], 
    lgb_data['y_train'],
    feature_names=lgb_data['feature_names']
)

# Test performance
y_pred_test = lgb_model.predict(lgb_data['X_test'])

# Calculate metrics manually
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
test_metrics = {
    'rmse': np.sqrt(mean_squared_error(lgb_data['y_test'], y_pred_test)),
    'mae': mean_absolute_error(lgb_data['y_test'], y_pred_test),
    'r2': r2_score(lgb_data['y_test'], y_pred_test)
}

print(f"\n📊 Test Performance:")
print(f"   RMSE: {test_metrics['rmse']:.6f}")
print(f"   MAE:  {test_metrics['mae']:.6f}")
print(f"   R²:   {test_metrics['r2']:.4f}")

# Feature importance kaydet
feature_importance = lgb_model.get_feature_importance(top_n=20)
feature_importance.to_csv('outputs/feature_importance.csv', index=False)
print(f"\n✅ Feature importance saved to outputs/feature_importance.csv")

# =============================================================================
# 4. WALK-FORWARD VALIDATION
# =============================================================================
print("\n🔄 STEP 4: WALK-FORWARD VALIDATION")
print("-" * 80)

# Validation için data hazırla
train_size = lgb_data['X_train'].shape[0]
dates_full = data['Date'].iloc[:train_size]

X_full_with_date = pd.DataFrame(lgb_data['X_train'], columns=lgb_data['feature_names'])
X_full_with_date['Date'] = dates_full.values
y_full = lgb_data['y_train']

# Walk-Forward Validator
wf_validator = WalkForwardValidator(
    train_window_months=12,
    test_window_months=1,
    step_months=1,
    min_train_size=200
)

# Folds oluştur
folds = wf_validator.create_folds(X_full_with_date, date_column='Date')

# Validate
wf_results = wf_validator.validate(
    model_class=LightGBMModel,
    X=X_full_with_date.drop('Date', axis=1),
    y=y_full,
    folds=folds,
    model_params=None,
    feature_names=lgb_data['feature_names'],
    verbose=False
)

# Walk-Forward sonuçlarını kaydet
wf_df = wf_validator.get_results_dataframe()
wf_df.to_csv('outputs/walk_forward_results.csv', index=False)
print(f"\n✅ Walk-Forward results saved to outputs/walk_forward_results.csv")

# Summary kaydet
summary = wf_results['summary']
summary_df = pd.DataFrame([summary])
summary_df.to_csv('outputs/walk_forward_summary.csv', index=False)
print(f"✅ Walk-Forward summary saved to outputs/walk_forward_summary.csv")

# =============================================================================
# 5. MONTE CARLO FORECASTING
# =============================================================================
print("\n🎲 STEP 5: MONTE CARLO FORECASTING (1000 scenarios)")
print("-" * 80)

# Scaler'ı al
scaler = lgb_data['preprocessor'].scaler
forecaster = RecursiveForecaster(lgb_model, scaler, lgb_data['feature_names'])

# Son veriyi al
last_sequence = lgb_data['X_test'][-1:].copy()

# Monte Carlo tahmin
mc_results = forecaster.forecast_monte_carlo(
    last_sequence=last_sequence,
    n_steps=30,
    n_simulations=1000,
    verbose=True
)

print("\n📊 MONTE CARLO RESULTS:")
print(f"   Median (Day 30): ${mc_results['median_forecast'][-1]:,.2f}")
print(f"   Mean:   ${mc_results['mean_forecast'][-1]:,.2f}")
print(f"   Std:    ${mc_results['std_forecast'][-1]:,.2f}")
print(f"   5th Percentile:  ${mc_results['percentile_05'][-1]:,.2f}")
print(f"   95th Percentile: ${mc_results['percentile_95'][-1]:,.2f}")

# Monte Carlo sonuçlarını kaydet
mc_forecast_df = pd.DataFrame({
    'Day': range(1, 31),
    'Median': mc_results['median_forecast'],
    'Mean': mc_results['mean_forecast'],
    'Std': mc_results['std_forecast'],
    'Percentile_05': mc_results['percentile_05'],
    'Percentile_25': mc_results['percentile_25'],
    'Percentile_75': mc_results['percentile_75'],
    'Percentile_95': mc_results['percentile_95']
})
mc_forecast_df.to_csv('outputs/monte_carlo_forecast.csv', index=False)
print(f"\n✅ Monte Carlo forecast saved to outputs/monte_carlo_forecast.csv")

# =============================================================================
# 6. DETERMINISTIC FORECAST (Eski sistem ile karşılaştırma)
# =============================================================================
print("\n🔮 STEP 6: DETERMINISTIC FORECAST (for comparison)")
print("-" * 80)

det_forecast = forecaster.forecast_recursive(
    last_sequence=last_sequence,
    n_steps=30,
    verbose=False
)

det_forecast_df = pd.DataFrame({
    'Day': range(1, 31),
    'Price': det_forecast
})
det_forecast_df.to_csv('outputs/deterministic_forecast.csv', index=False)
print(f"✅ Deterministic forecast saved to outputs/deterministic_forecast.csv")

print(f"\nDeterministic Final Price (Day 30): ${det_forecast[-1]:,.2f}")

# =============================================================================
# 7. COMPREHENSIVE COMPARISON
# =============================================================================
print("\n📊 STEP 7: COMPREHENSIVE COMPARISON")
print("-" * 80)

starting_price = det_forecast[0]

comparison_df = pd.DataFrame({
    'Metric': [
        'Starting Price',
        'Final Price (Day 30)',
        'Price Change %',
        'Confidence Interval 5%',
        'Confidence Interval 95%',
        'Uncertainty Range',
        'Median/Deterministic Diff'
    ],
    'Deterministic': [
        f"${starting_price:,.2f}",
        f"${det_forecast[-1]:,.2f}",
        f"{((det_forecast[-1] - starting_price) / starting_price * 100):.2f}%",
        'N/A',
        'N/A',
        'N/A',
        'N/A'
    ],
    'Monte Carlo': [
        f"${starting_price:,.2f}",
        f"${mc_results['median_forecast'][-1]:,.2f}",
        f"{((mc_results['median_forecast'][-1] - starting_price) / starting_price * 100):.2f}%",
        f"${mc_results['percentile_05'][-1]:,.2f}",
        f"${mc_results['percentile_95'][-1]:,.2f}",
        f"${mc_results['percentile_95'][-1] - mc_results['percentile_05'][-1]:,.2f}",
        f"${mc_results['median_forecast'][-1] - det_forecast[-1]:,.2f}"
    ]
})

comparison_df.to_csv('outputs/system_comparison.csv', index=False)
print(f"✅ System comparison saved to outputs/system_comparison.csv")

print("\n" + comparison_df.to_string(index=False))

# =============================================================================
# 8. EXECUTION SUMMARY
# =============================================================================
print("\n" + "="*80)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)

print("\n📁 Generated Files in outputs/:")
output_files = [
    'feature_importance.csv',
    'walk_forward_results.csv',
    'walk_forward_summary.csv',
    'monte_carlo_forecast.csv',
    'deterministic_forecast.csv',
    'system_comparison.csv'
]

for file in output_files:
    filepath = f'outputs/{file}'
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"   ✅ {file:40s} ({size:,} bytes)")
    else:
        print(f"   ❌ {file:40s} (not found)")

print("\n" + "="*80)
print("🎉 ALL IMPROVEMENTS INTEGRATED & SAVED!")
print("="*80)

# Execution timestamp kaydet
timestamp_df = pd.DataFrame({
    'execution_time': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'test_r2': [test_metrics['r2']],
    'wf_avg_r2': [summary['avg_r2']],
    'wf_consistency': [summary['consistency_score']],
    'mc_median_price': [mc_results['median_forecast'][-1]],
    'det_final_price': [det_forecast[-1]]
})
timestamp_df.to_csv('outputs/execution_log.csv', index=False)
print("\n✅ Execution log saved to outputs/execution_log.csv")
