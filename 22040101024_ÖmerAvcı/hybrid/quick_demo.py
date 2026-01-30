"""
Quick Demo - Professional Improvements
========================================
Yeni özellikleri hızlıca test etmek için mini demo.

Süre: ~2-3 dakika
"""

print("="*80)
print("🚀 QUICK DEMO: Professional Improvements")
print("="*80)

# 1. Sentiment API Test
print("\n" + "="*80)
print("TEST 1: Real-Time Fear & Greed Index")
print("="*80)

try:
    from src.sentiment_api import FearGreedIndexAPI
    
    fg_api = FearGreedIndexAPI()
    current = fg_api.get_current_index()
    
    if current:
        print(f"\n✅ API CONNECTED!")
        print(f"   Current Fear & Greed: {current['value']}/100")
        print(f"   Classification: {current['classification']}")
        print(f"   Timestamp: {current['timestamp']}")
        
        # Trading signal
        if current['value'] < 25:
            print("\n💡 Signal: EXTREME FEAR → Buy Opportunity")
        elif current['value'] > 75:
            print("\n⚠️ Signal: EXTREME GREED → Consider Taking Profits")
        else:
            print("\n😐 Signal: Neutral Zone")
    
    print("\n📊 Fetching last 7 days of historical data...")
    historical = fg_api.get_historical_data(days=7)
    print(historical.tail(7))
    
except Exception as e:
    print(f"❌ Sentiment API Error: {e}")
    print("   (Internet connection needed)")

# 2. Monte Carlo Preview
print("\n" + "="*80)
print("TEST 2: Monte Carlo Simulation Preview")
print("="*80)

import numpy as np

print("""
Monte Carlo simülasyonu aynı başlangıç noktasından 1000 farklı gelecek
senaryosu üretir. Her senaryoda farklı rastgele gürültü eklenir.

Örnek: 100 senaryo ile mini demo (gerçekte 1000 olur)
""")

# Dummy data ile mini simulation
np.random.seed(42)
initial_price = 50000
n_steps = 30
n_simulations = 100

all_scenarios = []
for _ in range(n_simulations):
    prices = [initial_price]
    for step in range(n_steps):
        # Basit random walk
        log_return = np.random.normal(0, 0.02)  # 2% daily vol
        new_price = prices[-1] * np.exp(log_return)
        prices.append(new_price)
    all_scenarios.append(prices[1:])

all_scenarios = np.array(all_scenarios)

# Percentiles
p5 = np.percentile(all_scenarios, 5, axis=0)
p50 = np.percentile(all_scenarios, 50, axis=0)
p95 = np.percentile(all_scenarios, 95, axis=0)

print(f"\n📊 Results (Day 30):")
print(f"   Starting Price: ${initial_price:,.2f}")
print(f"   Median Forecast: ${p50[-1]:,.2f}")
print(f"   5% Percentile (Worst Case): ${p5[-1]:,.2f}")
print(f"   95% Percentile (Best Case): ${p95[-1]:,.2f}")
print(f"   Price Range: ${p5[-1]:,.2f} - ${p95[-1]:,.2f}")

print(f"\n💡 Bu aralık, fiyatın %90 olasılıkla bu bandlarda kalacağını gösterir.")

# 3. Walk-Forward Validation Preview
print("\n" + "="*80)
print("TEST 3: Walk-Forward Validation Concept")
print("="*80)

print("""
Walk-Forward Validation, modelin zamana karşı tutarlılığını test eder.

Klasik Split (YANLIŞ):
   Train: 2020-2023 (karışık)
   Test: 2024 (karışık)
   ❌ Problem: Gelecek bilgisi sızabilir

Walk-Forward (DOĞRU):
   Fold 1: Train[2020-01 to 2020-12] → Test[2021-01]
   Fold 2: Train[2020-02 to 2021-01] → Test[2021-02]
   Fold 3: Train[2020-03 to 2021-02] → Test[2021-03]
   ...
   ✅ Her fold gerçek trading simüle eder

Consistency Score: Model'in farklı zaman dilimlerindeki performans istikrarı
- > 0.80: Çok tutarlı (mükemmel)
- > 0.60: Tutarlı (iyi)
- < 0.50: Tutarsız (piyasa rejimlerine hassas)
""")

# 4. Support/Resistance Preview
print("\n" + "="*80)
print("TEST 4: Support/Resistance Levels")
print("="*80)

print("""
Pivot Points (Klasik Teknik Analiz):

Örnek hesaplama:
   Yesterday: High=52000, Low=48000, Close=50000
   
   Pivot (P) = (52000 + 48000 + 50000) / 3 = 50000
   R1 = 2*P - Low = 2*50000 - 48000 = 52000
   S1 = 2*P - High = 2*50000 - 52000 = 48000
   
   Bugün fiyat bu seviyelerden DESTEK veya DİRENÇ bulur:
   - 52000'e yaklaşırsa → DİRENÇ (satış baskısı)
   - 48000'e yaklaşırsa → DESTEK (alım ilgisi)

Model bu seviyeleri öğrenerek tahmin kalitesini artırır.
""")

# Summary
print("\n" + "="*80)
print("✅ DEMO COMPLETED!")
print("="*80)

print("""
NEXT STEPS:
───────────
1. Tam test için:
   python test_comprehensive_improvements.py
   
2. Sadece Monte Carlo için:
   python test_improved_forecasting.py
   (forecasting.py'deki forecast_monte_carlo metodunu kullanacak şekilde güncelleyin)

3. API detayları için:
   python src/sentiment_api.py

4. Walk-Forward için:
   python src/walk_forward_validation.py

DOCUMENTATION:
──────────────
- PROFESSIONAL_IMPROVEMENTS.md → Detaylı açıklamalar
- IMPROVEMENTS.md → Önceki dokümantasyon
- README.md → Genel bilgi

REQUIREMENTS:
─────────────
pip install -r requirements.txt

ESTIMATED TIME:
───────────────
- Quick demo: 2-3 dakika
- Full test: 10-15 dakika
- Monte Carlo (1000 sim): 3-5 dakika
- Walk-Forward (5 folds): 3-5 dakika
""")
