"""
Sentiment API Integration
==========================
Gerçek piyasa sentiment verilerini çeker.

APIs:
1. Alternative.me Fear & Greed Index (Bitcoin)
2. CoinGlass (On-Chain Metrics) - Opsiyonel
3. Twitter Sentiment (Opsiyonel)

ÖNEM: Sentiment, Bitcoin fiyatını etkileyen en güçlü faktörlerden biri!
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class FearGreedIndexAPI:
    """
    Alternative.me Fear & Greed Index API wrapper.
    
    API DOC: https://alternative.me/crypto/fear-and-greed-index/
    
    BEDAVA ve API KEY GEREKMİYOR!
    
    Index Değerleri:
    - 0-24: Extreme Fear (Aşırı Korku)
    - 25-49: Fear (Korku)
    - 50-74: Greed (Açgözlülük)
    - 75-100: Extreme Greed (Aşırı Açgözlülük)
    
    Trading Mantığı:
    - Extreme Fear = Alım fırsatı (Warren Buffett: "Be greedy when others are fearful")
    - Extreme Greed = Satış sinyali (Piyasa zirvede olabilir)
    """
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self, cache_minutes=60):
        """
        Args:
            cache_minutes: Cache süresi (dakika). API'ye gereksiz istek atmamak için.
        """
        self.cache_minutes = cache_minutes
        self._cache = None
        self._cache_time = None
    
    def get_current_index(self):
        """
        Güncel Fear & Greed Index'i çeker.
        
        Returns:
            Dict: {
                'value': int (0-100),
                'classification': str ('Extreme Fear', 'Fear', etc.),
                'timestamp': datetime
            }
        """
        try:
            response = requests.get(f"{self.BASE_URL}?limit=1", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                
                return {
                    'value': int(latest['value']),
                    'classification': latest['value_classification'],
                    'timestamp': datetime.fromtimestamp(int(latest['timestamp']))
                }
            else:
                raise ValueError("API response format unexpected")
        
        except Exception as e:
            print(f"⚠️ Fear & Greed API error: {e}")
            return None
    
    def get_historical_data(self, days=365):
        """
        Geçmiş Fear & Greed verilerini çeker.
        
        Args:
            days: Kaç günlük veri (max: sınırsız)
            
        Returns:
            pd.DataFrame: Tarih ve değerler
        """
        # Cache kontrolü
        if self._cache is not None and self._cache_time is not None:
            if (datetime.now() - self._cache_time).total_seconds() < self.cache_minutes * 60:
                print("📦 Using cached Fear & Greed data...")
                return self._cache.copy()
        
        print(f"🌐 Fetching Fear & Greed Index (last {days} days)...")
        
        try:
            # API limiti yok ama nazik olalım
            if days > 365:
                print(f"⚠️ Requesting {days} days. This may take a while...")
            
            response = requests.get(f"{self.BASE_URL}?limit={days}", timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("API response missing 'data' field")
            
            # DataFrame'e çevir
            records = []
            for item in data['data']:
                records.append({
                    'Date': datetime.fromtimestamp(int(item['timestamp'])),
                    'Fear_Greed': int(item['value']),
                    'Classification': item['value_classification']
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Cache'e kaydet
            self._cache = df.copy()
            self._cache_time = datetime.now()
            
            print(f"✅ Fetched {len(df)} days of Fear & Greed data")
            print(f"   Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            
            return df
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return self._get_fallback_data(days)
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return self._get_fallback_data(days)
    
    def _get_fallback_data(self, days):
        """
        API başarısız olursa, realistic placeholder döndürür.
        
        NOT: Gerçek veri olmadığı için uyarı verir.
        """
        print("⚠️ API failed. Generating FALLBACK data...")
        print("⚠️ WARNING: This is NOT real data! Model accuracy will be limited.")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Realistic Fear & Greed pattern (sine wave + noise)
        # Gerçek piyasada ~50-60 gün döngüler var
        base_cycle = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, days))
        noise = np.random.normal(0, 10, days)
        values = np.clip(base_cycle + noise, 0, 100).astype(int)
        
        df = pd.DataFrame({
            'Date': dates,
            'Fear_Greed': values,
            'Classification': ['Fallback'] * days
        })
        
        return df
    
    def merge_with_price_data(self, price_df, on='Date'):
        """
        Fiyat verisi ile Fear & Greed'i merge eder.
        
        Args:
            price_df: Bitcoin fiyat verisi (Date kolonu içermeli)
            on: Merge key (default: 'Date')
            
        Returns:
            pd.DataFrame: Merged data
        """
        print("\n🔗 Merging Fear & Greed with price data...")
        
        # Fear & Greed data
        fg_data = self.get_historical_data(days=len(price_df) + 100)  # Biraz fazla al
        
        # Tarihleri normalize et (sadece gün)
        fg_data['Date'] = pd.to_datetime(fg_data['Date']).dt.date
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
        
        # Merge
        merged = price_df.merge(
            fg_data[['Date', 'Fear_Greed', 'Classification']], 
            on='Date', 
            how='left'
        )
        
        # Eksik değerleri forward fill (weekend veya API boşlukları)
        merged['Fear_Greed'] = merged['Fear_Greed'].fillna(method='ffill')
        merged['Fear_Greed'] = merged['Fear_Greed'].fillna(50)  # Eğer hala NaN varsa neutral değer
        
        print(f"✅ Merged successfully!")
        print(f"   Missing values filled: {merged['Fear_Greed'].isna().sum()} → 0")
        
        return merged


class OnChainMetricsAPI:
    """
    On-Chain metrics (blockchain data) için API wrapper.
    
    NOT: Bu metrikler genellikle ücretli API'ler gerektirir:
    - CryptoQuant
    - Glassnode
    - IntoTheBlock
    
    Burada basit bir template sunuyoruz.
    """
    
    def __init__(self, api_key=None):
        """
        Args:
            api_key: API anahtarı (varsa)
        """
        self.api_key = api_key
    
    def get_exchange_reserve(self):
        """
        Borsalardaki Bitcoin rezervi.
        
        MANTIK:
        - Rezerv düşüyor = Insanlar Bitcoin'i borsadan çekiyor (HODLing) → Bullish
        - Rezerv artıyor = Insanlar Bitcoin'i borsaya yatırıyor (satış) → Bearish
        
        NOT: Gerçek implementasyon için CryptoQuant API gerekir.
        """
        if not self.api_key:
            print("⚠️ No API key provided. Returning dummy data.")
            return None
        
        # Placeholder
        print("🚧 Exchange Reserve API - Not implemented yet.")
        return None
    
    def get_miner_flow(self):
        """
        Madenci akışı (miner netflow).
        
        MANTIK:
        - Madenciler satıyor (positive netflow) → Bearish pressure
        - Madenciler accumulate ediyor → Bullish
        
        NOT: Gerçek implementasyon için Glassnode gerekir.
        """
        if not self.api_key:
            print("⚠️ No API key provided. Returning dummy data.")
            return None
        
        print("🚧 Miner Flow API - Not implemented yet.")
        return None


class SentimentAggregator:
    """
    Tüm sentiment kaynaklarını bir araya getirir.
    """
    
    def __init__(self):
        self.fear_greed_api = FearGreedIndexAPI()
    
    def create_sentiment_features(self, df):
        """
        Sentiment-based özellikler oluşturur.
        
        Args:
            df: Bitcoin fiyat verisi (Date kolonu içermeli)
            
        Returns:
            pd.DataFrame: Sentiment features eklenmiş veri
        """
        print("\n" + "="*70)
        print("😨 SENTIMENT FEATURE ENGINEERING")
        print("="*70)
        
        # Fear & Greed merge
        df_with_fg = self.fear_greed_api.merge_with_price_data(df)
        
        # Ek türev özellikler
        print("\n📊 Creating derived sentiment features...")
        
        # 1. Fear & Greed değişim hızı (momentum)
        df_with_fg['FG_Change'] = df_with_fg['Fear_Greed'].diff()
        df_with_fg['FG_Change_Pct'] = df_with_fg['Fear_Greed'].pct_change()
        
        # 2. Hareketli ortalama (trend)
        df_with_fg['FG_MA_7'] = df_with_fg['Fear_Greed'].rolling(window=7).mean()
        df_with_fg['FG_MA_30'] = df_with_fg['Fear_Greed'].rolling(window=30).mean()
        
        # 3. Extremity indicator (ne kadar uç değerlerde)
        # 0'a yakın = Extreme Fear, 100'e yakın = Extreme Greed
        df_with_fg['FG_Extremity'] = np.abs(df_with_fg['Fear_Greed'] - 50)
        
        # 4. Regime flags (categorical)
        df_with_fg['FG_Regime'] = pd.cut(
            df_with_fg['Fear_Greed'],
            bins=[0, 25, 50, 75, 100],
            labels=['Extreme_Fear', 'Fear', 'Greed', 'Extreme_Greed']
        )
        
        # One-hot encoding için (opsiyonel)
        regime_dummies = pd.get_dummies(df_with_fg['FG_Regime'], prefix='FG')
        df_with_fg = pd.concat([df_with_fg, regime_dummies], axis=1)
        
        print("✅ Sentiment features created:")
        print("   - Fear_Greed (raw)")
        print("   - FG_Change, FG_Change_Pct (momentum)")
        print("   - FG_MA_7, FG_MA_30 (trend)")
        print("   - FG_Extremity (distance from neutral)")
        print("   - FG_Regime (categorical)")
        
        print("\n" + "="*70)
        
        return df_with_fg
    
    def print_current_sentiment(self):
        """
        Güncel sentiment durumunu yazdırır.
        """
        current = self.fear_greed_api.get_current_index()
        
        if current:
            print("\n" + "="*70)
            print("🌡️ CURRENT MARKET SENTIMENT")
            print("="*70)
            print(f"\n📊 Fear & Greed Index: {current['value']}/100")
            print(f"📝 Classification: {current['classification']}")
            print(f"🕒 Timestamp: {current['timestamp']}")
            
            # Trading signal
            if current['value'] < 25:
                print("\n💡 SIGNAL: Extreme Fear → POTENTIAL BUY OPPORTUNITY")
            elif current['value'] > 75:
                print("\n⚠️ SIGNAL: Extreme Greed → CONSIDER TAKING PROFITS")
            else:
                print("\n😐 SIGNAL: Neutral zone")
            
            print("="*70)


if __name__ == "__main__":
    # Test
    print("="*70)
    print("🧪 TESTING SENTIMENT API")
    print("="*70)
    
    # Fear & Greed Index
    fg_api = FearGreedIndexAPI()
    
    # Current
    current = fg_api.get_current_index()
    if current:
        print(f"\n✅ Current Fear & Greed: {current['value']} ({current['classification']})")
    
    # Historical
    historical = fg_api.get_historical_data(days=30)
    print(f"\n📊 Historical data shape: {historical.shape}")
    print(historical.head())
    
    # Aggregator test
    aggregator = SentimentAggregator()
    aggregator.print_current_sentiment()
