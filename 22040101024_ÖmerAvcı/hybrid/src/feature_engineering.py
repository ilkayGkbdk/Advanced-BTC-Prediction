"""
Feature Engineering Module
===========================
Bitcoin fiyat tahmini için gelişmiş özellik mühendisliği.

Özellikler:
1. Teknik İndikatörler: RSI, MACD, Bollinger Bands, ATR, VWAP
2. Cyclical Encoding: Haftanın günü, ayın günü (Sin/Cos dönüşümü)
3. Lag Features: Geçmiş fiyat bilgisi
4. Rolling Statistics: Hareketli ortalamalar, volatilite

NOT: Tüm hesaplamalar "look-ahead bias" (gelecek veri sızıntısı) 
     engellenecek şekilde yapılmalı!
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Tüm feature engineering işlemlerini yöneten sınıf.
    DRY prensibi: Her özellik tipi için ayrı metod.
    """
    
    def __init__(self, df):
        """
        Args:
            df: Raw data (Date, OHLCV + makro veriler içeren DataFrame)
        """
        self.df = df.copy()
        
    def add_technical_indicators(self):
        """
        Teknik analiz indikatörlerini ekler.
        
        Kullanılan İndikatörler:
        - RSI (14): Momentum göstergesi (0-100 arası)
        - MACD: Trend takip göstergesi
        - Bollinger Bands: Volatilite göstergesi
        - ATR (14): Ortalama True Range (volatilite)
        - VWAP: Volume Weighted Average Price
        
        NOT: 'ta' kütüphanesi kullanılıyor, otomatik NaN handling var.
        """
        print("📈 Teknik indikatörler ekleniyor...")
        
        try:
            # 1. RSI (Relative Strength Index)
            rsi = RSIIndicator(close=self.df['Close'], window=14)
            self.df['RSI'] = rsi.rsi()
            
            # 2. MACD (Moving Average Convergence Divergence)
            macd = MACD(close=self.df['Close'])
            self.df['MACD'] = macd.macd()
            self.df['MACD_Signal'] = macd.macd_signal()
            self.df['MACD_Diff'] = macd.macd_diff()
            
            # 3. Bollinger Bands
            bollinger = BollingerBands(close=self.df['Close'])
            self.df['BB_High'] = bollinger.bollinger_hband()
            self.df['BB_Low'] = bollinger.bollinger_lband()
            self.df['BB_Mid'] = bollinger.bollinger_mavg()
            self.df['BB_Width'] = self.df['BB_High'] - self.df['BB_Low']
            
            # 4. ATR (Average True Range) - Volatilite
            atr = AverageTrueRange(
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                window=14
            )
            self.df['ATR'] = atr.average_true_range()
            
            # 5. VWAP (Volume Weighted Average Price)
            # NOT: VWAP günlük resetlenir, burada basit versiyonu
            vwap = VolumeWeightedAveragePrice(
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                volume=self.df['Volume']
            )
            self.df['VWAP'] = vwap.volume_weighted_average_price()
            
            # 6. EMA (Exponential Moving Average) - Ekstra
            ema_50 = EMAIndicator(close=self.df['Close'], window=50)
            ema_200 = EMAIndicator(close=self.df['Close'], window=200)
            self.df['EMA_50'] = ema_50.ema_indicator()
            self.df['EMA_200'] = ema_200.ema_indicator()
            
            print(f"✅ {11} teknik indikatör eklendi")
            
        except Exception as e:
            print(f"⚠️ Teknik indikatör hesaplama hatası: {e}")
    
    def add_lag_features(self, n_lags=30):
        """
        Geçmiş fiyat bilgisini özellik olarak ekler (Lag Features).
        
        UYARI: Veri sızıntısı riski!
        - Eğitim sırasında gelecek bilgisi kullanılmamalı.
        - Bu yüzden sadece geçmiş (t-1, t-2, ...) değerleri kullanıyoruz.
        
        Args:
            n_lags: Kaç gün geriye gidilecek (default: 30)
        """
        print(f"🔙 {n_lags} günlük lag features ekleniyor...")
        
        # Close fiyatı için lag'ler
        for i in range(1, n_lags + 1):
            self.df[f'Close_Lag_{i}'] = self.df['Close'].shift(i)
        
        # Volume için de ekleyelim (3, 7, 14 gün)
        for i in [3, 7, 14]:
            self.df[f'Volume_Lag_{i}'] = self.df['Volume'].shift(i)
        
        print(f"✅ {n_lags + 3} lag feature eklendi")
    
    def add_rolling_features(self):
        """
        Hareketli pencere istatistikleri (Rolling Statistics).
        
        Özellikler:
        - Moving Averages (7, 14, 30 günlük)
        - Rolling Volatility (Standart sapma)
        - Rolling Return (Ortalama getiri)
        
        NOT: window parametresi - geçmiş kaç günün ortalaması
        """
        print("📊 Rolling statistics ekleniyor...")
        
        windows = [7, 14, 30]
        
        for window in windows:
            # Moving Average
            self.df[f'MA_{window}'] = self.df['Close'].rolling(window=window).mean()
            
            # Rolling Volatility (Standart Sapma)
            self.df[f'Volatility_{window}'] = self.df['Close'].rolling(window=window).std()
            
            # Rolling Return
            self.df[f'Return_{window}'] = self.df['Close'].pct_change(window)
        
        print(f"✅ {len(windows) * 3} rolling feature eklendi")
    
    def add_cyclical_features(self):
        """
        Zaman döngülerini Sin/Cos dönüşümü ile kodlar.
        
        Neden Cyclical Encoding?
        - Haftanın günü: Pazar (0) ve Cumartesi (6) birbirine yakın!
        - Ayın günü: 1. gün ve 31. gün döngüsel olarak yakın.
        - One-hot encoding bu ilişkiyi kaçırır, sin/cos kodar.
        
        Formül:
        - sin(2π * değer / max_değer)
        - cos(2π * değer / max_değer)
        """
        print("🔄 Cyclical time features ekleniyor...")
        
        # Date kolonunu datetime'a çevir
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Haftanın Günü (0=Pazartesi, 6=Pazar)
        self.df['Day_of_Week'] = self.df['Date'].dt.dayofweek
        self.df['Day_of_Week_Sin'] = np.sin(2 * np.pi * self.df['Day_of_Week'] / 7)
        self.df['Day_of_Week_Cos'] = np.cos(2 * np.pi * self.df['Day_of_Week'] / 7)
        
        # Ayın Günü (1-31 arası)
        self.df['Day_of_Month'] = self.df['Date'].dt.day
        self.df['Day_of_Month_Sin'] = np.sin(2 * np.pi * self.df['Day_of_Month'] / 31)
        self.df['Day_of_Month_Cos'] = np.cos(2 * np.pi * self.df['Day_of_Month'] / 31)
        
        # Ayın Kendisi (1-12)
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Month_Sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_Cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)
        
        # Yılın Çeyreği (1-4)
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        print("✅ Cyclical time features eklendi (Sin/Cos)")
    
    def add_price_momentum(self):
        """
        Fiyat momentum göstergeleri.
        
        - ROC: Rate of Change (Değişim oranı)
        - Price Position: Fiyatın yüksek/düşük aralığındaki konumu
        """
        print("⚡ Momentum features ekleniyor...")
        
        # Rate of Change (1, 7, 14, 30 günlük)
        for period in [1, 7, 14, 30]:
            self.df[f'ROC_{period}'] = (
                (self.df['Close'] - self.df['Close'].shift(period)) / 
                self.df['Close'].shift(period) * 100
            )
        
        # Price Position (Günlük high-low içinde nerede?)
        self.df['Price_Position'] = (
            (self.df['Close'] - self.df['Low']) / 
            (self.df['High'] - self.df['Low'] + 1e-10)  # Sıfıra bölmeyi engelle
        )
        
        print("✅ Momentum features eklendi")
    
    def add_volume_features(self):
        """
        Hacim (Volume) bazlı özellikler.
        
        NOT: Hacim, piyasa ilgisinin göstergesidir.
        Yüksek hacim = güçlü hareket sinyali.
        """
        print("📊 Volume features ekleniyor...")
        
        # Volume Moving Average
        self.df['Volume_MA_7'] = self.df['Volume'].rolling(window=7).mean()
        self.df['Volume_MA_30'] = self.df['Volume'].rolling(window=30).mean()
        
        # Volume Ratio (güncel hacim / ortalama hacim)
        self.df['Volume_Ratio_7'] = self.df['Volume'] / (self.df['Volume_MA_7'] + 1)
        self.df['Volume_Ratio_30'] = self.df['Volume'] / (self.df['Volume_MA_30'] + 1)
        
        # Volume Change
        self.df['Volume_Change'] = self.df['Volume'].pct_change()
        
        print("✅ Volume features eklendi")
    
    def add_macro_interactions(self):
        """
        Makroekonomik verilerin türevleri ve etkileşimleri.
        
        NOT: Bu değişkenler model için çok güçlü olabilir!
        - SPX-BTC korelasyonu
        - DXY-BTC ters korelasyon (Dolar güçlenirse Bitcoin zayıflar)
        - Faiz-BTC ilişkisi
        """
        print("🌍 Macro interaction features ekleniyor...")
        
        # SPX korelasyonu (varsa)
        if 'SPX_Close' in self.df.columns:
            self.df['SPX_Return'] = self.df['SPX_Close'].pct_change()
            self.df['BTC_SPX_Ratio'] = self.df['Close'] / (self.df['SPX_Close'] + 1)
        
        # DXY etkisi (varsa)
        if 'DXY' in self.df.columns:
            self.df['DXY_Change'] = self.df['DXY'].pct_change()
            self.df['BTC_DXY_Ratio'] = self.df['Close'] / (self.df['DXY'] + 1)
        
        # Faiz değişimi (varsa)
        if 'Treasury_10Y' in self.df.columns:
            self.df['Treasury_Change'] = self.df['Treasury_10Y'].diff()
        
        # Fear & Greed'in hareketli ortalaması
        if 'Fear_Greed' in self.df.columns:
            self.df['Fear_Greed_MA_7'] = self.df['Fear_Greed'].rolling(window=7).mean()
        
        print("✅ Macro interaction features eklendi")
    
    def add_support_resistance_levels(self, lookback_windows=[20, 50, 100]):
        """
        Support (destek) ve Resistance (direnç) seviyelerini hesaplar.
        
        LIQUIDITY MANTIK:
        Fiyat, geçmişte sıkça "test edilen" seviyelere geri dönme eğilimindedir.
        Bu seviyeler "likidite havuzları" olarak davranır.
        
        HESAPLAMA:
        1. Son N günlük high/low değerlerini al
        2. Pivot points hesapla (klasik teknik analiz)
        3. Fiyatın bu seviyelere uzaklığını hesapla
        
        PIVOT POINTS FORMÜLÜ:
        - Pivot (P) = (High + Low + Close) / 3
        - R1 (Resistance 1) = 2*P - Low
        - R2 = P + (High - Low)
        - S1 (Support 1) = 2*P - High
        - S2 = P - (High - Low)
        
        Args:
            lookback_windows: Farklı zaman periyotları (gün)
        """
        print("🎯 Support/Resistance levels ekleniyor...")
        
        for window in lookback_windows:
            # Rolling high/low
            rolling_high = self.df['High'].rolling(window=window).max()
            rolling_low = self.df['Low'].rolling(window=window).min()
            
            # Pivot Point (klasik formül)
            pivot = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
            
            # Resistance levels
            r1 = 2 * pivot - self.df['Low']
            r2 = pivot + (self.df['High'] - self.df['Low'])
            
            # Support levels
            s1 = 2 * pivot - self.df['High']
            s2 = pivot - (self.df['High'] - self.df['Low'])
            
            # Feature olarak ekle
            self.df[f'Pivot_{window}'] = pivot
            self.df[f'R1_{window}'] = r1
            self.df[f'R2_{window}'] = r2
            self.df[f'S1_{window}'] = s1
            self.df[f'S2_{window}'] = s2
            
            # Mevcut fiyatın bu seviyelere uzaklığı (normalized)
            self.df[f'Distance_to_R1_{window}'] = (r1 - self.df['Close']) / self.df['Close']
            self.df[f'Distance_to_S1_{window}'] = (self.df['Close'] - s1) / self.df['Close']
            
            # Fiyat pivot'un üstünde mi altında mı?
            self.df[f'Above_Pivot_{window}'] = (self.df['Close'] > pivot).astype(int)
            
            # Support/Resistance strength (kaç kez test edildi?)
            # Basitleştirilmiş: rolling pencerede kaç kez fiyat bu seviyelere yaklaştı
            tolerance = 0.02  # %2 tolerance
            
            def count_touches(series, level, tolerance):
                """Fiyatın belirli bir seviyeye kaç kez yaklaştığını sayar"""
                touches = ((series >= level * (1 - tolerance)) & 
                          (series <= level * (1 + tolerance))).astype(int)
                return touches.rolling(window=window).sum()
            
            self.df[f'R1_Strength_{window}'] = count_touches(self.df['High'], r1, tolerance)
            self.df[f'S1_Strength_{window}'] = count_touches(self.df['Low'], s1, tolerance)
        
        print(f"✅ Support/Resistance features eklendi ({len(lookback_windows)} pencere)")
    
    def add_liquidity_zones(self, volume_quantile=0.90):
        """
        Yüksek hacimli bölgeleri "likidite zonları" olarak işaretler.
        
        MANTIK:
        Yüksek hacimli fiyat seviyeleri = çok trade yapılmış = likidite var
        Fiyat bu bölgelere geri dönme eğilimindedir (value area).
        
        Args:
            volume_quantile: Hangi hacim seviyesi üstü "yüksek" sayılacak
        """
        print("💧 Liquidity zones ekleniyor...")
        
        # Yüksek hacim eşiği
        volume_threshold = self.df['Volume'].quantile(volume_quantile)
        
        # Yüksek hacimli günleri işaretle
        self.df['High_Volume_Day'] = (self.df['Volume'] > volume_threshold).astype(int)
        
        # Bu günlerdeki fiyat seviyeleri (OHLC ortalaması)
        self.df['High_Volume_Price'] = np.where(
            self.df['High_Volume_Day'] == 1,
            (self.df['Open'] + self.df['High'] + self.df['Low'] + self.df['Close']) / 4,
            np.nan
        )
        
        # Son N günlük yüksek hacimli fiyatın ortalaması (likidite merkezi)
        for window in [20, 50]:
            liquidity_center = self.df['High_Volume_Price'].rolling(window=window).mean()
            self.df[f'Liquidity_Center_{window}'] = liquidity_center
            
            # Mevcut fiyatın likidite merkezine uzaklığı
            self.df[f'Distance_to_Liquidity_{window}'] = (
                (self.df['Close'] - liquidity_center) / self.df['Close']
            )
        
        print("✅ Liquidity zone features eklendi")
    
    def create_all_features(self, n_lags=30):
        """
        Tüm feature engineering adımlarını sırayla uygular.
        
        Args:
            n_lags: Lag feature sayısı
            
        Returns:
            pd.DataFrame: Tüm özellikler eklenmiş veri seti
        """
        print("\n" + "="*60)
        print("🔧 FEATURE ENGINEERING BAŞLIYOR...")
        print("="*60 + "\n")
        
        self.add_technical_indicators()
        self.add_lag_features(n_lags=n_lags)
        self.add_rolling_features()
        self.add_cyclical_features()
        self.add_price_momentum()
        self.add_volume_features()
        self.add_macro_interactions()
        self.add_support_resistance_levels()
        self.add_liquidity_zones()
        
        # NaN değerleri kontrol et
        print("\n" + "-"*60)
        print("🔍 Eksik Değer Kontrolü...")
        print("-"*60)
        
        # İlk n_lags satırda NaN olması normal (lag features nedeniyle)
        # Bu satırları drop edelim
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        dropped_rows = initial_rows - len(self.df)
        
        print(f"✅ {dropped_rows} satır (başlangıç NaN'ları) silindi")
        print(f"📊 Kalan satır sayısı: {len(self.df)}")
        print(f"📊 Toplam özellik sayısı: {len(self.df.columns)}")
        
        print("\n" + "="*60)
        print("✅ FEATURE ENGINEERING TAMAMLANDI!")
        print("="*60 + "\n")
        
        return self.df
    
    def get_feature_names(self, exclude_cols=None):
        """
        Model için kullanılacak feature isimlerini döndürür.
        
        Args:
            exclude_cols: Hariç tutulacak kolonlar (Date, Close gibi)
            
        Returns:
            list: Feature isimleri
        """
        if exclude_cols is None:
            exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        features = [col for col in self.df.columns if col not in exclude_cols]
        return features


if __name__ == "__main__":
    # Test kodu
    from data_loader import DataLoader
    
    loader = DataLoader()
    raw_data = loader.merge_all_data()
    
    engineer = FeatureEngineer(raw_data)
    featured_data = engineer.create_all_features(n_lags=30)
    
    print(f"\n📊 Final Veri Şekli: {featured_data.shape}")
    print(f"\n📋 İlk 5 satır:")
    print(featured_data.head())
    
    features = engineer.get_feature_names()
    print(f"\n🎯 Toplam Feature Sayısı: {len(features)}")
    print(f"Feature listesi: {features[:20]}...")  # İlk 20 feature
