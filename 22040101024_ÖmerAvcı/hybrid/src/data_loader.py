"""
Data Loader Module
==================
Bitcoin fiyat tahmini için çeşitli kaynaklardan veri çeken modül.

Veri Kaynakları:
- Yahoo Finance: BTC-USD OHLCV verisi
- FRED API: Makroekonomik veriler (10Y Treasury, DXY gibi)
- Placeholder: Fear & Greed Index, Google Trends

NOT: Hafta sonu sorunu - Bitcoin 7/24 işlem görür ama makro veriler 
     (hisse senetleri, tahviller) hafta sonları yok. forward fill ile 
     son bilinen değer taşınır.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Tüm veri kaynaklarını yöneten merkezi sınıf.
    DRY prensibi: Her veri kaynağı için ayrı metod.
    """
    
    def __init__(self, start_date=None, end_date=None, fred_api_key=None):
        """
        Args:
            start_date: Başlangıç tarihi (str, 'YYYY-MM-DD')
            end_date: Bitiş tarihi (str, 'YYYY-MM-DD')
            fred_api_key: FRED API anahtarı (opsiyonel)
        """
        # Varsayılan olarak son 3 yıl
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.fred_api_key = fred_api_key
        
    def load_btc_data(self):
        """
        Yahoo Finance'den Bitcoin OHLCV verisi çeker.
        
        Returns:
            pd.DataFrame: BTC fiyat verisi (Date, Open, High, Low, Close, Volume)
        """
        print(f"📊 Bitcoin verisi indiriliyor: {self.start_date} -> {self.end_date}")
        
        try:
            df = yf.download('BTC-USD', 
                           start=self.start_date, 
                           end=self.end_date, 
                           progress=False)
            
            # MultiIndex kolonları düzelt
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            print(f"✅ Bitcoin verisi yüklendi: {len(df)} gün")
            return df
            
        except Exception as e:
            print(f"❌ Bitcoin verisi indirilemedi: {e}")
            return pd.DataFrame()
    
    def load_sp500_data(self):
        """
        S&P 500 endeksini yükler (korelasyon analizi için).
        
        NOT: Hafta sonları kapalı! forward fill gerekli.
        
        Returns:
            pd.DataFrame: S&P 500 kapanış fiyatı
        """
        print("📈 S&P 500 verisi indiriliyor...")
        
        try:
            df = yf.download('^GSPC', 
                           start=self.start_date, 
                           end=self.end_date, 
                           progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df = df[['Date', 'Close']].rename(columns={'Close': 'SPX_Close'})
            print(f"✅ S&P 500 verisi yüklendi: {len(df)} gün")
            return df
            
        except Exception as e:
            print(f"❌ S&P 500 verisi indirilemedi: {e}")
            return pd.DataFrame()
    
    def load_dxy_data(self):
        """
        US Dollar Index (DXY) - Dolar gücü göstergesi.
        
        NOT: DXY için Yahoo Finance sembolü: DX-Y.NYB
        
        Returns:
            pd.DataFrame: DXY değerleri
        """
        print("💵 DXY (Dolar Endeksi) verisi indiriliyor...")
        
        try:
            df = yf.download('DX-Y.NYB', 
                           start=self.start_date, 
                           end=self.end_date, 
                           progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df = df[['Date', 'Close']].rename(columns={'Close': 'DXY'})
            print(f"✅ DXY verisi yüklendi: {len(df)} gün")
            return df
            
        except Exception as e:
            print(f"⚠️ DXY verisi indirilemedi, varsayılan değerler kullanılacak: {e}")
            return pd.DataFrame()
    
    def load_treasury_10y(self):
        """
        ABD 10 Yıllık Tahvil Faizi (^TNX).
        
        UYARI: Tahvil piyasası hafta sonları kapalı!
        Missing values için forward fill şart.
        
        Returns:
            pd.DataFrame: 10Y Treasury Yield
        """
        print("🏦 10 Yıllık Tahvil Faizi indiriliyor...")
        
        try:
            df = yf.download('^TNX', 
                           start=self.start_date, 
                           end=self.end_date, 
                           progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df = df[['Date', 'Close']].rename(columns={'Close': 'Treasury_10Y'})
            print(f"✅ Tahvil faizi yüklendi: {len(df)} gün")
            return df
            
        except Exception as e:
            print(f"⚠️ Tahvil faizi indirilemedi, varsayılan değerler kullanılacak: {e}")
            return pd.DataFrame()
    
    def load_fred_data(self, series_id, column_name):
        """
        FRED API üzerinden makroekonomik veri çeker.
        
        ÖNEMLI: FRED verileri genelde AYLIK veya ÇEYREKLIK gelir!
        Bu verileri günlük BTC verisine uyarlamak için:
        1. Tarihleri birleştir (merge)
        2. method='ffill' ile boşlukları doldur (son bilinen değeri taşı)
        
        Args:
            series_id: FRED seri kodu (örn: 'DGS10' - 10Y Treasury)
            column_name: DataFrame'deki kolon adı
            
        Returns:
            pd.DataFrame: Günlük frekansa dönüştürülmüş veri
        """
        if not self.fred_api_key:
            print(f"⚠️ FRED API key yok, {column_name} için placeholder oluşturuluyor...")
            return pd.DataFrame()
        
        try:
            # FRED API entegrasyonu için pandas_datareader kullanılabilir
            from pandas_datareader import data as pdr
            
            df = pdr.DataReader(series_id, 'fred', self.start_date, self.end_date)
            df = df.reset_index()
            df.columns = ['Date', column_name]
            
            print(f"✅ FRED {series_id} verisi yüklendi")
            return df
            
        except ImportError:
            print("⚠️ pandas_datareader kurulu değil, manuel veri kullanılacak")
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ FRED API hatası: {e}")
            return pd.DataFrame()
    
    def create_fear_greed_placeholder(self, btc_df):
        """
        Fear & Greed Index için placeholder oluşturur.
        
        NOT: Gerçek veri için API (alternative.me/crypto/fear-and-greed-index/)
        kullanılabilir. Şimdilik 0-100 arası random değer.
        
        Args:
            btc_df: Bitcoin DataFrame (Date kolonu için)
            
        Returns:
            pd.DataFrame: Fear & Greed değerleri
        """
        print("😱 Fear & Greed Index (Placeholder) oluşturuluyor...")
        
        df = pd.DataFrame({
            'Date': btc_df['Date'],
            'Fear_Greed': np.random.randint(20, 80, size=len(btc_df))
        })
        
        print(f"✅ Fear & Greed placeholder: {len(df)} gün")
        return df
    
    def create_google_trends_placeholder(self, btc_df):
        """
        Google Trends "Bitcoin" arama hacmi için placeholder.
        
        NOT: Gerçek veri için pytrends kütüphanesi kullanılabilir.
        
        Args:
            btc_df: Bitcoin DataFrame (Date kolonu için)
            
        Returns:
            pd.DataFrame: Google Trends değerleri
        """
        print("🔍 Google Trends (Placeholder) oluşturuluyor...")
        
        df = pd.DataFrame({
            'Date': btc_df['Date'],
            'Google_Trends': np.random.randint(30, 100, size=len(btc_df))
        })
        
        print(f"✅ Google Trends placeholder: {len(df)} gün")
        return df
    
    def merge_all_data(self):
        """
        Tüm veri kaynaklarını birleştirir.
        
        KRITIK NOKTA: Hafta sonu problemi!
        - Bitcoin: 7/24 işlem var
        - SPX, Tahviller: Hafta sonu yok
        
        Çözüm: left join + forward fill (ffill)
        
        Returns:
            pd.DataFrame: Birleştirilmiş veri seti
        """
        print("\n" + "="*60)
        print("🔗 TÜM VERİLER BİRLEŞTİRİLİYOR...")
        print("="*60 + "\n")
        
        # 1. Bitcoin ana veri
        btc_df = self.load_btc_data()
        if btc_df.empty:
            raise ValueError("Bitcoin verisi yüklenemedi!")
        
        # 2. S&P 500
        spx_df = self.load_sp500_data()
        if not spx_df.empty:
            btc_df = btc_df.merge(spx_df, on='Date', how='left')
            # HAFTA SONU SORUNU: Forward fill ile doldur
            btc_df['SPX_Close'] = btc_df['SPX_Close'].ffill()
        
        # 3. DXY
        dxy_df = self.load_dxy_data()
        if not dxy_df.empty:
            btc_df = btc_df.merge(dxy_df, on='Date', how='left')
            btc_df['DXY'] = btc_df['DXY'].ffill()
        
        # 4. 10Y Treasury
        treasury_df = self.load_treasury_10y()
        if not treasury_df.empty:
            btc_df = btc_df.merge(treasury_df, on='Date', how='left')
            # UYARI: Tahvil faizi aylık güncellenebilir, ffill kritik!
            btc_df['Treasury_10Y'] = btc_df['Treasury_10Y'].ffill()
        
        # 5. Fear & Greed (Placeholder)
        fg_df = self.create_fear_greed_placeholder(btc_df)
        btc_df = btc_df.merge(fg_df, on='Date', how='left')
        
        # 6. Google Trends (Placeholder)
        gt_df = self.create_google_trends_placeholder(btc_df)
        btc_df = btc_df.merge(gt_df, on='Date', how='left')
        
        # Kalan eksik değerleri forward fill ile doldur
        btc_df = btc_df.ffill()
        
        # Başta kalan NaN'leri backward fill ile doldur
        btc_df = btc_df.bfill()
        
        print("\n✅ TÜM VERİLER BİRLEŞTİRİLDİ!")
        print(f"📊 Toplam Gün: {len(btc_df)}")
        print(f"📊 Toplam Özellik: {len(btc_df.columns)}")
        print(f"\nKolonlar: {list(btc_df.columns)}\n")
        
        # Eksik değer kontrolü
        missing = btc_df.isnull().sum()
        if missing.sum() > 0:
            print("⚠️ UYARI: Hala eksik değerler var!")
            print(missing[missing > 0])
        else:
            print("✅ Eksik değer yok, veri seti temiz!\n")
        
        return btc_df


if __name__ == "__main__":
    # Test kodu
    loader = DataLoader()
    data = loader.merge_all_data()
    print(data.head(10))
    print(f"\nVeri Şekli: {data.shape}")
