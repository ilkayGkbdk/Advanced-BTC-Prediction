"""
Preprocessing Module
====================
Veri ön işleme ve eğitim/test ayrımı.

KRITIK NOKTALAR:
1. Look-ahead Bias (Gelecek Veri Sızıntısı) Önleme
2. Log Returns kullanımı (fiyat gürültüsünü azaltır)
3. TimeSeriesSplit (zaman serisi için doğru CV)
4. MinMaxScaler (özellikle LSTM için gerekli)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Veri ön işleme ve train-test split işlemlerini yöneten sınıf.
    """
    
    def __init__(self, df, target_col='Close'):
        """
        Args:
            df: Feature engineering sonrası veri
            target_col: Tahmin edilecek hedef kolon (default: 'Close')
        """
        self.df = df.copy()
        self.target_col = target_col
        self.scaler_X = None
        self.scaler_y = None
        
    def create_log_returns(self):
        """
        Logaritmik getiri (Log Returns) oluşturur.
        
        NEDEN LOG RETURNS?
        1. Fiyat gürültüsünü azaltır
        2. Stasyonerlik sağlar (zaman serisi modelleme için kritik)
        3. Yüzde değişim mantığı: log(P_t) - log(P_t-1) ≈ (P_t - P_t-1) / P_t-1
        
        Formül: log_return = ln(P_t / P_t-1) = ln(P_t) - ln(P_t-1)
        
        NOT: Model log return tahmin edecek, sonra geriye fiyata çevireceğiz.
        """
        print("📉 Log Returns hesaplanıyor...")
        
        # Log return hesapla
        self.df['Log_Return'] = np.log(self.df[self.target_col] / 
                                       self.df[self.target_col].shift(1))
        
        # İlk satırda NaN oluşur (shift nedeniyle), drop edelim
        self.df = self.df.dropna(subset=['Log_Return'])
        
        print(f"✅ Log Returns oluşturuldu")
        print(f"   Mean: {self.df['Log_Return'].mean():.6f}")
        print(f"   Std: {self.df['Log_Return'].std():.6f}")
        
        return self.df
    
    def prepare_data(self, target='Log_Return', exclude_cols=None):
        """
        Veriyi X (features) ve y (target) olarak ayırır.
        
        ÖNEMLI: Veri sızıntısı (look-ahead bias) engelleme!
        - Target'ı hesaplarken gelecek bilgisi kullanılmaz
        - Sadece geçmiş verilerle tahmin yapılır
        
        Args:
            target: Hedef değişken (default: 'Log_Return')
            exclude_cols: X'ten hariç tutulacak kolonlar
            
        Returns:
            X, y: Features ve target
        """
        print("\n📊 X ve y ayrılıyor...")
        
        # Hedef değişken var mı kontrol et
        if target not in self.df.columns:
            raise ValueError(f"Hedef değişken '{target}' bulunamadı!")
        
        # Hariç tutulacak kolonlar
        if exclude_cols is None:
            exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                          'Log_Return', 'Day_of_Week', 'Day_of_Month', 'Month', 'Quarter']
        
        # Feature seçimi
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols].values
        y = self.df[target].values
        
        print(f"✅ X shape: {X.shape}")
        print(f"✅ y shape: {y.shape}")
        print(f"✅ Feature sayısı: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def timeseries_train_test_split(self, X, y, test_size=0.2):
        """
        Zaman serisi için train-test split.
        
        UYARI: Zaman serisi verilerinde RANDOM SPLIT YAPILMAZ!
        - Eğitim verisi: Geçmiş (örn: 2021-2023)
        - Test verisi: Gelecek (örn: 2024)
        
        Neden?
        - Gelecek tahmini yapıyoruz, geçmişe bakarak eğitmeliyiz
        - Random split yaparsak "gelecekteki bilgi" ile eğitmiş oluruz (veri sızıntısı!)
        
        Args:
            X: Features
            y: Target
            test_size: Test set oranı (default: 0.2)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n✂️ Train-Test Split (Zaman Serisi)...")
        
        # Split noktası
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"✅ Train: {X_train.shape[0]} örneklem ({(1-test_size)*100:.0f}%)")
        print(f"✅ Test: {X_test.shape[0]} örneklem ({test_size*100:.0f}%)")
        print(f"\n⚠️ VERİ SIZINTISI ENGELLEME:")
        print(f"   - Train verisi: İlk {split_idx} gün (GEÇMIŞ)")
        print(f"   - Test verisi: Son {len(X)-split_idx} gün (GELECEK)")
        print(f"   - Gelecek verisi eğitimde kullanılmadı! ✅")
        
        return X_train, X_test, y_train, y_test
    
    def get_timeseries_cv_splits(self, X, y, n_splits=5):
        """
        Zaman serisi için cross-validation split'leri oluşturur.
        
        TimeSeriesSplit:
        - Her split'te eğitim seti büyür, test seti ileriye kayar
        - Gerçek dünya simülasyonu: Her zaman geçmişten öğrenip geleceği tahmin ederiz
        
        Example (n_splits=3):
        Split 1: Train [0:100], Test [100:150]
        Split 2: Train [0:150], Test [150:200]
        Split 3: Train [0:200], Test [200:250]
        
        Args:
            X: Features
            y: Target
            n_splits: Kaç CV fold (default: 5)
            
        Returns:
            TimeSeriesSplit object
        """
        print(f"\n📅 TimeSeriesSplit ({n_splits} folds) hazırlanıyor...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        print(f"✅ {n_splits} fold oluşturuldu")
        print(f"   Kullanım: model.cross_val_score(cv=tscv)")
        
        return tscv
    
    def scale_features(self, X_train, X_test, scaler_type='minmax'):
        """
        Özellikleri ölçeklendirir (Scaling).
        
        NEDEN SCALING?
        1. LightGBM: Zorunlu değil ama yardımcı olur
        2. LSTM: ZORUNLU! (0-1 arası normalize edilmeli)
        3. Farklı ölçekli değişkenler (örn: Fiyat 50000, RSI 50) dengelenir
        
        ÖNEMLİ: Scaler sadece TRAIN verisi ile fit edilir!
        - Test verisine aynı transform uygulanır
        - Böylece veri sızıntısı engellenir
        
        Args:
            X_train: Eğitim features
            X_test: Test features
            scaler_type: 'minmax' veya 'standard'
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        print(f"\n📏 Feature Scaling ({scaler_type})...")
        
        if scaler_type == 'minmax':
            self.scaler_X = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler_X = StandardScaler()
        else:
            raise ValueError("scaler_type 'minmax' veya 'standard' olmalı")
        
        # FIT sadece train verisi ile!
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        # Test verisine aynı transform'u uygula
        X_test_scaled = self.scaler_X.transform(X_test)
        
        print(f"✅ Scaling tamamlandı")
        print(f"   Scaler: {scaler_type}")
        print(f"   Train Min-Max: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
        print(f"   Test Min-Max: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")
        print(f"\n⚠️ VERİ SIZINTISI ENGELLEME:")
        print(f"   - Scaler SADECE train verisi ile fit edildi ✅")
        print(f"   - Test verisi fit sırasında kullanılmadı ✅")
        
        return X_train_scaled, X_test_scaled
    
    def scale_target(self, y_train, y_test):
        """
        Hedef değişkeni ölçeklendirir (LSTM için gerekli).
        
        Args:
            y_train: Eğitim target
            y_test: Test target
            
        Returns:
            y_train_scaled, y_test_scaled
        """
        print("\n🎯 Target Scaling...")
        
        self.scaler_y = MinMaxScaler()
        
        # Target'ı 2D array'e çevir (scaler için gerekli)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # Fit sadece train ile
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        print(f"✅ Target scaling tamamlandı")
        print(f"   Train Min-Max: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
        
        return y_train_scaled.flatten(), y_test_scaled.flatten()
    
    def inverse_transform_target(self, y_scaled):
        """
        Ölçeklendirilmiş target'ı orijinal ölçeğe geri döndürür.
        
        Args:
            y_scaled: Scaled target değerleri
            
        Returns:
            Original scale target
        """
        if self.scaler_y is None:
            raise ValueError("Scaler henüz fit edilmemiş!")
        
        y_scaled = y_scaled.reshape(-1, 1)
        y_original = self.scaler_y.inverse_transform(y_scaled)
        return y_original.flatten()
    
    def inverse_transform_features(self, X_scaled):
        """
        Ölçeklendirilmiş features'ı orijinal ölçeğe geri döndürür.
        
        Args:
            X_scaled: Scaled features
            
        Returns:
            Original scale features
        """
        if self.scaler_X is None:
            raise ValueError("Scaler henüz fit edilmemiş!")
        
        return self.scaler_X.inverse_transform(X_scaled)
    
    def log_return_to_price(self, log_returns, initial_price):
        """
        Log return'leri fiyata çevirir.
        
        Formül: P_t = P_{t-1} * exp(log_return_t)
        
        Args:
            log_returns: Log return dizisi
            initial_price: Başlangıç fiyatı
            
        Returns:
            Fiyat dizisi
        """
        prices = [initial_price]
        
        for lr in log_returns:
            new_price = prices[-1] * np.exp(lr)
            prices.append(new_price)
        
        return np.array(prices[1:])  # İlk değeri çıkar
    
    def prepare_lstm_sequences(self, X, y, seq_length=60):
        """
        LSTM için sequence (dizi) oluşturur.
        
        LSTM'in çalışma mantığı:
        - Sequence: Son N günün verisi → Yarının tahmini
        - Örnek: Son 60 günü ver → 61. günü tahmin et
        
        Args:
            X: Features
            y: Target
            seq_length: Sequence uzunluğu (default: 60)
            
        Returns:
            X_sequences, y_sequences
        """
        print(f"\n🔗 LSTM Sequences oluşturuluyor (seq_length={seq_length})...")
        
        X_seq = []
        y_seq = []
        
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])  # Son N gün
            y_seq.append(y[i])  # N+1. günün değeri
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"✅ Sequences oluşturuldu")
        print(f"   X shape: {X_seq.shape} (samples, timesteps, features)")
        print(f"   y shape: {y_seq.shape}")
        
        return X_seq, y_seq


class FullPipeline:
    """
    Tüm preprocessing adımlarını tek seferde çalıştıran wrapper sınıf.
    """
    
    def __init__(self, featured_df, target_col='Close'):
        """
        Args:
            featured_df: Feature engineering sonrası DataFrame
            target_col: Tahmin edilecek kolon
        """
        self.preprocessor = DataPreprocessor(featured_df, target_col)
    
    def run_lightgbm_pipeline(self, test_size=0.2, scaler_type='minmax'):
        """
        LightGBM için tam preprocessing pipeline.
        
        Returns:
            Dict: Tüm gerekli veriler
        """
        print("\n" + "="*60)
        print("🔧 LIGHTGBM PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Log returns oluştur
        self.preprocessor.create_log_returns()
        
        # 2. X, y ayır
        X, y, feature_names = self.preprocessor.prepare_data(target='Log_Return')
        
        # 3. Train-test split (zaman serisi)
        X_train, X_test, y_train, y_test = \
            self.preprocessor.timeseries_train_test_split(X, y, test_size)
        
        # 4. Scaling (opsiyonel ama tavsiye edilir)
        X_train_scaled, X_test_scaled = \
            self.preprocessor.scale_features(X_train, X_test, scaler_type)
        
        print("\n✅ LightGBM preprocessing tamamlandı!")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'preprocessor': self.preprocessor,
            'original_df': self.preprocessor.df
        }
    
    def run_lstm_pipeline(self, test_size=0.2, seq_length=60):
        """
        LSTM için tam preprocessing pipeline.
        
        Returns:
            Dict: Tüm gerekli veriler
        """
        print("\n" + "="*60)
        print("🔧 LSTM PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Log returns oluştur
        self.preprocessor.create_log_returns()
        
        # 2. X, y ayır
        X, y, feature_names = self.preprocessor.prepare_data(target='Log_Return')
        
        # 3. Train-test split
        X_train, X_test, y_train, y_test = \
            self.preprocessor.timeseries_train_test_split(X, y, test_size)
        
        # 4. Scaling (LSTM için zorunlu!)
        X_train_scaled, X_test_scaled = \
            self.preprocessor.scale_features(X_train, X_test, 'minmax')
        
        # 5. Target scaling
        y_train_scaled, y_test_scaled = \
            self.preprocessor.scale_target(y_train, y_test)
        
        # 6. LSTM sequences
        X_train_seq, y_train_seq = \
            self.preprocessor.prepare_lstm_sequences(X_train_scaled, y_train_scaled, seq_length)
        X_test_seq, y_test_seq = \
            self.preprocessor.prepare_lstm_sequences(X_test_scaled, y_test_scaled, seq_length)
        
        print("\n✅ LSTM preprocessing tamamlandı!")
        
        return {
            'X_train': X_train_seq,
            'X_test': X_test_seq,
            'y_train': y_train_seq,
            'y_test': y_test_seq,
            'feature_names': feature_names,
            'preprocessor': self.preprocessor,
            'original_df': self.preprocessor.df,
            'seq_length': seq_length
        }


if __name__ == "__main__":
    # Test kodu
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    # Veri yükle
    loader = DataLoader()
    raw_data = loader.merge_all_data()
    
    # Feature engineering
    engineer = FeatureEngineer(raw_data)
    featured_data = engineer.create_all_features(n_lags=30)
    
    # LightGBM pipeline test
    pipeline = FullPipeline(featured_data)
    lgb_data = pipeline.run_lightgbm_pipeline()
    
    print(f"\n📊 LightGBM Hazır Veri:")
    print(f"   X_train: {lgb_data['X_train'].shape}")
    print(f"   y_train: {lgb_data['y_train'].shape}")
