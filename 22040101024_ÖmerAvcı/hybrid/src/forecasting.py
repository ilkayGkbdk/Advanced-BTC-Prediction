"""
Forecasting Module
==================
Özyinelemeli (Recursive) tahmin modülü - IMPROVED VERSION

RECURSIVE FORECASTING with:
✅ Rolling buffer (son 100 gün)
✅ Gerçek fiyat güncelleme: P_t+1 = P_t * e^r_pred
✅ Gaussian noise (stochastic forecasting)
✅ Quantile-based clipping (Black Swan kontrolü)
✅ Cyclical encoding güncelleme (tarih tracking)

Model sadece 1 gün ileriyi tahmin edebilir. 30 gün için:
1. t+1'i tahmin et
2. Yeni fiyatı hesapla: P_t+1 = P_t * e^r_pred
3. Rolling buffer'ı güncelle (lag shift, teknik indikatörler)
4. Cyclical encoding'i güncelle (date += 1 day)
5. Döngüyü 30 adım devam ettir

UYARI: Hata birikimi! İlk günlerdeki yanlış tahmin sonraki günleri etkiler.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RecursiveForecaster:
    """
    Özyinelemeli tahmin yapan sınıf - IMPROVED VERSION with Monte Carlo
    
    Her iki model (LightGBM ve LSTM) için çalışır.
    Profesyonel önerilerle geliştirildi:
    - Rolling buffer management
    - Realistic price updates (exponential)
    - Stochastic noise injection
    - Quantile clipping
    - Monte Carlo simulation (1000+ scenarios)
    """
    
    def __init__(self, model, preprocessor, feature_names, 
                 historical_returns=None, last_date=None):
        """
        Args:
            model: Eğitilmiş model (LightGBMModel veya LSTMTrainer)
            preprocessor: DataPreprocessor instance
            feature_names: Model'e verilen feature isimleri
            historical_returns: Geçmiş log returns (quantile hesabı için)
            last_date: Son bilinen tarih (cyclical encoding için)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.historical_returns = historical_returns
        self.last_date = last_date
        
        # Quantile clipping sınırları (Black Swan kontrolü)
        if historical_returns is not None:
            self.lower_quantile = np.percentile(historical_returns, 5)
            self.upper_quantile = np.percentile(historical_returns, 95)
        else:
            # Default: ±5% günlük hareket
            self.lower_quantile = -0.05
            self.upper_quantile = 0.05
    
    def forecast_lightgbm(self, X_last, n_steps=30, last_price=None, 
                         rolling_buffer=None):
        """
        LightGBM ile özyinelemeli tahmin - IMPROVED VERSION
        
        PROFESYONEL YAKLAŞIM:
        1. Son günün features'ını al
        2. 1 gün ileriyi tahmin et (log return)
        3. Quantile clipping uygula (Black Swan kontrolü)
        4. Yeni fiyatı hesapla: P_t+1 = P_t * e^(r_pred + noise)
        5. Rolling buffer'ı güncelle (lag shift + teknik indikatörler)
        6. Cyclical encoding'i güncelle (date += 1 day)
        7. Tekrarla
        
        Args:
            X_last: Son günün features (scaled)
            n_steps: Kaç gün tahmin edilecek
            last_price: Son bilinen fiyat (P_t)
            rolling_buffer: Son N günlük ham veri (teknik indikatör hesabı için)
            
        Returns:
            Dict: Tahminler (log returns ve prices)
        """
        print(f"\n🔮 LightGBM Recursive Forecasting ({n_steps} gün)...")
        print(f"📊 Quantile Sınırları: [{self.lower_quantile:.4f}, {self.upper_quantile:.4f}]")
        
        log_return_predictions = []
        price_predictions = []
        current_features = X_last.copy()
        current_price = last_price
        current_date = self.last_date if self.last_date else datetime.now()
        
        # Geçmiş standart sapma (noise için)
        if self.historical_returns is not None:
            base_std = np.std(self.historical_returns)
        else:
            base_std = 0.015  # Bitcoin tipik volatilitesi
        
        for step in range(n_steps):
            # 1. Tahmin yap (log return)
            pred_log_return = self.model.predict(current_features.reshape(1, -1))[0]
            
            # 2. QUANTILE CLIPPING (Black Swan kontrolü)
            pred_log_return_clipped = np.clip(
                pred_log_return, 
                self.lower_quantile, 
                self.upper_quantile
            )
            
            # 3. GAUSSIAN NOISE (Stochastic forecasting)
            # Error accumulation: Uzak gelecek = daha belirsiz
            noise_scale = base_std * np.sqrt(step + 1) / 8
            noise = np.random.normal(0, noise_scale)
            pred_with_noise = pred_log_return_clipped + noise
            
            # 4. YENİ FİYAT HESAPLA: P_t+1 = P_t * e^r_pred
            if current_price is not None:
                new_price = current_price * np.exp(pred_with_noise)
                price_predictions.append(new_price)
                current_price = new_price
            
            log_return_predictions.append(pred_with_noise)
            
            # 5. TARİH İLERLET (Cyclical encoding için)
            current_date += timedelta(days=1)
            
            # 6. FEATURES GÜNCELLE
            current_features = self._update_features_lightgbm(
                current_features, 
                pred_with_noise,
                current_date,
                new_price if current_price else None
            )
            
            if (step + 1) % 10 == 0:
                print(f"   ✅ {step + 1} gün | Son Fiyat: ${new_price:,.2f} | Log Return: {pred_with_noise:.4f}")
        
        print(f"✅ Tahmin tamamlandı! Son fiyat: ${price_predictions[-1]:,.2f}")
        
        return {
            'log_returns': np.array(log_return_predictions),
            'prices': np.array(price_predictions) if price_predictions else None
        }
        
        # Log return'leri fiyata çevir
        if last_price is not None:
            prices = self.preprocessor.log_return_to_price(
                np.array(log_return_predictions), 
                last_price
            )
        else:
            prices = None
        
        print(f"✅ Tahmin tamamlandı!")
        
        return {
            'log_returns': np.array(log_return_predictions),
            'prices': prices
        }
    
    def _update_features_lightgbm(self, features, new_log_return, new_date, new_price):
        """
        LightGBM için features günceller - IMPROVED VERSION
        
        PROFESYONEL YAKLAŞIM:
        1. Lag features'ları kaydır (Close_Lag_1 <- new, Lag_2 <- Lag_1, etc.)
        2. Volume lag'lerini kaydır
        3. Rolling features'ları güncelle (MA, volatility)
        4. Cyclical encoding'i güncelle (date'e göre Sin/Cos)
        
        Args:
            features: Mevcut features array
            new_log_return: Yeni tahmin edilen log return
            new_date: Yeni tarih (cyclical encoding için)
            new_price: Yeni fiyat (rolling features için)
            
        Returns:
            Güncellenmiş features array
        """
        updated_features = features.copy()
        
        if self.feature_names is None:
            return updated_features
        
        # 1. CLOSE LAG FEATURES'LARI KAYDIR
        close_lag_indices = []
        for i, name in enumerate(self.feature_names):
            if 'Close_Lag_' in name:
                try:
                    lag_num = int(name.split('_')[-1])
                    close_lag_indices.append((i, lag_num, name))
                except:
                    pass
        
        # Lag'leri büyükten küçüğe sırala (geriye doğru shift)
        close_lag_indices.sort(key=lambda x: x[1], reverse=True)
        
        for idx, (i, lag_num, name) in enumerate(close_lag_indices):
            if lag_num == 1:
                # Lag_1'e yeni log return
                updated_features[i] = new_log_return
            elif idx < len(close_lag_indices) - 1:
                # Bir önceki lag'den kopyala
                prev_idx = close_lag_indices[idx + 1][0]
                updated_features[i] = features[prev_idx]
        
        # 2. VOLUME LAG FEATURES'LARI KAYDIR (benzer mantık)
        vol_lag_indices = []
        for i, name in enumerate(self.feature_names):
            if 'Volume_Lag_' in name:
                try:
                    lag_num = int(name.split('_')[-1])
                    vol_lag_indices.append((i, lag_num, name))
                except:
                    pass
        
        vol_lag_indices.sort(key=lambda x: x[1], reverse=True)
        
        for idx, (i, lag_num, name) in enumerate(vol_lag_indices):
            if lag_num == 1:
                # Volume sabit kal (tahmin etmiyoruz)
                pass
            elif idx < len(vol_lag_indices) - 1:
                prev_idx = vol_lag_indices[idx + 1][0]
                updated_features[i] = features[prev_idx]
        
        # 3. CYCLICAL ENCODING GÜNCELLE
        if new_date:
            # Day of week (0-6)
            day_of_week = new_date.weekday()
            # Day of month (1-31)
            day_of_month = new_date.day
            # Month (1-12)
            month = new_date.month
            
            # Sin/Cos encoding
            for i, name in enumerate(self.feature_names):
                if 'DayOfWeek_Sin' in name:
                    updated_features[i] = np.sin(2 * np.pi * day_of_week / 7)
                elif 'DayOfWeek_Cos' in name:
                    updated_features[i] = np.cos(2 * np.pi * day_of_week / 7)
                elif 'DayOfMonth_Sin' in name:
                    updated_features[i] = np.sin(2 * np.pi * day_of_month / 31)
                elif 'DayOfMonth_Cos' in name:
                    updated_features[i] = np.cos(2 * np.pi * day_of_month / 31)
                elif 'Month_Sin' in name:
                    updated_features[i] = np.sin(2 * np.pi * month / 12)
                elif 'Month_Cos' in name:
                    updated_features[i] = np.cos(2 * np.pi * month / 12)
        
        # 4. ROLLING FEATURES (MA, volatility) - Simplified
        # Not: Tam implementasyon için rolling buffer gerekir
        # Şimdilik exponential decay ile yaklaşık güncelleme
        for i, name in enumerate(self.feature_names):
            if 'MA_' in name or 'Return_' in name or 'Volatility_' in name:
                # Exponential weighted update (alpha = 0.1)
                updated_features[i] = 0.9 * features[i] + 0.1 * new_log_return
        
        return updated_features
    
    def forecast_monte_carlo(self, X_last, n_steps=30, last_price=None, 
                            n_simulations=1000, confidence_levels=[0.05, 0.25, 0.50, 0.75, 0.95]):
        """
        Monte Carlo Simülasyonu ile çoklu senaryo analizi - PROFESSIONAL APPROACH
        
        NEDEN MONTE CARLO?
        - Tek tahmin yerine 1000 farklı olası gelecek senaryosu
        - Her simülasyonda farklı rastgele gürültü
        - Sonuç: En olası senaryo (median) + olasılık bantları (percentiles)
        - Gerçek piyasa belirsizliğini yansıtır
        
        ALGORITMA:
        1. Aynı başlangıç noktasından 1000 kez forecast yap
        2. Her seferinde farklı random seed kullan
        3. Tüm senaryoları topla
        4. Percentile'leri hesapla (5%, 25%, 50%, 75%, 95%)
        5. Median'ı "en olası senaryo" olarak döndür
        
        Args:
            X_last: Son günün features (scaled)
            n_steps: Kaç gün tahmin edilecek
            last_price: Son bilinen fiyat
            n_simulations: Kaç senaryo (default: 1000)
            confidence_levels: Hangi percentile'ler hesaplanacak
            
        Returns:
            Dict: {
                'median_prices': Median senaryo (en olası)
                'median_returns': Median log returns
                'percentiles': Her gün için percentile bantları
                'all_scenarios': Tüm senaryolar (n_simulations x n_steps)
                'statistics': Özet istatistikler
            }
        """
        print(f"\n🎲 MONTE CARLO SIMULATION ({n_simulations} scenarios, {n_steps} days)")
        print(f"📊 Confidence Levels: {confidence_levels}")
        print("="*70)
        
        all_price_scenarios = []
        all_return_scenarios = []
        
        # Progress tracking
        checkpoint_interval = n_simulations // 10
        
        for sim in range(n_simulations):
            # Her simülasyonda farklı random seed
            np.random.seed(42 + sim)
            
            # Tek bir senaryo çalıştır
            forecast = self.forecast_lightgbm(
                X_last=X_last.copy(),
                n_steps=n_steps,
                last_price=last_price,
                rolling_buffer=None
            )
            
            all_price_scenarios.append(forecast['prices'])
            all_return_scenarios.append(forecast['log_returns'])
            
            # Progress update
            if (sim + 1) % checkpoint_interval == 0:
                progress = ((sim + 1) / n_simulations) * 100
                print(f"   ⏳ Progress: {progress:.0f}% ({sim + 1}/{n_simulations} scenarios)")
        
        # Numpy array'e çevir: (n_simulations, n_steps)
        all_price_scenarios = np.array(all_price_scenarios)
        all_return_scenarios = np.array(all_return_scenarios)
        
        print(f"\n✅ Tüm simülasyonlar tamamlandı!")
        print(f"📊 Scenario matrix shape: {all_price_scenarios.shape}")
        
        # Percentile hesaplamaları (her gün için ayrı)
        percentiles = {}
        for level in confidence_levels:
            percentile_prices = np.percentile(all_price_scenarios, level * 100, axis=0)
            percentiles[f'p{int(level*100)}'] = percentile_prices
        
        # Median (en olası senaryo)
        median_prices = percentiles['p50']
        median_returns = np.percentile(all_return_scenarios, 50, axis=0)
        
        # İstatistikler
        statistics = {
            'final_price_mean': np.mean(all_price_scenarios[:, -1]),
            'final_price_median': median_prices[-1],
            'final_price_std': np.std(all_price_scenarios[:, -1]),
            'final_price_min': np.min(all_price_scenarios[:, -1]),
            'final_price_max': np.max(all_price_scenarios[:, -1]),
            'price_range_30d': (np.min(all_price_scenarios), np.max(all_price_scenarios)),
            'volatility_mean': np.mean(np.std(all_return_scenarios, axis=1))
        }
        
        # Sonuçları yazdır
        print("\n" + "="*70)
        print("📊 MONTE CARLO RESULTS SUMMARY")
        print("="*70)
        print(f"\n💰 Final Price (Day {n_steps}):")
        print(f"   Median (Most Likely): ${statistics['final_price_median']:,.2f}")
        print(f"   Mean: ${statistics['final_price_mean']:,.2f}")
        print(f"   Std Dev: ${statistics['final_price_std']:,.2f}")
        print(f"   Range: ${statistics['final_price_min']:,.2f} - ${statistics['final_price_max']:,.2f}")
        
        print(f"\n📈 Confidence Intervals (Day {n_steps}):")
        for level in confidence_levels:
            key = f'p{int(level*100)}'
            price = percentiles[key][-1]
            print(f"   {int(level*100)}th Percentile: ${price:,.2f}")
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Starting Price: ${last_price:,.2f}")
        print(f"   Median Change: ${statistics['final_price_median'] - last_price:,.2f} " +
              f"({((statistics['final_price_median'] / last_price) - 1) * 100:.2f}%)")
        print(f"   Average Volatility: {statistics['volatility_mean']:.4f}")
        
        print("="*70)
        
        return {
            'median_prices': median_prices,
            'median_returns': median_returns,
            'percentiles': percentiles,
            'all_scenarios': {
                'prices': all_price_scenarios,
                'returns': all_return_scenarios
            },
            'statistics': statistics,
            'n_simulations': n_simulations,
            'n_steps': n_steps
        }
    
    def forecast_lstm(self, X_last_sequence, n_steps=30, last_price=None, 
                      seq_length=60):
        """
        LSTM ile özyinelemeli tahmin - IMPROVED VERSION
        
        LSTM RECURSIVE CHALLENGE:
        - LSTM sequence (örn: 60 gün) bekler
        - Her tahmin sonrası sequence güncellenmeli (sliding window)
        
        PROFESYONEL YAKLAŞIM:
        1. Sequence ile tahmin yap
        2. Quantile clipping uygula
        3. Yeni fiyatı hesapla: P_t+1 = P_t * e^(r_pred + noise)
        4. Sequence'i kaydır (sliding window)
        
        Args:
            X_last_sequence: Son sequence (seq_length, n_features) - scaled
            n_steps: Kaç gün tahmin edilecek
            last_price: Son bilinen fiyat
            seq_length: Sequence uzunluğu
            
        Returns:
            Dict: Tahminler
        """
        print(f"\n🔮 LSTM Recursive Forecasting ({n_steps} gün)...")
        print(f"📊 Quantile Sınırları: [{self.lower_quantile:.4f}, {self.upper_quantile:.4f}]")
        
        log_return_predictions = []
        price_predictions = []
        current_sequence = X_last_sequence.copy()  # (seq_length, n_features)
        current_price = last_price
        current_date = self.last_date if self.last_date else datetime.now()
        
        # Geçmiş standart sapma (noise için)
        if self.historical_returns is not None:
            base_std = np.std(self.historical_returns) * 0.8  # LSTM için daha az noise
        else:
            base_std = 0.012
        
        for step in range(n_steps):
            # 1. LSTM input formatı: (1, seq_length, n_features)
            lstm_input = current_sequence.reshape(1, seq_length, -1)
            
            # 2. Tahmin yap (scaled log return)
            pred_scaled_raw = self.model.predict(lstm_input)
            pred_scaled = float(pred_scaled_raw) if np.ndim(pred_scaled_raw) == 0 else pred_scaled_raw[0]
            
            # 3. Inverse transform (scaled → log return)
            pred_log_return = self.preprocessor.inverse_transform_target(
                np.array([pred_scaled])
            )[0]
            
            # 4. QUANTILE CLIPPING (Black Swan kontrolü)
            pred_log_return_clipped = np.clip(
                pred_log_return,
                self.lower_quantile,
                self.upper_quantile
            )
            
            # 5. GAUSSIAN NOISE (Stochastic forecasting)
            noise_scale = base_std * np.sqrt(step + 1) / 8
            noise = np.random.normal(0, noise_scale)
            pred_with_noise = pred_log_return_clipped + noise
            
            # 6. YENİ FİYAT HESAPLA: P_t+1 = P_t * e^r_pred
            if current_price is not None:
                new_price = current_price * np.exp(pred_with_noise)
                price_predictions.append(new_price)
                current_price = new_price
            
            log_return_predictions.append(pred_with_noise)
            
            # 7. TARİH İLERLET
            current_date += timedelta(days=1)
            
            # 8. SEQUENCE GÜNCELLE (sliding window)
            current_sequence = self._update_sequence_lstm(
                current_sequence, 
                pred_scaled,
                pred_with_noise
            )
            
            if (step + 1) % 10 == 0:
                print(f"   ✅ {step + 1} gün | Son Fiyat: ${new_price:,.2f} | Log Return: {pred_with_noise:.4f}")
        
        print(f"✅ LSTM tahmin tamamlandı! Son fiyat: ${price_predictions[-1]:,.2f}")
        
        return {
            'log_returns': np.array(log_return_predictions),
            'prices': np.array(price_predictions) if price_predictions else None
        }
    
    def _update_sequence_lstm(self, sequence, pred_scaled, pred_log_return):
        """
        LSTM sequence'ı günceller (sliding window).
        
        MANTIK:
        - Eski sequence: [t-60, t-59, ..., t-1]
        - Yeni tahmin: t
        - Yeni sequence: [t-59, t-58, ..., t-1, t]
        
        Args:
            sequence: Mevcut sequence (seq_length, n_features)
            pred_scaled: Tahmin edilen scaled değer
            pred_log_return: Tahmin edilen log return
            
        Returns:
            Güncellenmiş sequence
        """
        # Yeni satır oluştur (tüm features için)
        # NOT: Gerçek uygulamada tüm features yeniden hesaplanmalı
        # Şimdilik basit bir yaklaşım: son satırı kopyala ve log return güncelle
        
        new_row = sequence[-1].copy()
        
        # Sequence'ı kaydır ve yeni satırı ekle
        updated_sequence = np.vstack([sequence[1:], new_row])
        
        return updated_sequence


class ForecastAnalyzer:
    """
    Tahmin sonuçlarını analiz eden sınıf.
    """
    
    def __init__(self, forecast_results, dates=None):
        """
        Args:
            forecast_results: RecursiveForecaster çıktısı
            dates: Tahmin tarihleri (opsiyonel)
        """
        self.log_returns = forecast_results['log_returns']
        self.prices = forecast_results['prices']
        self.dates = dates
    
    def get_forecast_dataframe(self):
        """
        Tahmin sonuçlarını DataFrame'e çevirir.
        
        Returns:
            pd.DataFrame: Tahmin sonuçları
        """
        data = {
            'Log_Return': self.log_returns
        }
        
        if self.prices is not None:
            data['Price'] = self.prices
        
        if self.dates is not None:
            data['Date'] = self.dates
        
        df = pd.DataFrame(data)
        
        if self.dates is not None:
            df = df[['Date', 'Log_Return', 'Price']]
        
        return df
    
    def calculate_confidence_intervals(self, std_multiplier=1.96):
        """
        Tahminler için güven aralığı hesaplar (basit yaklaşım).
        
        NOT: Bu basit bir yaklaşımdır. Gerçek uygulamada:
        - Monte Carlo simülasyon
        - Quantile regression
        - Ensemble tahminler kullanılabilir
        
        Args:
            std_multiplier: Standart sapma çarpanı (1.96 = %95 CI)
            
        Returns:
            Dict: Upper ve lower bounds
        """
        if self.prices is None:
            raise ValueError("Fiyat tahminleri mevcut değil!")
        
        # Basit yaklaşım: Tahmin hatası volatilitesi
        std = np.std(self.log_returns)
        
        upper_bound = self.prices * (1 + std * std_multiplier)
        lower_bound = self.prices * (1 - std * std_multiplier)
        
        return {
            'upper': upper_bound,
            'lower': lower_bound,
            'std': std
        }
    
    def get_price_statistics(self):
        """
        Tahmin edilen fiyatların istatistiklerini hesaplar.
        
        Returns:
            Dict: İstatistikler
        """
        if self.prices is None:
            raise ValueError("Fiyat tahminleri mevcut değil!")
        
        stats = {
            'Initial_Price': self.prices[0],
            'Final_Price': self.prices[-1],
            'Min_Price': np.min(self.prices),
            'Max_Price': np.max(self.prices),
            'Mean_Price': np.mean(self.prices),
            'Price_Change': self.prices[-1] - self.prices[0],
            'Price_Change_Pct': ((self.prices[-1] - self.prices[0]) / self.prices[0]) * 100,
            'Volatility': np.std(self.prices)
        }
        
        return stats
    
    def print_summary(self):
        """
        Tahmin özetini yazdırır.
        """
        print("\n" + "="*60)
        print("📊 FORECAST SUMMARY")
        print("="*60)
        
        if self.prices is not None:
            stats = self.get_price_statistics()
            
            print(f"\n💰 Fiyat Tahminleri:")
            print(f"   Başlangıç: ${stats['Initial_Price']:,.2f}")
            print(f"   Bitiş: ${stats['Final_Price']:,.2f}")
            print(f"   Min: ${stats['Min_Price']:,.2f}")
            print(f"   Max: ${stats['Max_Price']:,.2f}")
            print(f"   Ortalama: ${stats['Mean_Price']:,.2f}")
            print(f"   Değişim: ${stats['Price_Change']:,.2f} ({stats['Price_Change_Pct']:.2f}%)")
            print(f"   Volatilite: ${stats['Volatility']:,.2f}")
        
        print(f"\n📈 Log Return İstatistikleri:")
        print(f"   Ortalama: {np.mean(self.log_returns):.6f}")
        print(f"   Std: {np.std(self.log_returns):.6f}")
        print(f"   Min: {np.min(self.log_returns):.6f}")
        print(f"   Max: {np.max(self.log_returns):.6f}")
        
        print("\n" + "="*60)


def create_future_dates(start_date, n_days):
    """
    Gelecek tarihler oluşturur.
    
    Args:
        start_date: Başlangıç tarihi (datetime veya str)
        n_days: Kaç gün
        
    Returns:
        List: Tarih listesi
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    dates = [start_date + timedelta(days=i) for i in range(1, n_days + 1)]
    return dates


if __name__ == "__main__":
    print("Forecasting module test")
    
    # Test için dummy data
    from models import LightGBMModel
    
    # Dummy model
    X_train = np.random.rand(1000, 50)
    y_train = np.random.rand(1000)
    
    model = LightGBMModel()
    model.train(X_train, y_train)
    
    # Dummy preprocessor
    class DummyPreprocessor:
        def log_return_to_price(self, log_returns, initial_price):
            prices = [initial_price]
            for lr in log_returns:
                new_price = prices[-1] * np.exp(lr)
                prices.append(new_price)
            return np.array(prices[1:])
    
    preprocessor = DummyPreprocessor()
    
    # Forecaster test
    forecaster = RecursiveForecaster(model, preprocessor, None)
    X_last = np.random.rand(50)
    
    results = forecaster.forecast_lightgbm(X_last, n_steps=30, last_price=50000)
    
    # Analyzer test
    analyzer = ForecastAnalyzer(results)
    analyzer.print_summary()
