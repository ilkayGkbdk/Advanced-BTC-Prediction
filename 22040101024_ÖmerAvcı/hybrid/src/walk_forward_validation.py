"""
Walk-Forward Validation Module
================================
Zamansal veri için en uygun validation stratejisi.

WALK-FORWARD MANTIK:
- Klasik cross-validation zamansal yapıyı bozar (data leakage)
- Walk-forward: Gerçek trading senaryosunu simüle eder

ÖRNEK (1 yıl train, 1 ay test):
1. Train: 2020-01-01 to 2020-12-31 → Test: 2021-01-01 to 2021-01-31
2. Train: 2020-02-01 to 2021-01-31 → Test: 2021-02-01 to 2021-02-28
3. Train: 2020-03-01 to 2021-02-28 → Test: 2021-03-01 to 2021-03-31
...

AVANTAJLAR:
- Model her dönem yeniden eğitilir (regime change'e adapte olur)
- Gerçekçi performans ölçümü (future data leakage yok)
- Overfitting tespiti (test performansı train'den çok düşükse)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Walk-Forward validation için ana sınıf.
    
    Model'in zamanla nasıl perform ettiğini test eder.
    Regime change (piyasa koşullarındaki değişim) tespiti yapar.
    """
    
    def __init__(self, train_window_months=12, test_window_months=1, 
                 step_months=1, min_train_size=200):
        """
        Args:
            train_window_months: Eğitim penceresi (ay cinsinden)
            test_window_months: Test penceresi (ay cinsinden)
            step_months: Her adımda kaç ay ileri gidilecek
            min_train_size: Minimum eğitim seti büyüklüğü
        """
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.min_train_size = min_train_size
        
        self.results = []
        self.fold_metrics = []
    
    def create_folds(self, data, date_column='Date'):
        """
        Zamansal foldlar oluşturur.
        
        Args:
            data: DataFrame (date column içermeli)
            date_column: Tarih kolonu ismi
            
        Returns:
            List[Dict]: Her fold için train/test index'leri
        """
        print("\n" + "="*70)
        print("🔄 WALK-FORWARD FOLD CREATION")
        print("="*70)
        
        # Tarihleri sırala
        data = data.sort_values(date_column).reset_index(drop=True)
        data['Date'] = pd.to_datetime(data[date_column])
        
        min_date = data['Date'].min()
        max_date = data['Date'].max()
        
        print(f"\n📅 Dataset Range: {min_date.date()} to {max_date.date()}")
        print(f"📊 Total Days: {(max_date - min_date).days}")
        
        folds = []
        current_date = min_date + pd.DateOffset(months=self.train_window_months)
        
        fold_num = 1
        while current_date + pd.DateOffset(months=self.test_window_months) <= max_date:
            # Train period
            train_start = current_date - pd.DateOffset(months=self.train_window_months)
            train_end = current_date
            
            # Test period
            test_start = current_date
            test_end = current_date + pd.DateOffset(months=self.test_window_months)
            
            # Index'leri bul
            train_mask = (data['Date'] >= train_start) & (data['Date'] < train_end)
            test_mask = (data['Date'] >= test_start) & (data['Date'] < test_end)
            
            train_indices = data[train_mask].index.tolist()
            test_indices = data[test_mask].index.tolist()
            
            # Minimum train size kontrolü
            if len(train_indices) >= self.min_train_size and len(test_indices) > 0:
                fold_info = {
                    'fold': fold_num,
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_size': len(train_indices),
                    'test_size': len(test_indices)
                }
                
                folds.append(fold_info)
                
                print(f"\n✅ Fold {fold_num}:")
                print(f"   Train: {train_start.date()} to {train_end.date()} ({len(train_indices)} samples)")
                print(f"   Test:  {test_start.date()} to {test_end.date()} ({len(test_indices)} samples)")
                
                fold_num += 1
            
            # Pencereyi kaydır
            current_date += pd.DateOffset(months=self.step_months)
        
        print(f"\n📊 Total Folds Created: {len(folds)}")
        print("="*70)
        
        return folds
    
    def validate(self, model_class, X, y, folds, model_params=None, 
                 feature_names=None, verbose=True):
        """
        Walk-forward validation çalıştırır.
        
        Her fold için:
        1. Model eğit (train data)
        2. Tahmin yap (test data)
        3. Metrikleri kaydet
        4. Sonraki fold'a geç (yeni model)
        
        Args:
            model_class: Model sınıfı (LightGBMModel, etc.)
            X: Features DataFrame
            y: Target Series
            folds: create_folds() output'u
            model_params: Model parametreleri
            feature_names: Feature isimleri
            verbose: Detaylı çıktı
            
        Returns:
            Dict: Tüm fold'ların sonuçları
        """
        print("\n" + "="*70)
        print("🚀 WALK-FORWARD VALIDATION BAŞLIYOR")
        print("="*70)
        
        self.fold_metrics = []
        
        for fold_info in folds:
            fold_num = fold_info['fold']
            train_idx = fold_info['train_indices']
            test_idx = fold_info['test_indices']
            
            if verbose:
                print(f"\n📊 FOLD {fold_num}/{len(folds)}")
                print(f"   Train: {fold_info['train_start'].date()} to {fold_info['train_end'].date()}")
                print(f"   Test:  {fold_info['test_start'].date()} to {fold_info['test_end'].date()}")
            
            # Train/test split
            X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            X_test_fold = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
            y_test_fold = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
            
            # Model eğitimi
            model = model_class(params=model_params) if model_params else model_class()
            
            try:
                if verbose:
                    print("   🤖 Model eğitiliyor...")
                
                # Eğitim
                model.train(X_train_fold, y_train_fold, feature_names=feature_names)
                
                # Tahmin
                y_pred_fold = model.predict(X_test_fold)
                
                # Metrikler
                fold_metrics = {
                    'fold': fold_num,
                    'train_start': fold_info['train_start'],
                    'train_end': fold_info['train_end'],
                    'test_start': fold_info['test_start'],
                    'test_end': fold_info['test_end'],
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'rmse': np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)),
                    'mae': mean_absolute_error(y_test_fold, y_pred_fold),
                    'r2': r2_score(y_test_fold, y_pred_fold),
                    'mean_actual': np.mean(y_test_fold),
                    'mean_predicted': np.mean(y_pred_fold),
                    'std_actual': np.std(y_test_fold),
                    'std_predicted': np.std(y_pred_fold)
                }
                
                self.fold_metrics.append(fold_metrics)
                
                if verbose:
                    print(f"   ✅ RMSE: {fold_metrics['rmse']:.6f} | "
                          f"MAE: {fold_metrics['mae']:.6f} | "
                          f"R²: {fold_metrics['r2']:.4f}")
            
            except Exception as e:
                print(f"   ❌ Fold {fold_num} failed: {e}")
                continue
        
        # Özet istatistikler
        self._print_summary()
        
        return {
            'fold_metrics': self.fold_metrics,
            'summary': self._calculate_summary()
        }
    
    def _calculate_summary(self):
        """
        Tüm fold'ların ortalama metriklerini hesaplar.
        """
        if not self.fold_metrics:
            return {}
        
        df = pd.DataFrame(self.fold_metrics)
        
        summary = {
            'n_folds': len(self.fold_metrics),
            'avg_rmse': df['rmse'].mean(),
            'std_rmse': df['rmse'].std(),
            'avg_mae': df['mae'].mean(),
            'std_mae': df['mae'].std(),
            'avg_r2': df['r2'].mean(),
            'std_r2': df['r2'].std(),
            'min_r2': df['r2'].min(),
            'max_r2': df['r2'].max(),
            'rmse_range': (df['rmse'].min(), df['rmse'].max()),
            'consistency_score': 1 - (df['rmse'].std() / df['rmse'].mean())  # Yüksek = tutarlı
        }
        
        return summary
    
    def _print_summary(self):
        """
        Validation sonuçlarını yazdırır.
        """
        summary = self._calculate_summary()
        
        if not summary:
            print("\n⚠️ No results to summarize.")
            return
        
        print("\n" + "="*70)
        print("📊 WALK-FORWARD VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\n📈 Overall Performance ({summary['n_folds']} folds):")
        print(f"   Average RMSE: {summary['avg_rmse']:.6f} (±{summary['std_rmse']:.6f})")
        print(f"   Average MAE:  {summary['avg_mae']:.6f} (±{summary['std_mae']:.6f})")
        print(f"   Average R²:   {summary['avg_r2']:.4f} (±{summary['std_r2']:.4f})")
        print(f"   R² Range:     [{summary['min_r2']:.4f}, {summary['max_r2']:.4f}]")
        
        print(f"\n🎯 Model Consistency:")
        print(f"   Consistency Score: {summary['consistency_score']:.4f}")
        if summary['consistency_score'] > 0.8:
            print("   ✅ Model is VERY consistent across time periods")
        elif summary['consistency_score'] > 0.5:
            print("   ⚠️ Model shows MODERATE consistency")
        else:
            print("   ❌ Model is INCONSISTENT - possible regime changes!")
        
        print("\n" + "="*70)
    
    def plot_fold_performance(self):
        """
        Her fold'un performansını görselleştirir.
        
        NOT: matplotlib gerektirir.
        """
        if not self.fold_metrics:
            print("⚠️ No metrics to plot.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(self.fold_metrics)
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # RMSE over time
            axes[0].plot(df['fold'], df['rmse'], marker='o', linewidth=2)
            axes[0].set_title('RMSE Across Folds (Walk-Forward)', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Fold Number')
            axes[0].set_ylabel('RMSE')
            axes[0].grid(True, alpha=0.3)
            
            # R² over time
            axes[1].plot(df['fold'], df['r2'], marker='o', color='green', linewidth=2)
            axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1].set_title('R² Score Across Folds', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Fold Number')
            axes[1].set_ylabel('R² Score')
            axes[1].grid(True, alpha=0.3)
            
            # MAE over time
            axes[2].plot(df['fold'], df['mae'], marker='o', color='orange', linewidth=2)
            axes[2].set_title('MAE Across Folds', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Fold Number')
            axes[2].set_ylabel('MAE')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Plot generated successfully!")
            
        except ImportError:
            print("⚠️ matplotlib not installed. Install with: pip install matplotlib")
    
    def get_results_dataframe(self):
        """
        Sonuçları pandas DataFrame olarak döndürür.
        
        Returns:
            pd.DataFrame: Fold metrikleri
        """
        if not self.fold_metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(self.fold_metrics)
    
    def detect_regime_changes(self, threshold=0.3):
        """
        Regime change (piyasa rejimi değişimi) tespiti.
        
        Mantık: Ardışık fold'lar arasında R² veya RMSE'de büyük değişim
        varsa, piyasa koşulları değişmiş demektir.
        
        Args:
            threshold: Değişim eşiği (0.3 = %30 değişim)
            
        Returns:
            List[Dict]: Tespit edilen regime change'ler
        """
        if len(self.fold_metrics) < 2:
            return []
        
        df = pd.DataFrame(self.fold_metrics)
        
        # R² değişimi
        df['r2_change'] = df['r2'].pct_change().abs()
        # RMSE değişimi
        df['rmse_change'] = df['rmse'].pct_change().abs()
        
        # Regime change tespiti
        regime_changes = []
        
        for idx, row in df.iterrows():
            if idx == 0:
                continue
            
            if row['r2_change'] > threshold or row['rmse_change'] > threshold:
                change_info = {
                    'fold': row['fold'],
                    'date': row['test_start'],
                    'r2_change': row['r2_change'],
                    'rmse_change': row['rmse_change'],
                    'severity': 'HIGH' if max(row['r2_change'], row['rmse_change']) > 0.5 else 'MODERATE'
                }
                regime_changes.append(change_info)
        
        if regime_changes:
            print("\n" + "="*70)
            print("⚠️ REGIME CHANGES DETECTED")
            print("="*70)
            
            for change in regime_changes:
                print(f"\n📅 Date: {change['date'].date()}")
                print(f"   Fold: {change['fold']}")
                print(f"   R² Change: {change['r2_change']*100:.1f}%")
                print(f"   RMSE Change: {change['rmse_change']*100:.1f}%")
                print(f"   Severity: {change['severity']}")
        else:
            print("\n✅ No significant regime changes detected.")
        
        return regime_changes


if __name__ == "__main__":
    print("Walk-Forward Validation Module")
    print("Use this with your trained models for robust validation.")
