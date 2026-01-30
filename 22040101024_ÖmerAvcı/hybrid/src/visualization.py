"""
Visualization Module
====================
Profesyonel finans grafikleri için görselleştirme modülü.

Grafikler:
1. Gerçek vs Tahmin (Test Set)
2. Feature Importance (LightGBM)
3. LSTM Loss Curve
4. Gelecek Tahminleri (30 gün)
5. Güven Aralıkları
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Görsel stili
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FinancialVisualizer:
    """
    Finans grafiklerini oluşturan sınıf.
    """
    
    def __init__(self, figsize=(14, 8), save_dir='outputs'):
        """
        Args:
            figsize: Grafik boyutu
            save_dir: Grafiklerin kaydedileceği klasör
        """
        self.figsize = figsize
        self.save_dir = save_dir
    
    def plot_actual_vs_predicted(self, y_actual, y_pred, dates=None, 
                                 title='Actual vs Predicted', 
                                 model_name='Model',
                                 save_path=None):
        """
        Gerçek ve tahmin değerlerini karşılaştırır.
        
        Args:
            y_actual: Gerçek değerler
            y_pred: Tahmin değerleri
            dates: Tarihler (opsiyonel)
            title: Grafik başlığı
            model_name: Model ismi
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if dates is not None:
            ax.plot(dates, y_actual, label='Actual', linewidth=2, alpha=0.7)
            ax.plot(dates, y_pred, label=f'{model_name} Prediction', 
                   linewidth=2, alpha=0.7, linestyle='--')
        else:
            ax.plot(y_actual, label='Actual', linewidth=2, alpha=0.7)
            ax.plot(y_pred, label=f'{model_name} Prediction', 
                   linewidth=2, alpha=0.7, linestyle='--')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time' if dates is None else 'Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_price_predictions(self, actual_prices, pred_prices, 
                              actual_dates=None, pred_dates=None,
                              title='Bitcoin Price: Actual vs Forecast',
                              save_path=None):
        """
        Fiyat tahminlerini görselleştirir (gerçek + gelecek tahmin).
        
        Args:
            actual_prices: Gerçek fiyatlar (test set)
            pred_prices: Tahmin edilen fiyatlar (gelecek 30 gün)
            actual_dates: Gerçek fiyat tarihleri
            pred_dates: Tahmin tarihleri
            title: Başlık
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Gerçek fiyatlar
        if actual_dates is not None:
            ax.plot(actual_dates, actual_prices, label='Historical Data', 
                   linewidth=2, color='#2E86AB', alpha=0.8)
        else:
            ax.plot(actual_prices, label='Historical Data', 
                   linewidth=2, color='#2E86AB', alpha=0.8)
        
        # Tahmin fiyatları
        if pred_dates is not None:
            ax.plot(pred_dates, pred_prices, label='30-Day Forecast', 
                   linewidth=2.5, color='#E63946', marker='o', 
                   markersize=4, alpha=0.9, linestyle='--')
        else:
            # Tahminleri historical'ın devamı gibi göster
            start_idx = len(actual_prices)
            ax.plot(range(start_idx, start_idx + len(pred_prices)), 
                   pred_prices, label='30-Day Forecast', 
                   linewidth=2.5, color='#E63946', marker='o', 
                   markersize=4, alpha=0.9, linestyle='--')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Y ekseni formatı (fiyat için)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_forecast_with_confidence(self, forecast_prices, upper_bound, 
                                     lower_bound, dates=None,
                                     title='30-Day Forecast with Confidence Intervals',
                                     save_path=None):
        """
        Güven aralıklı tahmin grafiği.
        
        Args:
            forecast_prices: Tahmin fiyatları
            upper_bound: Üst güven sınırı
            lower_bound: Alt güven sınırı
            dates: Tarihler
            title: Başlık
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_axis = dates if dates is not None else range(len(forecast_prices))
        
        # Tahmin çizgisi
        ax.plot(x_axis, forecast_prices, label='Forecast', 
               linewidth=2.5, color='#E63946', marker='o', markersize=5)
        
        # Güven aralığı (shaded area)
        ax.fill_between(x_axis, lower_bound, upper_bound, 
                       alpha=0.3, color='#E63946', label='95% Confidence Interval')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Y ekseni formatı
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df, top_n=20,
                               title='Feature Importance (LightGBM)',
                               save_path=None):
        """
        Feature importance grafiği (LightGBM).
        
        ÖNEMLI: Hangi veri kaynaklarının en etkili olduğunu gösterir!
        - Faiz mi daha önemli?
        - RSI mi?
        - Lag features mı?
        
        Args:
            feature_importance_df: Feature importance DataFrame
            top_n: En önemli N özellik
            title: Başlık
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Top N özellik
        top_features = feature_importance_df.head(top_n)
        
        # Horizontal bar plot
        bars = ax.barh(top_features['Feature'], top_features['Importance'], 
                      color='#06A77D', alpha=0.8)
        
        # En önemli 3'ü vurgula
        for i in range(min(3, len(bars))):
            bars[-(i+1)].set_color('#E63946')
            bars[-(i+1)].set_alpha(1.0)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.invert_yaxis()  # En önemli üstte olsun
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_lstm_loss_curve(self, train_losses, val_losses=None,
                            title='LSTM Training Progress',
                            save_path=None):
        """
        LSTM eğitim loss grafiği.
        
        Args:
            train_losses: Training loss değerleri
            val_losses: Validation loss değerleri (opsiyonel)
            title: Başlık
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label='Training Loss', 
               linewidth=2, color='#2E86AB', alpha=0.8)
        
        if val_losses is not None:
            ax.plot(epochs, val_losses, label='Validation Loss', 
                   linewidth=2, color='#E63946', alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Loss azalıyor mu kontrol et
        if len(train_losses) > 10:
            trend = 'Decreasing ✅' if train_losses[-1] < train_losses[0] else 'Warning ⚠️'
            ax.text(0.02, 0.98, f'Trend: {trend}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, lgb_predictions, lstm_predictions, 
                             actual_values, dates=None,
                             title='Model Comparison: LightGBM vs LSTM',
                             save_path=None):
        """
        LightGBM ve LSTM modellerini karşılaştırır.
        
        Args:
            lgb_predictions: LightGBM tahminleri
            lstm_predictions: LSTM tahminleri
            actual_values: Gerçek değerler
            dates: Tarihler
            title: Başlık
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_axis = dates if dates is not None else range(len(actual_values))
        
        ax.plot(x_axis, actual_values, label='Actual', 
               linewidth=2.5, color='black', alpha=0.7)
        ax.plot(x_axis, lgb_predictions, label='LightGBM', 
               linewidth=2, color='#06A77D', alpha=0.7, linestyle='--')
        ax.plot(x_axis, lstm_predictions, label='LSTM', 
               linewidth=2, color='#E63946', alpha=0.7, linestyle=':')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date' if dates is not None else 'Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_actual, y_pred, model_name='Model',
                      save_path=None):
        """
        Hata (residual) analizi grafiği.
        
        Residual = Actual - Predicted
        İyi model: Residuals rastgele dağılmalı (pattern olmamalı)
        
        Args:
            y_actual: Gerçek değerler
            y_pred: Tahmin değerleri
            model_name: Model ismi
            save_path: Kayıt yolu
        """
        residuals = y_actual - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.6, color='#2E86AB')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title(f'{model_name} Residual Plot', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residual histogram
        axes[1].hist(residuals, bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title(f'{model_name} Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # İstatistikler ekle
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[1].text(0.02, 0.98, f'Mean: {mean_res:.6f}\nStd: {std_res:.6f}',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, lgb_metrics, lstm_metrics,
                               title='Model Performance Comparison',
                               save_path=None):
        """
        Model metriklerini karşılaştırır (bar chart).
        
        Args:
            lgb_metrics: LightGBM metrikleri (dict)
            lstm_metrics: LSTM metrikleri (dict)
            title: Başlık
            save_path: Kayıt yolu
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        colors = ['#2E86AB', '#E63946']
        
        for i, metric in enumerate(metrics):
            if metric in lgb_metrics and metric in lstm_metrics:
                values = [lgb_metrics[metric], lstm_metrics[metric]]
                bars = axes[i].bar(['LightGBM', 'LSTM'], values, 
                                  color=colors, alpha=0.8)
                
                # Daha iyi performans gösteren modeli vurgula
                if metric == 'R2':  # R2 için yüksek iyi
                    best_idx = np.argmax(values)
                else:  # Diğerleri için düşük iyi
                    best_idx = np.argmin(values)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
                
                axes[i].set_title(f'{metric}', fontsize=14, fontweight='bold')
                axes[i].set_ylabel('Value', fontsize=11)
                axes[i].grid(True, alpha=0.3, axis='y')
                
                # Değerleri bar üstünde göster
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.4f}',
                               ha='center', va='bottom', fontsize=10)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test kodu
    print("Visualization module test")
    
    # Dummy data
    np.random.seed(42)
    actual = np.cumsum(np.random.randn(100)) + 50000
    predicted = actual + np.random.randn(100) * 1000
    
    viz = FinancialVisualizer()
    viz.plot_actual_vs_predicted(actual, predicted, 
                                title='Test: Actual vs Predicted',
                                model_name='Test Model')
