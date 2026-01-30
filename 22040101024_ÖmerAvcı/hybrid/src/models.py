"""
Models Module
=============
LightGBM ve LSTM model sınıfları.

Her model için:
1. Train metodu
2. Predict metodu  
3. Evaluate metodu
4. Feature importance (LightGBM için)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class LightGBMModel:
    """
    LightGBM Regressor Wrapper.
    
    Avantajları:
    - Hızlı eğitim
    - Feature importance çıkarımı
    - Missing value handling
    - Kategorik değişken desteği
    """
    
    def __init__(self, params=None):
        """
        Args:
            params: LightGBM parametreleri (dict)
        """
        if params is None:
            # Varsayılan parametreler
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 1000,
                'random_state': 42
            }
        else:
            self.params = params
        
        self.model = None
        self.feature_names = None
        self.feature_importance_df = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              feature_names=None, early_stopping_rounds=50):
        """
        Modeli eğitir.
        
        Args:
            X_train: Eğitim features
            y_train: Eğitim target
            X_val: Validation features (opsiyonel)
            y_val: Validation target (opsiyonel)
            feature_names: Feature isimleri
            early_stopping_rounds: Erken durdurma (overfitting önleme)
        """
        print("\n🚀 LightGBM Eğitimi Başlıyor...")
        
        self.feature_names = feature_names
        
        # Model oluştur
        self.model = lgb.LGBMRegressor(**self.params)
        
        # Validation set varsa early stopping kullan
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            print(f"✅ Eğitim tamamlandı (Best iteration: {self.model.best_iteration_})")
        else:
            self.model.fit(X_train, y_train)
            print("✅ Eğitim tamamlandı")
        
        # Feature importance hesapla
        self._calculate_feature_importance()
    
    def _calculate_feature_importance(self):
        """
        Feature importance'ı hesaplar ve DataFrame'e kaydeder.
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is not None:
            self.feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            self.feature_importance_df = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(importances))],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
    
    def predict(self, X):
        """
        Tahmin yapar.
        
        Args:
            X: Features
            
        Returns:
            Tahmin değerleri
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Model performansını değerlendirir.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict: Metrikler
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }
        
        print("\n📊 LightGBM Performans Metrikleri:")
        print("="*50)
        for metric, value in metrics.items():
            print(f"   {metric:8s}: {value:.6f}")
        print("="*50)
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """
        En önemli N özelliği döndürür.
        
        Args:
            top_n: Kaç özellik gösterilecek
            
        Returns:
            DataFrame: Top N feature importance
        """
        if self.feature_importance_df is None:
            raise ValueError("Feature importance henüz hesaplanmadı!")
        
        return self.feature_importance_df.head(top_n)


class LSTMModel(nn.Module):
    """
    PyTorch LSTM Modeli.
    
    LSTM (Long Short-Term Memory):
    - Zaman serisi verileri için ideal
    - Uzun vadeli bağımlılıkları öğrenebilir
    - Sequence to value mapping
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, output_size=1):
        """
        Args:
            input_size: Feature sayısı
            hidden_size: LSTM hidden unit sayısı
            num_layers: LSTM katman sayısı
            dropout: Dropout oranı (overfitting önleme)
            output_size: Output boyutu (regression için 1)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM katmanları
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Son timestep'in output'unu al
        last_output = lstm_out[:, -1, :]
        
        # Fully connected
        output = self.fc(last_output)
        
        return output


class LSTMTrainer:
    """
    LSTM modelini eğitip değerlendiren wrapper sınıf.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, learning_rate=0.001):
        """
        Args:
            input_size: Feature sayısı
            hidden_size: LSTM hidden unit sayısı
            num_layers: LSTM katman sayısı
            dropout: Dropout oranı
            learning_rate: Öğrenme hızı
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️ Device: {self.device}")
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, patience=15):
        """
        LSTM modelini eğitir.
        
        Args:
            X_train: Training sequences (samples, seq_length, features)
            y_train: Training targets
            X_val: Validation sequences (opsiyonel)
            y_val: Validation targets (opsiyonel)
            epochs: Maksimum epoch sayısı
            batch_size: Batch boyutu
            patience: Early stopping patience
        """
        print(f"\n🚀 LSTM Eğitimi Başlıyor...")
        print(f"   Epochs: {epochs}, Batch Size: {batch_size}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # DataLoader oluştur
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Validation varsa hazırla
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            use_validation = True
        else:
            use_validation = False
        
        # Early stopping için
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Eğitim döngüsü
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs.squeeze(), y_val_tensor).item()
                    self.val_losses.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # En iyi modeli kaydet
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                
                # Progress
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\n⚠️ Early Stopping at epoch {epoch+1}")
                    # En iyi modeli yükle
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        print("✅ LSTM eğitimi tamamlandı")
    
    def predict(self, X):
        """
        LSTM ile tahmin yapar.
        
        Args:
            X: Input sequences
            
        Returns:
            Tahminler
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().squeeze()
    
    def evaluate(self, X_test, y_test):
        """
        LSTM performansını değerlendirir.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dict: Metrikler
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }
        
        print("\n📊 LSTM Performans Metrikleri:")
        print("="*50)
        for metric, value in metrics.items():
            print(f"   {metric:8s}: {value:.6f}")
        print("="*50)
        
        return metrics
    
    def save_model(self, path):
        """Model'i kaydeder."""
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model kaydedildi: {path}")
    
    def load_model(self, path):
        """Model'i yükler."""
        self.model.load_state_dict(torch.load(path))
        print(f"✅ Model yüklendi: {path}")


if __name__ == "__main__":
    # Test kodu - LightGBM
    print("Testing LightGBM...")
    
    # Dummy data
    X_train = np.random.rand(1000, 50)
    y_train = np.random.rand(1000)
    X_test = np.random.rand(200, 50)
    y_test = np.random.rand(200)
    
    # LightGBM test
    lgb_model = LightGBMModel()
    lgb_model.train(X_train, y_train)
    metrics = lgb_model.evaluate(X_test, y_test)
    
    # Test kodu - LSTM
    print("\n\nTesting LSTM...")
    
    # Dummy sequence data
    X_train_seq = np.random.rand(1000, 60, 50)  # (samples, timesteps, features)
    y_train_seq = np.random.rand(1000)
    X_test_seq = np.random.rand(200, 60, 50)
    y_test_seq = np.random.rand(200)
    
    # LSTM test
    lstm_trainer = LSTMTrainer(input_size=50, hidden_size=64, num_layers=2)
    lstm_trainer.train(X_train_seq, y_train_seq, epochs=20, batch_size=32)
    metrics = lstm_trainer.evaluate(X_test_seq, y_test_seq)
