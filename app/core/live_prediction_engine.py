"""
Live Prediction Engine
Real-time market prediction using machine learning and technical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class LivePredictionEngine:
    """Advanced live prediction system for market forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cache = {}
        self.prediction_cache = {}
        
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical analysis features"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_change'] = df['Close'] - df['Open']
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}']
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['Close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['Close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
        df['price_volume'] = df['Close'] * df['Volume']
        
        # Support and Resistance levels
        df['support'] = df['Low'].rolling(window=20).min()
        df['resistance'] = df['High'].rolling(window=20).max()
        df['support_distance'] = (df['Close'] - df['support']) / df['Close']
        df['resistance_distance'] = (df['resistance'] - df['Close']) / df['Close']
        
        # Trend features
        df['trend_5'] = df['Close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        df['trend_20'] = df['Close'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def prepare_prediction_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Prepare data for machine learning prediction"""
        try:
            # Fetch market data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return pd.DataFrame()
            
            # Create technical features
            df = self.create_technical_features(data)
            
            # Create target variables (future returns)
            df['target_1d'] = df['Close'].shift(-1) / df['Close'] - 1  # 1-day return
            df['target_5d'] = df['Close'].shift(-5) / df['Close'] - 1  # 5-day return
            df['target_20d'] = df['Close'].shift(-20) / df['Close'] - 1  # 20-day return
            
            # Direction targets (up/down)
            df['direction_1d'] = (df['target_1d'] > 0).astype(int)
            df['direction_5d'] = (df['target_5d'] > 0).astype(int)
            df['direction_20d'] = (df['target_20d'] > 0).astype(int)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error preparing data for {symbol}: {e}")
            return pd.DataFrame()
    
    def train_prediction_models(self, symbol: str) -> Dict:
        """Train multiple ML models for price prediction"""
        print(f"🤖 Training prediction models for {symbol}...")
        
        # Prepare data
        df = self.prepare_prediction_data(symbol)
        if df.empty:
            return {"error": "No data available for training"}
        
        # Select features (exclude target variables and non-numeric columns)
        feature_columns = [col for col in df.columns if col not in [
            'target_1d', 'target_5d', 'target_20d', 
            'direction_1d', 'direction_5d', 'direction_20d'
        ] and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_columns].fillna(0)
        
        results = {}
        
        # Train models for different time horizons
        for target in ['target_1d', 'target_5d', 'target_20d']:
            if target not in df.columns:
                continue
                
            y = df[target].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            target_results = {}
            
            for model_name, model in models.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                target_results[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'mse': mse,
                    'r2': r2,
                    'features': feature_columns
                }
            
            # Select best model based on R2 score
            best_model_name = max(target_results.keys(), key=lambda k: target_results[k]['r2'])
            results[target] = target_results[best_model_name]
            results[target]['best_model'] = best_model_name
        
        # Store models
        self.models[symbol] = results
        
        return {
            "symbol": symbol,
            "models_trained": list(results.keys()),
            "training_completed": datetime.now().isoformat(),
            "data_points": len(df),
            "features": len(feature_columns)
        }
    
    def make_live_prediction(self, symbol: str) -> Dict:
        """Make live predictions for a symbol"""
        try:
            # Check if models exist
            if symbol not in self.models:
                training_result = self.train_prediction_models(symbol)
                if "error" in training_result:
                    return training_result
            
            # Get latest data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")  # Get enough data for features
            
            if data.empty:
                return {"error": "No market data available"}
            
            # Create features
            df = self.create_technical_features(data)
            
            # Get latest row for prediction
            latest_data = df.iloc[-1:].copy()
            
            predictions = {}
            confidence_scores = {}
            
            # Make predictions for each time horizon
            for target, model_info in self.models[symbol].items():
                if 'model' not in model_info:
                    continue
                
                model = model_info['model']
                scaler = model_info['scaler']
                features = model_info['features']
                
                # Prepare features
                X = latest_data[features].fillna(0)
                X_scaled = scaler.transform(X)
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                # Calculate confidence (simplified)
                confidence = min(abs(model_info['r2']) * 100, 100)
                
                predictions[target] = {
                    'predicted_return': prediction,
                    'predicted_return_pct': prediction * 100,
                    'confidence': confidence,
                    'model_used': model_info['best_model'],
                    'r2_score': model_info['r2']
                }
            
            # Get current price info
            current_price = data['Close'].iloc[-1]
            previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            current_return = (current_price / previous_close - 1) * 100
            
            # Calculate predicted prices
            predicted_prices = {}
            for target, pred_info in predictions.items():
                days = int(target.split('_')[1].replace('d', ''))
                predicted_price = current_price * (1 + pred_info['predicted_return'])
                predicted_prices[f'{days}_day_price'] = predicted_price
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "current_return": current_return,
                "predictions": predictions,
                "predicted_prices": predicted_prices,
                "prediction_time": datetime.now().isoformat(),
                "market_status": self._get_market_status()
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_market_sentiment(self, symbols: List[str]) -> Dict:
        """Analyze overall market sentiment from multiple symbols"""
        sentiment_data = {}
        
        for symbol in symbols:
            prediction = self.make_live_prediction(symbol)
            if "error" not in prediction:
                # Extract sentiment indicators
                pred_1d = prediction['predictions'].get('target_1d', {}).get('predicted_return', 0)
                pred_5d = prediction['predictions'].get('target_5d', {}).get('predicted_return', 0)
                
                sentiment_data[symbol] = {
                    'short_term_sentiment': 'bullish' if pred_1d > 0 else 'bearish',
                    'medium_term_sentiment': 'bullish' if pred_5d > 0 else 'bearish',
                    'predicted_1d_return': pred_1d * 100,
                    'predicted_5d_return': pred_5d * 100
                }
        
        # Calculate overall sentiment
        bullish_short = sum(1 for data in sentiment_data.values() if data['short_term_sentiment'] == 'bullish')
        bullish_medium = sum(1 for data in sentiment_data.values() if data['medium_term_sentiment'] == 'bullish')
        
        total_symbols = len(sentiment_data)
        
        overall_sentiment = {
            'short_term_bullish_pct': (bullish_short / total_symbols * 100) if total_symbols > 0 else 0,
            'medium_term_bullish_pct': (bullish_medium / total_symbols * 100) if total_symbols > 0 else 0,
            'overall_sentiment': 'bullish' if bullish_short > total_symbols / 2 else 'bearish',
            'symbols_analyzed': total_symbols,
            'analysis_time': datetime.now().isoformat()
        }
        
        return {
            'individual_sentiment': sentiment_data,
            'overall_sentiment': overall_sentiment
        }
    
    def _get_market_status(self) -> str:
        """Get current market status"""
        now = datetime.now()
        hour = now.hour
        
        # Simplified market hours (US market)
        if 9 <= hour < 16:
            return "open"
        elif 4 <= hour < 9:
            return "pre_market"
        elif 16 <= hour < 20:
            return "after_hours"
        else:
            return "closed"
    
    def get_prediction_accuracy(self, symbol: str) -> Dict:
        """Calculate historical prediction accuracy"""
        if symbol not in self.models:
            return {"error": "No models trained for this symbol"}
        
        accuracy_metrics = {}
        
        for target, model_info in self.models[symbol].items():
            accuracy_metrics[target] = {
                'r2_score': model_info['r2'],
                'mse': model_info['mse'],
                'model_type': model_info['best_model'],
                'accuracy_rating': self._get_accuracy_rating(model_info['r2'])
            }
        
        return {
            'symbol': symbol,
            'accuracy_metrics': accuracy_metrics,
            'overall_rating': self._get_overall_rating(accuracy_metrics)
        }
    
    def _get_accuracy_rating(self, r2_score: float) -> str:
        """Convert R2 score to rating"""
        if r2_score >= 0.8:
            return "Excellent"
        elif r2_score >= 0.6:
            return "Good"
        elif r2_score >= 0.4:
            return "Fair"
        elif r2_score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_overall_rating(self, accuracy_metrics: Dict) -> str:
        """Get overall model rating"""
        if not accuracy_metrics:
            return "No Data"
        
        avg_r2 = np.mean([metrics['r2_score'] for metrics in accuracy_metrics.values()])
        return self._get_accuracy_rating(avg_r2)
    
    def batch_predict(self, symbols: List[str]) -> Dict:
        """Make predictions for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            print(f"🔮 Making prediction for {symbol}...")
            results[symbol] = self.make_live_prediction(symbol)
        
        return {
            'batch_predictions': results,
            'symbols_processed': len(symbols),
            'successful_predictions': len([r for r in results.values() if 'error' not in r]),
            'batch_time': datetime.now().isoformat()
        }