"""
Base Stage Analyzer
Defines the interface for all stage analysis components
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class StageResult:
    """Result from a single stage analysis"""
    stage_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    visual_indicator: str
    top_stocks: List[str] = field(default_factory=list)
    confidence: float = 0.0


class BaseStageAnalyzer(ABC):
    """
    Base class for all stage analyzers
    
    All stage analyzers must inherit from this class and implement:
    - run_all_stages() method
    - get_supported_stages() method
    """
    
    def __init__(self, blacklist_manager=None):
        self.name = self.__class__.__name__
        self.version = "1.0"
        self.blacklist_manager = blacklist_manager
        self.supported_stages = self.get_supported_stages()
        self.stage_top_stocks = defaultdict(list)
    
    @abstractmethod
    def run_all_stages(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> List[StageResult]:
        """
        Run all supported stages on the given DataFrame
        
        Args:
            df: OHLCV data
            symbol: Symbol being analyzed
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            List of StageResult objects
        """
        pass
    
    @abstractmethod
    def get_supported_stages(self) -> List[str]:
        """
        Return list of stage names this analyzer supports
        
        Returns:
            List of stage names
        """
        pass
    
    def run_specific_stage(self, stage_name: str, df: pd.DataFrame, 
                          symbol: str = "", is_crypto: bool = False) -> StageResult:
        """
        Run a specific stage and return detailed results
        
        Args:
            stage_name: Name of the stage to run
            df: OHLCV data
            symbol: Symbol being analyzed
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            StageResult object
        """
        if stage_name not in self.supported_stages:
            return StageResult(
                stage_name, False, 0.0, 
                {"error": f"Stage {stage_name} not supported by {self.name}"}, 
                "❌", [], 0.0
            )
        
        try:
            # Default implementation - subclasses should override for specific stages
            all_results = self.run_all_stages(df, symbol, is_crypto)
            for result in all_results:
                if result.stage_name == stage_name:
                    return result
            
            # Stage not found in results
            return StageResult(
                stage_name, False, 0.0, 
                {"error": f"Stage {stage_name} not found in results"}, 
                "❌", [], 0.0
            )
        except Exception as e:
            return StageResult(
                stage_name, False, 0.0, 
                {"error": str(e), "analyzer": self.name}, 
                "❌", [], 0.0
            )
    
    def get_stage_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this stage analyzer
        
        Returns:
            Dictionary with analyzer information
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_stages": self.supported_stages,
            "stage_count": len(self.supported_stages),
            "description": self.__doc__ or "No description available"
        }
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has required columns and data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        
        if not all(col in df.columns for col in required_columns):
            return False
        
        if len(df) < 10:  # Minimum data points
            return False
        
        if df[required_columns].isnull().any().any():
            return False
        
        if (df[required_columns] <= 0).any().any():
            return False
        
        return True
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate common technical indicators used across stages
        
        Args:
            df: OHLCV data
            
        Returns:
            Dictionary of technical indicators
        """
        indicators = {}
        
        try:
            # Moving averages
            if len(df) >= 20:
                indicators['sma_20'] = df['Close'].rolling(20).mean()
                indicators['ema_20'] = df['Close'].ewm(span=20).mean()
            
            if len(df) >= 50:
                indicators['sma_50'] = df['Close'].rolling(50).mean()
                indicators['ema_50'] = df['Close'].ewm(span=50).mean()
            
            # RSI
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            if len(df) >= 26:
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                indicators['macd'] = ema_12 - ema_26
                indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            if len(df) >= 20:
                sma_20 = df['Close'].rolling(20).mean()
                std_20 = df['Close'].rolling(20).std()
                indicators['bb_upper'] = sma_20 + (2 * std_20)
                indicators['bb_lower'] = sma_20 - (2 * std_20)
                indicators['bb_middle'] = sma_20
            
            # Volume indicators
            if 'Volume' in df.columns:
                indicators['volume_sma'] = df['Volume'].rolling(20).mean()
                
                # On-Balance Volume
                obv = [0]
                for i in range(1, len(df)):
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                        obv.append(obv[-1] + df['Volume'].iloc[i])
                    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                        obv.append(obv[-1] - df['Volume'].iloc[i])
                    else:
                        obv.append(obv[-1])
                indicators['obv'] = pd.Series(obv, index=df.index)
                
                # Volume Price Trend
                vpt = [0]
                for i in range(1, len(df)):
                    price_change = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                    vpt.append(vpt[-1] + df['Volume'].iloc[i] * price_change)
                indicators['vpt'] = pd.Series(vpt, index=df.index)
        
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels
        
        Args:
            df: OHLCV data
            window: Lookback window for calculations
            
        Returns:
            Dictionary with support/resistance levels
        """
        try:
            recent_data = df.tail(window)
            
            support_levels = []
            resistance_levels = []
            
            # Method 1: Recent lows/highs
            support_levels.append(recent_data['Low'].min())
            resistance_levels.append(recent_data['High'].max())
            
            # Method 2: Moving average support/resistance
            if len(df) >= 20:
                ma_20 = df['Close'].rolling(20).mean().iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                if current_price > ma_20:
                    support_levels.append(ma_20)
                else:
                    resistance_levels.append(ma_20)
            
            # Method 3: Bollinger Band levels
            if len(df) >= 20:
                sma = df['Close'].rolling(20).mean()
                std = df['Close'].rolling(20).std()
                bb_upper = (sma + 2 * std).iloc[-1]
                bb_lower = (sma - 2 * std).iloc[-1]
                
                support_levels.append(bb_lower)
                resistance_levels.append(bb_upper)
            
            return {
                'primary_support': min(support_levels) if support_levels else df['Low'].iloc[-1],
                'primary_resistance': max(resistance_levels) if resistance_levels else df['High'].iloc[-1],
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
        except Exception as e:
            current_price = df['Close'].iloc[-1]
            return {
                'primary_support': current_price * 0.95,
                'primary_resistance': current_price * 1.05,
                'support_levels': [current_price * 0.95],
                'resistance_levels': [current_price * 1.05]
            }
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various volatility metrics
        
        Args:
            df: OHLCV data
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            returns = df['Close'].pct_change().dropna()
            
            # Historical volatility (annualized)
            hist_vol = returns.std() * np.sqrt(252)
            
            # Recent volatility (last 10 periods)
            recent_vol = returns.tail(10).std() * np.sqrt(252)
            
            # Average True Range
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift(1))
            low_close = np.abs(df['Low'] - df['Close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1] if len(true_range) >= 14 else true_range.mean()
            
            # Bollinger Band width
            if len(df) >= 20:
                sma = df['Close'].rolling(20).mean()
                std = df['Close'].rolling(20).std()
                bb_width = ((sma + 2 * std) - (sma - 2 * std)) / sma
                current_bb_width = bb_width.iloc[-1]
            else:
                current_bb_width = 0.1
            
            return {
                'historical_volatility': hist_vol,
                'recent_volatility': recent_vol,
                'atr': atr,
                'atr_percentage': atr / df['Close'].iloc[-1],
                'bollinger_width': current_bb_width,
                'volatility_ratio': recent_vol / hist_vol if hist_vol > 0 else 1.0
            }
        except Exception as e:
            return {
                'historical_volatility': 0.2,
                'recent_volatility': 0.2,
                'atr': df['Close'].iloc[-1] * 0.02,
                'atr_percentage': 0.02,
                'bollinger_width': 0.1,
                'volatility_ratio': 1.0
            }
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate momentum indicators
        
        Args:
            df: OHLCV data
            
        Returns:
            Dictionary with momentum indicators
        """
        try:
            indicators = {}
            
            # RSI
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi'] = rsi.iloc[-1]
            else:
                indicators['rsi'] = 50.0
            
            # Stochastic
            if len(df) >= 14:
                low_14 = df['Low'].rolling(14).min()
                high_14 = df['High'].rolling(14).max()
                k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
                indicators['stochastic_k'] = k_percent.iloc[-1]
                indicators['stochastic_d'] = k_percent.rolling(3).mean().iloc[-1]
            else:
                indicators['stochastic_k'] = 50.0
                indicators['stochastic_d'] = 50.0
            
            # Williams %R
            if len(df) >= 14:
                high_14 = df['High'].rolling(14).max()
                low_14 = df['Low'].rolling(14).min()
                williams_r = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
                indicators['williams_r'] = williams_r.iloc[-1]
            else:
                indicators['williams_r'] = -50.0
            
            # MACD
            if len(df) >= 26:
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9).mean()
                
                indicators['macd'] = macd.iloc[-1]
                indicators['macd_signal'] = signal.iloc[-1]
                indicators['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
            else:
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_histogram'] = 0.0
            
            # Rate of Change
            if len(df) >= 12:
                roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-12]) / df['Close'].iloc[-12]) * 100
                indicators['roc_12'] = roc
            else:
                indicators['roc_12'] = 0.0
            
            return indicators
        except Exception as e:
            return {
                'rsi': 50.0,
                'stochastic_k': 50.0,
                'stochastic_d': 50.0,
                'williams_r': -50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'roc_12': 0.0
            }
    
    def calculate_volume_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volume analysis metrics
        
        Args:
            df: OHLCV data
            
        Returns:
            Dictionary with volume analysis
        """
        try:
            if 'Volume' not in df.columns:
                return {
                    'avg_volume': 0,
                    'volume_trend': 'neutral',
                    'volume_ratio': 1.0,
                    'volume_spike': False,
                    'accumulation_score': 0.5
                }
            
            volume = df['Volume']
            
            # Basic volume metrics
            avg_volume = volume.mean()
            recent_volume = volume.tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume trend
            if volume_ratio > 1.2:
                volume_trend = 'increasing'
            elif volume_ratio < 0.8:
                volume_trend = 'decreasing'
            else:
                volume_trend = 'neutral'
            
            # Volume spike detection
            volume_spike = volume.iloc[-1] > avg_volume * 2
            
            # Accumulation/Distribution analysis
            up_volume = 0
            down_volume = 0
            neutral_volume = 0
            
            for i in range(len(df)):
                vol = volume.iloc[i]
                if df['Close'].iloc[i] > df['Open'].iloc[i]:
                    up_volume += vol
                elif df['Close'].iloc[i] < df['Open'].iloc[i]:
                    down_volume += vol
                else:
                    neutral_volume += vol
            
            total_volume = up_volume + down_volume + neutral_volume
            if total_volume > 0:
                accumulation_score = up_volume / total_volume
            else:
                accumulation_score = 0.5
            
            # On-Balance Volume trend
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            
            obv_trend = 'neutral'
            if len(obv) >= 10:
                if obv[-1] > obv[-10]:
                    obv_trend = 'increasing'
                elif obv[-1] < obv[-10]:
                    obv_trend = 'decreasing'
            
            return {
                'avg_volume': avg_volume,
                'recent_volume': recent_volume,
                'volume_trend': volume_trend,
                'volume_ratio': volume_ratio,
                'volume_spike': volume_spike,
                'accumulation_score': accumulation_score,
                'up_volume_ratio': up_volume / total_volume if total_volume > 0 else 0.5,
                'down_volume_ratio': down_volume / total_volume if total_volume > 0 else 0.5,
                'obv_trend': obv_trend,
                'obv_current': obv[-1] if obv else 0
            }
        except Exception as e:
            return {
                'avg_volume': 0,
                'volume_trend': 'neutral',
                'volume_ratio': 1.0,
                'volume_spike': False,
                'accumulation_score': 0.5
            }
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} - {len(self.supported_stages)} stages"
    
    def __repr__(self) -> str:
        return f"<{self.name}(stages={len(self.supported_stages)})>"
