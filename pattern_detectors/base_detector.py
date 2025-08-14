"""
Base Pattern Detector
Defines the interface for all pattern detection components
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

class PatternType(Enum):
    """Pattern types - can be extended by adding new patterns"""
    # Original patterns
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    RISING_WEDGE = "Rising Wedge"
    FALLING_WEDGE = "Falling Wedge"
    BULL_FLAG = "Bull Flag"
    BEAR_FLAG = "Bear Flag"
    PENNANT = "Pennant"
    CUP_AND_HANDLE = "Cup and Handle"
    DOUBLE_BOTTOM = "Double Bottom"
    TRIPLE_BOTTOM = "Triple Bottom"
    HEAD_AND_SHOULDERS = "Head and Shoulders"
    INVERSE_HEAD_SHOULDERS = "Inverse H&S"
    
    # Advanced patterns
    HIDDEN_ACCUMULATION = "Hidden Accumulation"
    SMART_MONEY_FLOW = "Smart Money Flow"
    WHALE_ACCUMULATION = "Whale Accumulation"
    COILED_SPRING = "Coiled Spring"
    PRESSURE_COOKER = "Pressure Cooker"
    MOMENTUM_VACUUM = "Momentum Vacuum"
    VOLATILITY_CONTRACTION = "Volatility Contraction"
    SILENT_ACCUMULATION = "Silent Accumulation"
    INSTITUTIONAL_ABSORPTION = "Institutional Absorption"
    STEALTH_BREAKOUT = "Stealth Breakout"
    VOLUME_POCKET = "Volume Pocket"
    FIBONACCI_RETRACEMENT = "Fibonacci Retracement"
    ELLIOTT_WAVE_3 = "Elliott Wave 3 Setup"
    MOMENTUM_DIVERGENCE = "Momentum Divergence"
    RSI_DIVERGENCE = "RSI Divergence"
    STOCHASTIC_DIVERGENCE = "Stochastic Divergence"
    BOLLINGER_SQUEEZE = "Bollinger Band Squeeze"
    WILLIAMS_R_REVERSAL = "Williams %R Reversal"
    CCI_BULLISH = "CCI Bullish Setup"
    ICHIMOKU_CLOUD = "Ichimoku Cloud Breakout"
    VWAP_ACCUMULATION = "VWAP Accumulation Zone"
    DARK_POOL_ACTIVITY = "Dark Pool Activity"
    ALGORITHM_PATTERN = "Algorithm Recognition"
    FRACTAL_SUPPORT = "Fractal Support Level"
    GOLDEN_RATIO = "Golden Ratio Pattern"
    TRADINGVIEW_TRIPLE_SUPPORT = "TradingView Triple Support Stable"


class BasePatternDetector(ABC):
    """
    Base class for all pattern detectors
    
    All pattern detectors must inherit from this class and implement:
    - detect_patterns() method
    - get_supported_patterns() method
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0"
        self.supported_patterns = self.get_supported_patterns()
    
    @abstractmethod
    def detect_patterns(self, df: pd.DataFrame, is_crypto: bool = False) -> List[PatternType]:
        """
        Detect patterns in the given DataFrame
        
        Args:
            df: OHLCV data
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            List of detected PatternType enums
        """
        pass
    
    @abstractmethod
    def get_supported_patterns(self) -> List[PatternType]:
        """
        Return list of patterns this detector can find
        
        Returns:
            List of PatternType enums this detector supports
        """
        pass
    
    def detect_specific_pattern(self, df: pd.DataFrame, pattern_type: PatternType, 
                               is_crypto: bool = False) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect a specific pattern and return detailed results
        
        Args:
            df: OHLCV data
            pattern_type: Specific pattern to detect
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            Tuple of (pattern_found, confidence_score, details_dict)
        """
        if pattern_type not in self.supported_patterns:
            return False, 0.0, {"error": f"Pattern {pattern_type.value} not supported by {self.name}"}
        
        try:
            # Default implementation - subclasses should override for specific patterns
            detected_patterns = self.detect_patterns(df, is_crypto)
            if pattern_type in detected_patterns:
                return True, 0.7, {"detector": self.name, "pattern": pattern_type.value}
            else:
                return False, 0.0, {"detector": self.name, "pattern": pattern_type.value}
        except Exception as e:
            return False, 0.0, {"error": str(e), "detector": self.name}
    
    def get_pattern_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this pattern detector
        
        Returns:
            Dictionary with detector information
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_patterns": [p.value for p in self.supported_patterns],
            "pattern_count": len(self.supported_patterns),
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
        Calculate common technical indicators used across patterns
        
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
        
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} - {len(self.supported_patterns)} patterns"
    
    def __repr__(self) -> str:
        return f"<{self.name}(patterns={len(self.supported_patterns)})>"
