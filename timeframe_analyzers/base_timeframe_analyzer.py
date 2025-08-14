"""
Base Timeframe Analyzer
Defines the interface for all multi-timeframe analysis components
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass

# Import shared data structures
class TimeFrame(Enum):
    D1 = 1
    D3 = 3
    D6 = 6
    D11 = 11
    D21 = 21
    D33 = 33
    D55 = 55
    D89 = 89

class PatternType(Enum):
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

class ConsolidationType(Enum):
    TIGHT = "TIGHT_CONSOLIDATION"
    MODERATE = "MODERATE_CONSOLIDATION"
    WIDE = "WIDE_CONSOLIDATION"
    LOOSE = "LOOSE_CONSOLIDATION"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    COILED = "COILED_SPRING"
    ACCUMULATION = "ACCUMULATION_ZONE"

class PatternMaturity(Enum):
    EARLY = "EARLY"
    DEVELOPING = "DEVELOPING"
    MATURE = "MATURE"
    READY = "READY"
    EXPLOSIVE = "EXPLOSIVE"
    IMMINENT = "IMMINENT"

@dataclass
class PatternCombination:
    patterns: List[PatternType]
    combined_score: float
    timeframe: TimeFrame
    confidence: float
    historical_success_rate: float
    top_stocks: List[str] = None
    risk_level: str = "MEDIUM"
    expected_gain: float = 0.0
    
    def __post_init__(self):
        if self.top_stocks is None:
            self.top_stocks = []


class BaseTimeframeAnalyzer(ABC):
    """
    Base class for all timeframe analyzers
    
    All timeframe analyzers must inherit from this class and implement:
    - analyze_timeframe() method
    - find_best_combination() method
    - determine_consolidation_type() method
    """
    
    def __init__(self, learning_system=None, blacklist_manager=None):
        self.name = self.__class__.__name__
        self.version = "1.0"
        self.learning_system = learning_system
        self.blacklist_manager = blacklist_manager
        self.supported_timeframes = self.get_supported_timeframes()
    
    @abstractmethod
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame, is_crypto: bool = False) -> List[PatternType]:
        """
        Analyze patterns for a specific timeframe
        
        Args:
            df: OHLCV data
            timeframe: TimeFrame to analyze
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            List of detected PatternType enums
        """
        pass
    
    @abstractmethod
    def find_best_combination(self, timeframe_patterns: Dict[TimeFrame, List[PatternType]], 
                             is_crypto: bool = False) -> PatternCombination:
        """
        Find the best pattern combination across timeframes
        
        Args:
            timeframe_patterns: Dictionary of patterns per timeframe
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            PatternCombination object with best combination
        """
        pass
    
    @abstractmethod
    def determine_consolidation_type(self, df: pd.DataFrame, is_crypto: bool = False) -> ConsolidationType:
        """
        Determine the type of consolidation
        
        Args:
            df: OHLCV data
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            ConsolidationType enum
        """
        pass
    
    def get_supported_timeframes(self) -> List[TimeFrame]:
        """
        Return list of timeframes this analyzer supports
        
        Returns:
            List of TimeFrame enums
        """
        return [TimeFrame.D1, TimeFrame.D3, TimeFrame.D6, TimeFrame.D11, 
                TimeFrame.D21, TimeFrame.D33, TimeFrame.D55, TimeFrame.D89]
    
    def determine_maturity(self, stage_results: List, timeframe_patterns: Dict[TimeFrame, List[PatternType]]) -> PatternMaturity:
        """
        Determine pattern maturity based on analysis results
        
        Args:
            stage_results: List of stage analysis results
            timeframe_patterns: Dictionary of patterns per timeframe
            
        Returns:
            PatternMaturity enum
        """
        try:
            # Count strong patterns across timeframes
            total_patterns = sum(len(patterns) for patterns in timeframe_patterns.values())
            strong_patterns = sum(1 for patterns in timeframe_patterns.values() if len(patterns) >= 2)
            
            # Check stage results
            passed_stages = sum(1 for result in stage_results if hasattr(result, 'passed') and result.passed)
            
            if total_patterns >= 8 and strong_patterns >= 4 and passed_stages >= 8:
                return PatternMaturity.EXPLOSIVE
            elif total_patterns >= 6 and strong_patterns >= 3 and passed_stages >= 7:
                return PatternMaturity.IMMINENT
            elif total_patterns >= 4 and strong_patterns >= 2 and passed_stages >= 6:
                return PatternMaturity.READY
            elif total_patterns >= 3 and passed_stages >= 4:
                return PatternMaturity.MATURE
            elif total_patterns >= 2 and passed_stages >= 2:
                return PatternMaturity.DEVELOPING
            else:
                return PatternMaturity.EARLY
        except Exception:
            return PatternMaturity.DEVELOPING
    
    def calculate_entry_setup(self, df: pd.DataFrame, stage_results: List, is_crypto: bool = False) -> Dict[str, float]:
        """
        Calculate entry setup details
        
        Args:
            df: OHLCV data
            stage_results: List of stage analysis results
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            Dictionary with entry setup details
        """
        try:
            current_price = df['Close'].iloc[-1]
            support = df['Low'].tail(20).min()
            resistance = df['High'].tail(20).max()
            
            entry_price = current_price * 0.995  # Slight discount
            stop_loss = support * 0.98
            target_1 = resistance * 1.02
            target_2 = resistance * 1.05
            
            risk = entry_price - stop_loss
            reward_1 = target_1 - entry_price
            risk_reward_1 = reward_1 / risk if risk > 0 else 2.0
            
            return {
                'current': float(current_price),
                'entry': float(entry_price),
                'stop_loss': float(stop_loss),
                'target1': float(target_1),
                'target2': float(target_2),
                'risk_reward_1': float(risk_reward_1),
                'position_size_pct': 5.0 if is_crypto else 3.0
            }
        except Exception:
            current_price = df['Close'].iloc[-1]
            return {
                'current': float(current_price),
                'entry': float(current_price * 0.995),
                'stop_loss': float(current_price * 0.95),
                'target1': float(current_price * 1.05),
                'target2': float(current_price * 1.10),
                'risk_reward_1': 2.0,
                'position_size_pct': 5.0
            }
    
    def create_visual_summary(self, stage_results: List) -> str:
        """
        Create visual summary based on stage results
        
        Args:
            stage_results: List of stage analysis results
            
        Returns:
            Visual summary string
        """
        try:
            passed_stages = sum(1 for result in stage_results if hasattr(result, 'passed') and result.passed)
            total_stages = len(stage_results)
            
            if passed_stages >= total_stages * 0.8:
                return "🔥💎🚀"
            elif passed_stages >= total_stages * 0.6:
                return "⭐📈💰"
            elif passed_stages >= total_stages * 0.4:
                return "📊⚡🎯"
            else:
                return "📉⚠️🔍"
        except Exception:
            return "📊"
    
    def calculate_game_score(self, stage_results: List, timeframe_patterns: Dict[TimeFrame, List[PatternType]]) -> Tuple[float, str]:
        """
        Calculate game score and rarity level
        
        Args:
            stage_results: List of stage analysis results
            timeframe_patterns: Dictionary of patterns per timeframe
            
        Returns:
            Tuple of (game_score, rarity_level)
        """
        try:
            # Base score from stage results
            base_score = 100
            if stage_results:
                avg_score = np.mean([getattr(result, 'score', 50) for result in stage_results])
                base_score = avg_score
            
            # Pattern bonus
            total_patterns = sum(len(patterns) for patterns in timeframe_patterns.values())
            pattern_bonus = min(200, total_patterns * 15)
            
            # Timeframe bonus
            timeframe_bonus = len(timeframe_patterns) * 10
            
            game_score = base_score + pattern_bonus + timeframe_bonus
            
            # Determine rarity
            if game_score >= 500:
                rarity = "🏆 LEGENDARY"
            elif game_score >= 400:
                rarity = "💎 EPIC"
            elif game_score >= 300:
                rarity = "⭐ RARE"
            elif game_score >= 200:
                rarity = "🔵 UNCOMMON"
            else:
                rarity = "⚪ COMMON"
            
            return game_score, rarity
        except Exception:
            return 250.0, "🔵 UNCOMMON"
    
    def detect_special_pattern(self, all_patterns: List[PatternType]) -> str:
        """
        Detect special pattern combinations
        
        Args:
            all_patterns: List of all detected patterns
            
        Returns:
            Special pattern name or empty string
        """
        try:
            pattern_names = [p.value for p in all_patterns]
            
            # Define special combinations
            special_combinations = {
                "🔥 ACCUMULATION ZONE": ["Hidden Accumulation", "Smart Money Flow", "Institutional Absorption"],
                "💎 BREAKOUT IMMINENT": ["Coiled Spring", "Pressure Cooker", "Volume Pocket"],
                "🚀 ROCKET FUEL": ["Momentum Vacuum", "Elliott Wave 3 Setup"],
                "⚡ STEALTH MODE": ["Silent Accumulation", "Dark Pool Activity"],
                "🌟 PERFECT STORM": ["Fibonacci Retracement", "Golden Ratio Pattern"],
                "🎯 PRECISION ENTRY": ["TradingView Triple Support Stable", "Fractal Support Level"],
            }
            
            for special_name, required_patterns in special_combinations.items():
                if any(pattern in pattern_names for pattern in required_patterns):
                    return special_name
            
            return ""
        except Exception:
            return ""
    
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
    
    def get_analyzer_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this timeframe analyzer
        
        Returns:
            Dictionary with analyzer information
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_timeframes": [tf.value for tf in self.supported_timeframes],
            "timeframe_count": len(self.supported_timeframes),
            "description": self.__doc__ or "No description available"
        }
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} - {len(self.supported_timeframes)} timeframes"
    
    def __repr__(self) -> str:
        return f"<{self.name}(timeframes={len(self.supported_timeframes)})>"
