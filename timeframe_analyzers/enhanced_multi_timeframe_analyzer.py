"""
Enhanced Multi-Timeframe Analyzer
Implements sophisticated multi-timeframe pattern analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from .base_timeframe_analyzer import BaseTimeframeAnalyzer, TimeFrame, PatternType, ConsolidationType, PatternMaturity, PatternCombination


class EnhancedMultiTimeframeAnalyzer(BaseTimeframeAnalyzer):
    """
    Enhanced multi-timeframe analyzer with sophisticated pattern correlation
    """
    
    def __init__(self, learning_system=None, blacklist_manager=None):
        super().__init__(learning_system, blacklist_manager)
        self.version = "6.1"
        self.pattern_correlation_matrix = self._build_pattern_correlation_matrix()
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame, is_crypto: bool = False) -> List[PatternType]:
        """
        Analyze patterns for a specific timeframe
        """
        if not self.validate_data(df):
            return []
        
        patterns = []
        
        # Adjust data for timeframe
        tf_df = self._adjust_data_for_timeframe(df, timeframe)
        
        if len(tf_df) < 10:
            return []
        
        # Pattern detection based on timeframe characteristics
        patterns.extend(self._detect_timeframe_specific_patterns(tf_df, timeframe, is_crypto))
        
        # Technical indicator patterns
        patterns.extend(self._detect_technical_patterns(tf_df, timeframe, is_crypto))
        
        # Volume-based patterns (if volume data available)
        if 'Volume' in tf_df.columns:
            patterns.extend(self._detect_volume_patterns(tf_df, timeframe, is_crypto))
        
        return patterns
    
    def find_best_combination(self, timeframe_patterns: Dict[TimeFrame, List[PatternType]], 
                             is_crypto: bool = False) -> PatternCombination:
        """
        Find the best pattern combination across timeframes
        """
        if not timeframe_patterns:
            return PatternCombination(
                patterns=[], 
                combined_score=50.0, 
                timeframe=TimeFrame.D1,
                confidence=50.0,
                historical_success_rate=0.5
            )
        
        best_score = 0
        best_combination = None
        
        # Evaluate all possible combinations
        for timeframe, patterns in timeframe_patterns.items():
            if not patterns:
                continue
            
            score = self._calculate_combination_score(patterns, timeframe, is_crypto)
            
            if score > best_score:
                best_score = score
                best_combination = PatternCombination(
                    patterns=patterns,
                    combined_score=score,
                    timeframe=timeframe,
                    confidence=min(95, score * 1.2),
                    historical_success_rate=self._get_historical_success_rate(patterns, is_crypto),
                    risk_level=self._assess_risk_level(patterns, timeframe),
                    expected_gain=self._estimate_expected_gain(patterns, timeframe, is_crypto)
                )
        
        if best_combination is None:
            # Return default combination with first available patterns
            first_timeframe = list(timeframe_patterns.keys())[0]
            first_patterns = timeframe_patterns[first_timeframe]
            
            best_combination = PatternCombination(
                patterns=first_patterns,
                combined_score=60.0,
                timeframe=first_timeframe,
                confidence=60.0,
                historical_success_rate=0.6
            )
        
        return best_combination
    
    def determine_consolidation_type(self, df: pd.DataFrame, is_crypto: bool = False) -> ConsolidationType:
        """
        Determine the type of consolidation
        """
        try:
            if len(df) < 20:
                return ConsolidationType.MODERATE
            
            # Calculate price range and volatility
            high = df['High'].tail(20).max()
            low = df['Low'].tail(20).min()
            avg_price = df['Close'].tail(20).mean()
            
            price_range_pct = (high - low) / avg_price
            
            # Calculate volatility
            returns = df['Close'].pct_change().tail(20).dropna()
            volatility = returns.std()
            
            # Volume analysis if available
            volume_factor = 1.0
            if 'Volume' in df.columns:
                volume_cv = df['Volume'].tail(20).std() / df['Volume'].tail(20).mean()
                volume_factor = 1 + volume_cv
            
            # Determine consolidation type
            adjusted_range = price_range_pct * volume_factor
            
            if adjusted_range < 0.03:
                return ConsolidationType.TIGHT
            elif adjusted_range < 0.06:
                return ConsolidationType.COILED
            elif adjusted_range < 0.10:
                return ConsolidationType.MODERATE
            elif adjusted_range < 0.15:
                return ConsolidationType.WIDE
            elif volatility > 0.05:
                return ConsolidationType.HIGH_VOLATILITY
            else:
                return ConsolidationType.ACCUMULATION
                
        except Exception:
            return ConsolidationType.MODERATE
    
    # ================== HELPER METHODS ==================
    
    def _adjust_data_for_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame) -> pd.DataFrame:
        """Adjust data sampling for different timeframes"""
        try:
            tf_value = timeframe.value
            
            if tf_value == 1:
                return df
            elif tf_value <= len(df):
                # Sample every tf_value days
                return df.iloc[::tf_value].copy()
            else:
                # If timeframe is larger than data, return last portion
                return df.tail(max(10, len(df) // 2)).copy()
        except Exception:
            return df
    
    def _detect_timeframe_specific_patterns(self, df: pd.DataFrame, timeframe: TimeFrame, is_crypto: bool) -> List[PatternType]:
        """Detect patterns specific to timeframe characteristics"""
        patterns = []
        
        try:
            # Short-term patterns (D1, D3)
            if timeframe.value <= 3:
                if self._detect_momentum_acceleration(df):
                    patterns.append(PatternType.MOMENTUM_DIVERGENCE)
                
                if self._detect_breakout_preparation(df):
                    patterns.append(PatternType.STEALTH_BREAKOUT)
            
            # Medium-term patterns (D6, D11, D21)
            elif timeframe.value <= 21:
                if self._detect_accumulation_pattern(df):
                    patterns.append(PatternType.HIDDEN_ACCUMULATION)
                
                if self._detect_consolidation_pattern(df):
                    patterns.append(PatternType.COILED_SPRING)
            
            # Long-term patterns (D33, D55, D89)
            else:
                if self._detect_institutional_activity(df):
                    patterns.append(PatternType.INSTITUTIONAL_ABSORPTION)
                
                if self._detect_trend_continuation(df):
                    patterns.append(PatternType.ELLIOTT_WAVE_3)
        
        except Exception:
            pass
        
        return patterns
    
    def _detect_technical_patterns(self, df: pd.DataFrame, timeframe: TimeFrame, is_crypto: bool) -> List[PatternType]:
        """Detect technical indicator based patterns"""
        patterns = []
        
        try:
            # RSI analysis
            if len(df) >= 14:
                rsi = self._calculate_rsi(df['Close'])
                if self._detect_rsi_divergence(df['Close'], rsi):
                    patterns.append(PatternType.RSI_DIVERGENCE)
            
            # MACD analysis
            if len(df) >= 26:
                if self._detect_macd_bullish_setup(df['Close']):
                    patterns.append(PatternType.MOMENTUM_VACUUM)
            
            # Bollinger Bands
            if len(df) >= 20:
                if self._detect_bollinger_squeeze(df['Close']):
                    patterns.append(PatternType.BOLLINGER_SQUEEZE)
            
            # Support/Resistance
            if self._detect_support_level_strength(df):
                patterns.append(PatternType.FRACTAL_SUPPORT)
        
        except Exception:
            pass
        
        return patterns
    
    def _detect_volume_patterns(self, df: pd.DataFrame, timeframe: TimeFrame, is_crypto: bool) -> List[PatternType]:
        """Detect volume-based patterns"""
        patterns = []
        
        try:
            if 'Volume' not in df.columns:
                return patterns
            
            # Volume accumulation
            if self._detect_volume_accumulation(df):
                patterns.append(PatternType.VWAP_ACCUMULATION)
            
            # Large volume with price stability
            if self._detect_smart_money_volume(df):
                patterns.append(PatternType.SMART_MONEY_FLOW)
            
            # Volume pocket (low volume before breakout)
            if self._detect_volume_pocket_pattern(df):
                patterns.append(PatternType.VOLUME_POCKET)
        
        except Exception:
            pass
        
        return patterns
    
    def _calculate_combination_score(self, patterns: List[PatternType], timeframe: TimeFrame, is_crypto: bool) -> float:
        """Calculate score for pattern combination"""
        try:
            if not patterns:
                return 0
            
            base_score = len(patterns) * 15
            
            # Timeframe bonus
            tf_multiplier = {
                TimeFrame.D1: 1.0,
                TimeFrame.D3: 1.1,
                TimeFrame.D6: 1.2,
                TimeFrame.D11: 1.3,
                TimeFrame.D21: 1.4,
                TimeFrame.D33: 1.3,
                TimeFrame.D55: 1.2,
                TimeFrame.D89: 1.1
            }.get(timeframe, 1.0)
            
            # Pattern quality bonus
            high_quality_patterns = [
                PatternType.ELLIOTT_WAVE_3,
                PatternType.FIBONACCI_RETRACEMENT,
                PatternType.INSTITUTIONAL_ABSORPTION,
                PatternType.SMART_MONEY_FLOW,
                PatternType.TRADINGVIEW_TRIPLE_SUPPORT
            ]
            
            quality_bonus = sum(10 for p in patterns if p in high_quality_patterns)
            
            # Crypto bonus
            crypto_bonus = 5 if is_crypto else 0
            
            total_score = (base_score + quality_bonus + crypto_bonus) * tf_multiplier
            
            return min(100, total_score)
        
        except Exception:
            return 50.0
    
    def _get_historical_success_rate(self, patterns: List[PatternType], is_crypto: bool) -> float:
        """Get historical success rate for pattern combination"""
        try:
            if not patterns or not self.learning_system:
                return 0.65 if is_crypto else 0.60
            
            success_rates = []
            for pattern in patterns:
                score, metadata = self.learning_system.get_advanced_pattern_score(
                    pattern.value, is_crypto
                )
                if metadata.get('sample_size', 0) > 0:
                    success_rates.append(score)
            
            if success_rates:
                return np.mean(success_rates)
            else:
                return 0.65 if is_crypto else 0.60
        
        except Exception:
            return 0.60
    
    def _assess_risk_level(self, patterns: List[PatternType], timeframe: TimeFrame) -> str:
        """Assess risk level for pattern combination"""
        try:
            high_risk_patterns = [
                PatternType.MOMENTUM_DIVERGENCE,
                PatternType.STEALTH_BREAKOUT,
                PatternType.VOLATILITY_CONTRACTION
            ]
            
            low_risk_patterns = [
                PatternType.INSTITUTIONAL_ABSORPTION,
                PatternType.SMART_MONEY_FLOW,
                PatternType.FRACTAL_SUPPORT
            ]
            
            high_risk_count = sum(1 for p in patterns if p in high_risk_patterns)
            low_risk_count = sum(1 for p in patterns if p in low_risk_patterns)
            
            # Longer timeframes generally lower risk
            tf_risk_adjustment = 0 if timeframe.value >= 21 else 1
            
            net_risk = high_risk_count - low_risk_count + tf_risk_adjustment
            
            if net_risk <= -2:
                return "LOW"
            elif net_risk >= 2:
                return "HIGH"
            else:
                return "MEDIUM"
        
        except Exception:
            return "MEDIUM"
    
    def _estimate_expected_gain(self, patterns: List[PatternType], timeframe: TimeFrame, is_crypto: bool) -> float:
        """Estimate expected gain for pattern combination"""
        try:
            base_gain = 0.08 if is_crypto else 0.05
            
            # Pattern-based adjustments
            gain_multipliers = {
                PatternType.ELLIOTT_WAVE_3: 1.5,
                PatternType.FIBONACCI_RETRACEMENT: 1.3,
                PatternType.INSTITUTIONAL_ABSORPTION: 1.2,
                PatternType.MOMENTUM_VACUUM: 1.4,
                PatternType.COILED_SPRING: 1.6
            }
            
            max_multiplier = 1.0
            for pattern in patterns:
                multiplier = gain_multipliers.get(pattern, 1.0)
                max_multiplier = max(max_multiplier, multiplier)
            
            # Timeframe adjustment
            tf_adjustment = min(1.5, timeframe.value / 21)
            
            expected_gain = base_gain * max_multiplier * tf_adjustment
            
            return min(0.25, expected_gain)  # Cap at 25%
        
        except Exception:
            return 0.10
    
    def _build_pattern_correlation_matrix(self) -> Dict[PatternType, List[PatternType]]:
        """Build correlation matrix for pattern combinations"""
        return {
            PatternType.HIDDEN_ACCUMULATION: [
                PatternType.SMART_MONEY_FLOW,
                PatternType.INSTITUTIONAL_ABSORPTION,
                PatternType.SILENT_ACCUMULATION
            ],
            PatternType.COILED_SPRING: [
                PatternType.PRESSURE_COOKER,
                PatternType.BOLLINGER_SQUEEZE,
                PatternType.VOLATILITY_CONTRACTION
            ],
            PatternType.MOMENTUM_VACUUM: [
                PatternType.STEALTH_BREAKOUT,
                PatternType.MOMENTUM_DIVERGENCE,
                PatternType.RSI_DIVERGENCE
            ]
        }
    
    # ================== PATTERN DETECTION METHODS ==================
    
    def _detect_momentum_acceleration(self, df: pd.DataFrame) -> bool:
        """Detect momentum acceleration pattern"""
        try:
            if len(df) < 10:
                return False
            
            recent_momentum = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
            earlier_momentum = (df['Close'].iloc[-5] - df['Close'].iloc[-10]) / df['Close'].iloc[-10]
            
            return recent_momentum > earlier_momentum and recent_momentum > 0.02
        except Exception:
            return False
    
    def _detect_breakout_preparation(self, df: pd.DataFrame) -> bool:
        """Detect breakout preparation pattern"""
        try:
            if len(df) < 15:
                return False
            
            resistance = df['High'].tail(10).max()
            current_price = df['Close'].iloc[-1]
            
            # Price near resistance with decreasing volume
            near_resistance = (resistance - current_price) / current_price < 0.02
            
            if 'Volume' in df.columns:
                volume_decrease = df['Volume'].tail(3).mean() < df['Volume'].tail(10).mean()
                return near_resistance and volume_decrease
            else:
                return near_resistance
        except Exception:
            return False
    
    def _detect_accumulation_pattern(self, df: pd.DataFrame) -> bool:
        """Detect accumulation pattern"""
        try:
            if len(df) < 20:
                return False
            
            # Price consolidation with potential volume increase
            price_range = (df['High'].tail(15).max() - df['Low'].tail(15).min()) / df['Close'].tail(15).mean()
            
            if 'Volume' in df.columns:
                volume_trend = df['Volume'].tail(5).mean() / df['Volume'].tail(15).mean()
                return price_range < 0.08 and volume_trend > 1.1
            else:
                return price_range < 0.06
        except Exception:
            return False
    
    def _detect_consolidation_pattern(self, df: pd.DataFrame) -> bool:
        """Detect consolidation pattern"""
        try:
            if len(df) < 15:
                return False
            
            volatility = df['Close'].tail(10).pct_change().std()
            historical_volatility = df['Close'].pct_change().std()
            
            return volatility < historical_volatility * 0.7
        except Exception:
            return False
    
    def _detect_institutional_activity(self, df: pd.DataFrame) -> bool:
        """Detect institutional activity pattern"""
        try:
            if 'Volume' not in df.columns or len(df) < 20:
                return False
            
            # Large volume with minimal price impact
            avg_volume = df['Volume'].mean()
            large_volume_days = (df['Volume'] > avg_volume * 1.5).tail(10).sum()
            
            price_stability = df['Close'].tail(10).std() / df['Close'].tail(10).mean()
            
            return large_volume_days >= 3 and price_stability < 0.05
        except Exception:
            return False
    
    def _detect_trend_continuation(self, df: pd.DataFrame) -> bool:
        """Detect trend continuation pattern"""
        try:
            if len(df) < 30:
                return False
            
            # Long-term uptrend with recent pullback
            long_term_trend = df['Close'].iloc[-1] > df['Close'].iloc[-30]
            recent_pullback = df['Close'].iloc[-5:].min() < df['Close'].iloc[-10]
            
            return long_term_trend and recent_pullback
        except Exception:
            return False
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices))
    
    def _detect_rsi_divergence(self, prices: pd.Series, rsi: pd.Series) -> bool:
        """Detect RSI divergence"""
        try:
            if len(prices) < 20:
                return False
            
            # Bullish divergence: price making lower lows, RSI making higher lows
            price_trend = prices.iloc[-1] < prices.iloc[-10]
            rsi_trend = rsi.iloc[-1] > rsi.iloc[-10]
            
            return price_trend and rsi_trend and rsi.iloc[-1] < 50
        except Exception:
            return False
    
    def _detect_macd_bullish_setup(self, prices: pd.Series) -> bool:
        """Detect MACD bullish setup"""
        try:
            if len(prices) < 26:
                return False
            
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            # MACD crossing above signal line
            return macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
        except Exception:
            return False
    
    def _detect_bollinger_squeeze(self, prices: pd.Series) -> bool:
        """Detect Bollinger Band squeeze"""
        try:
            if len(prices) < 20:
                return False
            
            sma = prices.rolling(20).mean()
            std = prices.rolling(20).std()
            bb_width = (2 * std) / sma
            
            current_width = bb_width.iloc[-1]
            avg_width = bb_width.mean()
            
            return current_width < avg_width * 0.7
        except Exception:
            return False
    
    def _detect_support_level_strength(self, df: pd.DataFrame) -> bool:
        """Detect strong support level"""
        try:
            if len(df) < 20:
                return False
            
            support_level = df['Low'].tail(20).min()
            tolerance = support_level * 0.02
            
            touches = sum(1 for low in df['Low'].tail(20) 
                         if support_level - tolerance <= low <= support_level + tolerance)
            
            return touches >= 3
        except Exception:
            return False
    
    def _detect_volume_accumulation(self, df: pd.DataFrame) -> bool:
        """Detect volume accumulation"""
        try:
            if 'Volume' not in df.columns or len(df) < 15:
                return False
            
            recent_volume = df['Volume'].tail(5).mean()
            historical_volume = df['Volume'].mean()
            
            return recent_volume > historical_volume * 1.2
        except Exception:
            return False
    
    def _detect_smart_money_volume(self, df: pd.DataFrame) -> bool:
        """Detect smart money volume patterns"""
        try:
            if 'Volume' not in df.columns or len(df) < 10:
                return False
            
            # Higher volume on down days
            down_days = df[df['Close'] < df['Open']]
            up_days = df[df['Close'] > df['Open']]
            
            if len(down_days) == 0 or len(up_days) == 0:
                return False
            
            avg_down_volume = down_days['Volume'].mean()
            avg_up_volume = up_days['Volume'].mean()
            
            return avg_down_volume > avg_up_volume * 1.1
        except Exception:
            return False
    
    def _detect_volume_pocket_pattern(self, df: pd.DataFrame) -> bool:
        """Detect volume pocket pattern"""
        try:
            if 'Volume' not in df.columns or len(df) < 10:
                return False
            
            recent_volume = df['Volume'].tail(3).mean()
            previous_volume = df['Volume'].iloc[-10:-3].mean()
            
            return recent_volume < previous_volume * 0.8
        except Exception:
            return False
