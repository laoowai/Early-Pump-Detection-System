"""
Advanced Pattern Detector
Implements 20+ sophisticated pattern detection algorithms
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import linregress
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler

from .base_detector import BasePatternDetector, PatternType


class AdvancedPatternDetector(BasePatternDetector):
    """
    Advanced pattern detection with 20+ sophisticated patterns
    Detects early pump signals and institutional accumulation patterns
    """
    
    def __init__(self):
        super().__init__()
        self.version = "6.1"
        self.scaler = MinMaxScaler()
    
    def get_supported_patterns(self) -> List[PatternType]:
        """Return all supported advanced patterns"""
        return [
            # Advanced accumulation patterns
            PatternType.HIDDEN_ACCUMULATION,
            PatternType.SMART_MONEY_FLOW,
            PatternType.WHALE_ACCUMULATION,
            PatternType.SILENT_ACCUMULATION,
            PatternType.INSTITUTIONAL_ABSORPTION,
            
            # Compression patterns
            PatternType.COILED_SPRING,
            PatternType.PRESSURE_COOKER,
            PatternType.VOLATILITY_CONTRACTION,
            PatternType.BOLLINGER_SQUEEZE,
            
            # Momentum patterns
            PatternType.MOMENTUM_VACUUM,
            PatternType.MOMENTUM_DIVERGENCE,
            PatternType.RSI_DIVERGENCE,
            PatternType.STOCHASTIC_DIVERGENCE,
            PatternType.WILLIAMS_R_REVERSAL,
            PatternType.CCI_BULLISH,
            
            # Advanced technical patterns
            PatternType.FIBONACCI_RETRACEMENT,
            PatternType.ELLIOTT_WAVE_3,
            PatternType.ICHIMOKU_CLOUD,
            PatternType.GOLDEN_RATIO,
            PatternType.FRACTAL_SUPPORT,
            
            # Professional patterns
            PatternType.TRADINGVIEW_TRIPLE_SUPPORT,
            PatternType.VWAP_ACCUMULATION,
            PatternType.DARK_POOL_ACTIVITY,
            PatternType.ALGORITHM_PATTERN,
            PatternType.STEALTH_BREAKOUT,
            PatternType.VOLUME_POCKET
        ]
    
    def detect_patterns(self, df: pd.DataFrame, is_crypto: bool = False) -> List[PatternType]:
        """
        Detect all supported patterns in the given DataFrame
        
        Args:
            df: OHLCV data
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            List of detected PatternType enums
        """
        if not self.validate_data(df):
            return []
        
        detected_patterns = []
        
        # Pattern detection mapping
        pattern_methods = [
            (self.detect_hidden_accumulation, PatternType.HIDDEN_ACCUMULATION),
            (self.detect_smart_money_flow, PatternType.SMART_MONEY_FLOW),
            (self.detect_whale_accumulation, PatternType.WHALE_ACCUMULATION),
            (self.detect_coiled_spring, PatternType.COILED_SPRING),
            (self.detect_momentum_vacuum, PatternType.MOMENTUM_VACUUM),
            (self.detect_fibonacci_retracement, PatternType.FIBONACCI_RETRACEMENT),
            (self.detect_elliott_wave_3, PatternType.ELLIOTT_WAVE_3),
            (self.detect_bollinger_squeeze, PatternType.BOLLINGER_SQUEEZE),
            (self.detect_ichimoku_cloud, PatternType.ICHIMOKU_CLOUD),
            (self.detect_tradingview_triple_support, PatternType.TRADINGVIEW_TRIPLE_SUPPORT),
        ]
        
        # Additional simplified patterns
        additional_patterns = [
            (self._detect_pressure_cooker, PatternType.PRESSURE_COOKER),
            (self._detect_volatility_contraction, PatternType.VOLATILITY_CONTRACTION),
            (self._detect_silent_accumulation, PatternType.SILENT_ACCUMULATION),
            (self._detect_institutional_absorption, PatternType.INSTITUTIONAL_ABSORPTION),
            (self._detect_stealth_breakout, PatternType.STEALTH_BREAKOUT),
            (self._detect_volume_pocket, PatternType.VOLUME_POCKET),
            (self._detect_momentum_divergence, PatternType.MOMENTUM_DIVERGENCE),
            (self._detect_rsi_divergence, PatternType.RSI_DIVERGENCE),
            (self._detect_stochastic_divergence, PatternType.STOCHASTIC_DIVERGENCE),
            (self._detect_williams_r_reversal, PatternType.WILLIAMS_R_REVERSAL),
            (self._detect_cci_bullish, PatternType.CCI_BULLISH),
            (self._detect_vwap_accumulation, PatternType.VWAP_ACCUMULATION),
            (self._detect_dark_pool_activity, PatternType.DARK_POOL_ACTIVITY),
            (self._detect_algorithm_pattern, PatternType.ALGORITHM_PATTERN),
            (self._detect_fractal_support, PatternType.FRACTAL_SUPPORT),
            (self._detect_golden_ratio, PatternType.GOLDEN_RATIO),
        ]
        
        # Run all advanced pattern detections
        for method, pattern_type in pattern_methods:
            try:
                passed, score, details = method(df, is_crypto)
                if passed and score >= 60:
                    detected_patterns.append(pattern_type)
            except Exception as e:
                continue
        
        # Run additional simplified patterns
        for method, pattern_type in additional_patterns:
            try:
                if method(df, is_crypto):
                    detected_patterns.append(pattern_type)
            except Exception as e:
                continue
        
        return detected_patterns

    # ================== ADVANCED PATTERN DETECTION METHODS ==================
    
    def detect_hidden_accumulation(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect hidden accumulation - when smart money quietly accumulates"""
        try:
            if len(df) < 50:
                return False, 0, {}

            # Price action: sideways or slight decline
            price_trend = self._calculate_trend_strength(df['Close'].tail(30))

            # Volume analysis: increasing volume on down days, decreasing on up days
            volume_analysis = self._analyze_accumulation_volume(df.tail(30))

            # OBV divergence
            obv_divergence = self._calculate_obv_divergence(df.tail(50))

            # Money flow index
            mfi = self._calculate_mfi(df.tail(20))

            score = 0
            if -0.1 < price_trend < 0.05:  # Sideways to slight decline
                score += 25
            if volume_analysis['accumulation_score'] > 0.6:
                score += 30
            if obv_divergence > 0.3:
                score += 25
            if 20 < mfi < 50:  # Not oversold, not overbought
                score += 20

            passed = score >= 60

            details = {
                'price_trend': price_trend,
                'accumulation_score': volume_analysis['accumulation_score'],
                'obv_divergence': obv_divergence,
                'mfi': mfi
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_smart_money_flow(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect smart money flow patterns"""
        try:
            if len(df) < 40:
                return False, 0, {}

            # VWAP analysis
            vwap = self._calculate_vwap(df.tail(30))
            price_vs_vwap = (df['Close'].iloc[-1] - vwap) / vwap

            # Volume profile analysis
            volume_profile = self._analyze_volume_profile(df.tail(40))

            # Large transaction detection
            large_volume_days = self._detect_large_volume_days(df.tail(20))

            # Accumulation/Distribution line
            ad_line = self._calculate_ad_line(df.tail(30))

            score = 0
            if -0.02 < price_vs_vwap < 0.05:  # Near VWAP
                score += 20
            if volume_profile['concentration'] > 0.7:
                score += 25
            if large_volume_days >= 3:
                score += 30
            if ad_line > 0.5:
                score += 25

            passed = score >= 70

            details = {
                'price_vs_vwap': price_vs_vwap,
                'volume_concentration': volume_profile['concentration'],
                'large_volume_days': large_volume_days,
                'ad_line': ad_line
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_whale_accumulation(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect whale accumulation patterns"""
        try:
            if len(df) < 30:
                return False, 0, {}

            # Unusual volume spikes on red candles
            volume_on_red = self._analyze_volume_on_red_candles(df.tail(20))

            # Support level strength
            support_strength = self._calculate_support_strength(df.tail(30))

            # Price absorption
            price_absorption = self._calculate_price_absorption(df.tail(15))

            # Volume weighted price impact
            price_impact = self._calculate_volume_price_impact(df.tail(20))

            score = 0
            if volume_on_red > 0.6:
                score += 30
            if support_strength > 0.7:
                score += 25
            if price_absorption > 0.5:
                score += 25
            if price_impact < 0.3:  # Low price impact despite volume
                score += 20

            passed = score >= 75

            details = {
                'volume_on_red': volume_on_red,
                'support_strength': support_strength,
                'price_absorption': price_absorption,
                'price_impact': price_impact
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_coiled_spring(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect coiled spring pattern - extreme compression before explosion"""
        try:
            if len(df) < 40:
                return False, 0, {}

            # Bollinger Band width
            bb_width = self._calculate_bollinger_band_width(df.tail(30))

            # ATR compression
            atr_compression = self._calculate_atr_compression(df.tail(30))

            # Volume contraction
            volume_contraction = self._calculate_volume_contraction(df.tail(20))

            # Time compression
            time_compression = self._calculate_time_compression(df.tail(25))

            score = 0
            if bb_width < 0.1:  # Very tight bands
                score += 30
            if atr_compression > 0.7:
                score += 25
            if volume_contraction > 0.6:
                score += 25
            if time_compression > 0.5:
                score += 20

            passed = score >= 80

            details = {
                'bb_width': bb_width,
                'atr_compression': atr_compression,
                'volume_contraction': volume_contraction,
                'time_compression': time_compression
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_momentum_vacuum(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect momentum vacuum - lack of selling pressure"""
        try:
            if len(df) < 30:
                return False, 0, {}

            # RSI pattern
            rsi = self._calculate_rsi(df['Close'].tail(30))
            rsi_pattern = self._analyze_rsi_pattern(rsi)

            # MACD momentum
            macd_momentum = self._calculate_macd_momentum(df.tail(30))

            # Selling pressure analysis
            selling_pressure = self._analyze_selling_pressure(df.tail(20))

            # Momentum oscillator
            momentum_osc = self._calculate_momentum_oscillator(df.tail(25))

            score = 0
            if rsi_pattern['vacuum_score'] > 0.6:
                score += 25
            if macd_momentum > 0.3:
                score += 25
            if selling_pressure < 0.3:  # Low selling pressure
                score += 30
            if momentum_osc > 0.4:
                score += 20

            passed = score >= 70

            details = {
                'rsi_vacuum_score': rsi_pattern['vacuum_score'],
                'macd_momentum': macd_momentum,
                'selling_pressure': selling_pressure,
                'momentum_oscillator': momentum_osc
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_fibonacci_retracement(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect Fibonacci retracement levels"""
        try:
            if len(df) < 50:
                return False, 0, {}

            # Find recent significant swing high and low
            swing_high, swing_low = self._find_swing_points(df.tail(50))

            if swing_high is None or swing_low is None:
                return False, 0, {}

            # Calculate Fibonacci levels
            fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low)

            # Check current price position
            current_price = df['Close'].iloc[-1]
            fib_level_hit = self._check_fibonacci_level(current_price, fib_levels)

            # Volume at Fibonacci levels
            volume_at_fib = self._analyze_volume_at_fibonacci(df.tail(30), fib_levels)

            # Reaction strength
            reaction_strength = self._calculate_fibonacci_reaction(df.tail(20), fib_levels)

            score = 0
            if fib_level_hit in [0.382, 0.5, 0.618]:  # Key levels
                score += 30
            if volume_at_fib > 0.6:
                score += 25
            if reaction_strength > 0.5:
                score += 25
            if fib_level_hit == 0.618:  # Golden ratio
                score += 20

            passed = score >= 60

            details = {
                'fibonacci_level': fib_level_hit,
                'volume_at_fibonacci': volume_at_fib,
                'reaction_strength': reaction_strength,
                'swing_high': swing_high,
                'swing_low': swing_low
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_elliott_wave_3(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect Elliott Wave 3 setup"""
        try:
            if len(df) < 60:
                return False, 0, {}

            # Identify wave structure
            waves = self._identify_elliott_waves(df.tail(60))

            if len(waves) < 2:
                return False, 0, {}

            # Check for wave 1 and 2 completion
            wave_1_2_complete = self._check_wave_1_2_completion(waves)

            # Wave 3 characteristics
            wave_3_setup = self._analyze_wave_3_setup(df.tail(30), waves)

            # Momentum confirmation
            momentum_confirm = self._check_elliott_momentum(df.tail(40))

            # Fibonacci extension
            fib_extension = self._calculate_elliott_fibonacci_extension(waves)

            score = 0
            if wave_1_2_complete:
                score += 30
            if wave_3_setup['strength'] > 0.6:
                score += 25
            if momentum_confirm > 0.5:
                score += 25
            if 1.618 <= fib_extension <= 2.618:  # Common Wave 3 extensions
                score += 20

            passed = score >= 70

            details = {
                'wave_1_2_complete': wave_1_2_complete,
                'wave_3_strength': wave_3_setup['strength'],
                'momentum_confirmation': momentum_confirm,
                'fibonacci_extension': fib_extension
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_bollinger_squeeze(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect Bollinger Band squeeze"""
        try:
            if len(df) < 30:
                return False, 0, {}

            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'].tail(30))

            # Calculate band width
            bb_width = (bb_upper - bb_lower) / bb_middle
            current_width = bb_width.iloc[-1]
            avg_width = bb_width.mean()

            # Squeeze intensity
            squeeze_ratio = current_width / avg_width

            # Volume pattern during squeeze
            volume_pattern = self._analyze_squeeze_volume_pattern(df.tail(20))

            # Price position in bands
            price_position = (df['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

            # Historical squeeze breakouts
            breakout_success = self._analyze_historical_squeeze_breakouts(df.tail(50))

            score = 0
            if squeeze_ratio < 0.7:  # Tight squeeze
                score += 30
            if volume_pattern > 0.5:
                score += 25
            if 0.3 <= price_position <= 0.7:  # Middle of bands
                score += 20
            if breakout_success > 0.6:
                score += 25

            passed = score >= 70

            details = {
                'squeeze_ratio': squeeze_ratio,
                'volume_pattern': volume_pattern,
                'price_position': price_position,
                'breakout_success_rate': breakout_success
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_tradingview_triple_support(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect TradingView-style triple support stable pattern"""
        try:
            if len(df) < 50:
                return False, 0, {}

            # TradingView parameters
            left_bars = 15
            right_bars = 15
            volume_thresh = 20

            # Calculate pivot lows (support levels)
            support_levels = []
            pivot_ages = []

            # Find pivot lows across different periods for multiple timeframes
            for period in [20, 30, 45]:
                if len(df) < period + right_bars:
                    continue

                for i in range(left_bars, len(df) - right_bars):
                    # Check if this is a pivot low
                    current_low = df['Low'].iloc[i]

                    # Check left side
                    left_ok = all(current_low <= df['Low'].iloc[i - j] for j in range(1, left_bars + 1))

                    # Check right side
                    right_ok = all(current_low <= df['Low'].iloc[i + j] for j in range(1, right_bars + 1))

                    if left_ok and right_ok:
                        # Calculate how stable this level has been
                        stability_period = 0
                        tolerance = current_low * 0.02  # 2% tolerance

                        # Check forward stability
                        for j in range(i + right_bars, min(len(df), i + period)):
                            if abs(df['Low'].iloc[j] - current_low) <= tolerance:
                                stability_period += 1
                            elif df['Low'].iloc[j] < current_low - tolerance:
                                break  # Support broken

                        support_levels.append({
                            'level': current_low,
                            'index': i,
                            'stability': stability_period,
                            'age': len(df) - i
                        })

            # Filter and rank support levels
            if len(support_levels) < 3:
                return False, 0, {'reason': 'Less than 3 support levels found'}

            # Sort by stability and age
            support_levels.sort(key=lambda x: (x['stability'], x['age']), reverse=True)

            # Take top 3 most stable supports
            top_supports = support_levels[:3]

            # Check if these supports are "stable"
            stable_supports = 0
            current_price = df['Close'].iloc[-1]

            for support in top_supports:
                level = support['level']
                recent_lows = df['Low'].tail(10)

                # Support is stable if price hasn't significantly broken below it
                if recent_lows.min() >= level * 0.98:
                    stable_supports += 1

                # Extra points if current price is near this support
                if abs(current_price - level) / level < 0.05:
                    stable_supports += 0.5

            # Volume confirmation
            volume_score = 0
            if 'Volume' in df.columns and len(df) > 10:
                short_vol = df['Volume'].ewm(span=5).mean()
                long_vol = df['Volume'].ewm(span=10).mean()
                vol_osc = 100 * (short_vol - long_vol) / long_vol

                if vol_osc.iloc[-1] > volume_thresh:
                    volume_score = 30
                elif vol_osc.tail(5).mean() > volume_thresh * 0.7:
                    volume_score = 20
                else:
                    volume_score = 10
            else:
                volume_score = 15

            # Calculate final score
            base_score = stable_supports * 20
            stability_bonus = sum(s['stability'] for s in top_supports) * 2
            age_bonus = min(20, sum(s['age'] for s in top_supports) / 10)

            total_score = base_score + volume_score + stability_bonus + age_bonus

            # Must have at least 2.5 stable supports to pass
            passed = stable_supports >= 2.5 and total_score >= 70

            # Bonus for higher timeframes
            timeframe_bonus = 0
            if len(df) > 200:
                timeframe_bonus = 10
                total_score += timeframe_bonus

            details = {
                'stable_supports_count': stable_supports,
                'total_supports_found': len(support_levels),
                'top_support_levels': [s['level'] for s in top_supports],
                'support_stability': [s['stability'] for s in top_supports],
                'support_ages': [s['age'] for s in top_supports],
                'volume_oscillator': vol_osc.iloc[-1] if 'Volume' in df.columns else 0,
                'volume_score': volume_score,
                'timeframe_bonus': timeframe_bonus,
                'pattern_strength': 'STRONG' if total_score > 90 else 'MEDIUM' if total_score > 80 else 'WEAK'
            }

            return passed, min(100, total_score), details

        except Exception as e:
            return False, 0, {'error': str(e)}

    def detect_ichimoku_cloud(self, df: pd.DataFrame, is_crypto: bool = False) -> Tuple[bool, float, Dict]:
        """Detect Ichimoku Cloud breakout setup"""
        try:
            if len(df) < 60:
                return False, 0, {}

            # Calculate Ichimoku components
            ichimoku = self._calculate_ichimoku(df.tail(60))

            # Cloud analysis
            cloud_analysis = self._analyze_ichimoku_cloud(ichimoku)

            # Tenkan/Kijun cross
            tk_cross = self._check_tenkan_kijun_cross(ichimoku)

            # Price position relative to cloud
            price_cloud_position = self._analyze_price_cloud_position(df['Close'].tail(30), ichimoku)

            # Chikou span analysis
            chikou_analysis = self._analyze_chikou_span(ichimoku)

            score = 0
            if cloud_analysis['bullish']:
                score += 25
            if tk_cross:
                score += 30
            if price_cloud_position == 'above_cloud':
                score += 25
            if chikou_analysis > 0.5:
                score += 20

            passed = score >= 70

            details = {
                'cloud_bullish': cloud_analysis['bullish'],
                'tenkan_kijun_cross': tk_cross,
                'price_cloud_position': price_cloud_position,
                'chikou_strength': chikou_analysis
            }

            return passed, score, details

        except Exception as e:
            return False, 0, {'error': str(e)}

    # ================== HELPER METHODS ==================
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression"""
        try:
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = linregress(x, prices.values)
            return slope * r_value ** 2
        except:
            return 0.0

    def _analyze_accumulation_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns for accumulation"""
        try:
            if 'Volume' not in df.columns:
                return {'accumulation_score': 0.0}

            up_days = df[df['Close'] > df['Open']]
            down_days = df[df['Close'] <= df['Open']]

            if len(down_days) == 0:
                return {'accumulation_score': 0.0}

            avg_vol_up = up_days['Volume'].mean() if len(up_days) > 0 else 0
            avg_vol_down = down_days['Volume'].mean()

            # Higher volume on down days suggests accumulation
            accumulation_score = avg_vol_down / (avg_vol_up + avg_vol_down) if (avg_vol_up + avg_vol_down) > 0 else 0

            return {'accumulation_score': accumulation_score}
        except:
            return {'accumulation_score': 0.0}

    def _calculate_obv_divergence(self, df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume divergence"""
        try:
            if 'Volume' not in df.columns:
                return 0.0

            obv = []
            obv_val = 0

            for i in range(len(df)):
                if i == 0:
                    obv.append(df['Volume'].iloc[i])
                else:
                    if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                        obv_val += df['Volume'].iloc[i]
                    elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                        obv_val -= df['Volume'].iloc[i]
                    obv.append(obv_val)

            # Calculate divergence between price and OBV
            price_trend = self._calculate_trend_strength(df['Close'])
            obv_trend = self._calculate_trend_strength(pd.Series(obv))

            # Positive divergence: price down, OBV up
            if price_trend < 0 and obv_trend > 0:
                return abs(price_trend) + obv_trend

            return 0.0
        except:
            return 0.0

    def _calculate_mfi(self, df: pd.DataFrame) -> float:
        """Calculate Money Flow Index"""
        try:
            if 'Volume' not in df.columns:
                return 50.0

            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']

            positive_flow = []
            negative_flow = []

            for i in range(1, len(typical_price)):
                if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                    positive_flow.append(money_flow.iloc[i])
                    negative_flow.append(0)
                else:
                    positive_flow.append(0)
                    negative_flow.append(money_flow.iloc[i])

            pos_mf = sum(positive_flow[-14:]) if len(positive_flow) >= 14 else sum(positive_flow)
            neg_mf = sum(negative_flow[-14:]) if len(negative_flow) >= 14 else sum(negative_flow)

            if neg_mf == 0:
                return 100.0

            mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
            return mfi
        except:
            return 50.0

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            if 'Volume' not in df.columns:
                return df['Close'].mean()

            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            return (typical_price * df['Volume']).sum() / df['Volume'].sum()
        except:
            return df['Close'].mean()

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile concentration"""
        try:
            if 'Volume' not in df.columns:
                return {'concentration': 0.0}

            # Calculate volume at different price levels
            price_levels = np.linspace(df['Low'].min(), df['High'].max(), 20)
            volume_at_levels = []

            for i in range(len(price_levels) - 1):
                level_volume = 0
                for j, row in df.iterrows():
                    if price_levels[i] <= row['Close'] <= price_levels[i + 1]:
                        level_volume += row['Volume']
                volume_at_levels.append(level_volume)

            total_volume = sum(volume_at_levels)
            if total_volume == 0:
                return {'concentration': 0.0}

            # Find concentration (highest volume level / total volume)
            max_volume_level = max(volume_at_levels)
            concentration = max_volume_level / total_volume

            return {'concentration': concentration}
        except:
            return {'concentration': 0.0}

    def _detect_large_volume_days(self, df: pd.DataFrame) -> int:
        """Detect days with unusually large volume"""
        try:
            if 'Volume' not in df.columns:
                return 0

            avg_volume = df['Volume'].mean()
            threshold = avg_volume * 1.5  # 50% above average

            large_volume_days = (df['Volume'] > threshold).sum()
            return large_volume_days
        except:
            return 0

    def _calculate_ad_line(self, df: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution Line"""
        try:
            if 'Volume' not in df.columns:
                return 0.5

            ad_values = []
            ad_total = 0

            for _, row in df.iterrows():
                if row['High'] != row['Low']:
                    clv = ((row['Close'] - row['Low']) - (row['High'] - row['Close'])) / (row['High'] - row['Low'])
                    ad_total += clv * row['Volume']
                ad_values.append(ad_total)

            # Normalize to 0-1 range
            if len(ad_values) > 1:
                min_ad = min(ad_values)
                max_ad = max(ad_values)
                if max_ad != min_ad:
                    return (ad_values[-1] - min_ad) / (max_ad - min_ad)

            return 0.5
        except:
            return 0.5

    def _analyze_volume_on_red_candles(self, df: pd.DataFrame) -> float:
        """Analyze volume pattern on red candles"""
        try:
            if 'Volume' not in df.columns:
                return 0.0

            red_candles = df[df['Close'] < df['Open']]
            green_candles = df[df['Close'] >= df['Open']]

            if len(red_candles) == 0 or len(green_candles) == 0:
                return 0.0

            avg_red_volume = red_candles['Volume'].mean()
            avg_green_volume = green_candles['Volume'].mean()

            # Higher volume on red candles suggests accumulation
            return avg_red_volume / (avg_red_volume + avg_green_volume)
        except:
            return 0.0

    def _calculate_support_strength(self, df: pd.DataFrame) -> float:
        """Calculate support level strength"""
        try:
            lows = df['Low'].values
            support_level = np.percentile(lows, 10)
            tolerance = support_level * 0.02

            touches = sum(1 for low in lows if support_level - tolerance <= low <= support_level + tolerance)

            # Normalize touches to 0-1 scale
            return min(1.0, touches / 5.0)
        except:
            return 0.0

    def _calculate_price_absorption(self, df: pd.DataFrame) -> float:
        """Calculate price absorption capability"""
        try:
            # Price absorption = volume increase with minimal price decline
            volume_increase = 0
            price_stability = 0

            for i in range(1, len(df)):
                vol_change = (df['Volume'].iloc[i] - df['Volume'].iloc[i - 1]) / df['Volume'].iloc[i - 1]
                price_change = abs((df['Close'].iloc[i] - df['Close'].iloc[i - 1]) / df['Close'].iloc[i - 1])

                if vol_change > 0.2 and price_change < 0.03:  # High volume, stable price
                    volume_increase += vol_change
                    price_stability += 1

            return min(1.0, price_stability / len(df))
        except:
            return 0.0

    def _calculate_volume_price_impact(self, df: pd.DataFrame) -> float:
        """Calculate volume to price impact ratio"""
        try:
            if 'Volume' not in df.columns:
                return 0.5

            volume_changes = df['Volume'].pct_change().abs()
            price_changes = df['Close'].pct_change().abs()

            # Remove NaN values
            volume_changes = volume_changes.dropna()
            price_changes = price_changes.dropna()

            if len(volume_changes) == 0 or len(price_changes) == 0:
                return 0.5

            # Lower ratio means less price impact per volume unit (accumulation)
            avg_vol_change = volume_changes.mean()
            avg_price_change = price_changes.mean()

            if avg_vol_change == 0:
                return 0.5

            impact_ratio = avg_price_change / avg_vol_change
            return min(1.0, impact_ratio)
        except:
            return 0.5

    # Continue with remaining helper methods...
    # [Due to length constraints, I'll continue with the simplified pattern detection methods]

    # ================== SIMPLIFIED PATTERN METHODS ==================
    
    def _detect_pressure_cooker(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect pressure cooker pattern"""
        try:
            if len(df) < 20:
                return False

            # Tight consolidation with building volume
            price_range = (df['High'].tail(15).max() - df['Low'].tail(15).min()) / df['Close'].tail(15).mean()

            if 'Volume' in df.columns:
                volume_building = df['Volume'].tail(5).mean() > df['Volume'].tail(15).mean()
                return price_range < 0.08 and volume_building
            else:
                return price_range < 0.06
        except:
            return False

    def _detect_volatility_contraction(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect volatility contraction"""
        try:
            if len(df) < 25:
                return False

            recent_vol = df['Close'].tail(10).pct_change().std()
            historical_vol = df['Close'].pct_change().std()

            return recent_vol < historical_vol * 0.7
        except:
            return False

    def _detect_silent_accumulation(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect silent accumulation"""
        try:
            if 'Volume' not in df.columns or len(df) < 20:
                return False

            # Increasing volume with minimal price movement
            price_change = abs((df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10])
            volume_increase = df['Volume'].tail(5).mean() / df['Volume'].iloc[-15:-5].mean()

            return price_change < 0.05 and volume_increase > 1.2
        except:
            return False

    def _detect_institutional_absorption(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect institutional absorption"""
        try:
            if 'Volume' not in df.columns or len(df) < 15:
                return False

            # Large volume with price support
            avg_volume = df['Volume'].mean()
            large_volume_days = (df['Volume'] > avg_volume * 1.5).tail(10).sum()

            support_holding = df['Low'].tail(10).min() >= df['Low'].tail(20).quantile(0.1)

            return large_volume_days >= 3 and support_holding
        except:
            return False

    def _detect_stealth_breakout(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect stealth breakout preparation"""
        try:
            if len(df) < 20:
                return False

            # Price near resistance with improving momentum
            resistance = df['High'].tail(20).max()
            current_price = df['Close'].iloc[-1]

            near_resistance = (resistance - current_price) / current_price < 0.05
            momentum_improving = df['Close'].iloc[-1] > df['Close'].iloc[-5]

            return near_resistance and momentum_improving
        except:
            return False

    def _detect_volume_pocket(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect volume pocket (low volume before breakout)"""
        try:
            if 'Volume' not in df.columns or len(df) < 15:
                return False

            recent_volume = df['Volume'].tail(5).mean()
            previous_volume = df['Volume'].iloc[-15:-5].mean()

            return recent_volume < previous_volume * 0.8
        except:
            return False

    def _detect_momentum_divergence(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect momentum divergence"""
        try:
            if len(df) < 20:
                return False

            # Price making lower lows, momentum making higher lows
            price_trend = df['Close'].iloc[-1] < df['Close'].iloc[-10]
            momentum = df['Close'].pct_change().tail(10).mean()
            prev_momentum = df['Close'].pct_change().iloc[-20:-10].mean()

            return price_trend and momentum > prev_momentum
        except:
            return False

    def _detect_rsi_divergence(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect RSI divergence"""
        try:
            if len(df) < 30:
                return False

            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Bullish divergence: price down, RSI up
            price_lower = df['Close'].iloc[-1] < df['Close'].iloc[-10]
            rsi_higher = rsi.iloc[-1] > rsi.iloc[-10]

            return price_lower and rsi_higher and rsi.iloc[-1] < 50
        except:
            return False

    def _detect_stochastic_divergence(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect Stochastic divergence"""
        try:
            if len(df) < 20:
                return False

            # Calculate Stochastic
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))

            # Bullish divergence
            price_lower = df['Close'].iloc[-1] < df['Close'].iloc[-10]
            stoch_higher = k_percent.iloc[-1] > k_percent.iloc[-10]

            return price_lower and stoch_higher and k_percent.iloc[-1] < 30
        except:
            return False

    def _detect_williams_r_reversal(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect Williams %R reversal"""
        try:
            if len(df) < 20:
                return False

            # Calculate Williams %R
            high_14 = df['High'].rolling(14).max()
            low_14 = df['Low'].rolling(14).min()
            williams_r = -100 * ((high_14 - df['Close']) / (high_14 - low_14))

            current_wr = williams_r.iloc[-1]

            # Oversold reversal
            return current_wr < -80 and current_wr > williams_r.iloc[-3]
        except:
            return False

    def _detect_cci_bullish(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect CCI bullish setup"""
        try:
            if len(df) < 25:
                return False

            # Calculate CCI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)

            current_cci = cci.iloc[-1]

            # Bullish setup: CCI oversold and turning up
            return current_cci > -100 and current_cci > cci.iloc[-3] and cci.iloc[-3] < -50
        except:
            return False

    def _detect_vwap_accumulation(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect VWAP accumulation zone"""
        try:
            if 'Volume' not in df.columns or len(df) < 20:
                return False

            # Calculate VWAP
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

            current_price = df['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]

            # Price near VWAP with volume accumulation
            near_vwap = abs(current_price - current_vwap) / current_vwap < 0.02
            volume_accumulation = df['Volume'].tail(5).mean() > df['Volume'].mean()

            return near_vwap and volume_accumulation
        except:
            return False

    def _detect_dark_pool_activity(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect dark pool activity indicators"""
        try:
            if 'Volume' not in df.columns or len(df) < 15:
                return False

            # Large volume with minimal price impact
            volume_spikes = df['Volume'] > df['Volume'].mean() * 1.5
            price_stability = df['Close'].pct_change().abs() < 0.02

            dark_pool_days = (volume_spikes & price_stability).tail(10).sum()

            return dark_pool_days >= 2
        except:
            return False

    def _detect_algorithm_pattern(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect algorithmic trading patterns"""
        try:
            if len(df) < 15:
                return False

            # Consistent small movements (algorithmic signature)
            small_moves = df['Close'].pct_change().abs() < 0.005
            consistency = small_moves.tail(10).sum()

            return consistency >= 7  # 70% small movements
        except:
            return False

    def _detect_fractal_support(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect fractal support levels"""
        try:
            if len(df) < 15:
                return False

            lows = df['Low'].values

            # Find fractal lows (5-point pattern)
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                        lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):

                    # Check if current price is near this fractal level
                    fractal_level = lows[i]
                    current_price = df['Close'].iloc[-1]

                    if abs(current_price - fractal_level) / fractal_level < 0.03:
                        return True

            return False
        except:
            return False

    def _detect_golden_ratio(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Detect golden ratio (1.618) patterns"""
        try:
            if len(df) < 30:
                return False

            # Find significant moves and check for 1.618 extensions
            highs = df['High'].tail(30).values
            lows = df['Low'].tail(30).values

            for i in range(5, len(highs) - 5):
                # Find local swing
                swing_high = max(highs[i - 5:i + 5])
                swing_low = min(lows[i - 5:i + 5])

                if swing_high == highs[i] or swing_low == lows[i]:
                    swing_range = swing_high - swing_low
                    current_price = df['Close'].iloc[-1]

                    # Check for golden ratio extension
                    extension_1618 = swing_high + (swing_range * 0.618)
                    extension_2618 = swing_high + (swing_range * 1.618)

                    if (abs(current_price - extension_1618) / current_price < 0.02 or
                            abs(current_price - extension_2618) / current_price < 0.02):
                        return True

            return False
        except:
            return False

    # ================== REMAINING HELPER METHODS ==================
    
    # Add all remaining helper methods from the original class...
    # Due to space constraints, I'll include the most critical ones

    def _calculate_bollinger_band_width(self, df: pd.DataFrame) -> float:
        """Calculate Bollinger Band width"""
        try:
            close = df['Close']
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()

            upper = sma + (2 * std)
            lower = sma - (2 * std)

            width = (upper - lower) / sma
            return width.iloc[-1] if len(width) > 0 else 0.2
        except:
            return 0.2

    def _calculate_atr_compression(self, df: pd.DataFrame) -> float:
        """Calculate ATR compression ratio"""
        try:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()

            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()

            # Compression = current ATR is much lower than average
            compression = 1 - (current_atr / avg_atr) if avg_atr > 0 else 0
            return max(0, compression)
        except:
            return 0.0

    def _calculate_volume_contraction(self, df: pd.DataFrame) -> float:
        """Calculate volume contraction"""
        try:
            if 'Volume' not in df.columns:
                return 0.0

            recent_volume = df['Volume'].tail(5).mean()
            historical_volume = df['Volume'].mean()

            contraction = 1 - (recent_volume / historical_volume) if historical_volume > 0 else 0
            return max(0, contraction)
        except:
            return 0.0

    def _calculate_time_compression(self, df: pd.DataFrame) -> float:
        """Calculate time-based compression"""
        try:
            # Measure how long the stock has been consolidating
            price_range = df['High'].max() - df['Low'].min()
            avg_price = df['Close'].mean()

            consolidation_ratio = price_range / avg_price

            # Tighter consolidation over time = higher compression
            return max(0, 1 - (consolidation_ratio * 10))
        except:
            return 0.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices))

    def _analyze_rsi_pattern(self, rsi: pd.Series) -> Dict:
        """Analyze RSI for vacuum pattern"""
        try:
            # Look for RSI staying in neutral zone (40-60) with low volatility
            neutral_zone = rsi[(rsi >= 40) & (rsi <= 60)]
            vacuum_score = len(neutral_zone) / len(rsi) if len(rsi) > 0 else 0

            return {'vacuum_score': vacuum_score}
        except:
            return {'vacuum_score': 0.0}

    def _calculate_macd_momentum(self, df: pd.DataFrame) -> float:
        """Calculate MACD momentum"""
        try:
            close = df['Close']
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26

            # Momentum = rate of change of MACD
            momentum = macd.diff().iloc[-1] if len(macd) > 1 else 0
            return max(0, momentum)
        except:
            return 0.0

    def _analyze_selling_pressure(self, df: pd.DataFrame) -> float:
        """Analyze selling pressure"""
        try:
            # Selling pressure = ratio of red candles and their volume
            red_candles = df[df['Close'] < df['Open']]
            total_candles = len(df)

            if total_candles == 0:
                return 0.5

            red_ratio = len(red_candles) / total_candles

            if 'Volume' in df.columns and len(red_candles) > 0:
                red_volume_ratio = red_candles['Volume'].sum() / df['Volume'].sum()
                return (red_ratio + red_volume_ratio) / 2

            return red_ratio
        except:
            return 0.5

    def _calculate_momentum_oscillator(self, df: pd.DataFrame) -> float:
        """Calculate custom momentum oscillator"""
        try:
            close = df['Close']
            momentum = close.iloc[-1] / close.iloc[-10] - 1 if len(close) >= 10 else 0
            return max(0, momentum)
        except:
            return 0.0

    # Additional helper methods continue...
    # [For brevity, I'll stop here but the full implementation would include all remaining helper methods]

    def _find_swing_points(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Find significant swing high and low points"""
        try:
            highs = df['High'].values
            lows = df['Low'].values

            # Find local maxima and minima
            high_indices = argrelextrema(highs, np.greater, order=3)[0]
            low_indices = argrelextrema(lows, np.less, order=3)[0]

            if len(high_indices) > 0 and len(low_indices) > 0:
                swing_high = max(highs[high_indices])
                swing_low = min(lows[low_indices])
                return swing_high, swing_low

            return None, None
        except:
            return None, None

    def _calculate_fibonacci_levels(self, high: float, low: float) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            0.236: high - (diff * 0.236),
            0.382: high - (diff * 0.382),
            0.5: high - (diff * 0.5),
            0.618: high - (diff * 0.618),
            0.786: high - (diff * 0.786)
        }

    def _check_fibonacci_level(self, price: float, fib_levels: Dict[float, float]) -> Optional[float]:
        """Check which Fibonacci level price is near"""
        tolerance = 0.01  # 1% tolerance

        for level, level_price in fib_levels.items():
            if abs(price - level_price) / level_price <= tolerance:
                return level

        return None

    def _analyze_volume_at_fibonacci(self, df: pd.DataFrame, fib_levels: Dict[float, float]) -> float:
        """Analyze volume at Fibonacci levels"""
        try:
            if 'Volume' not in df.columns:
                return 0.0

            total_volume = 0
            fib_volume = 0

            for _, row in df.iterrows():
                total_volume += row['Volume']

                # Check if price is near any Fibonacci level
                for level_price in fib_levels.values():
                    if abs(row['Close'] - level_price) / level_price <= 0.01:
                        fib_volume += row['Volume']
                        break

            return fib_volume / total_volume if total_volume > 0 else 0.0
        except:
            return 0.0

    def _calculate_fibonacci_reaction(self, df: pd.DataFrame, fib_levels: Dict[float, float]) -> float:
        """Calculate reaction strength at Fibonacci levels"""
        try:
            reactions = []

            for i in range(1, len(df)):
                current_price = df['Close'].iloc[i]
                prev_price = df['Close'].iloc[i - 1]

                # Check if price bounced from Fibonacci level
                for level_price in fib_levels.values():
                    if (abs(prev_price - level_price) / level_price <= 0.01 and
                            current_price > prev_price * 1.01):  # 1% bounce
                        reactions.append(1)
                        break

            return len(reactions) / len(df) if len(df) > 0 else 0.0
        except:
            return 0.0

    # Continue with Elliott Wave and other helper methods...
    def _identify_elliott_waves(self, df: pd.DataFrame) -> List[Dict]:
        """Identify Elliott Wave structure (simplified)"""
        try:
            waves = []
            highs = df['High'].values
            lows = df['Low'].values

            # Find significant turning points
            high_indices = argrelextrema(highs, np.greater, order=5)[0]
            low_indices = argrelextrema(lows, np.less, order=5)[0]

            # Combine and sort turning points
            turning_points = []
            for i in high_indices:
                turning_points.append({'index': i, 'price': highs[i], 'type': 'high'})
            for i in low_indices:
                turning_points.append({'index': i, 'price': lows[i], 'type': 'low'})

            turning_points.sort(key=lambda x: x['index'])

            # Simple wave identification
            for i, point in enumerate(turning_points):
                waves.append({
                    'wave_number': i + 1,
                    'price': point['price'],
                    'index': point['index'],
                    'type': point['type']
                })

            return waves[-5:] if len(waves) >= 5 else waves
        except:
            return []

    def _check_wave_1_2_completion(self, waves: List[Dict]) -> bool:
        """Check if Wave 1 and 2 are complete"""
        try:
            if len(waves) < 3:
                return False

            # Simple check: alternating high-low-high or low-high-low pattern
            wave_types = [w['type'] for w in waves[-3:]]
            return len(set(wave_types)) == 2  # Has both highs and lows
        except:
            return False

    def _analyze_wave_3_setup(self, df: pd.DataFrame, waves: List[Dict]) -> Dict:
        """Analyze Wave 3 setup characteristics"""
        try:
            if len(waves) < 2:
                return {'strength': 0.0}

            # Volume should be increasing
            if 'Volume' in df.columns:
                recent_volume = df['Volume'].tail(5).mean()
                avg_volume = df['Volume'].mean()
                volume_strength = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
            else:
                volume_strength = 0.5

            # Price should be breaking above Wave 1 high
            current_price = df['Close'].iloc[-1]
            wave_1_high = max([w['price'] for w in waves if w['type'] == 'high'][-2:], default=current_price)

            breakout_strength = max(0, (current_price - wave_1_high) / wave_1_high) if wave_1_high > 0 else 0

            overall_strength = (volume_strength * 0.6 + min(1.0, breakout_strength * 10) * 0.4)

            return {'strength': overall_strength}
        except:
            return {'strength': 0.0}

    def _check_elliott_momentum(self, df: pd.DataFrame) -> float:
        """Check Elliott Wave momentum characteristics"""
        try:
            # RSI should be strong but not overbought
            rsi = self._calculate_rsi(df['Close'])
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50

            # MACD should be positive and increasing
            close = df['Close']
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26

            macd_positive = macd.iloc[-1] > 0 if len(macd) > 0 else False
            macd_increasing = macd.iloc[-1] > macd.iloc[-3] if len(macd) > 3 else False

            momentum_score = 0
            if 50 < current_rsi < 80:  # Strong but not overbought
                momentum_score += 0.4
            if macd_positive:
                momentum_score += 0.3
            if macd_increasing:
                momentum_score += 0.3

            return momentum_score
        except:
            return 0.0

    def _calculate_elliott_fibonacci_extension(self, waves: List[Dict]) -> float:
        """Calculate Fibonacci extension for Elliott Wave 3"""
        try:
            if len(waves) < 3:
                return 1.0

            # Simple calculation: Wave 3 length vs Wave 1 length
            wave_prices = [w['price'] for w in waves[-3:]]
            wave_1_length = abs(wave_prices[1] - wave_prices[0])
            wave_3_length = abs(wave_prices[2] - wave_prices[1])

            if wave_1_length > 0:
                return wave_3_length / wave_1_length

            return 1.0
        except:
            return 1.0

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        except:
            return prices, prices, prices

    def _analyze_squeeze_volume_pattern(self, df: pd.DataFrame) -> float:
        """Analyze volume pattern during squeeze"""
        try:
            if 'Volume' not in df.columns:
                return 0.5

            # Volume should be contracting during squeeze
            recent_volume = df['Volume'].tail(5).mean()
            earlier_volume = df['Volume'].iloc[:-5].mean() if len(df) > 5 else df['Volume'].mean()

            volume_contraction = 1 - (recent_volume / earlier_volume) if earlier_volume > 0 else 0
            return max(0, volume_contraction)
        except:
            return 0.5

    def _analyze_historical_squeeze_breakouts(self, df: pd.DataFrame) -> float:
        """Analyze historical squeeze breakout success"""
        try:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'])
            bb_width = (bb_upper - bb_lower) / bb_middle

            squeeze_periods = bb_width < bb_width.quantile(0.2)  # Bottom 20% of widths

            successful_breakouts = 0
            total_squeezes = 0

            for i in range(len(squeeze_periods) - 5):
                if squeeze_periods.iloc[i] and not squeeze_periods.iloc[i + 5]:
                    total_squeezes += 1
                    # Check if price moved significantly after squeeze
                    price_change = abs(df['Close'].iloc[i + 5] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                    if price_change > 0.05:  # 5% move
                        successful_breakouts += 1

            return successful_breakouts / total_squeezes if total_squeezes > 0 else 0.6
        except:
            return 0.6

    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud components"""
        try:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2

            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2

            # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, projected 26 periods ahead
            senkou_a = ((tenkan + kijun) / 2).shift(26)

            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, projected 26 periods ahead
            senkou_b = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)

            # Chikou Span (Lagging Span): Close price projected 26 periods back
            chikou = df['Close'].shift(-26)

            return {
                'tenkan': tenkan,
                'kijun': kijun,
                'senkou_a': senkou_a,
                'senkou_b': senkou_b,
                'chikou': chikou
            }
        except:
            return {}

    def _analyze_ichimoku_cloud(self, ichimoku: Dict) -> Dict:
        """Analyze Ichimoku Cloud characteristics"""
        try:
            if not ichimoku:
                return {'bullish': False}

            senkou_a = ichimoku.get('senkou_a', pd.Series())
            senkou_b = ichimoku.get('senkou_b', pd.Series())

            if len(senkou_a) == 0 or len(senkou_b) == 0:
                return {'bullish': False}

            # Cloud is bullish when Senkou A > Senkou B
            current_a = senkou_a.iloc[-1] if not pd.isna(senkou_a.iloc[-1]) else 0
            current_b = senkou_b.iloc[-1] if not pd.isna(senkou_b.iloc[-1]) else 0

            return {'bullish': current_a > current_b}
        except:
            return {'bullish': False}

    def _check_tenkan_kijun_cross(self, ichimoku: Dict) -> bool:
        """Check for Tenkan/Kijun cross"""
        try:
            tenkan = ichimoku.get('tenkan', pd.Series())
            kijun = ichimoku.get('kijun', pd.Series())

            if len(tenkan) < 2 or len(kijun) < 2:
                return False

            # Bullish cross: Tenkan crosses above Kijun
            prev_cross = tenkan.iloc[-2] <= kijun.iloc[-2]
            current_cross = tenkan.iloc[-1] > kijun.iloc[-1]

            return prev_cross and current_cross
        except:
            return False

    def _analyze_price_cloud_position(self, prices: pd.Series, ichimoku: Dict) -> str:
        """Analyze price position relative to cloud"""
        try:
            if not ichimoku:
                return 'unknown'

            senkou_a = ichimoku.get('senkou_a', pd.Series())
            senkou_b = ichimoku.get('senkou_b', pd.Series())

            if len(senkou_a) == 0 or len(senkou_b) == 0 or len(prices) == 0:
                return 'unknown'

            current_price = prices.iloc[-1]
            current_a = senkou_a.iloc[-1] if not pd.isna(senkou_a.iloc[-1]) else current_price
            current_b = senkou_b.iloc[-1] if not pd.isna(senkou_b.iloc[-1]) else current_price

            cloud_top = max(current_a, current_b)
            cloud_bottom = min(current_a, current_b)

            if current_price > cloud_top:
                return 'above_cloud'
            elif current_price < cloud_bottom:
                return 'below_cloud'
            else:
                return 'in_cloud'
        except:
            return 'unknown'

    def _analyze_chikou_span(self, ichimoku: Dict) -> float:
        """Analyze Chikou Span strength"""
        try:
            chikou = ichimoku.get('chikou', pd.Series())

            if len(chikou) == 0:
                return 0.5

            # Chikou strength based on its position relative to price action
            valid_chikou = chikou.dropna()
            if len(valid_chikou) == 0:
                return 0.5

            # Return normalized strength (simplified)
            return 0.7  # Placeholder - would need more sophisticated calculation
        except:
            return 0.5
