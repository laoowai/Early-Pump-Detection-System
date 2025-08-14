#!/usr/bin/env python3
"""
Pattern Analyzer Game v6.0 - Professional Trading Edition
Enhanced with 20 New Advanced Patterns for Early Pump Detection
Optimized Performance & Structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import logging
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy.stats import linregress, pearsonr
from scipy.signal import argrelextrema, find_peaks
from enum import Enum
import statistics
from tabulate import tabulate
import sys
from collections import defaultdict
import random
import os
import ta
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
import platform  # NEW: For M1 detection

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================== ENHANCED CONFIGURATION ==================
BLACKLISTED_STOCKS = {
    # Chinese stocks (already pumped)
    '002916', '002780', '002594', '002415', '002273',
    '000661', '000725', '000792', '000858', '000903',
    '600519', '600276', '600809', '600887', '600900',
    '603259', '603288', '603501', '603658', '603899',
}

BLACKLISTED_CRYPTO = {
    # Crypto pairs (already pumped or low quality)
    'SFUND_USDT', 'METIS_USDT', 'LUNA_USDT', 'UST_USDT',
    'FTT_USDT', 'CEL_USDT', 'LUNC_USDT', 'USTC_USDT',
}

# Enhanced game messages
GAME_MESSAGES = [
    "🎰 Rolling the dice for patterns...",
    "🎲 Shuffling the deck of stocks...",
    "🎯 Hunting for hidden treasures...",
    "🔮 Crystal ball says...",
    "🎪 Welcome to the Pattern Circus!",
    "🎮 Level up! New patterns unlocked!",
    "💎 Mining for diamond patterns...",
    "🚀 Preparing for moon mission...",
    "🎊 Party time! Patterns everywhere!",
    "🎭 The show must go on...",
    "⚡ Lightning strikes twice...",
    "🔥 Fire in the hole!",
    "💰 Money machine activated...",
    "🌟 Star alignment detected...",
    "🏆 Champion mode engaged..."
]

# Professional pattern combinations
PROFESSIONAL_PATTERNS = {
    "🔥 ACCUMULATION ZONE": ["Hidden Accumulation", "Smart Money Flow", "Institutional Absorption"],
    "💎 BREAKOUT IMMINENT": ["Coiled Spring", "Pressure Cooker", "Volume Pocket"],
    "🚀 ROCKET FUEL": ["Fuel Tank Pattern", "Ignition Sequence", "Momentum Vacuum"],
    "⚡ STEALTH MODE": ["Silent Accumulation", "Whale Whispers", "Dark Pool Activity"],
    "🌟 PERFECT STORM": ["Confluence Zone", "Multiple Timeframe Sync", "Technical Nirvana"],
    "🏆 MASTER SETUP": ["Professional Grade", "Institutional Quality", "Expert Level"],
    "💰 MONEY MAGNET": ["Cash Flow Positive", "Profit Engine", "Revenue Stream"],
    "🎯 PRECISION ENTRY": ["Surgical Strike", "Sniper Entry", "Laser Focus"],
    "🔮 ORACLE VISION": ["Future Sight", "Crystal Clear", "Prophetic Signal"],
    "👑 ROYAL FLUSH": ["Perfect Hand", "Maximum Odds", "Ultimate Setup"],
    "📊 TRADINGVIEW MASTER": ["TradingView Triple Support", "Volume Confirmation", "Multiple Timeframes"],  # NEW!
    "🟦 TRIPLE SUPPORT KING": ["Triple Support Stable", "Support", "MACD"],  # NEW!
}


class MarketType(Enum):
    CHINESE_STOCK = "Chinese A-Share"
    CRYPTO = "Cryptocurrency"
    BOTH = "Both Markets"


class TimeFrame(Enum):
    D1 = 1
    D3 = 3
    D6 = 6
    D11 = 11
    D21 = 21
    D33 = 33
    D55 = 55  # New longer timeframe
    D89 = 89  # Fibonacci timeframe


class PatternType(Enum):
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
    RISING_SUPPORT_COMPRESSION = "Rising Support Compression"
    MACD_SUPPORT_BOUNCE = "MACD Support Bounce"
    MACD_RISING_PRICE_DECLINE = "MACD Rising Price Decline"
    VOLUME_BREAKOUT = "Volume Breakout"
    PRICE_COMPRESSION = "Price Compression"
    SUPPORT_MACD_DIVERGENCE = "Support + MACD Divergence"

    # NEW ADVANCED PATTERNS FOR EARLY PUMP DETECTION
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
    TRADINGVIEW_TRIPLE_SUPPORT = "TradingView Triple Support Stable"  # NEW!

STRONG_PATTERNS = {
    PatternType.HIDDEN_ACCUMULATION,
    PatternType.SMART_MONEY_FLOW,
    PatternType.WHALE_ACCUMULATION,
    PatternType.COILED_SPRING,
    PatternType.PRESSURE_COOKER,
    PatternType.MOMENTUM_VACUUM,
    PatternType.STEALTH_BREAKOUT,
    PatternType.BOLLINGER_SQUEEZE,
    PatternType.FIBONACCI_RETRACEMENT,
    PatternType.ELLIOTT_WAVE_3,
    PatternType.TRADINGVIEW_TRIPLE_SUPPORT,
    PatternType.ICHIMOKU_CLOUD,
}

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
class StageResult:
    stage_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    visual_indicator: str
    top_stocks: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PatternCombination:
    patterns: List[PatternType]
    combined_score: float
    timeframe: TimeFrame
    confidence: float
    historical_success_rate: float
    top_stocks: List[str] = field(default_factory=list)
    risk_level: str = "MEDIUM"
    expected_gain: float = 0.0


@dataclass
class MultiTimeframeAnalysis:
    symbol: str
    market_type: MarketType
    timeframe_patterns: Dict[TimeFrame, List[PatternType]]
    best_combination: PatternCombination
    consolidation_type: ConsolidationType
    maturity: PatternMaturity
    stage_results: List[StageResult]
    overall_score: float
    prediction_confidence: float
    entry_setup: Dict[str, float]
    visual_summary: str
    game_score: float = 0.0
    rarity_level: str = "Common"
    special_pattern: str = ""
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    technical_indicators: Dict[str, float] = field(default_factory=dict)


class EnhancedBlacklistManager:
    """Enhanced blacklist management with dynamic scoring"""

    def __init__(self):
        self.blacklisted_stocks = BLACKLISTED_STOCKS.copy()
        self.blacklisted_crypto = BLACKLISTED_CRYPTO.copy()
        self.performance_tracker = defaultdict(list)
        self.dynamic_blacklist = set()

    def is_blacklisted(self, symbol: str, market_type: MarketType) -> bool:
        """Enhanced blacklist checking"""
        if market_type == MarketType.CRYPTO:
            return symbol.upper() in self.blacklisted_crypto or symbol in self.dynamic_blacklist
        else:
            return symbol in self.blacklisted_stocks or symbol in self.dynamic_blacklist

    def add_performance_data(self, symbol: str, performance: float):
        """Track symbol performance for dynamic blacklisting"""
        self.performance_tracker[symbol].append(performance)

        # Auto-blacklist consistently poor performers
        if len(self.performance_tracker[symbol]) >= 5:
            avg_performance = np.mean(self.performance_tracker[symbol])
            if avg_performance < -0.1:  # Consistent 10% losses
                self.dynamic_blacklist.add(symbol)
                logger.info(f"Auto-blacklisted {symbol} for poor performance")


class ProfessionalLearningSystem:
    """Enhanced learning system with advanced pattern tracking"""

    def __init__(self, results_dir: str = "professional_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.pattern_performance = defaultdict(lambda: {
            "success": 0, "total": 0, "avg_gain": [], "max_gain": 0,
            "win_rate": 0.0, "avg_hold_time": 0, "volatility": 0.0,
            "symbols": [], "timeframes": [], "market_conditions": []
        })

        self.market_regime = {
            "bull": 0.0, "bear": 0.0, "sideways": 0.0, "volatile": 0.0
        }

        self.load_advanced_results()

    def load_advanced_results(self):
        """Load and analyze all previous results with advanced metrics"""
        json_files = list(self.results_dir.glob("*.json"))
        logger.info(f"🧠 Advanced learning from {len(json_files)} sessions")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.analyze_advanced_results(data)
            except Exception as e:
                logger.debug(f"Error loading {json_file}: {e}")

    def analyze_advanced_results(self, data: Dict):
        """Advanced result analysis with market regime detection"""
        if 'pattern_statistics' in data:
            for pattern_name, stats in data['pattern_statistics'].items():
                if 'gains' in stats and stats['gains']:
                    gains = np.array(stats['gains'])
                    successes = sum(1 for g in gains if g > 0)

                    perf = self.pattern_performance[pattern_name]
                    perf['success'] += successes
                    perf['total'] += len(gains)
                    perf['avg_gain'].extend(gains)
                    perf['max_gain'] = max(perf['max_gain'], np.max(gains) if len(gains) > 0 else 0)
                    perf['win_rate'] = perf['success'] / max(1, perf['total'])
                    perf['volatility'] = np.std(gains) if len(gains) > 1 else 0

                    if 'symbols' in stats:
                        perf['symbols'].extend(stats['symbols'])

    def get_advanced_pattern_score(self, pattern: str, is_crypto: bool = False,
                                   market_regime: str = "neutral") -> Tuple[float, Dict]:
        """Get advanced pattern scoring with market regime consideration"""
        base_score = 0.6 if is_crypto else 0.5

        if pattern in self.pattern_performance:
            perf = self.pattern_performance[pattern]
            if perf['total'] >= 3:
                # Advanced scoring algorithm
                win_rate = perf['win_rate']
                avg_gain = np.mean(perf['avg_gain']) if perf['avg_gain'] else 0
                max_gain = perf['max_gain']
                volatility = perf['volatility']

                # Composite score with risk adjustment
                score = (win_rate * 0.4 +
                         min(avg_gain / 100, 0.3) * 0.3 +
                         min(max_gain / 200, 0.2) * 0.2 +
                         max(0, (1 - volatility / 50)) * 0.1)

                # Market regime adjustment
                if market_regime == "bull" and avg_gain > 0:
                    score *= 1.2
                elif market_regime == "bear" and win_rate > 0.6:
                    score *= 1.1

                metadata = {
                    'win_rate': win_rate,
                    'avg_gain': avg_gain,
                    'max_gain': max_gain,
                    'volatility': volatility,
                    'sample_size': perf['total']
                }

                return min(1.0, score), metadata

        return base_score, {'sample_size': 0}


class AdvancedPatternDetector:
    """Advanced pattern detection with 20 new sophisticated patterns"""

    def __init__(self):
        self.scaler = MinMaxScaler()

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
            pivot_ages = []  # How long each level has been stable

            # Find pivot lows across different periods for multiple timeframes
            for period in [20, 30, 45]:  # Different lookback periods
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
                            'age': len(df) - i  # How old this support is
                        })

            # Filter and rank support levels
            if len(support_levels) < 3:
                return False, 0, {'reason': 'Less than 3 support levels found'}

            # Sort by stability and age
            support_levels.sort(key=lambda x: (x['stability'], x['age']), reverse=True)

            # Take top 3 most stable supports
            top_supports = support_levels[:3]

            # Check if these supports are "stable" (haven't changed recently)
            stable_supports = 0
            current_price = df['Close'].iloc[-1]

            for support in top_supports:
                # Check if support level is still valid (not broken)
                level = support['level']
                recent_lows = df['Low'].tail(10)

                # Support is stable if price hasn't significantly broken below it
                if recent_lows.min() >= level * 0.98:  # 2% tolerance
                    stable_supports += 1

                # Extra points if current price is near this support
                if abs(current_price - level) / level < 0.05:
                    stable_supports += 0.5

            # Volume confirmation (TradingView style)
            volume_score = 0
            if 'Volume' in df.columns and len(df) > 10:
                # EMA volume oscillator
                short_vol = df['Volume'].ewm(span=5).mean()
                long_vol = df['Volume'].ewm(span=10).mean()
                vol_osc = 100 * (short_vol - long_vol) / long_vol

                # Recent volume above threshold suggests accumulation
                if vol_osc.iloc[-1] > volume_thresh:
                    volume_score = 30
                elif vol_osc.tail(5).mean() > volume_thresh * 0.7:
                    volume_score = 20
                else:
                    volume_score = 10
            else:
                volume_score = 15  # Default when no volume data

            # Calculate final score
            base_score = stable_supports * 20  # Up to 60 for 3 stable supports
            stability_bonus = sum(s['stability'] for s in top_supports) * 2
            age_bonus = min(20, sum(s['age'] for s in top_supports) / 10)

            total_score = base_score + volume_score + stability_bonus + age_bonus

            # Must have at least 2.5 stable supports to pass
            passed = stable_supports >= 2.5 and total_score >= 70

            # Bonus for higher timeframes (stronger signal)
            timeframe_bonus = 0
            if len(df) > 200:  # Longer history suggests higher timeframe
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

    # Helper methods for pattern detection
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

    # Additional helper methods for remaining patterns...
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

    # Continue with remaining helper methods...
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

            # Simple wave identification (needs more sophisticated logic for real Elliott Wave)
            for i, point in enumerate(turning_points):
                waves.append({
                    'wave_number': i + 1,
                    'price': point['price'],
                    'index': point['index'],
                    'type': point['type']
                })

            return waves[-5:] if len(waves) >= 5 else waves  # Return last 5 waves
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

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
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
            # This is a simplified version - in practice, you'd analyze multiple historical squeezes
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
            # This is a simplified calculation
            valid_chikou = chikou.dropna()
            if len(valid_chikou) == 0:
                return 0.5

            # Return normalized strength (simplified)
            return 0.7  # Placeholder - would need more sophisticated calculation
        except:
            return 0.5


class EnhancedStageAnalyzer:
    """Enhanced stage analyzer with new patterns integration"""

    def __init__(self, blacklist_manager: EnhancedBlacklistManager):
        self.blacklist = blacklist_manager
        self.pattern_detector = AdvancedPatternDetector()
        self.stages = [
            "Smart Money Detection",  # Enhanced
            "Accumulation Analysis",  # New
            "Technical Confluence",  # Enhanced
            "Volume Profiling",  # Enhanced
            "Momentum Analysis",  # Enhanced
            "Pattern Recognition",  # New advanced patterns
            "Risk Assessment",  # Enhanced
            "Entry Optimization",  # Enhanced
            "Breakout Probability",  # New
            "Professional Grade"  # New
        ]
        self.stage_top_stocks = defaultdict(list)

    def run_all_stages(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> List[StageResult]:
        """Run all enhanced stages"""
        results = []

        # Stage 1: Smart Money Detection
        results.append(self.stage_1_smart_money_detection(df, symbol, is_crypto))

        # Stage 2: Accumulation Analysis
        results.append(self.stage_2_accumulation_analysis(df, symbol, is_crypto))

        # Stage 3: Technical Confluence
        results.append(self.stage_3_technical_confluence(df, symbol, is_crypto))

        # Stage 4: Volume Profiling
        results.append(self.stage_4_volume_profiling(df, symbol, is_crypto))

        # Stage 5: Momentum Analysis
        results.append(self.stage_5_momentum_analysis(df, symbol, is_crypto))

        # Stage 6: Advanced Pattern Recognition
        results.append(self.stage_6_pattern_recognition(df, symbol, is_crypto))

        # Stage 7: Risk Assessment
        results.append(self.stage_7_risk_assessment(df, symbol, is_crypto))

        # Stage 8: Entry Optimization
        results.append(self.stage_8_entry_optimization(df, symbol, is_crypto))

        # Stage 9: Breakout Probability
        results.append(self.stage_9_breakout_probability(df, symbol, is_crypto))

        # Stage 10: Professional Grade
        results.append(self.stage_10_professional_grade(df, symbol, is_crypto))

        return results

    def stage_1_smart_money_detection(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Detect smart money activity"""
        try:
            # Use advanced pattern detector
            hidden_acc_passed, hidden_acc_score, hidden_acc_details = self.pattern_detector.detect_hidden_accumulation(
                df, is_crypto)
            smart_money_passed, smart_money_score, smart_money_details = self.pattern_detector.detect_smart_money_flow(
                df, is_crypto)
            whale_passed, whale_score, whale_details = self.pattern_detector.detect_whale_accumulation(df, is_crypto)

            # Combined scoring
            total_score = (hidden_acc_score * 0.4 + smart_money_score * 0.35 + whale_score * 0.25)
            passed = hidden_acc_passed or smart_money_passed or whale_passed

            # Enhanced confidence calculation
            confidence = min(100, total_score * 1.2)

            indicator = "💰" if passed else "❌"

            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Smart Money Detection"].append((symbol, total_score))

            details = {
                'hidden_accumulation': hidden_acc_details,
                'smart_money_flow': smart_money_details,
                'whale_accumulation': whale_details,
                'combined_analysis': {
                    'total_score': total_score,
                    'smart_money_probability': confidence
                }
            }

            return StageResult(
                "Smart Money Detection",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Smart Money Detection", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_2_accumulation_analysis(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced accumulation zone analysis"""
        try:
            # Multiple accumulation detection methods
            coiled_passed, coiled_score, coiled_details = self.pattern_detector.detect_coiled_spring(df, is_crypto)
            vacuum_passed, vacuum_score, vacuum_details = self.pattern_detector.detect_momentum_vacuum(df, is_crypto)
            bb_squeeze_passed, bb_squeeze_score, bb_squeeze_details = self.pattern_detector.detect_bollinger_squeeze(df,
                                                                                                                     is_crypto)

            # Professional accumulation scoring
            accumulation_factors = []

            # Factor 1: Price stability with volume increase
            if 'Volume' in df.columns and len(df) >= 20:
                price_stability = 1 - (df['Close'].tail(20).std() / df['Close'].tail(20).mean())
                volume_trend = df['Volume'].tail(10).mean() / df['Volume'].iloc[-20:-10].mean()

                if price_stability > 0.95 and volume_trend > 1.1:
                    accumulation_factors.append(0.8)
                elif price_stability > 0.9 and volume_trend > 1.05:
                    accumulation_factors.append(0.6)
                else:
                    accumulation_factors.append(0.3)
            else:
                accumulation_factors.append(0.3)

            # Factor 2: Support level holding
            support_strength = self._calculate_advanced_support_strength(df)
            accumulation_factors.append(support_strength)

            # Factor 3: Insider accumulation indicators
            insider_score = self._detect_insider_accumulation(df)
            accumulation_factors.append(insider_score)

            # Combined accumulation score
            base_accumulation = np.mean(accumulation_factors) * 100
            pattern_bonus = (coiled_score + vacuum_score + bb_squeeze_score) / 3

            total_score = base_accumulation * 0.6 + pattern_bonus * 0.4
            passed = total_score >= 65 or any([coiled_passed, vacuum_passed, bb_squeeze_passed])

            confidence = min(100, total_score * 1.1)
            indicator = "🏗️" if passed else "❌"

            if passed and total_score > 75 and symbol:
                self.stage_top_stocks["Accumulation Analysis"].append((symbol, total_score))

            details = {
                'price_stability': accumulation_factors[0] if accumulation_factors else 0,
                'support_strength': support_strength,
                'insider_accumulation': insider_score,
                'coiled_spring': coiled_details,
                'momentum_vacuum': vacuum_details,
                'bollinger_squeeze': bb_squeeze_details,
                'accumulation_probability': confidence
            }

            return StageResult(
                "Accumulation Analysis",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Accumulation Analysis", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_3_technical_confluence(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced technical confluence analysis"""
        try:
            confluence_signals = []

            # Fibonacci analysis
            fib_passed, fib_score, fib_details = self.pattern_detector.detect_fibonacci_retracement(df, is_crypto)
            confluence_signals.append(('fibonacci', fib_score / 100))

            # Elliott Wave analysis
            ew_passed, ew_score, ew_details = self.pattern_detector.detect_elliott_wave_3(df, is_crypto)
            confluence_signals.append(('elliott_wave', ew_score / 100))

            # Ichimoku analysis
            ichimoku_passed, ichimoku_score, ichimoku_details = self.pattern_detector.detect_ichimoku_cloud(df,
                                                                                                            is_crypto)
            confluence_signals.append(('ichimoku', ichimoku_score / 100))

            # Multiple timeframe alignment
            mtf_alignment = self._calculate_mtf_alignment(df)
            confluence_signals.append(('mtf_alignment', mtf_alignment))

            # Support/Resistance confluence
            sr_confluence = self._calculate_sr_confluence(df)
            confluence_signals.append(('sr_confluence', sr_confluence))

            # Calculate confluence strength
            valid_signals = [score for name, score in confluence_signals if score > 0.3]
            confluence_strength = np.mean(valid_signals) if valid_signals else 0

            # Bonus for multiple confirming signals
            strong_signals = len([score for name, score in confluence_signals if score > 0.6])
            confluence_bonus = min(0.3, strong_signals * 0.1)

            total_score = (confluence_strength + confluence_bonus) * 100
            passed = confluence_strength >= 0.6 and strong_signals >= 2

            confidence = min(100, total_score * 1.15)
            indicator = "🎯" if passed else "❌"

            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Technical Confluence"].append((symbol, total_score))

            details = {
                'fibonacci_analysis': fib_details,
                'elliott_wave_analysis': ew_details,
                'ichimoku_analysis': ichimoku_details,
                'mtf_alignment': mtf_alignment,
                'sr_confluence': sr_confluence,
                'confluence_strength': confluence_strength,
                'strong_signals_count': strong_signals,
                'confluence_probability': confidence
            }

            return StageResult(
                "Technical Confluence",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Technical Confluence", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_4_volume_profiling(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced volume profile analysis"""
        try:
            if 'Volume' not in df.columns:
                return StageResult("Volume Profiling", False, 30, {"error": "No volume data"}, "❌", [], 0)

            # Volume profile analysis
            volume_profile = self._calculate_detailed_volume_profile(df)

            # Volume accumulation patterns
            volume_accumulation = self._analyze_volume_accumulation_patterns(df)

            # Institutional volume detection
            institutional_volume = self._detect_institutional_volume_patterns(df)

            # Volume breakout potential
            breakout_potential = self._calculate_volume_breakout_potential(df)

            # Dark pool activity estimation
            dark_pool_score = self._estimate_dark_pool_activity(df)

            # Combined volume score
            volume_scores = [
                volume_profile['strength'],
                volume_accumulation,
                institutional_volume,
                breakout_potential,
                dark_pool_score
            ]

            total_score = np.mean(volume_scores) * 100

            # Enhanced criteria for crypto
            threshold = 60 if is_crypto else 65
            passed = total_score >= threshold and volume_profile['concentration'] > 0.4

            confidence = min(100, total_score * 1.2)
            indicator = "📊" if passed else "❌"

            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Volume Profiling"].append((symbol, total_score))

            details = {
                'volume_profile': volume_profile,
                'volume_accumulation': volume_accumulation,
                'institutional_volume': institutional_volume,
                'breakout_potential': breakout_potential,
                'dark_pool_score': dark_pool_score,
                'volume_strength': total_score / 100,
                'volume_confidence': confidence
            }

            return StageResult(
                "Volume Profiling",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Volume Profiling", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_5_momentum_analysis(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced momentum analysis with multiple oscillators"""
        try:
            momentum_indicators = {}

            # RSI with divergence analysis
            rsi = self._calculate_rsi_with_divergence(df)
            momentum_indicators['rsi'] = rsi

            # Stochastic with pattern recognition
            stochastic = self._calculate_stochastic_with_patterns(df)
            momentum_indicators['stochastic'] = stochastic

            # Williams %R with reversal detection
            williams_r = self._calculate_williams_r_with_reversal(df)
            momentum_indicators['williams_r'] = williams_r

            # CCI with trend analysis
            cci = self._calculate_cci_with_trend(df)
            momentum_indicators['cci'] = cci

            # Custom momentum composite
            momentum_composite = self._calculate_momentum_composite(df)
            momentum_indicators['composite'] = momentum_composite

            # Calculate overall momentum score
            momentum_scores = []
            for indicator, data in momentum_indicators.items():
                if isinstance(data, dict) and 'score' in data:
                    momentum_scores.append(data['score'])
                elif isinstance(data, (int, float)):
                    momentum_scores.append(data)

            if momentum_scores:
                total_score = np.mean(momentum_scores)
            else:
                total_score = 50

            # Enhanced momentum criteria
            bullish_count = sum(1 for indicator, data in momentum_indicators.items()
                                if isinstance(data, dict) and data.get('bullish', False))

            passed = total_score >= 60 and bullish_count >= 3

            confidence = min(100, total_score * 1.1 + bullish_count * 5)
            indicator = "🚀" if passed else "❌"

            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Momentum Analysis"].append((symbol, total_score))

            details = momentum_indicators.copy()
            details.update({
                'overall_momentum_score': total_score,
                'bullish_indicators_count': bullish_count,
                'momentum_confidence': confidence
            })

            return StageResult(
                "Momentum Analysis",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Momentum Analysis", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_6_pattern_recognition(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced pattern recognition using machine learning concepts"""
        try:
            detected_patterns = []
            pattern_scores = []

            # Traditional patterns with enhanced detection
            traditional_patterns = self._detect_enhanced_traditional_patterns(df, is_crypto)
            detected_patterns.extend(traditional_patterns['patterns'])
            pattern_scores.append(traditional_patterns['score'])

            # Fractal patterns
            fractal_patterns = self._detect_fractal_patterns(df)
            if fractal_patterns['detected']:
                detected_patterns.extend(fractal_patterns['patterns'])
                pattern_scores.append(fractal_patterns['score'])

            # Algorithm recognition patterns
            algo_patterns = self._detect_algorithmic_patterns(df)
            if algo_patterns['detected']:
                detected_patterns.extend(algo_patterns['patterns'])
                pattern_scores.append(algo_patterns['score'])

            # Harmonic patterns (simplified)
            harmonic_patterns = self._detect_harmonic_patterns(df)
            if harmonic_patterns['detected']:
                detected_patterns.extend(harmonic_patterns['patterns'])
                pattern_scores.append(harmonic_patterns['score'])

            # Calculate pattern recognition score
            if pattern_scores:
                total_score = np.mean(pattern_scores)
                pattern_strength = len(detected_patterns) * 10
                total_score = min(100, total_score + pattern_strength)
            else:
                total_score = 0

            passed = len(detected_patterns) >= 2 and total_score >= 60

            confidence = min(100, total_score * 1.1 + len(detected_patterns) * 3)
            indicator = "🔍" if passed else "❌"

            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Pattern Recognition"].append((symbol, total_score))

            details = {
                'detected_patterns': detected_patterns,
                'traditional_patterns': traditional_patterns,
                'fractal_patterns': fractal_patterns,
                'algorithmic_patterns': algo_patterns,
                'harmonic_patterns': harmonic_patterns,
                'pattern_count': len(detected_patterns),
                'pattern_strength': total_score,
                'recognition_confidence': confidence
            }

            return StageResult(
                "Pattern Recognition",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Pattern Recognition", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_7_risk_assessment(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Comprehensive risk assessment"""
        try:
            risk_factors = {}

            # Volatility risk
            volatility_risk = self._calculate_volatility_risk(df, is_crypto)
            risk_factors['volatility'] = volatility_risk

            # Liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(df)
            risk_factors['liquidity'] = liquidity_risk

            # Technical risk (support breakdown probability)
            technical_risk = self._calculate_technical_risk(df)
            risk_factors['technical'] = technical_risk

            # Market structure risk
            market_structure_risk = self._calculate_market_structure_risk(df, is_crypto)
            risk_factors['market_structure'] = market_structure_risk

            # Correlation risk (for crypto)
            if is_crypto:
                correlation_risk = self._calculate_correlation_risk(df)
                risk_factors['correlation'] = correlation_risk

            # Calculate overall risk score (lower is better)
            risk_scores = list(risk_factors.values())
            overall_risk = np.mean([r['score'] for r in risk_scores if isinstance(r, dict) and 'score' in r])

            # Risk-adjusted score (higher is better)
            risk_adjusted_score = max(0, 100 - overall_risk * 100)

            # Risk level classification
            if overall_risk < 0.3:
                risk_level = "LOW"
                risk_indicator = "🟢"
            elif overall_risk < 0.6:
                risk_level = "MEDIUM"
                risk_indicator = "🟡"
            else:
                risk_level = "HIGH"
                risk_indicator = "🔴"

            # Pass if risk is acceptable
            passed = overall_risk < 0.7 and risk_adjusted_score >= 40

            confidence = min(100, risk_adjusted_score * 1.2)

            if passed and risk_adjusted_score > 60 and symbol:
                self.stage_top_stocks["Risk Assessment"].append((symbol, risk_adjusted_score))

            details = risk_factors.copy()
            details.update({
                'overall_risk': overall_risk,
                'risk_level': risk_level,
                'risk_adjusted_score': risk_adjusted_score,
                'risk_confidence': confidence
            })

            return StageResult(
                "Risk Assessment",
                passed,
                risk_adjusted_score,
                details,
                risk_indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Risk Assessment", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_8_entry_optimization(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced entry point optimization"""
        try:
            entry_analysis = {}

            # Optimal entry timing
            entry_timing = self._calculate_optimal_entry_timing(df, is_crypto)
            entry_analysis['timing'] = entry_timing

            # Entry zone analysis
            entry_zone = self._calculate_entry_zone(df, is_crypto)
            entry_analysis['zone'] = entry_zone

            # Risk-reward optimization
            risk_reward = self._optimize_risk_reward(df, is_crypto)
            entry_analysis['risk_reward'] = risk_reward

            # Market microstructure analysis
            microstructure = self._analyze_market_microstructure(df)
            entry_analysis['microstructure'] = microstructure

            # Execution probability
            execution_prob = self._calculate_execution_probability(df, is_crypto)
            entry_analysis['execution'] = execution_prob

            # Calculate entry optimization score
            entry_scores = []
            for analysis in entry_analysis.values():
                if isinstance(analysis, dict) and 'score' in analysis:
                    entry_scores.append(analysis['score'])

            if entry_scores:
                total_score = np.mean(entry_scores) * 100
            else:
                total_score = 50

            # Enhanced criteria
            passed = (total_score >= 65 and
                      risk_reward.get('ratio', 0) >= 2.0 and
                      entry_timing.get('optimal', False))

            confidence = min(100, total_score * 1.15)
            indicator = "🎯" if passed else "❌"

            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Entry Optimization"].append((symbol, total_score))

            details = entry_analysis.copy()
            details.update({
                'entry_score': total_score,
                'entry_confidence': confidence,
                'recommendation': 'BUY' if passed else 'WAIT'
            })

            return StageResult(
                "Entry Optimization",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Entry Optimization", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_9_breakout_probability(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Calculate breakout probability using advanced methods"""
        try:
            breakout_factors = {}

            # Historical breakout analysis
            historical_breakouts = self._analyze_historical_breakouts(df)
            breakout_factors['historical'] = historical_breakouts

            # Volume breakout indicators
            volume_breakout = self._analyze_volume_breakout_indicators(df)
            breakout_factors['volume'] = volume_breakout

            # Price action breakout signals
            price_action = self._analyze_price_action_breakout(df)
            breakout_factors['price_action'] = price_action

            # Technical breakout probability
            technical_breakout = self._calculate_technical_breakout_probability(df)
            breakout_factors['technical'] = technical_breakout

            # Time-based breakout analysis
            time_analysis = self._analyze_time_based_breakout_factors(df)
            breakout_factors['time'] = time_analysis

            # Machine learning-inspired breakout prediction
            ml_prediction = self._ml_inspired_breakout_prediction(df, is_crypto)
            breakout_factors['ml_prediction'] = ml_prediction

            # Calculate overall breakout probability
            breakout_scores = []
            for factor in breakout_factors.values():
                if isinstance(factor, dict) and 'probability' in factor:
                    breakout_scores.append(factor['probability'])

            if breakout_scores:
                breakout_probability = np.mean(breakout_scores)
                total_score = breakout_probability * 100
            else:
                breakout_probability = 0.5
                total_score = 50

            # Enhanced breakout criteria
            high_probability_factors = sum(1 for factor in breakout_factors.values()
                                           if isinstance(factor, dict) and factor.get('probability', 0) > 0.7)

            passed = breakout_probability >= 0.65 and high_probability_factors >= 3

            confidence = min(100, total_score * 1.2 + high_probability_factors * 5)
            indicator = "💥" if passed else "❌"

            if passed and total_score > 75 and symbol:
                self.stage_top_stocks["Breakout Probability"].append((symbol, total_score))

            details = breakout_factors.copy()
            details.update({
                'breakout_probability': breakout_probability,
                'breakout_score': total_score,
                'high_probability_factors': high_probability_factors,
                'breakout_confidence': confidence,
                'expected_breakout_timeframe': self._estimate_breakout_timeframe(breakout_factors),
                'breakout_target': self._calculate_breakout_target(df, breakout_probability)
            })

            return StageResult(
                "Breakout Probability",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Breakout Probability", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_10_professional_grade(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Final professional-grade evaluation"""
        try:
            professional_criteria = {}

            # Institutional quality check
            institutional_quality = self._assess_institutional_quality(df, is_crypto)
            professional_criteria['institutional_quality'] = institutional_quality

            # Professional trader grade
            trader_grade = self._calculate_professional_trader_grade(df)
            professional_criteria['trader_grade'] = trader_grade

            # Hedge fund criteria
            hedge_fund_criteria = self._evaluate_hedge_fund_criteria(df, is_crypto)
            professional_criteria['hedge_fund'] = hedge_fund_criteria

            # Risk management grade
            risk_management = self._grade_risk_management_profile(df)
            professional_criteria['risk_management'] = risk_management

            # Execution excellence
            execution_grade = self._grade_execution_excellence(df, is_crypto)
            professional_criteria['execution'] = execution_grade

            # Alpha generation potential
            alpha_potential = self._calculate_alpha_generation_potential(df, is_crypto)
            professional_criteria['alpha_potential'] = alpha_potential

            # Calculate professional grade
            professional_scores = []
            for criteria in professional_criteria.values():
                if isinstance(criteria, dict) and 'grade' in criteria:
                    professional_scores.append(criteria['grade'])
                elif isinstance(criteria, (int, float)):
                    professional_scores.append(criteria)

            if professional_scores:
                professional_grade = np.mean(professional_scores)
                total_score = professional_grade
            else:
                professional_grade = 50
                total_score = 50

            # Professional grade classification
            if professional_grade >= 90:
                grade_level = "INSTITUTIONAL"
                grade_indicator = "👑"
            elif professional_grade >= 80:
                grade_level = "PROFESSIONAL"
                grade_indicator = "🏆"
            elif professional_grade >= 70:
                grade_level = "ADVANCED"
                grade_indicator = "⭐"
            elif professional_grade >= 60:
                grade_level = "INTERMEDIATE"
                grade_indicator = "📈"
            else:
                grade_level = "BASIC"
                grade_indicator = "📊"

            # Pass if meets professional standards
            passed = professional_grade >= 75 and alpha_potential.get('score', 0) >= 70

            confidence = min(100, professional_grade * 1.1)

            if passed and professional_grade > 80 and symbol:
                self.stage_top_stocks["Professional Grade"].append((symbol, professional_grade))

            details = professional_criteria.copy()
            details.update({
                'professional_grade': professional_grade,
                'grade_level': grade_level,
                'professional_confidence': confidence,
                'recommendation': 'STRONG BUY' if professional_grade >= 85 else 'BUY' if passed else 'HOLD'
            })

            return StageResult(
                "Professional Grade",
                passed,
                total_score,
                details,
                grade_indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Professional Grade", False, 0, {"error": str(e)}, "❌", [], 0)

    # Helper methods for enhanced stages
    def _calculate_advanced_support_strength(self, df: pd.DataFrame) -> float:
        """Calculate advanced support strength"""
        try:
            if len(df) < 30:
                return 0.3

            # Multiple support level analysis
            lows = df['Low'].tail(30).values

            # Calculate multiple potential support levels
            support_levels = [
                np.percentile(lows, 5),  # Strong support
                np.percentile(lows, 10),  # Medium support
                np.percentile(lows, 15)  # Weak support
            ]

            support_strength = 0
            for level in support_levels:
                tolerance = level * 0.02
                touches = sum(1 for low in lows if level - tolerance <= low <= level + tolerance)
                strength = min(1.0, touches / 5.0)  # Normalize to max 1.0
                support_strength = max(support_strength, strength)

            return support_strength
        except:
            return 0.3

    def _detect_insider_accumulation(self, df: pd.DataFrame) -> float:
        """Detect potential insider accumulation patterns"""
        try:
            if 'Volume' not in df.columns or len(df) < 20:
                return 0.3

            # Unusual volume patterns
            avg_volume = df['Volume'].mean()
            unusual_days = 0

            for i in range(len(df)):
                volume_ratio = df['Volume'].iloc[i] / avg_volume
                price_change = abs((df['Close'].iloc[i] - df['Open'].iloc[i]) / df['Open'].iloc[i])

                # High volume with minimal price change suggests accumulation
                if volume_ratio > 1.5 and price_change < 0.02:
                    unusual_days += 1

            insider_score = min(1.0, unusual_days / 10.0)
            return insider_score
        except:
            return 0.3

    def _calculate_mtf_alignment(self, df: pd.DataFrame) -> float:
        """Calculate multiple timeframe alignment"""
        try:
            if len(df) < 50:
                return 0.3

            # Simple MTF analysis using different MA periods
            ma_5 = df['Close'].rolling(5).mean()
            ma_20 = df['Close'].rolling(20).mean()
            ma_50 = df['Close'].rolling(50).mean()

            current_price = df['Close'].iloc[-1]

            # Check alignment
            alignment_score = 0
            if current_price > ma_5.iloc[-1]:
                alignment_score += 0.33
            if ma_5.iloc[-1] > ma_20.iloc[-1]:
                alignment_score += 0.33
            if ma_20.iloc[-1] > ma_50.iloc[-1]:
                alignment_score += 0.34

            return alignment_score
        except:
            return 0.3

    def _calculate_sr_confluence(self, df: pd.DataFrame) -> float:
        """Calculate support/resistance confluence"""
        try:
            if len(df) < 40:
                return 0.3

            # Find significant highs and lows
            highs = df['High'].values
            lows = df['Low'].values

            # Use rolling maxima/minima to find levels
            resistance_levels = []
            support_levels = []

            window = 10
            for i in range(window, len(df) - window):
                local_high = df['High'].iloc[i - window:i + window].max()
                local_low = df['Low'].iloc[i - window:i + window].min()

                if df['High'].iloc[i] == local_high:
                    resistance_levels.append(local_high)
                if df['Low'].iloc[i] == local_low:
                    support_levels.append(local_low)

            # Calculate confluence based on level clustering
            current_price = df['Close'].iloc[-1]
            confluence_score = 0

            # Check proximity to support/resistance levels
            for level in resistance_levels + support_levels:
                distance = abs(current_price - level) / current_price
                if distance < 0.03:  # Within 3%
                    confluence_score += 1

            return min(1.0, confluence_score / 5.0)
        except:
            return 0.3

    def _calculate_detailed_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed volume profile"""
        try:
            if 'Volume' not in df.columns:
                return {'strength': 0.3, 'concentration': 0.0}

            # Create price levels
            price_min = df['Low'].min()
            price_max = df['High'].max()
            price_levels = np.linspace(price_min, price_max, 20)

            volume_at_levels = []

            for i in range(len(price_levels) - 1):
                level_volume = 0
                for _, row in df.iterrows():
                    if price_levels[i] <= row['Close'] <= price_levels[i + 1]:
                        level_volume += row['Volume']
                volume_at_levels.append(level_volume)

            total_volume = sum(volume_at_levels)
            if total_volume == 0:
                return {'strength': 0.3, 'concentration': 0.0}

            # Find Point of Control (highest volume level)
            max_volume_index = np.argmax(volume_at_levels)
            poc_volume = volume_at_levels[max_volume_index]

            concentration = poc_volume / total_volume
            strength = min(1.0, concentration * 3)  # Amplify for scoring

            return {
                'strength': strength,
                'concentration': concentration,
                'poc_level': price_levels[max_volume_index],
                'volume_distribution': volume_at_levels
            }
        except:
            return {'strength': 0.3, 'concentration': 0.0}

    def _analyze_volume_accumulation_patterns(self, df: pd.DataFrame) -> float:
        """Analyze volume accumulation patterns"""
        try:
            if 'Volume' not in df.columns or len(df) < 15:
                return 0.3

            accumulation_score = 0

            # Check for increasing volume on down days
            down_days = df[df['Close'] < df['Open']]
            up_days = df[df['Close'] >= df['Open']]

            if len(down_days) > 0 and len(up_days) > 0:
                avg_down_volume = down_days['Volume'].mean()
                avg_up_volume = up_days['Volume'].mean()

                if avg_down_volume > avg_up_volume:
                    accumulation_score += 0.4

            # Check for volume trend
            volume_trend = df['Volume'].tail(10).mean() / df['Volume'].iloc[-20:-10].mean()
            if volume_trend > 1.1:
                accumulation_score += 0.3

            # OBV analysis
            obv = self._calculate_simple_obv(df)
            if len(obv) > 10:
                obv_trend = obv[-1] > obv[-10]
                if obv_trend:
                    accumulation_score += 0.3

            return min(1.0, accumulation_score)
        except:
            return 0.3

    def _calculate_simple_obv(self, df: pd.DataFrame) -> List[float]:
        """Calculate simple On-Balance Volume"""
        try:
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            return obv
        except:
            return [0]

    def _detect_institutional_volume_patterns(self, df: pd.DataFrame) -> float:
        """Detect institutional volume patterns"""
        try:
            if 'Volume' not in df.columns:
                return 0.3

            # Look for block trading patterns
            avg_volume = df['Volume'].mean()
            std_volume = df['Volume'].std()

            # Institutional blocks (volume > 2 std above mean)
            institutional_threshold = avg_volume + (2 * std_volume)
            institutional_days = (df['Volume'] > institutional_threshold).sum()

            # Score based on frequency of institutional activity
            institutional_score = min(1.0, institutional_days / len(df) * 10)

            return institutional_score
        except:
            return 0.3

    def _calculate_volume_breakout_potential(self, df: pd.DataFrame) -> float:
        """Calculate volume breakout potential"""
        try:
            if 'Volume' not in df.columns or len(df) < 20:
                return 0.3

            # Volume contraction followed by expansion
            recent_volume = df['Volume'].tail(5).mean()
            previous_volume = df['Volume'].iloc[-15:-5].mean()

            # Volume pocket (low volume before breakout)
            volume_pocket_score = 0
            if recent_volume < previous_volume * 0.8:
                volume_pocket_score = 0.4

            # Volume spike potential
            max_recent_volume = df['Volume'].tail(10).max()
            avg_volume = df['Volume'].mean()

            spike_potential = min(1.0, max_recent_volume / avg_volume / 3)

            breakout_potential = volume_pocket_score + spike_potential * 0.6
            return min(1.0, breakout_potential)
        except:
            return 0.3

    def _estimate_dark_pool_activity(self, df: pd.DataFrame) -> float:
        """Estimate dark pool activity (simplified heuristic)"""
        try:
            if 'Volume' not in df.columns or len(df) < 10:
                return 0.3

            # Look for price stability with volume increase
            price_volatility = df['Close'].tail(10).std() / df['Close'].tail(10).mean()
            volume_increase = df['Volume'].tail(5).mean() / df['Volume'].mean()

            # Dark pool signature: low volatility + high volume
            if price_volatility < 0.02 and volume_increase > 1.2:
                return 0.8
            elif price_volatility < 0.03 and volume_increase > 1.1:
                return 0.6
            else:
                return 0.3
        except:
            return 0.3

    def _calculate_rsi_with_divergence(self, df: pd.DataFrame) -> Dict:
        """Calculate RSI with divergence analysis"""
        try:
            if len(df) < 30:
                return {'score': 50, 'bullish': False, 'divergence': False}

            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]

            # Check for bullish divergence
            price_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10]
            rsi_trend = rsi.iloc[-1] - rsi.iloc[-10]

            bullish_divergence = price_trend < 0 and rsi_trend > 0

            # RSI scoring
            if 30 <= current_rsi <= 50:
                rsi_score = 80
            elif 50 < current_rsi <= 70:
                rsi_score = 60
            else:
                rsi_score = 30

            if bullish_divergence:
                rsi_score += 20

            return {
                'score': min(100, rsi_score),
                'value': current_rsi,
                'bullish': current_rsi > 50 and not current_rsi > 70,
                'divergence': bullish_divergence,
                'oversold': current_rsi < 30
            }
        except:
            return {'score': 50, 'bullish': False, 'divergence': False}

    def _calculate_stochastic_with_patterns(self, df: pd.DataFrame) -> Dict:
        """Calculate Stochastic with pattern recognition"""
        try:
            if len(df) < 20:
                return {'score': 50, 'bullish': False}

            # Calculate Stochastic
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(3).mean()

            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]

            # Pattern recognition
            bullish_cross = k_percent.iloc[-2] <= d_percent.iloc[-2] and k_percent.iloc[-1] > d_percent.iloc[-1]
            oversold_bounce = current_k < 20 and current_k > k_percent.iloc[-3]

            # Scoring
            stoch_score = 50
            if bullish_cross:
                stoch_score += 25
            if oversold_bounce:
                stoch_score += 25
            if 20 < current_k < 80:
                stoch_score += 10

            return {
                'score': min(100, stoch_score),
                'k_value': current_k,
                'd_value': current_d,
                'bullish': current_k > current_d and current_k > 20,
                'bullish_cross': bullish_cross,
                'oversold_bounce': oversold_bounce
            }
        except:
            return {'score': 50, 'bullish': False}

    def _calculate_williams_r_with_reversal(self, df: pd.DataFrame) -> Dict:
        """Calculate Williams %R with reversal detection"""
        try:
            if len(df) < 20:
                return {'score': 50, 'bullish': False}

            # Calculate Williams %R
            high_14 = df['High'].rolling(14).max()
            low_14 = df['Low'].rolling(14).min()
            williams_r = -100 * ((high_14 - df['Close']) / (high_14 - low_14))

            current_wr = williams_r.iloc[-1]

            # Reversal detection
            oversold_reversal = current_wr < -80 and current_wr > williams_r.iloc[-3]
            bullish_momentum = current_wr > -50

            # Scoring
            wr_score = 50
            if oversold_reversal:
                wr_score += 30
            if bullish_momentum:
                wr_score += 20
            if -80 < current_wr < -20:
                wr_score += 10

            return {
                'score': min(100, wr_score),
                'value': current_wr,
                'bullish': current_wr > -50,
                'oversold_reversal': oversold_reversal,
                'oversold': current_wr < -80
            }
        except:
            return {'score': 50, 'bullish': False}

    def _calculate_cci_with_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate CCI with trend analysis"""
        try:
            if len(df) < 25:
                return {'score': 50, 'bullish': False}

            # Calculate CCI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)

            current_cci = cci.iloc[-1]

            # Trend analysis
            cci_trend = current_cci - cci.iloc[-5] if len(cci) > 5 else 0
            bullish_trend = cci_trend > 0

            # Scoring
            cci_score = 50
            if -100 < current_cci < 100:
                cci_score += 20
            if bullish_trend:
                cci_score += 20
            if current_cci > 0:
                cci_score += 10

            return {
                'score': min(100, cci_score),
                'value': current_cci,
                'bullish': current_cci > 0 and bullish_trend,
                'trend': 'bullish' if bullish_trend else 'bearish',
                'oversold': current_cci < -100
            }
        except:
            return {'score': 50, 'bullish': False}

    def _calculate_momentum_composite(self, df: pd.DataFrame) -> Dict:
        """Calculate composite momentum indicator"""
        try:
            if len(df) < 30:
                return {'score': 50, 'bullish': False}

            # Multiple momentum calculations
            momentum_5 = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
            momentum_10 = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
            momentum_20 = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100

            # Rate of change
            roc = df['Close'].pct_change(10).iloc[-1] * 100

            # Composite scoring
            momentum_scores = []

            if momentum_5 > 0:
                momentum_scores.append(60 + min(40, momentum_5 * 2))
            else:
                momentum_scores.append(40 + max(-40, momentum_5 * 2))

            if momentum_10 > 0:
                momentum_scores.append(60 + min(40, momentum_10))
            else:
                momentum_scores.append(40 + max(-40, momentum_10))

            composite_score = np.mean(momentum_scores)

            return {
                'score': min(100, max(0, composite_score)),
                'momentum_5d': momentum_5,
                'momentum_10d': momentum_10,
                'momentum_20d': momentum_20,
                'roc': roc,
                'bullish': composite_score > 60
            }
        except:
            return {'score': 50, 'bullish': False}

    # Continue with remaining helper methods...
    def _detect_enhanced_traditional_patterns(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Detect traditional patterns with enhanced algorithms"""
        try:
            patterns = []
            total_score = 0

            # Enhanced triangle detection
            triangle_result = self._detect_enhanced_triangles(df)
            if triangle_result['detected']:
                patterns.extend(triangle_result['patterns'])
                total_score += triangle_result['score']

            # Enhanced flag and pennant detection
            flag_result = self._detect_enhanced_flags(df)
            if flag_result['detected']:
                patterns.extend(flag_result['patterns'])
                total_score += flag_result['score']

            # Enhanced cup and handle detection
            cup_result = self._detect_enhanced_cup_handle(df)
            if cup_result['detected']:
                patterns.extend(cup_result['patterns'])
                total_score += cup_result['score']

            return {
                'patterns': patterns,
                'score': min(100, total_score / max(1, len(patterns))),
                'pattern_count': len(patterns)
            }
        except:
            return {'patterns': [], 'score': 0, 'pattern_count': 0}

    def _detect_enhanced_triangles(self, df: pd.DataFrame) -> Dict:
        """Enhanced triangle pattern detection"""
        try:
            if len(df) < 30:
                return {'detected': False, 'patterns': [], 'score': 0}

            patterns = []
            scores = []

            # Get recent data for pattern analysis
            recent_data = df.tail(30)

            # Calculate trendlines for highs and lows
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            x = np.arange(len(highs))

            try:
                # High trendline
                high_slope, high_intercept, high_r, _, _ = linregress(x, highs)
                # Low trendline
                low_slope, low_intercept, low_r, _, _ = linregress(x, lows)

                # Classify triangle type
                if abs(high_slope) < 0.01 and low_slope > 0.01:  # Ascending triangle
                    patterns.append('Ascending Triangle')
                    scores.append(70 + abs(low_r) * 30)
                elif high_slope < -0.01 and abs(low_slope) < 0.01:  # Descending triangle
                    patterns.append('Descending Triangle')
                    scores.append(60 + abs(high_r) * 30)
                elif high_slope < -0.01 and low_slope > 0.01:  # Symmetrical triangle
                    patterns.append('Symmetrical Triangle')
                    scores.append(65 + (abs(high_r) + abs(low_r)) * 15)

            except:
                pass

            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': max(scores) if scores else 0
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_enhanced_flags(self, df: pd.DataFrame) -> Dict:
        """Enhanced flag and pennant detection"""
        try:
            if len(df) < 25:
                return {'detected': False, 'patterns': [], 'score': 0}

            patterns = []
            scores = []

            # Look for flagpole and flag components
            recent_data = df.tail(25)

            # Check for strong move (flagpole)
            flagpole_move = (recent_data['Close'].iloc[10] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[
                0]

            if abs(flagpole_move) > 0.05:  # 5% move for flagpole
                # Analyze consolidation period (flag)
                flag_data = recent_data.iloc[10:]
                flag_range = (flag_data['High'].max() - flag_data['Low'].min()) / flag_data['Close'].mean()

                if flag_range < 0.08:  # Tight consolidation
                    if flagpole_move > 0:
                        patterns.append('Bull Flag')
                        scores.append(75)
                    else:
                        patterns.append('Bear Flag')
                        scores.append(65)

            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': max(scores) if scores else 0
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_enhanced_cup_handle(self, df: pd.DataFrame) -> Dict:
        """Enhanced cup and handle detection"""
        try:
            if len(df) < 50:
                return {'detected': False, 'patterns': [], 'score': 0}

            patterns = []
            scores = []

            # Cup and handle requires longer timeframe
            cup_data = df.tail(50)

            # Find potential cup formation
            cup_low = cup_data['Low'].min()
            cup_low_idx = cup_data['Low'].idxmin()

            # Check cup depth and shape
            left_high = cup_data['High'].iloc[:10].max()
            right_high = cup_data['High'].iloc[-10:].max()

            cup_depth = (left_high - cup_low) / left_high

            if 0.12 < cup_depth < 0.5:  # Reasonable cup depth
                # Look for handle formation
                handle_data = cup_data.iloc[-15:]
                handle_low = handle_data['Low'].min()

                # Handle should be shallow retracement
                handle_depth = (right_high - handle_low) / right_high

                if handle_depth < cup_depth * 0.5:  # Handle < 50% of cup depth
                    patterns.append('Cup and Handle')
                    scores.append(80)

            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': max(scores) if scores else 0
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_fractal_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect fractal patterns"""
        try:
            if len(df) < 20:
                return {'detected': False, 'patterns': [], 'score': 0}

            patterns = []
            score = 0

            # Simple fractal detection (5-point pattern)
            highs = df['High'].values
            lows = df['Low'].values

            # Bullish fractals (support)
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                        lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
                    patterns.append('Bullish Fractal')
                    score += 20

            # Bearish fractals (resistance)
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                        highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
                    patterns.append('Bearish Fractal')
                    score += 15

            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': min(100, score)
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_algorithmic_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect algorithmic trading patterns"""
        try:
            if len(df) < 20 or 'Volume' not in df.columns:
                return {'detected': False, 'patterns': [], 'score': 0}

            patterns = []
            score = 0

            # Look for algorithmic signatures

            # 1. Consistent volume at specific times
            volume_consistency = df['Volume'].std() / df['Volume'].mean()
            if volume_consistency < 0.5:  # Very consistent volume
                patterns.append('Algorithm Volume Pattern')
                score += 25

            # 2. Price clustering at round numbers
            closes = df['Close'].values
            round_number_hits = 0
            for price in closes[-10:]:
                if price % 0.1 < 0.02 or price % 0.1 > 0.98:  # Near round numbers
                    round_number_hits += 1

            if round_number_hits >= 5:
                patterns.append('Price Clustering Algorithm')
                score += 20

            # 3. Micro-movements pattern
            small_moves = sum(1 for i in range(1, len(df))
                              if abs(df['Close'].iloc[i] - df['Close'].iloc[i - 1]) / df['Close'].iloc[i - 1] < 0.005)

            if small_moves / len(df) > 0.7:
                patterns.append('Micro-Movement Algorithm')
                score += 15

            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': min(100, score)
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_harmonic_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect harmonic patterns (simplified)"""
        try:
            if len(df) < 30:
                return {'detected': False, 'patterns': [], 'score': 0}

            patterns = []
            score = 0

            # Simplified Gartley pattern detection
            highs = df['High'].values
            lows = df['Low'].values

            # Find significant turning points
            turning_points = []

            for i in range(5, len(df) - 5):
                if highs[i] == max(highs[i - 5:i + 5]):
                    turning_points.append(('high', i, highs[i]))
                elif lows[i] == min(lows[i - 5:i + 5]):
                    turning_points.append(('low', i, lows[i]))

            # Look for ABCD pattern
            if len(turning_points) >= 4:
                # Simple ABCD ratio check
                recent_points = turning_points[-4:]

                if len(set([p[0] for p in recent_points])) == 2:  # Alternating highs/lows
                    patterns.append('ABCD Pattern')
                    score += 30

            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': min(100, score)
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    # Risk assessment helper methods
    def _calculate_volatility_risk(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate volatility risk - IMPROVED THRESHOLDS"""
        try:
            if len(df) < 20:
                return {'score': 0.3, 'level': 'MEDIUM'}  # More lenient default

            # Calculate different volatility measures
            returns = df['Close'].pct_change().dropna()

            # Historical volatility
            hist_vol = returns.std() * np.sqrt(252)  # Annualized

            # Recent volatility
            recent_vol = returns.tail(10).std() * np.sqrt(252)

            # Volatility of volatility
            vol_of_vol = returns.rolling(10).std().std()

            # IMPROVED: More lenient thresholds
            if is_crypto:
                high_vol_threshold = 1.5  # Increased from 1.0
                medium_vol_threshold = 0.8  # Increased from 0.5
            else:
                high_vol_threshold = 0.6  # Increased from 0.4
                medium_vol_threshold = 0.35  # Increased from 0.2

            # Risk scoring (0 = low risk, 1 = high risk)
            if hist_vol > high_vol_threshold:
                vol_risk = 0.7  # Reduced from 0.8
                level = 'HIGH'
            elif hist_vol > medium_vol_threshold:
                vol_risk = 0.4  # Reduced from 0.5
                level = 'MEDIUM'
            else:
                vol_risk = 0.2
                level = 'LOW'

            return {
                'score': vol_risk,
                'level': level,
                'historical_volatility': hist_vol,
                'recent_volatility': recent_vol,
                'volatility_of_volatility': vol_of_vol
            }
        except:
            return {'score': 0.3, 'level': 'MEDIUM'}

    def _calculate_liquidity_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate liquidity risk"""
        try:
            if 'Volume' not in df.columns or len(df) < 20:
                return {'score': 0.5, 'level': 'MEDIUM'}

            # Volume-based liquidity measures
            avg_volume = df['Volume'].mean()
            volume_std = df['Volume'].std()

            # Bid-ask spread estimation (using high-low as proxy)
            spread_proxy = ((df['High'] - df['Low']) / df['Close']).mean()

            # Volume consistency
            volume_consistency = 1 - (volume_std / avg_volume) if avg_volume > 0 else 0

            # Liquidity risk scoring
            liquidity_score = 0

            if avg_volume < df['Volume'].quantile(0.25):  # Low volume
                liquidity_score += 0.4
            if spread_proxy > 0.05:  # High spread
                liquidity_score += 0.3
            if volume_consistency < 0.5:  # Inconsistent volume
                liquidity_score += 0.3

            if liquidity_score > 0.7:
                level = 'HIGH'
            elif liquidity_score > 0.4:
                level = 'MEDIUM'
            else:
                level = 'LOW'

            return {
                'score': liquidity_score,
                'level': level,
                'average_volume': avg_volume,
                'spread_proxy': spread_proxy,
                'volume_consistency': volume_consistency
            }
        except:
            return {'score': 0.5, 'level': 'MEDIUM'}

    def _calculate_technical_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate technical breakdown risk"""
        try:
            if len(df) < 30:
                return {'score': 0.5, 'level': 'MEDIUM'}

            risk_factors = []

            # Support level risk
            support_level = df['Low'].tail(20).min()
            current_price = df['Close'].iloc[-1]
            distance_to_support = (current_price - support_level) / current_price

            if distance_to_support < 0.05:  # Very close to support
                risk_factors.append(0.6)
            elif distance_to_support < 0.1:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)

            # Trend deterioration risk
            ma_20 = df['Close'].rolling(20).mean()
            trend_strength = (df['Close'].iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1]

            if trend_strength < -0.05:
                risk_factors.append(0.7)
            elif trend_strength < 0:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)

            # Volume confirmation risk
            if 'Volume' in df.columns:
                volume_trend = df['Volume'].tail(5).mean() / df['Volume'].mean()
                if volume_trend < 0.8:  # Declining volume
                    risk_factors.append(0.4)
                else:
                    risk_factors.append(0.1)
            else:
                risk_factors.append(0.3)

            technical_risk = np.mean(risk_factors)

            if technical_risk > 0.6:
                level = 'HIGH'
            elif technical_risk > 0.3:
                level = 'MEDIUM'
            else:
                level = 'LOW'

            return {
                'score': technical_risk,
                'level': level,
                'support_distance': distance_to_support,
                'trend_strength': trend_strength
            }
        except:
            return {'score': 0.5, 'level': 'MEDIUM'}

    def _calculate_market_structure_risk(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate market structure risk"""
        try:
            if len(df) < 50:
                return {'score': 0.5, 'level': 'MEDIUM'}

            # Market structure analysis
            structure_risks = []

            # Higher high, higher low pattern
            recent_highs = df['High'].tail(30)
            recent_lows = df['Low'].tail(30)

            uptrend_intact = (recent_highs.iloc[-1] >= recent_highs.iloc[-10] and
                              recent_lows.iloc[-1] >= recent_lows.iloc[-10])

            if not uptrend_intact:
                structure_risks.append(0.5)
            else:
                structure_risks.append(0.1)

            # Market regime detection
            volatility_regime = df['Close'].pct_change().tail(20).std()
            historical_vol = df['Close'].pct_change().std()

            if volatility_regime > historical_vol * 1.5:  # High vol regime
                structure_risks.append(0.6)
            else:
                structure_risks.append(0.2)

            # Correlation breakdown (simplified)
            if is_crypto:
                # Crypto markets can be more correlated
                structure_risks.append(0.4)
            else:
                structure_risks.append(0.2)

            market_structure_risk = np.mean(structure_risks)

            if market_structure_risk > 0.6:
                level = 'HIGH'
            elif market_structure_risk > 0.3:
                level = 'MEDIUM'
            else:
                level = 'LOW'

            return {
                'score': market_structure_risk,
                'level': level,
                'uptrend_intact': uptrend_intact,
                'volatility_regime': 'HIGH' if volatility_regime > historical_vol * 1.5 else 'NORMAL'
            }
        except:
            return {'score': 0.5, 'level': 'MEDIUM'}

    def _calculate_correlation_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate correlation risk for crypto"""
        try:
            # Simplified correlation risk for crypto
            # In practice, you'd correlate with Bitcoin or market indices

            # Use price momentum as proxy for correlation risk
            momentum = df['Close'].pct_change().tail(10).mean()

            if momentum < -0.02:  # Negative momentum
                corr_risk = 0.7
                level = 'HIGH'
            elif momentum < 0:
                corr_risk = 0.4
                level = 'MEDIUM'
            else:
                corr_risk = 0.2
                level = 'LOW'

            return {
                'score': corr_risk,
                'level': level,
                'momentum_proxy': momentum
            }
        except:
            return {'score': 0.5, 'level': 'MEDIUM'}

    # Entry optimization helper methods
    def _calculate_optimal_entry_timing(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate optimal entry timing"""
        try:
            if len(df) < 20:
                return {'score': 0.5, 'optimal': False}

            timing_factors = []

            # RSI timing
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]

                if 30 <= current_rsi <= 50:  # Ideal entry zone
                    timing_factors.append(0.8)
                elif 50 < current_rsi <= 60:
                    timing_factors.append(0.6)
                else:
                    timing_factors.append(0.3)
            else:
                timing_factors.append(0.5)

            # Support proximity timing
            support_level = df['Low'].tail(20).min()
            current_price = df['Close'].iloc[-1]
            support_distance = (current_price - support_level) / current_price

            if support_distance < 0.03:  # Very close to support
                timing_factors.append(0.9)
            elif support_distance < 0.05:
                timing_factors.append(0.7)
            else:
                timing_factors.append(0.4)

            # Volume timing
            if 'Volume' in df.columns:
                recent_volume = df['Volume'].tail(3).mean()
                avg_volume = df['Volume'].mean()
                volume_factor = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
                timing_factors.append(volume_factor)
            else:
                timing_factors.append(0.5)

            timing_score = np.mean(timing_factors)
            optimal = timing_score >= 0.7

            return {
                'score': timing_score,
                'optimal': optimal,
                'rsi_timing': timing_factors[0] if timing_factors else 0.5,
                'support_timing': timing_factors[1] if len(timing_factors) > 1 else 0.5,
                'volume_timing': timing_factors[2] if len(timing_factors) > 2 else 0.5
            }
        except:
            return {'score': 0.5, 'optimal': False}

    def _calculate_entry_zone(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate optimal entry zone"""
        try:
            current_price = df['Close'].iloc[-1]

            # Support levels
            support_1 = df['Low'].tail(20).min()
            support_2 = df['Low'].tail(50).min() if len(df) >= 50 else support_1

            # Entry zone calculation
            entry_zone_low = support_1 * 1.01  # 1% above support
            entry_zone_high = current_price * 0.99  # 1% below current

            zone_width = (entry_zone_high - entry_zone_low) / current_price

            # Zone quality assessment
            if zone_width > 0.05:  # Wide zone
                zone_quality = 0.4
            elif zone_width > 0.02:  # Medium zone
                zone_quality = 0.7
            else:  # Tight zone
                zone_quality = 0.9

            in_zone = entry_zone_low <= current_price <= entry_zone_high

            return {
                'score': zone_quality,
                'entry_low': entry_zone_low,
                'entry_high': entry_zone_high,
                'zone_width': zone_width,
                'in_entry_zone': in_zone,
                'zone_quality': 'EXCELLENT' if zone_quality > 0.8 else 'GOOD' if zone_quality > 0.6 else 'FAIR'
            }
        except:
            return {'score': 0.5, 'zone_quality': 'UNKNOWN'}

    def _optimize_risk_reward(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Optimize risk-reward ratio"""
        try:
            current_price = df['Close'].iloc[-1]

            # Support level for stop loss
            support_level = df['Low'].tail(20).min()
            stop_loss = support_level * 0.98  # 2% below support

            # Resistance levels for targets
            resistance_1 = df['High'].tail(20).max()
            resistance_2 = df['High'].tail(50).max() if len(df) >= 50 else resistance_1 * 1.1

            # Calculate risk and rewards
            risk = current_price - stop_loss
            reward_1 = resistance_1 - current_price
            reward_2 = resistance_2 - current_price

            # Risk-reward ratios
            rr_ratio_1 = reward_1 / risk if risk > 0 else 0
            rr_ratio_2 = reward_2 / risk if risk > 0 else 0

            # Optimization score
            if rr_ratio_1 >= 3.0:
                rr_score = 1.0
            elif rr_ratio_1 >= 2.0:
                rr_score = 0.8
            elif rr_ratio_1 >= 1.5:
                rr_score = 0.6
            else:
                rr_score = 0.3

            return {
                'score': rr_score,
                'ratio': rr_ratio_1,
                'ratio_target_2': rr_ratio_2,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target_1': resistance_1,
                'target_2': resistance_2,
                'risk_amount': risk,
                'reward_1': reward_1,
                'reward_2': reward_2
            }
        except:
            return {'score': 0.5, 'ratio': 1.0}

    def _analyze_market_microstructure(self, df: pd.DataFrame) -> Dict:
        """Analyze market microstructure"""
        try:
            if len(df) < 10:
                return {'score': 0.5, 'structure': 'UNKNOWN'}

            # Bid-ask spread proxy
            spread_proxy = ((df['High'] - df['Low']) / df['Close']).tail(10).mean()

            # Price impact estimation
            if 'Volume' in df.columns:
                price_changes = df['Close'].pct_change().abs()
                volume_changes = df['Volume'].pct_change().abs()

                # Correlation between volume and price impact
                impact_correlation = price_changes.corr(volume_changes)

                # Low correlation suggests good liquidity
                if impact_correlation < 0.3:
                    microstructure_score = 0.8
                    structure = 'GOOD'
                elif impact_correlation < 0.6:
                    microstructure_score = 0.6
                    structure = 'FAIR'
                else:
                    microstructure_score = 0.3
                    structure = 'POOR'
            else:
                microstructure_score = 0.5
                structure = 'UNKNOWN'

            return {
                'score': microstructure_score,
                'structure': structure,
                'spread_proxy': spread_proxy,
                'liquidity_assessment': structure
            }
        except:
            return {'score': 0.5, 'structure': 'UNKNOWN'}

    def _calculate_execution_probability(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate execution probability"""
        try:
            execution_factors = []

            # Volume adequacy
            if 'Volume' in df.columns:
                avg_volume = df['Volume'].mean()
                recent_volume = df['Volume'].tail(5).mean()
                volume_adequacy = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
                execution_factors.append(volume_adequacy)
            else:
                execution_factors.append(0.5)

            # Price stability
            price_volatility = df['Close'].tail(10).std() / df['Close'].tail(10).mean()
            if price_volatility < 0.02:
                stability_factor = 0.9
            elif price_volatility < 0.05:
                stability_factor = 0.7
            else:
                stability_factor = 0.4
            execution_factors.append(stability_factor)

            # Market hours factor (simplified - assume always trading hours)
            execution_factors.append(0.8)

            execution_probability = np.mean(execution_factors)

            if execution_probability > 0.8:
                execution_rating = 'EXCELLENT'
            elif execution_probability > 0.6:
                execution_rating = 'GOOD'
            else:
                execution_rating = 'FAIR'

            return {
                'score': execution_probability,
                'probability': execution_probability,
                'rating': execution_rating,
                'volume_adequacy': execution_factors[0],
                'price_stability': execution_factors[1]
            }
        except:
            return {'score': 0.5, 'probability': 0.5, 'rating': 'UNKNOWN'}

    # Additional helper methods for remaining stages...
    def _analyze_historical_breakouts(self, df: pd.DataFrame) -> Dict:
        """Analyze historical breakout patterns"""
        try:
            if len(df) < 60:
                return {'probability': 0.5, 'success_rate': 0.5}

            # Simple breakout analysis
            breakout_successes = 0
            total_breakouts = 0

            # Look for historical resistance breakouts
            for i in range(20, len(df) - 10):
                # Define resistance level
                resistance = df['High'].iloc[i - 20:i].max()

                # Check if price broke above resistance
                if df['Close'].iloc[i] > resistance * 1.02:  # 2% breakout
                    total_breakouts += 1

                    # Check if breakout was successful (price stayed above for 5 days)
                    future_lows = df['Low'].iloc[i + 1:i + 6]
                    if len(future_lows) > 0 and future_lows.min() > resistance:
                        breakout_successes += 1

            success_rate = breakout_successes / max(1, total_breakouts)

            return {
                'probability': success_rate,
                'success_rate': success_rate,
                'total_breakouts': total_breakouts,
                'successful_breakouts': breakout_successes
            }
        except:
            return {'probability': 0.5, 'success_rate': 0.5}

    def _analyze_volume_breakout_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyze volume breakout indicators"""
        try:
            if 'Volume' not in df.columns or len(df) < 20:
                return {'probability': 0.5, 'volume_surge': False}

            # Volume analysis for breakout
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['Volume'].tail(3).mean()

            volume_surge = recent_volume > avg_volume * 1.5
            volume_building = df['Volume'].tail(5).mean() > df['Volume'].tail(10).mean()

            probability_score = 0.5
            if volume_surge:
                probability_score += 0.3
            if volume_building:
                probability_score += 0.2

            return {
                'probability': min(1.0, probability_score),
                'volume_surge': volume_surge,
                'volume_building': volume_building,
                'volume_ratio': recent_volume / avg_volume if avg_volume > 0 else 1.0
            }
        except:
            return {'probability': 0.5, 'volume_surge': False}

    def _analyze_price_action_breakout(self, df: pd.DataFrame) -> Dict:
        """Analyze price action breakout signals"""
        try:
            if len(df) < 20:
                return {'probability': 0.5, 'setup_quality': 'UNKNOWN'}

            # Price action analysis
            current_price = df['Close'].iloc[-1]
            resistance_level = df['High'].tail(20).max()

            # Distance to resistance
            distance_to_resistance = (resistance_level - current_price) / current_price

            # Recent price momentum
            momentum = (current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]

            probability_factors = []

            # Close to resistance
            if distance_to_resistance < 0.02:
                probability_factors.append(0.8)
            elif distance_to_resistance < 0.05:
                probability_factors.append(0.6)
            else:
                probability_factors.append(0.3)

            # Positive momentum
            if momentum > 0.02:
                probability_factors.append(0.8)
            elif momentum > 0:
                probability_factors.append(0.6)
            else:
                probability_factors.append(0.2)

            # Higher highs pattern
            recent_highs = df['High'].tail(5)
            higher_highs = all(recent_highs.iloc[i] >= recent_highs.iloc[i - 1] for i in range(1, len(recent_highs)))

            if higher_highs:
                probability_factors.append(0.8)
            else:
                probability_factors.append(0.4)

            breakout_probability = np.mean(probability_factors)

            if breakout_probability > 0.8:
                setup_quality = 'EXCELLENT'
            elif breakout_probability > 0.6:
                setup_quality = 'GOOD'
            else:
                setup_quality = 'FAIR'

            return {
                'probability': breakout_probability,
                'setup_quality': setup_quality,
                'distance_to_resistance': distance_to_resistance,
                'momentum': momentum,
                'higher_highs': higher_highs
            }
        except:
            return {'probability': 0.5, 'setup_quality': 'UNKNOWN'}

    def _calculate_technical_breakout_probability(self, df: pd.DataFrame) -> Dict:
        """Calculate technical breakout probability"""
        try:
            if len(df) < 30:
                return {'probability': 0.5, 'indicators': {}}

            indicators = {}

            # RSI momentum
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                rsi_momentum = rsi.iloc[-1] > rsi.iloc[-5]
                indicators['rsi_momentum'] = rsi_momentum
            else:
                indicators['rsi_momentum'] = False

            # MACD momentum
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26

            macd_positive = macd.iloc[-1] > 0
            macd_improving = macd.iloc[-1] > macd.iloc[-3]

            indicators['macd_positive'] = macd_positive
            indicators['macd_improving'] = macd_improving

            # Bollinger Band position
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            upper_band = sma_20 + (2 * std_20)

            near_upper_band = df['Close'].iloc[-1] > upper_band.iloc[-1] * 0.95
            indicators['near_upper_band'] = near_upper_band

            # Calculate composite probability
            positive_indicators = sum(indicators.values())
            total_indicators = len(indicators)

            technical_probability = positive_indicators / total_indicators if total_indicators > 0 else 0.5

            return {
                'probability': technical_probability,
                'indicators': indicators,
                'positive_count': positive_indicators,
                'total_count': total_indicators
            }
        except:
            return {'probability': 0.5, 'indicators': {}}

    def _analyze_time_based_breakout_factors(self, df: pd.DataFrame) -> Dict:
        """Analyze time-based breakout factors"""
        try:
            if len(df) < 30:
                return {'probability': 0.5, 'time_factors': {}}

            time_factors = {}

            # Consolidation duration
            consolidation_days = 0
            current_high = df['High'].tail(20).max()
            current_low = df['Low'].tail(20).min()
            consolidation_range = (current_high - current_low) / df['Close'].iloc[-1]

            if consolidation_range < 0.1:  # Tight consolidation
                for i in range(min(50, len(df))):
                    if (df['High'].iloc[-i - 1] <= current_high * 1.02 and
                            df['Low'].iloc[-i - 1] >= current_low * 0.98):
                        consolidation_days += 1
                    else:
                        break

            time_factors['consolidation_days'] = consolidation_days

            # Optimal breakout timing (simplified)
            if 15 <= consolidation_days <= 45:
                time_score = 0.8
            elif 10 <= consolidation_days <= 60:
                time_score = 0.6
            else:
                time_score = 0.4

            time_factors['time_score'] = time_score

            # Weekly/monthly timing (simplified - use day of data as proxy)
            # In practice, you'd use actual date/time analysis
            time_factors['timing_optimal'] = True  # Placeholder

            return {
                'probability': time_score,
                'time_factors': time_factors
            }
        except:
            return {'probability': 0.5, 'time_factors': {}}

    def _ml_inspired_breakout_prediction(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Machine learning inspired breakout prediction"""
        try:
            if len(df) < 50:
                return {'probability': 0.5, 'confidence': 0.5}

            # Feature engineering for ML-inspired analysis
            features = []

            # Price features
            returns = df['Close'].pct_change()
            features.extend([
                returns.tail(5).mean(),  # Recent return
                returns.tail(10).std(),  # Recent volatility
                (df['Close'].iloc[-1] - df['Close'].rolling(20).mean().iloc[-1]) / df['Close'].iloc[-1]
                # Distance from MA
            ])

            # Volume features
            if 'Volume' in df.columns:
                volume_ma = df['Volume'].rolling(20).mean()
                features.extend([
                    df['Volume'].iloc[-1] / volume_ma.iloc[-1],  # Volume ratio
                    df['Volume'].tail(5).mean() / volume_ma.iloc[-1]  # Recent volume trend
                ])
            else:
                features.extend([1.0, 1.0])

            # Technical indicator features
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1] / 100)

            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            features.append(min(1.0, max(-1.0, macd.iloc[-1] / df['Close'].iloc[-1])))

            # Simple ML-inspired scoring (weighted sum)
            weights = [0.15, -0.1, 0.2, 0.15, 0.1, 0.2, 0.1]  # Hand-tuned weights

            if len(features) == len(weights):
                ml_score = sum(f * w for f, w in zip(features, weights))
                # Normalize to 0-1 range
                ml_probability = max(0, min(1, (ml_score + 0.5)))
            else:
                ml_probability = 0.5

            # Confidence based on feature consistency
            feature_consistency = 1 - np.std(features) if len(features) > 1 else 0.5

            return {
                'probability': ml_probability,
                'confidence': feature_consistency,
                'features': features,
                'ml_score': ml_score if 'ml_score' in locals() else 0.0
            }
        except:
            return {'probability': 0.5, 'confidence': 0.5}

    def _estimate_breakout_timeframe(self, breakout_factors: Dict) -> str:
        """Estimate breakout timeframe"""
        try:
            # Analyze various factors to estimate timeframe
            high_prob_count = sum(1 for factor in breakout_factors.values()
                                  if isinstance(factor, dict) and factor.get('probability', 0) > 0.7)

            if high_prob_count >= 4:
                return "1-5 days"
            elif high_prob_count >= 2:
                return "1-2 weeks"
            else:
                return "2-4 weeks"
        except:
            return "Unknown"

    def _calculate_breakout_target(self, df: pd.DataFrame, breakout_probability: float) -> float:
        """Calculate breakout price target"""
        try:
            current_price = df['Close'].iloc[-1]
            resistance_level = df['High'].tail(20).max()

            # Base target using measured move
            consolidation_range = resistance_level - df['Low'].tail(20).min()
            base_target = resistance_level + consolidation_range

            # Adjust based on probability
            probability_multiplier = 0.5 + (breakout_probability * 0.5)
            adjusted_target = current_price + ((base_target - current_price) * probability_multiplier)

            return adjusted_target
        except:
            return df['Close'].iloc[-1] * 1.1

    # Professional grade helper methods
    def _assess_institutional_quality(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Assess institutional investment quality"""
        try:
            if len(df) < 60:
                return {'grade': 50, 'quality': 'UNKNOWN'}

            quality_factors = []

            # Liquidity assessment
            if 'Volume' in df.columns:
                avg_volume = df['Volume'].mean()
                volume_consistency = 1 - (df['Volume'].std() / avg_volume) if avg_volume > 0 else 0
                quality_factors.append(volume_consistency * 100)
            else:
                quality_factors.append(50)

            # Price stability
            price_stability = 1 - (df['Close'].tail(30).std() / df['Close'].tail(30).mean())
            quality_factors.append(price_stability * 100)

            # Trend consistency
            ma_20 = df['Close'].rolling(20).mean()
            trend_consistency = (df['Close'] > ma_20).tail(20).mean()
            quality_factors.append(trend_consistency * 100)

            # Market cap proxy (using price level)
            # Higher prices often indicate larger market cap
            price_level_score = min(100, max(0, (df['Close'].iloc[-1] / 10) * 20))
            if not is_crypto:  # Different scoring for stocks vs crypto
                quality_factors.append(price_level_score)
            else:
                quality_factors.append(70)  # Neutral score for crypto

            institutional_grade = np.mean(quality_factors)

            if institutional_grade > 80:
                quality = 'INSTITUTIONAL'
            elif institutional_grade > 60:
                quality = 'PROFESSIONAL'
            else:
                quality = 'RETAIL'

            return {
                'grade': institutional_grade,
                'quality': quality,
                'liquidity_score': quality_factors[0],
                'stability_score': quality_factors[1],
                'trend_score': quality_factors[2]
            }
        except:
            return {'grade': 50, 'quality': 'UNKNOWN'}

    def _calculate_professional_trader_grade(self, df: pd.DataFrame) -> Dict:
        """Calculate professional trader grade - IMPROVED SCORING"""
        try:
            if len(df) < 30:
                return {'grade': 60, 'level': 'INTERMEDIATE'}  # Better default

            trader_criteria = []

            # Risk-reward assessment - IMPROVED
            support_level = df['Low'].tail(20).min()
            resistance_level = df['High'].tail(20).max()
            current_price = df['Close'].iloc[-1]

            risk = current_price - support_level
            reward = resistance_level - current_price
            rr_ratio = reward / risk if risk > 0 else 2.0  # Better default

            # IMPROVED: More generous R/R scoring
            if rr_ratio >= 2.5:
                trader_criteria.append(90)
            elif rr_ratio >= 2.0:
                trader_criteria.append(85)
            elif rr_ratio >= 1.5:
                trader_criteria.append(75)
            elif rr_ratio >= 1.0:
                trader_criteria.append(65)
            else:
                trader_criteria.append(50)

            # Technical setup quality - IMPROVED
            # Multiple timeframe alignment
            ma_5 = df['Close'].rolling(5).mean().iloc[-1]
            ma_20 = df['Close'].rolling(20).mean().iloc[-1]

            alignment_score = 60  # Better base
            if current_price > ma_5 > ma_20:
                alignment_score = 90
            elif current_price > ma_5:
                alignment_score = 80
            elif current_price > ma_20:
                alignment_score = 70

            trader_criteria.append(alignment_score)

            # Entry precision - IMPROVED
            distance_to_support = (current_price - support_level) / current_price
            if distance_to_support < 0.02:
                precision_score = 95
            elif distance_to_support < 0.05:
                precision_score = 85
            elif distance_to_support < 0.08:
                precision_score = 75
            else:
                precision_score = 60  # Better than 50

            trader_criteria.append(precision_score)

            # Volume confirmation - IMPROVED
            if 'Volume' in df.columns:
                volume_ratio = df['Volume'].tail(5).mean() / df['Volume'].mean()
                volume_score = min(95, max(60, volume_ratio * 70))  # Better scaling
            else:
                volume_score = 70  # Better default

            trader_criteria.append(volume_score)

            professional_grade = np.mean(trader_criteria)

            # IMPROVED: Better level classification
            if professional_grade >= 85:
                level = 'EXPERT'
            elif professional_grade >= 75:
                level = 'PROFESSIONAL'
            elif professional_grade >= 65:
                level = 'ADVANCED'
            elif professional_grade >= 55:
                level = 'INTERMEDIATE'
            else:
                level = 'BASIC'

            return {
                'grade': professional_grade,
                'level': level,
                'risk_reward_grade': trader_criteria[0],
                'technical_grade': trader_criteria[1],
                'precision_grade': trader_criteria[2],
                'volume_grade': trader_criteria[3]
            }
        except:
            return {'grade': 65, 'level': 'INTERMEDIATE'}  # Better default

    def _evaluate_hedge_fund_criteria(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Evaluate hedge fund investment criteria"""
        try:
            if len(df) < 90:
                return {'grade': 50, 'suitable': False}

            hedge_fund_factors = []

            # Absolute return potential
            max_gain_90d = (df['High'].tail(90).max() - df['Close'].tail(90).iloc[0]) / df['Close'].tail(90).iloc[0]
            if max_gain_90d > 0.5:  # 50% potential
                hedge_fund_factors.append(90)
            elif max_gain_90d > 0.3:
                hedge_fund_factors.append(75)
            else:
                hedge_fund_factors.append(50)

            # Alpha generation potential
            # Simplified: compare to moving average performance
            market_proxy = df['Close'].rolling(60).mean()
            alpha_proxy = (df['Close'].iloc[-1] - market_proxy.iloc[-60]) / market_proxy.iloc[-60]

            if alpha_proxy > 0.1:
                hedge_fund_factors.append(85)
            elif alpha_proxy > 0:
                hedge_fund_factors.append(70)
            else:
                hedge_fund_factors.append(45)

            # Sharpe ratio estimation
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 30:
                sharpe_proxy = returns.mean() / returns.std() if returns.std() > 0 else 0
                sharpe_score = min(100, max(0, (sharpe_proxy + 1) * 50))
            else:
                sharpe_score = 50

            hedge_fund_factors.append(sharpe_score)

            # Liquidity for large positions
            if 'Volume' in df.columns:
                liquidity_score = min(100, (df['Volume'].mean() / 1000000) * 20)  # Simplified
            else:
                liquidity_score = 40

            hedge_fund_factors.append(liquidity_score)

            # Diversification benefit (inverse correlation to market)
            # Simplified: use price momentum as proxy
            momentum = df['Close'].pct_change().tail(20).mean()
            diversification_score = 100 - abs(momentum * 1000)  # Simplified
            diversification_score = max(0, min(100, diversification_score))

            hedge_fund_factors.append(diversification_score)

            hedge_fund_grade = np.mean(hedge_fund_factors)
            suitable = hedge_fund_grade >= 70

            return {
                'grade': hedge_fund_grade,
                'suitable': suitable,
                'return_potential': hedge_fund_factors[0],
                'alpha_potential': hedge_fund_factors[1],
                'risk_adjusted_return': hedge_fund_factors[2],
                'liquidity_score': hedge_fund_factors[3],
                'diversification_score': hedge_fund_factors[4]
            }
        except:
            return {'grade': 50, 'suitable': False}

    def _grade_risk_management_profile(self, df: pd.DataFrame) -> Dict:
        """Grade risk management profile"""
        try:
            if len(df) < 30:
                return {'grade': 50, 'profile': 'UNKNOWN'}

            risk_factors = []

            # Volatility assessment
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std()

            if volatility < 0.02:  # Low volatility
                volatility_grade = 90
                vol_profile = 'LOW'
            elif volatility < 0.05:
                volatility_grade = 70
                vol_profile = 'MEDIUM'
            else:
                volatility_grade = 40
                vol_profile = 'HIGH'

            risk_factors.append(volatility_grade)

            # Drawdown analysis
            rolling_max = df['Close'].expanding().max()
            drawdown = (df['Close'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())

            if max_drawdown < 0.1:  # 10% max drawdown
                drawdown_grade = 90
            elif max_drawdown < 0.2:
                drawdown_grade = 70
            else:
                drawdown_grade = 40

            risk_factors.append(drawdown_grade)

            # Support level strength
            support_tests = 0
            support_level = df['Low'].tail(30).min()
            tolerance = support_level * 0.02

            for low in df['Low'].tail(30):
                if support_level <= low <= support_level + tolerance:
                    support_tests += 1

            support_grade = min(100, support_tests * 20)
            risk_factors.append(support_grade)

            # Liquidity risk
            if 'Volume' in df.columns:
                volume_consistency = 1 - (df['Volume'].std() / df['Volume'].mean())
                liquidity_grade = volume_consistency * 100
            else:
                liquidity_grade = 50

            risk_factors.append(liquidity_grade)

            risk_management_grade = np.mean(risk_factors)

            if risk_management_grade > 80:
                profile = 'CONSERVATIVE'
            elif risk_management_grade > 60:
                profile = 'MODERATE'
            else:
                profile = 'AGGRESSIVE'

            return {
                'grade': risk_management_grade,
                'profile': profile,
                'volatility_grade': risk_factors[0],
                'volatility_profile': vol_profile,
                'drawdown_grade': risk_factors[1],
                'max_drawdown': max_drawdown,
                'support_grade': risk_factors[2],
                'liquidity_grade': risk_factors[3]
            }
        except:
            return {'grade': 50, 'profile': 'UNKNOWN'}

    def _grade_execution_excellence(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Grade execution excellence"""
        try:
            if len(df) < 20:
                return {'grade': 50, 'excellence': 'UNKNOWN'}

            execution_factors = []

            # Bid-ask spread estimation
            spread_proxy = ((df['High'] - df['Low']) / df['Close']).tail(10).mean()

            if spread_proxy < 0.01:  # Tight spread
                spread_grade = 90
            elif spread_proxy < 0.03:
                spread_grade = 70
            else:
                spread_grade = 50

            execution_factors.append(spread_grade)

            # Volume depth
            if 'Volume' in df.columns:
                volume_depth = df['Volume'].tail(10).mean()
                # Normalize volume (simplified)
                depth_grade = min(100, max(20, volume_depth / 100000 * 20))
            else:
                depth_grade = 50

            execution_factors.append(depth_grade)

            # Price impact estimation
            price_changes = df['Close'].pct_change().abs().tail(10)
            avg_price_impact = price_changes.mean()

            if avg_price_impact < 0.005:  # Low impact
                impact_grade = 90
            elif avg_price_impact < 0.01:
                impact_grade = 70
            else:
                impact_grade = 50

            execution_factors.append(impact_grade)

            # Execution timing (market hours proxy)
            # Simplified: assume optimal timing
            timing_grade = 80  # Placeholder
            execution_factors.append(timing_grade)

            execution_grade = np.mean(execution_factors)

            if execution_grade > 85:
                excellence = 'EXCELLENT'
            elif execution_grade > 70:
                excellence = 'GOOD'
            elif execution_grade > 55:
                excellence = 'FAIR'
            else:
                excellence = 'POOR'

            return {
                'grade': execution_grade,
                'excellence': excellence,
                'spread_grade': execution_factors[0],
                'depth_grade': execution_factors[1],
                'impact_grade': execution_factors[2],
                'timing_grade': execution_factors[3]
            }
        except:
            return {'grade': 50, 'excellence': 'UNKNOWN'}

    def _calculate_alpha_generation_potential(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate alpha generation potential"""
        try:
            if len(df) < 60:
                return {'score': 50, 'potential': 'MEDIUM'}

            alpha_factors = []

            # Relative strength vs market (simplified)
            # Use 60-day performance as proxy
            performance_60d = (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]

            if performance_60d > 0.2:  # 20% outperformance
                alpha_factors.append(90)
            elif performance_60d > 0.1:
                alpha_factors.append(75)
            elif performance_60d > 0:
                alpha_factors.append(60)
            else:
                alpha_factors.append(30)

            # Momentum factor
            momentum_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
            momentum_score = min(100, max(0, 50 + momentum_20d * 500))
            alpha_factors.append(momentum_score)

            # Technical factor (trend strength)
            ma_20 = df['Close'].rolling(20).mean()
            trend_strength = (df['Close'].iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1]
            trend_score = min(100, max(0, 50 + trend_strength * 1000))
            alpha_factors.append(trend_score)

            # Quality factor (earnings growth proxy using price appreciation)
            growth_proxy = (df['Close'].tail(20).mean() - df['Close'].iloc[-40:-20].mean()) / df['Close'].iloc[
                                                                                              -40:-20].mean()
            quality_score = min(100, max(0, 50 + growth_proxy * 500))
            alpha_factors.append(quality_score)

            # Size factor (smaller stocks often have higher alpha potential)
            # Use price level as inverse proxy for size
            size_factor = max(20, min(100, 100 - (df['Close'].iloc[-1] / 1000) * 10))
            alpha_factors.append(size_factor)

            alpha_score = np.mean(alpha_factors)

            if alpha_score > 80:
                potential = 'HIGH'
            elif alpha_score > 60:
                potential = 'MEDIUM'
            else:
                potential = 'LOW'

            # Expected alpha (simplified calculation)
            expected_alpha = max(0, (alpha_score - 50) / 50 * 0.15)  # Up to 15% alpha

            return {
                'score': alpha_score,
                'potential': potential,
                'expected_alpha': expected_alpha,
                'performance_factor': alpha_factors[0],
                'momentum_factor': alpha_factors[1],
                'technical_factor': alpha_factors[2],
                'quality_factor': alpha_factors[3],
                'size_factor': alpha_factors[4]
            }
        except:
            return {'score': 50, 'potential': 'MEDIUM'}


class EnhancedMultiTimeframeAnalyzer:
    """Enhanced analyzer with all new features integrated"""

    def __init__(self, learning_system: ProfessionalLearningSystem, blacklist_manager: EnhancedBlacklistManager):
        self.learning_system = learning_system
        self.blacklist = blacklist_manager
        self.stage_analyzer = EnhancedStageAnalyzer(blacklist_manager)
        self.pattern_detector = AdvancedPatternDetector()

        # Enhanced timeframes including Fibonacci numbers
        self.timeframes = [
            TimeFrame.D1, TimeFrame.D3, TimeFrame.D6, TimeFrame.D11,
            TimeFrame.D21, TimeFrame.D33, TimeFrame.D55, TimeFrame.D89
        ]

    def analyze_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame, is_crypto: bool = False) -> List[PatternType]:
        """Enhanced timeframe analysis with new patterns"""
        patterns_found = []

        if timeframe.value > 1:
            df_resampled = self.resample_data(df, timeframe.value)
        else:
            df_resampled = df.copy()

        if len(df_resampled) < 10:
            return patterns_found

        # Advanced pattern detection using the new pattern detector
        advanced_patterns = self._detect_all_advanced_patterns(df_resampled, is_crypto)
        patterns_found.extend(advanced_patterns)

        # Traditional patterns with enhanced detection
        traditional_patterns = self._detect_enhanced_traditional_patterns(df_resampled, is_crypto)
        patterns_found.extend(traditional_patterns)

        return patterns_found

    def _detect_all_advanced_patterns(self, df: pd.DataFrame, is_crypto: bool = False) -> List[PatternType]:
        """Detect all 20 new advanced patterns"""
        patterns = []

        # Pattern detection mapping
        pattern_methods = [
            (self.pattern_detector.detect_hidden_accumulation, PatternType.HIDDEN_ACCUMULATION),
            (self.pattern_detector.detect_smart_money_flow, PatternType.SMART_MONEY_FLOW),
            (self.pattern_detector.detect_whale_accumulation, PatternType.WHALE_ACCUMULATION),
            (self.pattern_detector.detect_coiled_spring, PatternType.COILED_SPRING),
            (self.pattern_detector.detect_momentum_vacuum, PatternType.MOMENTUM_VACUUM),
            (self.pattern_detector.detect_fibonacci_retracement, PatternType.FIBONACCI_RETRACEMENT),
            (self.pattern_detector.detect_elliott_wave_3, PatternType.ELLIOTT_WAVE_3),
            (self.pattern_detector.detect_bollinger_squeeze, PatternType.BOLLINGER_SQUEEZE),
            (self.pattern_detector.detect_ichimoku_cloud, PatternType.ICHIMOKU_CLOUD),
            (self.pattern_detector.detect_tradingview_triple_support, PatternType.TRADINGVIEW_TRIPLE_SUPPORT),  # NEW!
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
            (self._detect_tradingview_triple_support_simple, PatternType.TRADINGVIEW_TRIPLE_SUPPORT),  # NEW!
        ]

        # Run all advanced pattern detections
        for method, pattern_type in pattern_methods:
            try:
                passed, score, details = method(df, is_crypto)
                if passed and score >= 60:
                    patterns.append(pattern_type)
            except:
                continue

        # Run additional simplified patterns
        for method, pattern_type in additional_patterns:
            try:
                if method(df, is_crypto):
                    patterns.append(pattern_type)
            except:
                continue

        return patterns

    # Simplified pattern detection methods for the additional patterns
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

    def _detect_tradingview_triple_support_simple(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Simplified TradingView triple support detection"""
        try:
            if len(df) < 40:
                return False

            # Find multiple support levels that are holding
            current_price = df['Close'].iloc[-1]

            # Look for support levels at different periods
            support_levels = []

            # Short-term support (last 15 days)
            short_support = df['Low'].tail(15).min()
            if current_price > short_support * 1.02:  # Price above short support
                support_levels.append(short_support)

            # Medium-term support (last 30 days)
            medium_support = df['Low'].tail(30).min()
            if current_price > medium_support * 1.02:  # Price above medium support
                support_levels.append(medium_support)

            # Long-term support (last 45 days)
            if len(df) >= 45:
                long_support = df['Low'].tail(45).min()
                if current_price > long_support * 1.02:  # Price above long support
                    support_levels.append(long_support)

            # Check if supports are distinct but related
            distinct_supports = []
            for level in support_levels:
                is_distinct = True
                for existing in distinct_supports:
                    if abs(level - existing) / existing < 0.05:  # Within 5%
                        is_distinct = False
                        break
                if is_distinct:
                    distinct_supports.append(level)

            # Need at least 2 distinct stable support levels
            if len(distinct_supports) < 2:
                return False

            # Volume confirmation - simplified
            volume_ok = True
            if 'Volume' in df.columns:
                recent_volume = df['Volume'].tail(5).mean()
                avg_volume = df['Volume'].mean()
                volume_ok = recent_volume >= avg_volume * 0.8  # Not too low volume

            # Check price stability near supports
            price_stable = True
            for support in distinct_supports:
                distance = (current_price - support) / current_price
                if distance > 0.1:  # More than 10% above support
                    price_stable = False
                    break

            return len(distinct_supports) >= 2 and volume_ok and price_stable

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

    def _detect_enhanced_traditional_patterns(self, df: pd.DataFrame, is_crypto: bool = False) -> List[PatternType]:
        """Enhanced traditional pattern detection"""
        patterns = []

        try:
            # Enhanced triangle patterns
            if self._detect_enhanced_ascending_triangle(df, is_crypto):
                patterns.append(PatternType.ASCENDING_TRIANGLE)
            if self._detect_enhanced_descending_triangle(df, is_crypto):
                patterns.append(PatternType.DESCENDING_TRIANGLE)
            if self._detect_enhanced_symmetrical_triangle(df, is_crypto):
                patterns.append(PatternType.SYMMETRICAL_TRIANGLE)

            # Enhanced wedge patterns
            if self._detect_enhanced_rising_wedge(df, is_crypto):
                patterns.append(PatternType.RISING_WEDGE)
            if self._detect_enhanced_falling_wedge(df, is_crypto):
                patterns.append(PatternType.FALLING_WEDGE)

            # Enhanced flag patterns
            if self._detect_enhanced_bull_flag(df, is_crypto):
                patterns.append(PatternType.BULL_FLAG)
            if self._detect_enhanced_bear_flag(df, is_crypto):
                patterns.append(PatternType.BEAR_FLAG)

            # Enhanced reversal patterns
            if self._detect_enhanced_double_bottom(df, is_crypto):
                patterns.append(PatternType.DOUBLE_BOTTOM)
            if self._detect_enhanced_triple_bottom(df, is_crypto):
                patterns.append(PatternType.TRIPLE_BOTTOM)
            if self._detect_enhanced_inverse_head_shoulders(df, is_crypto):
                patterns.append(PatternType.INVERSE_HEAD_SHOULDERS)

        except Exception as e:
            logger.debug(f"Error in traditional pattern detection: {e}")

        return patterns

    def _detect_enhanced_ascending_triangle(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced ascending triangle detection"""
        try:
            if len(df) < 20:
                return False

            highs = df['High'].tail(20).values
            lows = df['Low'].tail(20).values
            x = np.arange(len(highs))

            # Flat resistance line
            high_std = np.std(highs[-10:]) / np.mean(highs[-10:])

            # Rising support line
            low_slope, _, low_r, _, _ = linregress(x, lows)

            threshold = 0.05 if is_crypto else 0.03
            min_slope = 0.001 if is_crypto else 0.002
            min_r = 0.3 if is_crypto else 0.4

            return (high_std < threshold and
                    low_slope > min_slope and
                    abs(low_r) > min_r)
        except:
            return False

    def _detect_enhanced_descending_triangle(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced descending triangle detection"""
        try:
            if len(df) < 20:
                return False

            highs = df['High'].tail(20).values
            lows = df['Low'].tail(20).values
            x = np.arange(len(highs))

            # Flat support line
            low_std = np.std(lows[-10:]) / np.mean(lows[-10:])

            # Falling resistance line
            high_slope, _, high_r, _, _ = linregress(x, highs)

            threshold = 0.05 if is_crypto else 0.03
            max_slope = -0.001 if is_crypto else -0.002
            min_r = 0.3 if is_crypto else 0.4

            return (low_std < threshold and
                    high_slope < max_slope and
                    abs(high_r) > min_r)
        except:
            return False

    def _detect_enhanced_symmetrical_triangle(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced symmetrical triangle detection"""
        try:
            if len(df) < 25:
                return False

            highs = df['High'].tail(25).values
            lows = df['Low'].tail(25).values
            x = np.arange(len(highs))

            # Falling resistance and rising support
            high_slope, _, high_r, _, _ = linregress(x, highs)
            low_slope, _, low_r, _, _ = linregress(x, lows)

            # Convergence
            convergence = abs(high_slope) > 0.001 and low_slope > 0.001
            correlation = abs(high_r) > 0.3 and abs(low_r) > 0.3

            return convergence and correlation
        except:
            return False

    def _detect_enhanced_rising_wedge(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced rising wedge detection"""
        try:
            if len(df) < 25:
                return False

            highs = df['High'].tail(25).values
            lows = df['Low'].tail(25).values
            x = np.arange(len(highs))

            # Both lines rising, but support rising faster
            high_slope, _, high_r, _, _ = linregress(x, highs)
            low_slope, _, low_r, _, _ = linregress(x, lows)

            both_rising = high_slope > 0 and low_slope > 0
            convergence = low_slope > high_slope
            correlation = abs(high_r) > 0.4 and abs(low_r) > 0.4

            return both_rising and convergence and correlation
        except:
            return False

    def _detect_enhanced_falling_wedge(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced falling wedge detection"""
        try:
            if len(df) < 25:
                return False

            highs = df['High'].tail(25).values
            lows = df['Low'].tail(25).values
            x = np.arange(len(highs))

            # Both lines falling, but resistance falling faster
            high_slope, _, high_r, _, _ = linregress(x, highs)
            low_slope, _, low_r, _, _ = linregress(x, lows)

            both_falling = high_slope < 0 and low_slope < 0
            convergence = high_slope < low_slope  # Resistance falling faster
            correlation = abs(high_r) > 0.4 and abs(low_r) > 0.4

            return both_falling and convergence and correlation
        except:
            return False

    def _detect_enhanced_bull_flag(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced bull flag detection"""
        try:
            if len(df) < 25:
                return False

            # Look for flagpole (strong upward move)
            flagpole_start = 15
            flagpole_move = (df['Close'].iloc[-flagpole_start] - df['Close'].iloc[-25]) / df['Close'].iloc[-25]

            if flagpole_move < 0.08:  # Need significant upward move
                return False

            # Look for flag (consolidation)
            flag_data = df.tail(flagpole_start)
            flag_range = (flag_data['High'].max() - flag_data['Low'].min()) / flag_data['Close'].mean()

            # Flag should be tight consolidation
            max_flag_range = 0.1 if is_crypto else 0.08

            return flag_range < max_flag_range
        except:
            return False

    def _detect_enhanced_bear_flag(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced bear flag detection"""
        try:
            if len(df) < 25:
                return False

            # Look for flagpole (strong downward move)
            flagpole_start = 15
            flagpole_move = (df['Close'].iloc[-25] - df['Close'].iloc[-flagpole_start]) / df['Close'].iloc[-25]

            if flagpole_move < 0.08:  # Need significant downward move
                return False

            # Look for flag (consolidation)
            flag_data = df.tail(flagpole_start)
            flag_range = (flag_data['High'].max() - flag_data['Low'].min()) / flag_data['Close'].mean()

            # Flag should be tight consolidation
            max_flag_range = 0.1 if is_crypto else 0.08

            return flag_range < max_flag_range
        except:
            return False

    def _detect_enhanced_double_bottom(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced double bottom detection"""
        try:
            if len(df) < 40:
                return False

            lows = df['Low'].tail(40).values

            # Find two significant lows
            low_indices = argrelextrema(lows, np.less, order=5)[0]

            if len(low_indices) < 2:
                return False

            # Get the two most recent lows
            last_two_lows = low_indices[-2:]
            low1_price = lows[last_two_lows[0]]
            low2_price = lows[last_two_lows[1]]

            # Lows should be at similar levels
            tolerance = 0.05 if is_crypto else 0.03
            similar_levels = abs(low1_price - low2_price) / low1_price < tolerance

            # Should have a peak between the lows
            between_data = lows[last_two_lows[0]:last_two_lows[1]]
            peak_between = max(between_data) > max(low1_price, low2_price) * 1.05

            return similar_levels and peak_between
        except:
            return False

    def _detect_enhanced_triple_bottom(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced triple bottom detection"""
        try:
            if len(df) < 60:
                return False

            lows = df['Low'].tail(60).values

            # Find three significant lows
            low_indices = argrelextrema(lows, np.less, order=5)[0]

            if len(low_indices) < 3:
                return False

            # Get the three most recent lows
            last_three_lows = low_indices[-3:]
            low_prices = [lows[i] for i in last_three_lows]

            # All lows should be at similar levels
            tolerance = 0.06 if is_crypto else 0.04
            max_low = max(low_prices)
            min_low = min(low_prices)

            similar_levels = (max_low - min_low) / min_low < tolerance

            return similar_levels
        except:
            return False

    def _detect_enhanced_inverse_head_shoulders(self, df: pd.DataFrame, is_crypto: bool = False) -> bool:
        """Enhanced inverse head and shoulders detection"""
        try:
            if len(df) < 50:
                return False

            lows = df['Low'].tail(50).values

            # Find three significant lows for head and shoulders
            low_indices = argrelextrema(lows, np.less, order=5)[0]

            if len(low_indices) < 3:
                return False

            # Get last three lows (left shoulder, head, right shoulder)
            shoulders_head = low_indices[-3:]
            left_shoulder = lows[shoulders_head[0]]
            head = lows[shoulders_head[1]]
            right_shoulder = lows[shoulders_head[2]]

            # Head should be lower than both shoulders
            head_lower = head < left_shoulder and head < right_shoulder

            # Shoulders should be approximately equal
            tolerance = 0.08 if is_crypto else 0.06
            shoulders_equal = abs(left_shoulder - right_shoulder) / left_shoulder < tolerance

            return head_lower and shoulders_equal
        except:
            return False

    def resample_data(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Enhanced data resampling"""
        try:
            resampled = pd.DataFrame()

            for i in range(0, len(df), days):
                chunk = df.iloc[i:i + days]
                if len(chunk) > 0:
                    row = {
                        'Open': chunk['Open'].iloc[0],
                        'High': chunk['High'].max(),
                        'Low': chunk['Low'].min(),
                        'Close': chunk['Close'].iloc[-1]
                    }
                    if 'Volume' in df.columns:
                        row['Volume'] = chunk['Volume'].sum()

                    resampled = pd.concat([resampled, pd.DataFrame([row])], ignore_index=True)

            return resampled
        except Exception:
            return df

    def find_best_combination(self, timeframe_patterns: Dict[TimeFrame, List[PatternType]],
                              is_crypto: bool = False) -> PatternCombination:
        """Enhanced pattern combination finder - IMPROVED SCORING"""
        all_combinations = []

        for tf, patterns in timeframe_patterns.items():
            if patterns:
                for pattern in patterns:
                    success_rate, metadata = self.learning_system.get_advanced_pattern_score(pattern.value, is_crypto)

                    # IMPROVED: Better base scoring
                    base_score = 70 if is_crypto else 65  # Increased from 60/50
                    score = base_score + success_rate * 40  # Better multiplier

                    # Enhanced risk assessment
                    risk_level = "LOW" if success_rate > 0.6 else "MEDIUM" if success_rate > 0.4 else "HIGH"
                    expected_gain = metadata.get('avg_gain', 0.08) if isinstance(metadata,
                                                                                 dict) else 0.08  # Better default

                    # IMPROVED: Higher confidence baseline
                    confidence = max(55, success_rate * 120)  # Increased baseline and multiplier

                    all_combinations.append(PatternCombination(
                        patterns=[pattern],
                        combined_score=score,
                        timeframe=tf,
                        confidence=confidence,
                        historical_success_rate=success_rate,
                        top_stocks=[],
                        risk_level=risk_level,
                        expected_gain=expected_gain
                    ))

                # Multi-pattern combinations - IMPROVED
                if len(patterns) >= 2:
                    for i in range(len(patterns)):
                        for j in range(i + 1, len(patterns)):
                            combo = [patterns[i], patterns[j]]
                            avg_success = np.mean([
                                self.learning_system.get_advanced_pattern_score(p.value, is_crypto)[0]
                                for p in combo
                            ])

                            # IMPROVED: Better combo scoring with bigger bonus
                            combo_score = 80 + avg_success * 70  # Increased from 70 + 60
                            confidence = max(70, avg_success * 140)  # Higher confidence for combos

                            all_combinations.append(PatternCombination(
                                patterns=combo,
                                combined_score=combo_score,
                                timeframe=tf,
                                confidence=confidence,
                                historical_success_rate=avg_success,
                                top_stocks=[],
                                risk_level="LOW" if avg_success > 0.5 else "MEDIUM",  # More generous
                                expected_gain=avg_success * 0.18  # Higher expected gain
                            ))

        if all_combinations:
            return max(all_combinations, key=lambda x: x.combined_score)
        else:
            # Enhanced default
            default_pattern = PatternType.CRYPTO_PUMP if is_crypto else PatternType.MIXED_BULLISH
            return PatternCombination(
                patterns=[default_pattern],
                combined_score=55,  # Increased from 45
                timeframe=TimeFrame.D1,
                confidence=50,  # Increased from 40
                historical_success_rate=0.5,  # Increased from 0.4
                top_stocks=[],
                risk_level="MEDIUM",
                expected_gain=0.10  # Increased from 0.08
            )

    def determine_consolidation_type(self, df: pd.DataFrame, is_crypto: bool = False) -> ConsolidationType:
        """Enhanced consolidation type determination"""
        try:
            if len(df) < 20:
                return ConsolidationType.HIGH_VOLATILITY

            recent = df.tail(20)
            high = recent['High'].max()
            low = recent['Low'].min()
            avg = recent['Close'].mean()

            range_pct = (high - low) / avg * 100

            # Enhanced classification with new types
            if is_crypto:
                if range_pct <= 15:
                    return ConsolidationType.COILED
                elif range_pct <= 25:
                    return ConsolidationType.TIGHT
                elif range_pct <= 40:
                    return ConsolidationType.MODERATE
                elif range_pct <= 60:
                    return ConsolidationType.WIDE
                else:
                    return ConsolidationType.HIGH_VOLATILITY
            else:
                if range_pct <= 8:
                    return ConsolidationType.COILED
                elif range_pct <= 15:
                    return ConsolidationType.TIGHT
                elif range_pct <= 25:
                    return ConsolidationType.MODERATE
                elif range_pct <= 40:
                    return ConsolidationType.WIDE
                else:
                    return ConsolidationType.HIGH_VOLATILITY

            # Check for accumulation pattern
            if 'Volume' in df.columns:
                volume_increasing = recent['Volume'].tail(10).mean() > recent['Volume'].mean()
                price_stable = range_pct < (20 if is_crypto else 12)

                if volume_increasing and price_stable:
                    return ConsolidationType.ACCUMULATION

        except:
            return ConsolidationType.HIGH_VOLATILITY

    def determine_maturity(self, stage_results: List[StageResult], patterns: Dict) -> PatternMaturity:
        """Enhanced pattern maturity determination"""
        passed_stages = sum(1 for r in stage_results if r.passed)
        avg_score = np.mean([r.score for r in stage_results])
        avg_confidence = np.mean([r.confidence for r in stage_results])

        # Enhanced maturity levels
        if passed_stages >= 9 and avg_score >= 80 and avg_confidence >= 85:
            return PatternMaturity.IMMINENT
        elif passed_stages >= 8 and avg_score >= 75:
            return PatternMaturity.EXPLOSIVE
        elif passed_stages >= 7 and avg_score >= 65:
            return PatternMaturity.READY
        elif passed_stages >= 5 and avg_score >= 55:
            return PatternMaturity.MATURE
        elif passed_stages >= 3:
            return PatternMaturity.DEVELOPING
        else:
            return PatternMaturity.EARLY

    def create_visual_summary(self, stage_results: List[StageResult]) -> str:
        """Enhanced visual indicator summary"""
        indicators = []

        for result in stage_results:
            if result.passed and result.visual_indicator not in ["✅", "❌"]:
                indicators.append(result.visual_indicator)

        # Add special indicators based on combinations
        if len(indicators) >= 8:
            indicators.append("🔥")  # Fire for hot setup
        if len(indicators) >= 6:
            indicators.append("⚡")  # Lightning for strong setup

        return " ".join(indicators) if indicators else "📊"

    def calculate_entry_setup(self, df: pd.DataFrame, stage_results: List[StageResult],
                              is_crypto: bool = False) -> Dict[str, float]:
        """Enhanced entry and exit level calculation"""
        current_price = df['Close'].iloc[-1]

        # Enhanced support calculation
        support_levels = []

        # Technical support
        tech_support = df['Low'].tail(20).min()
        support_levels.append(tech_support)

        # Moving average support
        if len(df) >= 20:
            ma_20 = df['Close'].rolling(20).mean().iloc[-1]
            if ma_20 < current_price:
                support_levels.append(ma_20)

        # Volume-weighted support
        if 'Volume' in df.columns and len(df) >= 30:
            vwap = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).tail(30).sum() / df['Volume'].tail(
                30).sum()
            if vwap < current_price:
                support_levels.append(vwap)

        # Use strongest support
        primary_support = max(support_levels) if support_levels else current_price * 0.95

        # Enhanced entry calculation
        entry_premium = 0.005 if is_crypto else 0.002
        entry = current_price * (1 - entry_premium)

        # Enhanced stop loss
        stop_buffer = 0.03 if is_crypto else 0.02
        stop_loss = primary_support * (1 - stop_buffer)

        # Enhanced targets with Fibonacci levels
        if is_crypto:
            target1 = current_price * 1.08  # 8% target
            target2 = current_price * 1.15  # 15% target
            target3 = current_price * 1.25  # 25% target
            target4 = current_price * 1.40  # 40% target (crypto bonus)
        else:
            target1 = current_price * 1.05  # 5% target
            target2 = current_price * 1.10  # 10% target
            target3 = current_price * 1.18  # 18% target
            target4 = current_price * 1.30  # 30% target (extended)

        # Risk-reward calculations
        risk = entry - stop_loss
        reward1 = target1 - entry
        reward2 = target2 - entry

        risk_reward_1 = reward1 / risk if risk > 0 else 0
        risk_reward_2 = reward2 / risk if risk > 0 else 0

        # Position sizing recommendation (Kelly Criterion inspired)
        win_rate = 0.6  # Base assumption
        avg_win = reward1
        avg_loss = risk

        if avg_loss > 0:
            kelly_percentage = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_percentage = max(0, min(0.25, kelly_percentage))  # Cap at 25%
        else:
            kelly_percentage = 0.05

        return {
            'current': round(float(current_price), 4 if is_crypto else 2),
            'entry': round(float(entry), 4 if is_crypto else 2),
            'stop_loss': round(float(stop_loss), 4 if is_crypto else 2),
            'support_level': round(float(primary_support), 4 if is_crypto else 2),
            'target1': round(float(target1), 4 if is_crypto else 2),
            'target2': round(float(target2), 4 if is_crypto else 2),
            'target3': round(float(target3), 4 if is_crypto else 2),
            'target4': round(float(target4), 4 if is_crypto else 2),
            'risk_amount': round(float(risk), 4 if is_crypto else 2),
            'risk_reward_1': round(float(risk_reward_1), 2),
            'risk_reward_2': round(float(risk_reward_2), 2),
            'position_size_pct': round(float(kelly_percentage * 100), 1)
        }

    def calculate_game_score(self, stage_results: List[StageResult], patterns: Dict) -> Tuple[float, str]:
        """Enhanced game scoring system"""
        base_score = sum(r.score for r in stage_results if r.passed)

        # Pattern bonuses
        pattern_count = sum(len(p) for p in patterns.values())
        pattern_bonus = pattern_count * 75

        # Advanced pattern bonuses
        advanced_pattern_bonus = 0
        advanced_patterns = [
            PatternType.HIDDEN_ACCUMULATION, PatternType.SMART_MONEY_FLOW,
            PatternType.WHALE_ACCUMULATION, PatternType.COILED_SPRING,
            PatternType.ELLIOTT_WAVE_3, PatternType.FIBONACCI_RETRACEMENT
        ]

        for tf_patterns in patterns.values():
            for pattern in tf_patterns:
                if pattern in advanced_patterns:
                    advanced_pattern_bonus += 100

        # Confidence bonus
        confidence_bonus = np.mean([r.confidence for r in stage_results]) * 2

        # Professional grade bonus
        professional_stages = ["Professional Grade", "Risk Assessment", "Entry Optimization"]
        professional_bonus = sum(r.score for r in stage_results
                                 if r.stage_name in professional_stages and r.passed) * 2

        # Random excitement factor
        excitement_factor = random.randint(0, 200)

        game_score = (base_score + pattern_bonus + advanced_pattern_bonus +
                      confidence_bonus + professional_bonus + excitement_factor)

        # Enhanced rarity classification
        if game_score > 1500:
            rarity = "👑 LEGENDARY+"
        elif game_score > 1200:
            rarity = "🌟 LEGENDARY"
        elif game_score > 900:
            rarity = "💎 EPIC+"
        elif game_score > 700:
            rarity = "💎 EPIC"
        elif game_score > 500:
            rarity = "💜 RARE+"
        elif game_score > 350:
            rarity = "💜 RARE"
        else:
            rarity = "⚪ COMMON"

        return game_score, rarity

    def detect_special_pattern(self, patterns: List[PatternType]) -> str:
        """Enhanced special pattern detection"""
        pattern_names = [p.value for p in patterns]

        # Check for professional patterns first
        for special_name, required_patterns in PROFESSIONAL_PATTERNS.items():
            matches = sum(1 for rp in required_patterns
                          if any(rp.lower() in pn.lower() for pn in pattern_names))
            if matches >= 2:
                return special_name

        # Check original patterns
        for special_name, required_patterns in STRONG_PATTERNS.items():
            matches = sum(1 for rp in required_patterns
                          if any(rp.lower() in pn.lower() for pn in pattern_names))
            if matches >= 2:
                return special_name

        # Advanced pattern combinations
        advanced_combo_patterns = {
            "🧠 INSTITUTIONAL GRADE": ["Hidden Accumulation", "Smart Money", "Whale"],
            "⚡ BREAKOUT MASTER": ["Coiled Spring", "Pressure Cooker", "Volume Pocket"],
            "📊 TECHNICAL PERFECTION": ["Fibonacci", "Elliott Wave", "Ichimoku"],
            "🎯 SNIPER PRECISION": ["VWAP", "Fractal", "Golden Ratio"],
            "🚀 MOMENTUM BEAST": ["RSI Divergence", "Momentum", "Stochastic"],
            "🟦 TRADINGVIEW KING": ["TradingView Triple Support", "Support", "Volume"],  # NEW!
            "💎 TRIPLE THREAT": ["Triple Support", "Stable", "Multiple Timeframe"]  # NEW!
        }

        for special_name, required_patterns in advanced_combo_patterns.items():
            matches = sum(1 for rp in required_patterns
                          if any(rp.lower() in pn.lower() for pn in pattern_names))
            if matches >= 2:
                return special_name

        return ""


def detect_m1_optimization() -> Dict[str, Any]:
    """Detect M1 MacBook and return optimization settings"""
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'cpu_count': mp.cpu_count(),
        'is_m1': False,
        'optimal_processes': 4,
        'chunk_multiplier': 1
    }

    # Detect M1/M2 MacBook
    if (platform.system() == 'Darwin' and
            (platform.machine() == 'arm64' or 'Apple' in platform.processor())):
        system_info['is_m1'] = True

        # M1-specific optimizations
        cpu_count = mp.cpu_count()
        if cpu_count >= 10:  # M1 Ultra/Max
            system_info['optimal_processes'] = min(12, cpu_count - 2)
            system_info['chunk_multiplier'] = 3
        elif cpu_count >= 8:  # M1 Pro
            system_info['optimal_processes'] = min(10, cpu_count - 1)
            system_info['chunk_multiplier'] = 2
        else:  # M1 Standard
            system_info['optimal_processes'] = min(8, cpu_count)
            system_info['chunk_multiplier'] = 2
    else:
        # Intel/AMD optimization
        cpu_count = mp.cpu_count()
        system_info['optimal_processes'] = min(6, cpu_count - 1)
        system_info['chunk_multiplier'] = 1

    return system_info


class ProfessionalPatternAnalyzer:
    """Enhanced main analyzer with professional features"""

    def __init__(self, data_dir: str = "Chinese_Market/data"):
        self.data_dir = Path(data_dir)
        self.blacklist = EnhancedBlacklistManager()
        self.learning_system = ProfessionalLearningSystem()
        self.multi_tf_analyzer = EnhancedMultiTimeframeAnalyzer(self.learning_system, self.blacklist)
        self.results = []
        self.used_symbols = set()

        # Enhanced path configuration
        self.stock_paths = {
            'shanghai': self.data_dir / 'shanghai_6xx',
            'shenzhen': self.data_dir / 'shenzhen_0xx',
            'beijing': self.data_dir / 'beijing_8xx'  # Additional exchange
        }

        # Multiple crypto path attempts - FIXED PATHS
        self.crypto_paths = [
            self.data_dir / 'huobi' / 'spot_usdt' / '1d',  # CORRECT PATH
            self.data_dir / 'huobi' / 'csv' / 'spot',
            self.data_dir / 'csv' / 'huobi' / 'spot',
            self.data_dir / 'crypto' / 'spot',
            self.data_dir / 'binance' / 'spot'
        ]

        self.crypto_path = self._find_crypto_path()

    def _find_crypto_path(self) -> Optional[Path]:
        """Find available crypto data path"""
        for path in self.crypto_paths:
            if path.exists():
                logger.info(f"Found crypto data at: {path}")
                return path

        logger.warning("No crypto data path found")
        return None

    def load_data(self, symbol: str, market_type: MarketType) -> Optional[pd.DataFrame]:
        """Enhanced data loading with better error handling"""
        try:
            if market_type == MarketType.CRYPTO:
                if not self.crypto_path:
                    return None

                # Multiple naming conventions
                possible_names = [
                    f"{symbol}.csv",
                    f"{symbol.upper()}.csv",
                    f"{symbol.lower()}.csv",
                    f"{symbol}_USDT.csv",
                    f"{symbol.replace('_USDT', '')}_USDT.csv"
                ]

                file_path = None
                for name in possible_names:
                    test_path = self.crypto_path / name
                    if test_path.exists():
                        file_path = test_path
                        break

                if not file_path:
                    return None
            else:
                file_path = None
                for folder in self.stock_paths.values():
                    test_path = folder / f"{symbol}.csv"
                    if test_path.exists():
                        file_path = test_path
                        break

                if not file_path:
                    return None

            # Enhanced data loading
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Column standardization
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'volume': 'Volume', 'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            }

            df = df.rename(columns=column_mapping)

            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_columns):
                return None

            # Data cleaning
            df = df.sort_index()
            df = df.dropna(subset=required_columns)
            df = df[(df[required_columns] > 0).all(axis=1)]

            # Remove extreme outliers
            for col in required_columns:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df = df[(df[col] >= q01) & (df[col] <= q99)]

            if 'Volume' in df.columns:
                df = df[df['Volume'] > 0]

            return df if len(df) >= 30 else None

        except Exception as e:
            logger.debug(f"Error loading {symbol}: {e}")
            return None

    def get_all_symbols(self, market_type: MarketType) -> List[str]:
        """Enhanced symbol collection - UNLIMITED"""
        symbols = []

        if market_type in [MarketType.CHINESE_STOCK, MarketType.BOTH]:
            for exchange, folder in self.stock_paths.items():
                if folder.exists():
                    logger.info(f"Loading {exchange} stocks from: {folder}")
                    csv_files = list(folder.glob("*.csv"))
                    logger.info(f"Found {len(csv_files)} files in {exchange}")

                    for file_path in csv_files:
                        symbol = file_path.stem
                        if not self.blacklist.is_blacklisted(symbol, MarketType.CHINESE_STOCK):
                            symbols.append(symbol)

        if market_type in [MarketType.CRYPTO, MarketType.BOTH]:
            if self.crypto_path and self.crypto_path.exists():
                logger.info(f"Loading crypto symbols from: {self.crypto_path}")
                csv_files = list(self.crypto_path.glob("*.csv"))
                logger.info(f"Found {len(csv_files)} crypto files")

                for file_path in csv_files:
                    symbol = file_path.stem
                    check_symbol = symbol.replace('_USDT', '') if '_USDT' in symbol else symbol
                    if not self.blacklist.is_blacklisted(check_symbol, MarketType.CRYPTO):
                        symbols.append(symbol)
            else:
                logger.warning(f"Crypto path not found or doesn't exist: {self.crypto_path}")

        # Don't shuffle - maintain order for debugging
        logger.info(f"Total symbols collected: {len(symbols)}")
        return list(set(symbols))

    def analyze_symbol(self, symbol: str, market_type: MarketType) -> Optional[MultiTimeframeAnalysis]:
        """Enhanced symbol analysis"""
        try:
            is_crypto = market_type == MarketType.CRYPTO

            # Blacklist check
            check_symbol = symbol.replace('_USDT', '') if '_USDT' in symbol else symbol
            if self.blacklist.is_blacklisted(check_symbol, market_type):
                return None

            df = self.load_data(symbol, market_type)
            if df is None:
                return None

            # Enhanced minimum data requirements
            min_len = 60 if is_crypto else 120
            if len(df) < min_len:
                return None

            # Run enhanced stage analysis
            stage_results = self.multi_tf_analyzer.stage_analyzer.run_all_stages(df, symbol, is_crypto)

            # Enhanced filtering - MORE AGGRESSIVE FOR BETTER PERFORMANCE
            passed_stages = sum(1 for r in stage_results if r.passed)
            avg_stage_score = np.mean([r.score for r in stage_results])

            # IMPROVED: More lenient filtering
            if passed_stages < (2 if is_crypto else 3):  # Reduced from 3/4
                return None

            # IMPROVED: Also accept high average scores even with fewer passed stages
            if avg_stage_score < (45 if is_crypto else 40):  # More lenient
                return None

            # Multi-timeframe pattern analysis
            timeframe_patterns = {}
            for tf in self.multi_tf_analyzer.timeframes:
                patterns = self.multi_tf_analyzer.analyze_timeframe(df, tf, is_crypto)
                if patterns:
                    timeframe_patterns[tf] = patterns

            # Ensure minimum pattern requirements
            total_patterns = sum(len(patterns) for patterns in timeframe_patterns.values())
            if total_patterns < (2 if is_crypto else 3):
                return None

            best_combo = self.multi_tf_analyzer.find_best_combination(timeframe_patterns, is_crypto)
            consolidation = self.multi_tf_analyzer.determine_consolidation_type(df, is_crypto)
            maturity = self.multi_tf_analyzer.determine_maturity(stage_results, timeframe_patterns)

            # Enhanced scoring - IMPROVED CONFIDENCE CALCULATION
            stage_scores = [r.score for r in stage_results]
            confidence_scores = [r.confidence for r in stage_results]

            # IMPROVED: Better scoring weights and bonuses
            base_score = np.mean(stage_scores) * 0.5 + np.mean(confidence_scores) * 0.3

            # Pattern count bonus
            total_patterns = sum(len(patterns) for patterns in timeframe_patterns.values())
            pattern_bonus = min(25, total_patterns * 3)  # Up to 25 bonus points

            # Stage pass bonus
            stage_bonus = min(20, passed_stages * 2.5)  # Up to 20 bonus points

            overall_score = base_score + pattern_bonus + stage_bonus

            # IMPROVED: Better confidence calculation
            prediction_confidence = min(95, best_combo.confidence + pattern_bonus + stage_bonus)

            # Quality boost for crypto and good setups
            if is_crypto:
                overall_score = max(50, overall_score * 1.1)
                prediction_confidence = max(55, prediction_confidence)
            else:
                overall_score = max(45, overall_score)
                prediction_confidence = max(50, prediction_confidence)

            entry_setup = self.multi_tf_analyzer.calculate_entry_setup(df, stage_results, is_crypto)
            visual = self.multi_tf_analyzer.create_visual_summary(stage_results)
            game_score, rarity = self.multi_tf_analyzer.calculate_game_score(stage_results, timeframe_patterns)

            # Enhanced special pattern detection
            all_patterns = []
            for patterns in timeframe_patterns.values():
                all_patterns.extend(patterns)
            special_pattern = self.multi_tf_analyzer.detect_special_pattern(all_patterns)

            # Enhanced risk assessment
            risk_assessment = self._calculate_comprehensive_risk(df, stage_results, is_crypto)

            # Enhanced technical indicators
            technical_indicators = self._calculate_technical_indicators(df, is_crypto)

            return MultiTimeframeAnalysis(
                symbol=symbol,
                market_type=market_type,
                timeframe_patterns=timeframe_patterns,
                best_combination=best_combo,
                consolidation_type=consolidation,
                maturity=maturity,
                stage_results=stage_results,
                overall_score=overall_score,
                prediction_confidence=prediction_confidence,  # Use our improved confidence
                entry_setup=entry_setup,
                visual_summary=visual,
                game_score=game_score,
                rarity_level=rarity,
                special_pattern=special_pattern,
                risk_assessment=risk_assessment,
                technical_indicators=technical_indicators
            )

        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            return None

    def _calculate_comprehensive_risk(self, df: pd.DataFrame, stage_results: List[StageResult],
                                      is_crypto: bool = False) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment - IMPROVED"""
        try:
            risk_factors = {}

            # Market risk - IMPROVED
            volatility = df['Close'].pct_change().std() * np.sqrt(252)
            # More lenient volatility thresholds
            vol_threshold_high = 1.0 if is_crypto else 0.5
            vol_threshold_medium = 0.6 if is_crypto else 0.3

            risk_factors['volatility'] = {
                'value': volatility,
                'level': 'HIGH' if volatility > vol_threshold_high else 'MEDIUM' if volatility > vol_threshold_medium else 'LOW'
            }

            # Liquidity risk - IMPROVED
            if 'Volume' in df.columns:
                volume_consistency = 1 - (df['Volume'].std() / df['Volume'].mean())
                # More generous liquidity assessment
                risk_factors['liquidity'] = {
                    'value': volume_consistency,
                    'level': 'LOW' if volume_consistency > 0.5 else 'MEDIUM' if volume_consistency > 0.3 else 'HIGH'
                }
            else:
                risk_factors['liquidity'] = {'value': 0.6, 'level': 'MEDIUM'}  # Better default

            # Technical risk - IMPROVED
            support_distance = (df['Close'].iloc[-1] - df['Low'].tail(20).min()) / df['Close'].iloc[-1]
            # More lenient technical risk
            risk_factors['technical'] = {
                'value': support_distance,
                'level': 'LOW' if support_distance > 0.08 else 'MEDIUM' if support_distance > 0.04 else 'HIGH'
            }

            # Overall risk score - IMPROVED calculation
            risk_scores = []
            for factor in risk_factors.values():
                if factor['level'] == 'LOW':
                    risk_scores.append(0.2)
                elif factor['level'] == 'MEDIUM':
                    risk_scores.append(0.5)
                else:
                    risk_scores.append(0.8)

            overall_risk = np.mean(risk_scores)

            # More generous overall assessment
            if overall_risk < 0.4:
                overall_level = 'LOW'
            elif overall_risk < 0.7:
                overall_level = 'MEDIUM'
            else:
                overall_level = 'HIGH'

            return {
                'factors': risk_factors,
                'overall_score': overall_risk,
                'overall_level': overall_level
            }
        except:
            return {'overall_level': 'MEDIUM', 'overall_score': 0.4}  # Better default

    def _calculate_technical_indicators(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict[str, float]:
        """Calculate key technical indicators"""
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
                indicators.update({'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0})

            # Moving averages
            if len(df) >= 20:
                ma_20 = df['Close'].rolling(20).mean().iloc[-1]
                indicators['ma_20'] = ma_20
                indicators['price_vs_ma20'] = (df['Close'].iloc[-1] - ma_20) / ma_20
            else:
                indicators.update({'ma_20': df['Close'].iloc[-1], 'price_vs_ma20': 0.0})

            # Bollinger Bands
            if len(df) >= 20:
                sma_20 = df['Close'].rolling(20).mean()
                std_20 = df['Close'].rolling(20).std()
                bb_upper = sma_20 + (2 * std_20)
                bb_lower = sma_20 - (2 * std_20)

                indicators['bb_upper'] = bb_upper.iloc[-1]
                indicators['bb_lower'] = bb_lower.iloc[-1]
                indicators['bb_position'] = (df['Close'].iloc[-1] - bb_lower.iloc[-1]) / (
                            bb_upper.iloc[-1] - bb_lower.iloc[-1])
            else:
                current_price = df['Close'].iloc[-1]
                indicators.update(
                    {'bb_upper': current_price * 1.05, 'bb_lower': current_price * 0.95, 'bb_position': 0.5})

            return indicators
        except:
            return {'rsi': 50.0, 'macd': 0.0}

    def run_analysis(self, market_type: MarketType = MarketType.BOTH,
                     max_symbols: int = None, num_processes: int = None) -> List[MultiTimeframeAnalysis]:
        """Enhanced analysis with M1-optimized parallel processing"""
        print(f"\n🎮 {random.choice(GAME_MESSAGES)}")
        print(f"📊 Market: {market_type.value}")
        print(f"🧠 Learning from {len(self.learning_system.pattern_performance)} advanced patterns")

        symbols = self.get_all_symbols(market_type)

        # IMPROVED: Better limit handling
        if max_symbols and max_symbols < len(symbols):
            print(f"⚠️  Limiting analysis to {max_symbols} symbols (from {len(symbols)} available)")
            symbols = symbols[:max_symbols]
        else:
            print(f"📈 Analyzing ALL {len(symbols)} symbols")

        print(
            f"🚫 Blacklisted: {len(self.blacklist.blacklisted_stocks)} stocks, {len(self.blacklist.blacklisted_crypto)} crypto")

        results = []
        successful_analyses = 0
        failed_analyses = 0

        # M1-OPTIMIZED: Enhanced parallel processing
        system_info = detect_m1_optimization()  # Get system info

        if num_processes is None:
            # Auto-detect optimal processes using our M1 detection
            num_processes = system_info['optimal_processes']

        print(f"🚀 M1-Optimized: Using {num_processes} parallel processes (CPU cores: {mp.cpu_count()})")

        # M1-OPTIMIZED: Use parallel processing for smaller datasets too
        if len(symbols) > 50 and num_processes > 1:  # Lowered threshold from 100

            # M1-OPTIMIZED: Dynamic chunk sizing based on system capabilities
            system_info = detect_m1_optimization()
            base_chunk_size = max(5, len(symbols) // (num_processes * system_info['chunk_multiplier']))
            optimal_chunk_size = min(base_chunk_size, 50)  # Cap at 50 for memory efficiency

            if system_info['is_m1']:
                print(f"💪 M1 Turbo Mode: {num_processes} workers, {optimal_chunk_size} symbols per chunk")
            else:
                print(f"💻 Parallel Mode: {num_processes} workers, {optimal_chunk_size} symbols per chunk")

            # Create more evenly distributed chunks
            symbol_chunks = []
            for i in range(0, len(symbols), optimal_chunk_size):
                chunk = symbols[i:i + optimal_chunk_size]
                if chunk:  # Only add non-empty chunks
                    symbol_chunks.append(chunk)

            print(f"📦 Created {len(symbol_chunks)} processing chunks")

            # M1-OPTIMIZED: Use ProcessPoolExecutor for better M1 performance
            with ThreadPoolExecutor(max_workers=num_processes) as executor:
                futures = []

                # Submit all tasks
                for chunk_idx, chunk in enumerate(symbol_chunks):
                    for symbol_idx, symbol in enumerate(chunk):
                        # Auto-detect market type
                        if market_type == MarketType.BOTH:
                            if any(symbol.endswith(suffix) for suffix in ['_USDT', 'USDT', 'BTC', 'ETH']):
                                sym_market = MarketType.CRYPTO
                            elif self.crypto_path and (self.crypto_path / f"{symbol}.csv").exists():
                                sym_market = MarketType.CRYPTO
                            else:
                                sym_market = MarketType.CHINESE_STOCK
                        else:
                            sym_market = market_type

                        future = executor.submit(self.analyze_symbol, symbol, sym_market)
                        futures.append((future, symbol, sym_market, chunk_idx))

                print(f"🔄 Submitted {len(futures)} analysis tasks to M1 processor")

                # M1-OPTIMIZED: Collect results with enhanced progress tracking and memory management
                completed_count = 0
                batch_results = []  # Collect in batches for memory efficiency

                for future, symbol, sym_market, chunk_idx in futures:
                    try:
                        # M1-OPTIMIZED: Reduced timeout for faster processing
                        analysis = future.result(timeout=15 if system_info['is_m1'] else 20)
                        completed_count += 1

                        if analysis:
                            batch_results.append(analysis)
                            successful_analyses += 1

                            # M1-OPTIMIZED: Process in batches to manage memory
                            if len(batch_results) >= 100:
                                results.extend(batch_results)
                                batch_results = []  # Clear batch to free memory
                        else:
                            failed_analyses += 1

                        # Enhanced progress tracking with M1 performance metrics
                        if completed_count % 50 == 0:
                            progress_pct = (completed_count / len(futures)) * 100
                            if system_info['is_m1']:
                                print(
                                    f"🚀 M1 Turbo: {completed_count}/{len(futures)} ({progress_pct:.1f}%) | ✅ {successful_analyses} | ❌ {failed_analyses}")
                            else:
                                print(
                                    f"💻 Progress: {completed_count}/{len(futures)} ({progress_pct:.1f}%) | ✅ {successful_analyses} | ❌ {failed_analyses}")

                    except Exception as e:
                        failed_analyses += 1
                        completed_count += 1
                        if completed_count % 100 == 0:  # Less frequent error logging
                            logger.debug(f"Analysis failed for {symbol}: {e}")
                        continue

                # Add any remaining batch results
                if batch_results:
                    results.extend(batch_results)
        else:
            # Sequential processing for very small datasets
            print(f"📝 Sequential processing for {len(symbols)} symbols")
            for i, symbol in enumerate(symbols):
                if i % 50 == 0:
                    progress_pct = (i / len(symbols)) * 100
                    print(
                        f"Progress: {i}/{len(symbols)} ({progress_pct:.1f}%) | ✅ {successful_analyses} | ❌ {failed_analyses}")

                # Auto-detect market type
                if market_type == MarketType.BOTH:
                    if any(symbol.endswith(suffix) for suffix in ['_USDT', 'USDT', 'BTC', 'ETH']):
                        sym_market = MarketType.CRYPTO
                    elif self.crypto_path and (self.crypto_path / f"{symbol}.csv").exists():
                        sym_market = MarketType.CRYPTO
                    else:
                        sym_market = MarketType.CHINESE_STOCK
                else:
                    sym_market = market_type

                analysis = self.analyze_symbol(symbol, sym_market)
                if analysis:
                    results.append(analysis)
                    successful_analyses += 1
                else:
                    failed_analyses += 1

        print(f"\n📊 M1-OPTIMIZED ANALYSIS SUMMARY:")
        print(f"   📈 Total Processed: {len(symbols)}")
        print(f"   ✅ Successful: {successful_analyses}")
        print(f"   ❌ Failed/Filtered: {failed_analyses}")
        print(f"   📊 Success Rate: {(successful_analyses / len(symbols) * 100):.1f}%")

        # Performance metrics
        if system_info['is_m1']:
            print(f"   🚀 M1 Performance: {num_processes} cores utilized")
            print(f"   ⚡ M1 Efficiency: {system_info['chunk_multiplier']}x chunk optimization")
        else:
            print(f"   💻 Standard Performance: {num_processes} threads utilized")

        # Enhanced sorting with multiple criteria
        results.sort(key=lambda x: (
                x.overall_score * 0.3 +
                x.prediction_confidence * 0.25 +
                x.game_score * 0.1 +
                len(x.best_combination.patterns) * 15 +
                (100 if x.special_pattern else 0) +
                (50 if x.maturity == PatternMaturity.IMMINENT else 0)
        ), reverse=True)

        self.results = results
        return results

    def get_predictions(self, top_n: int = 50) -> List[MultiTimeframeAnalysis]:
        """Enhanced prediction selection"""
        # Multi-tier filtering
        tier_1 = [r for r in self.results if
                  r.special_pattern and r.maturity in [PatternMaturity.READY, PatternMaturity.EXPLOSIVE,
                                                       PatternMaturity.IMMINENT]]
        tier_2 = [r for r in self.results if r.overall_score >= 70 and r.prediction_confidence >= 70]
        tier_3 = [r for r in self.results if r.overall_score >= 60 and len(r.best_combination.patterns) >= 3]
        tier_4 = [r for r in self.results if r.overall_score >= 50]

        # Combine tiers with limits
        predictions = []
        predictions.extend(tier_1[:15])  # Top special patterns
        predictions.extend([r for r in tier_2 if r not in predictions][:15])  # High quality
        predictions.extend([r for r in tier_3 if r not in predictions][:10])  # Good patterns
        predictions.extend([r for r in tier_4 if r not in predictions][:10])  # Decent setups

        return predictions[:top_n]

    def print_results(self):
        """Enhanced results display"""
        if not self.results:
            print("\n❌ No patterns found")
            self._print_troubleshooting_guide()
            return

        print("\n" + "=" * 140)
        print("🎮 PROFESSIONAL PATTERN ANALYZER v6.0 - ANALYSIS COMPLETE!")
        print("=" * 140)

        # Market breakdown
        crypto_results = [r for r in self.results if r.market_type == MarketType.CRYPTO]
        stock_results = [r for r in self.results if r.market_type == MarketType.CHINESE_STOCK]

        print(f"\n📊 MARKET BREAKDOWN:")
        print(f"   🪙 Cryptocurrency: {len(crypto_results)} opportunities")
        print(f"   🏮 Chinese Stocks: {len(stock_results)} opportunities")
        print(f"   📈 Total Analyzed: {len(self.results)} patterns")

        # Advanced pattern statistics
        self._print_advanced_pattern_stats()

        # Special pattern showcase
        self._print_special_patterns()

        # Professional grade analysis
        self._print_professional_analysis()

        # Top predictions with enhanced display
        self._print_enhanced_predictions()

        # Performance analytics
        self._print_performance_analytics()

    def _print_troubleshooting_guide(self):
        """Print troubleshooting information"""
        print("\n💡 TROUBLESHOOTING GUIDE:")
        print("   • Check data file paths and formats")
        print(f"   • Crypto path: {self.crypto_path}")
        print(f"   • Stock paths: {list(self.stock_paths.values())}")
        print("   • Ensure minimum 60-120 days of data per symbol")
        print("   • Try lowering quality thresholds in settings")
        print("   • Check for data file corruption or format issues")

    def _print_advanced_pattern_stats(self):
        """Print advanced pattern statistics"""
        print("\n🔬 ADVANCED PATTERN ANALYSIS:")

        # Count advanced patterns
        advanced_pattern_counts = defaultdict(int)
        for result in self.results:
            for patterns in result.timeframe_patterns.values():
                for pattern in patterns:
                    if pattern.value in [p.value for p in [
                        PatternType.HIDDEN_ACCUMULATION, PatternType.SMART_MONEY_FLOW,
                        PatternType.WHALE_ACCUMULATION, PatternType.COILED_SPRING,
                        PatternType.ELLIOTT_WAVE_3, PatternType.FIBONACCI_RETRACEMENT,
                        PatternType.BOLLINGER_SQUEEZE, PatternType.ICHIMOKU_CLOUD
                    ]]:
                        advanced_pattern_counts[pattern.value] += 1

        if advanced_pattern_counts:
            print("   🧠 Top Advanced Patterns:")
            for pattern, count in sorted(advanced_pattern_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
                print(f"      {pattern}: {count} occurrences")

        # Maturity distribution
        maturity_counts = defaultdict(int)
        for result in self.results:
            maturity_counts[result.maturity.value] += 1

        print(f"\n   📊 Pattern Maturity Distribution:")
        for maturity, count in maturity_counts.items():
            print(f"      {maturity}: {count}")

    def _print_special_patterns(self):
        """Print special pattern combinations"""
        print("\n🔥 SPECIAL PATTERN COMBINATIONS:")

        special_results = [r for r in self.results if r.special_pattern]

        if special_results:
            pattern_groups = defaultdict(list)
            for r in special_results:
                pattern_groups[r.special_pattern].append(r.symbol)

            for pattern_name, symbols in pattern_groups.items():
                unique_symbols = list(set(symbols))[:12]
                print(f"\n{pattern_name}:")
                print(f"   🎯 Opportunities: {', '.join(unique_symbols)}")

                # Show best example
                best_example = max([r for r in special_results if r.special_pattern == pattern_name],
                                   key=lambda x: x.overall_score)
                print(f"   ⭐ Best Setup: {best_example.symbol} (Score: {best_example.overall_score:.1f})")
        else:
            print("   No special pattern combinations detected in this scan")

    def _print_professional_analysis(self):
        """Print professional-grade analysis"""
        print("\n🏆 PROFESSIONAL GRADE ANALYSIS:")

        # Grade distribution
        professional_grades = []
        for result in self.results:
            for stage_result in result.stage_results:
                if stage_result.stage_name == "Professional Grade":
                    professional_grades.append(stage_result.score)
                    break

        if professional_grades:
            avg_grade = np.mean(professional_grades)
            max_grade = max(professional_grades)

            print(f"   📊 Average Professional Grade: {avg_grade:.1f}")
            print(f"   🏆 Highest Professional Grade: {max_grade:.1f}")

            # Grade categories
            institutional_count = sum(1 for g in professional_grades if g >= 85)
            professional_count = sum(1 for g in professional_grades if 70 <= g < 85)
            intermediate_count = sum(1 for g in professional_grades if 55 <= g < 70)

            print(f"   👑 Institutional Grade: {institutional_count}")
            print(f"   🏆 Professional Grade: {professional_count}")
            print(f"   ⭐ Intermediate Grade: {intermediate_count}")

    def _print_enhanced_predictions(self):
        """Print enhanced predictions table"""
        print("\n🔮 TOP 30 PROFESSIONAL PREDICTIONS:")

        predictions = self.get_predictions(30)

        if predictions:
            headers = [
                "Rank", "Symbol", "Market", "Score", "Conf%", "Special",
                "Maturity", "R/R", "Risk", "Entry", "Target", "Patterns"
            ]

            rows = []
            for i, pred in enumerate(predictions, 1):
                pattern_count = sum(len(p) for p in pred.timeframe_patterns.values())

                rows.append([
                    i,
                    pred.symbol[:12],
                    "Crypto" if pred.market_type == MarketType.CRYPTO else "Stock",
                    f"{pred.overall_score:.1f}",
                    f"{pred.prediction_confidence:.0f}",
                    pred.special_pattern[:12] if pred.special_pattern else "-",
                    pred.maturity.value[:8],
                    pred.entry_setup.get('risk_reward_1', 0),
                    pred.risk_assessment.get('overall_level', 'MED')[:3],
                    pred.entry_setup['entry'],
                    pred.entry_setup['target1'],
                    pattern_count
                ])

            print(tabulate(rows, headers=headers, tablefmt="grid"))

            # Detailed top 5
            print("\n⭐ TOP 5 DETAILED ANALYSIS:")
            for i, pred in enumerate(predictions[:5], 1):
                self._print_detailed_analysis(pred, i)

    def _print_detailed_analysis(self, pred: MultiTimeframeAnalysis, rank: int):
        """Print detailed analysis for a single prediction"""
        print(f"\n{rank}. {pred.symbol} - {pred.rarity_level}")
        print(f"   📊 Market: {pred.market_type.value}")
        print(f"   🎯 Overall Score: {pred.overall_score:.1f} | Confidence: {pred.prediction_confidence:.1f}%")
        print(f"   🔥 Game Score: {pred.game_score:.0f}")

        if pred.special_pattern:
            print(f"   ⚡ SPECIAL: {pred.special_pattern}")

        print(f"   📈 Maturity: {pred.maturity.value} | Consolidation: {pred.consolidation_type.value}")
        print(
            f"   💰 Entry: {pred.entry_setup['entry']} → Target: {pred.entry_setup['target1']} (R/R: {pred.entry_setup.get('risk_reward_1', 0)})")
        print(f"   ⚠️  Risk Level: {pred.risk_assessment.get('overall_level', 'UNKNOWN')}")
        print(f"   🎮 Visual: {pred.visual_summary}")

        # Show top patterns
        all_patterns = []
        for patterns in pred.timeframe_patterns.values():
            all_patterns.extend([p.value for p in patterns])

        if all_patterns:
            top_patterns = list(set(all_patterns))[:4]
            print(f"   📊 Key Patterns: {', '.join(top_patterns)}")

        # Show professional metrics
        professional_stage = next((r for r in pred.stage_results if r.stage_name == "Professional Grade"), None)
        if professional_stage:
            grade_level = professional_stage.details.get('grade_level', 'UNKNOWN')
            print(f"   🏆 Professional Grade: {grade_level} ({professional_stage.score:.1f})")

    def _print_performance_analytics(self):
        """Print performance analytics"""
        print("\n📊 PERFORMANCE ANALYTICS:")

        # Risk-reward distribution
        risk_rewards = [r.entry_setup.get('risk_reward_1', 0) for r in self.results]
        if risk_rewards:
            avg_rr = np.mean(risk_rewards)
            good_rr_count = sum(1 for rr in risk_rewards if rr >= 2.0)
            excellent_rr_count = sum(1 for rr in risk_rewards if rr >= 3.0)

            print(f"   📈 Average Risk/Reward: {avg_rr:.2f}")
            print(f"   ✅ Good R/R Setups (≥2.0): {good_rr_count}")
            print(f"   🏆 Excellent R/R Setups (≥3.0): {excellent_rr_count}")

        # Confidence distribution
        confidences = [r.prediction_confidence for r in self.results]
        if confidences:
            avg_confidence = np.mean(confidences)
            high_confidence = sum(1 for c in confidences if c >= 80)

            print(f"   🎯 Average Confidence: {avg_confidence:.1f}%")
            print(f"   🔥 High Confidence Setups (≥80%): {high_confidence}")

        # Pattern diversity
        unique_patterns = set()
        for result in self.results:
            for patterns in result.timeframe_patterns.values():
                unique_patterns.update(p.value for p in patterns)

        print(f"   🎨 Pattern Diversity: {len(unique_patterns)} unique patterns detected")

    def save_results(self, filename: str = None):
        """Enhanced results saving"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"professional_analysis_v6_{timestamp}.json"

        output_path = Path("professional_results") / filename
        output_path.parent.mkdir(exist_ok=True)

        def convert_numpy(obj):
            """Enhanced numpy conversion"""
            if isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                return convert_numpy(obj.__dict__)
            return obj

        # Enhanced data structure
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '6.0',
                'analyzer': 'Professional Pattern Analyzer',
                'game_message': random.choice(GAME_MESSAGES),
                'total_symbols_analyzed': len(self.results),
                'crypto_count': len([r for r in self.results if r.market_type == MarketType.CRYPTO]),
                'stock_count': len([r for r in self.results if r.market_type == MarketType.CHINESE_STOCK])
            },
            'market_analysis': [],
            'special_patterns': defaultdict(list),
            'professional_grades': {},
            'performance_metrics': {},
            'top_predictions': []
        }

        # Special patterns tracking
        for result in self.results:
            if result.special_pattern:
                data['special_patterns'][result.special_pattern].append({
                    'symbol': result.symbol,
                    'score': result.overall_score,
                    'confidence': result.prediction_confidence
                })

        # Professional grades
        professional_grades = []
        for result in self.results:
            for stage_result in result.stage_results:
                if stage_result.stage_name == "Professional Grade":
                    professional_grades.append(stage_result.score)
                    break

        if professional_grades:
            data['professional_grades'] = {
                'average': np.mean(professional_grades),
                'maximum': max(professional_grades),
                'institutional_count': sum(1 for g in professional_grades if g >= 85),
                'professional_count': sum(1 for g in professional_grades if 70 <= g < 85)
            }

        # Performance metrics
        risk_rewards = [r.entry_setup.get('risk_reward_1', 0) for r in self.results]
        confidences = [r.prediction_confidence for r in self.results]

        data['performance_metrics'] = {
            'average_risk_reward': np.mean(risk_rewards) if risk_rewards else 0,
            'good_risk_reward_count': sum(1 for rr in risk_rewards if rr >= 2.0),
            'average_confidence': np.mean(confidences) if confidences else 0,
            'high_confidence_count': sum(1 for c in confidences if c >= 80)
        }

        # Save top results with full details
        for result in self.results[:100]:
            result_data = {
                'symbol': result.symbol,
                'market_type': result.market_type.value,
                'overall_score': float(result.overall_score),
                'prediction_confidence': float(result.prediction_confidence),
                'game_score': float(result.game_score),
                'rarity_level': result.rarity_level,
                'special_pattern': result.special_pattern,
                'maturity': result.maturity.value,
                'consolidation_type': result.consolidation_type.value,
                'visual_summary': result.visual_summary,
                'entry_setup': convert_numpy(result.entry_setup),
                'risk_assessment': convert_numpy(result.risk_assessment),
                'technical_indicators': convert_numpy(result.technical_indicators),
                'timeframe_patterns': {
                    str(tf.value): [p.value for p in patterns]
                    for tf, patterns in result.timeframe_patterns.items()
                },
                'stage_results': [
                    convert_numpy({
                        'stage': r.stage_name,
                        'passed': r.passed,
                        'score': r.score,
                        'confidence': r.confidence,
                        'indicator': r.visual_indicator,
                        'details': r.details
                    })
                    for r in result.stage_results
                ],
                'best_combination': convert_numpy({
                    'patterns': [p.value for p in result.best_combination.patterns],
                    'score': result.best_combination.combined_score,
                    'confidence': result.best_combination.confidence,
                    'risk_level': result.best_combination.risk_level,
                    'expected_gain': result.best_combination.expected_gain
                })
            }
            data['market_analysis'].append(convert_numpy(result_data))

        # Convert defaultdict to regular dict
        data['special_patterns'] = dict(data['special_patterns'])

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Professional analysis saved to: {output_path}")
        print(f"📊 Saved {len(data['market_analysis'])} detailed analyses")


def main():
    """Enhanced main execution with M1 optimization"""
    # Detect system capabilities
    system_info = detect_m1_optimization()

    print("\n" + "=" * 100)
    print("   🎮 PROFESSIONAL PATTERN ANALYZER v6.1")
    print("   🧠 Advanced AI-Powered Trading Analysis")
    print("   💎 25+ New Patterns | Enhanced Risk Management")
    print("   🟦 NEW: TradingView Triple Support Detection!")
    if system_info['is_m1']:
        print("   🚀 M1/M2 MacBook Optimization ACTIVE!")
    print("=" * 100)

    print(f"\n{random.choice(GAME_MESSAGES)}")

    # Show system info
    print(f"\n💻 System Detection:")
    print(f"   🖥️  Platform: {system_info['platform']}")
    print(f"   🔧 Processor: {system_info['processor'] or system_info['machine']}")
    print(f"   ⚙️  CPU Cores: {system_info['cpu_count']}")
    if system_info['is_m1']:
        print(f"   🚀 M1/M2 Detected: Optimal processes = {system_info['optimal_processes']}")
    else:
        print(f"   💻 Intel/AMD: Optimal processes = {system_info['optimal_processes']}")

    print("\n📊 Select Your Professional Quest:")
    print("1. 🏮 Chinese A-Shares Professional Analysis")
    print("2. 🪙 Cryptocurrency Advanced Scanning")
    print("3. 🌍 Global Market Domination (Both)")
    print("4. 🎯 Quick Scan (Limited Symbols)")
    print("5. 🚀 Full Professional Scan (All Symbols)")

    choice = input("\nChoose your trading destiny (1-5, default=3): ").strip() or '3'

    market_map = {
        '1': MarketType.CHINESE_STOCK,
        '2': MarketType.CRYPTO,
        '3': MarketType.BOTH,
        '4': MarketType.BOTH,
        '5': MarketType.BOTH
    }

    market_type = market_map.get(choice, MarketType.BOTH)

    # M1-OPTIMIZED: Set limits based on choice and system capabilities
    if choice == '4':
        max_symbols = 300
        num_processes = max(4, system_info['optimal_processes'] // 2)  # Conservative for quick scan
    elif choice == '5':
        max_symbols = None
        num_processes = system_info['optimal_processes']  # Use all available power
    else:
        max_symbols = None
        num_processes = system_info['optimal_processes']  # Use optimal for system

    analyzer = ProfessionalPatternAnalyzer()

    print(f"\n🎮 Professional Configuration:")
    print(f"   🌍 Market: {market_type.value}")
    print(f"   🎯 Patterns: 50+ types (25+ advanced)")
    print(f"   ⏱️ Timeframes: 8 levels (including Fibonacci)")
    print(f"   🏆 Stages: 10 professional challenges")
    print(f"   🚫 Blacklist: Dynamic + Static")
    print(f"   🔥 Special Combos: {len(PROFESSIONAL_PATTERNS)} professional patterns")
    print(f"   🧠 AI Features: ML-inspired analysis")
    if system_info['is_m1']:
        print(f"   🚀 M1 Processing: {num_processes} threads (OPTIMIZED)")
    else:
        print(f"   ⚡ Processing: {num_processes} threads")
    print(f"   🟦 NEW: TradingView Triple Support Stable Pattern!")

    # Performance warning with M1 consideration
    if max_symbols is None or max_symbols > 1000:
        if system_info['is_m1']:
            estimated_time = "5-15 minutes (M1 optimized)"
        else:
            estimated_time = "10-30 minutes"
        print(f"\n⚠️  WARNING: Full scan may take {estimated_time}")
        proceed = input("Continue? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Analysis cancelled.")
            return

    start_time = datetime.now()

    try:
        results = analyzer.run_analysis(market_type, max_symbols, num_processes)

        analysis_time = datetime.now() - start_time
        symbols_per_second = len(analyzer.get_all_symbols(market_type)) / analysis_time.total_seconds()

        print(f"\n⏱️  Analysis completed in {analysis_time}")
        if system_info['is_m1']:
            print(f"🚀 M1 Performance: {symbols_per_second:.1f} symbols/second")
        else:
            print(f"💻 Performance: {symbols_per_second:.1f} symbols/second")

        analyzer.print_results()

        save_choice = input("\n💾 Save professional analysis? (y/n): ").strip().lower()
        if save_choice == 'y':
            analyzer.save_results()

        print(f"\n🎮 PROFESSIONAL ANALYSIS COMPLETE!")
        print(f"🏆 Discovered {len(results)} professional-grade opportunities!")

        if results:
            crypto_count = sum(1 for r in results if r.market_type == MarketType.CRYPTO)
            stock_count = sum(1 for r in results if r.market_type == MarketType.CHINESE_STOCK)
            special_count = sum(1 for r in results if r.special_pattern)
            imminent_count = sum(1 for r in results if r.maturity == PatternMaturity.IMMINENT)
            tradingview_count = sum(1 for r in results
                                    for patterns in r.timeframe_patterns.values()
                                    for p in patterns
                                    if 'TradingView' in p.value)

            print(f"   🪙 Crypto Opportunities: {crypto_count}")
            print(f"   🏮 Stock Opportunities: {stock_count}")
            print(f"   🔥 Special Patterns: {special_count}")
            print(f"   ⚡ Imminent Setups: {imminent_count}")
            print(f"   🟦 TradingView Patterns: {tradingview_count}")

            # Risk summary
            low_risk = sum(1 for r in results if r.risk_assessment.get('overall_level') == 'LOW')
            medium_risk = sum(1 for r in results if r.risk_assessment.get('overall_level') == 'MEDIUM')
            high_risk = sum(1 for r in results if r.risk_assessment.get('overall_level') == 'HIGH')

            print(f"   🟢 Low Risk: {low_risk}")
            print(f"   🟡 Medium Risk: {medium_risk}")
            print(f"   🔴 High Risk: {high_risk}")

        print(f"\n⭐ Return tomorrow for fresh professional insights!")
        if system_info['is_m1']:
            print(f"🚀 Your M1 MacBook delivered exceptional performance!")
        print(f"💰 Trade wisely and prosper!")

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Analysis paused by user")
        print(f"💾 Partial results may be available")
    except Exception as e:
        print(f"\n❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🎮 Professional analysis paused! Your trading edge awaits!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Professional analyzer crashed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
