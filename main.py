#!/usr/bin/env python3
"""
Professional Pattern Analyzer v6.1 - Main Body
Auto-Discovery Modular Trading Analysis System
ENHANCED with Better Error Handling and Initialization
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
import platform
import importlib
import inspect
import glob
import traceback

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================== CORE CONFIGURATION ==================
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
    "📊 TRADINGVIEW MASTER": ["TradingView Triple Support", "Volume Confirmation", "Multiple Timeframes"],
    "🟦 TRIPLE SUPPORT KING": ["Triple Support Stable", "Support", "MACD"],
}

# ================== CORE DATA STRUCTURES ==================
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
    D55 = 55
    D89 = 89


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


# ================== AUTO-DISCOVERY SYSTEM - ENHANCED ==================
class ComponentRegistry:
    """Enhanced auto-discovery registry for all components with better error handling"""
    
    def __init__(self):
        self.pattern_detectors = {}
        self.stage_analyzers = {}
        self.pattern_analyzers = {}
        self.timeframe_analyzers = {}
        self.initialization_errors = []
        self.discover_all_components()
    
    def discover_all_components(self):
        """Auto-discover all components from folders with enhanced error handling"""
        logger.info("🔍 Starting component auto-discovery...")
        
        self.discover_pattern_detectors()
        self.discover_stage_analyzers()
        self.discover_pattern_analyzers()
        self.discover_timeframe_analyzers()
        
        # Log summary
        total_components = (len(self.pattern_detectors) + len(self.stage_analyzers) + 
                          len(self.pattern_analyzers) + len(self.timeframe_analyzers))
        
        if self.initialization_errors:
            logger.warning(f"⚠️  Component discovery completed with {len(self.initialization_errors)} errors")
            for error in self.initialization_errors:
                logger.debug(f"   Error: {error}")
        else:
            logger.info(f"✅ Component discovery completed successfully - {total_components} components loaded")
    
    def discover_pattern_detectors(self):
        """Auto-discover pattern detectors with enhanced error handling"""
        try:
            folder_path = Path("pattern_detectors")
            if not folder_path.exists():
                self.initialization_errors.append("pattern_detectors folder not found")
                return
                
            for file_path in folder_path.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                
                module_name = f"pattern_detectors.{file_path.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all detector classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, 'detect_patterns') and name != 'BasePatternDetector':
                            self.pattern_detectors[name] = obj
                            logger.info(f"🔍 Auto-discovered pattern detector: {name}")
                except Exception as e:
                    error_msg = f"Failed to load pattern detector {file_path.name}: {e}"
                    self.initialization_errors.append(error_msg)
                    logger.debug(error_msg)
        except Exception as e:
            error_msg = f"Error in pattern detector discovery: {e}"
            self.initialization_errors.append(error_msg)
            logger.warning(error_msg)
    
    def discover_stage_analyzers(self):
        """Auto-discover stage analyzers with enhanced error handling"""
        try:
            folder_path = Path("stage_analyzers")
            if not folder_path.exists():
                self.initialization_errors.append("stage_analyzers folder not found")
                return
                
            for file_path in folder_path.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                
                module_name = f"stage_analyzers.{file_path.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all analyzer classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, 'run_all_stages') and name != 'BaseStageAnalyzer':
                            self.stage_analyzers[name] = obj
                            logger.info(f"📊 Auto-discovered stage analyzer: {name}")
                except Exception as e:
                    error_msg = f"Failed to load stage analyzer {file_path.name}: {e}"
                    self.initialization_errors.append(error_msg)
                    logger.debug(error_msg)
        except Exception as e:
            error_msg = f"Error in stage analyzer discovery: {e}"
            self.initialization_errors.append(error_msg)
            logger.warning(error_msg)
    
    def discover_pattern_analyzers(self):
        """Auto-discover pattern analyzers with enhanced error handling"""
        try:
            folder_path = Path("pattern_analyzers")
            if not folder_path.exists():
                self.initialization_errors.append("pattern_analyzers folder not found")
                return
                
            for file_path in folder_path.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                
                module_name = f"pattern_analyzers.{file_path.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all analyzer classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, 'analyze_symbol') and name != 'BasePatternAnalyzer':
                            self.pattern_analyzers[name] = obj
                            logger.info(f"🔬 Auto-discovered pattern analyzer: {name}")
                except Exception as e:
                    error_msg = f"Failed to load pattern analyzer {file_path.name}: {e}"
                    self.initialization_errors.append(error_msg)
                    logger.debug(error_msg)
        except Exception as e:
            error_msg = f"Error in pattern analyzer discovery: {e}"
            self.initialization_errors.append(error_msg)
            logger.warning(error_msg)
    
    def discover_timeframe_analyzers(self):
        """Auto-discover timeframe analyzers with enhanced error handling"""
        try:
            folder_path = Path("timeframe_analyzers")
            if not folder_path.exists():
                self.initialization_errors.append("timeframe_analyzers folder not found (optional)")
                return
                
            for file_path in folder_path.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                
                module_name = f"timeframe_analyzers.{file_path.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all analyzer classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, 'analyze_timeframe') and name != 'BaseTimeframeAnalyzer':
                            self.timeframe_analyzers[name] = obj
                            logger.info(f"⏰ Auto-discovered timeframe analyzer: {name}")
                except Exception as e:
                    error_msg = f"Failed to load timeframe analyzer {file_path.name}: {e}"
                    self.initialization_errors.append(error_msg)
                    logger.debug(error_msg)
        except Exception as e:
            error_msg = f"Error in timeframe analyzer discovery: {e}"
            self.initialization_errors.append(error_msg)
            logger.debug(error_msg)  # Debug level since timeframe analyzers are optional
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of component discovery"""
        return {
            'pattern_detectors': len(self.pattern_detectors),
            'stage_analyzers': len(self.stage_analyzers),
            'pattern_analyzers': len(self.pattern_analyzers),
            'timeframe_analyzers': len(self.timeframe_analyzers),
            'total_components': (len(self.pattern_detectors) + len(self.stage_analyzers) + 
                               len(self.pattern_analyzers) + len(self.timeframe_analyzers)),
            'initialization_errors': len(self.initialization_errors),
            'error_details': self.initialization_errors
        }


# ================== BLACKLIST MANAGER ==================
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
            if avg_performance < -0.1:
                self.dynamic_blacklist.add(symbol)
                logger.info(f"Auto-blacklisted {symbol} for poor performance")


# ================== LEARNING SYSTEM ==================
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
                win_rate = perf['win_rate']
                avg_gain = np.mean(perf['avg_gain']) if perf['avg_gain'] else 0
                max_gain = perf['max_gain']
                volatility = perf['volatility']

                score = (win_rate * 0.4 +
                         min(avg_gain / 100, 0.3) * 0.3 +
                         min(max_gain / 200, 0.2) * 0.2 +
                         max(0, (1 - volatility / 50)) * 0.1)

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


# ================== SYSTEM OPTIMIZATION ==================
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

    if (platform.system() == 'Darwin' and
            (platform.machine() == 'arm64' or 'Apple' in platform.processor())):
        system_info['is_m1'] = True

        cpu_count = mp.cpu_count()
        if cpu_count >= 10:
            system_info['optimal_processes'] = min(12, cpu_count - 2)
            system_info['chunk_multiplier'] = 3
        elif cpu_count >= 8:
            system_info['optimal_processes'] = min(10, cpu_count - 1)
            system_info['chunk_multiplier'] = 2
        else:
            system_info['optimal_processes'] = min(8, cpu_count)
            system_info['chunk_multiplier'] = 2
    else:
        cpu_count = mp.cpu_count()
        system_info['optimal_processes'] = min(6, cpu_count - 1)
        system_info['chunk_multiplier'] = 1

    return system_info


# ================== MAIN ORCHESTRATOR - ENHANCED ==================
class ProfessionalTradingOrchestrator:
    """Enhanced main orchestrator that coordinates all components with better error handling"""
    
    def __init__(self, data_dir: str = "Chinese_Market/data"):
        self.data_dir = Path(data_dir)
        self.component_registry = ComponentRegistry()
        self.blacklist_manager = EnhancedBlacklistManager()
        self.learning_system = ProfessionalLearningSystem()
        self.results = []
        self.initialization_status = {}
        
        # Initialize components with enhanced error handling
        self._initialize_components()
        
        # Setup data paths
        self.stock_paths = {
            'shanghai': self.data_dir / 'shanghai_6xx',
            'shenzhen': self.data_dir / 'shenzhen_0xx',
            'beijing': self.data_dir / 'beijing_8xx'
        }
        
        self.crypto_paths = [
            self.data_dir / 'huobi' / 'spot_usdt' / '1d',
            self.data_dir / 'huobi' / 'csv' / 'spot',
            self.data_dir / 'csv' / 'huobi' / 'spot',
            self.data_dir / 'crypto' / 'spot',
            self.data_dir / 'binance' / 'spot'
        ]
        
        self.crypto_path = self._find_crypto_path()
    
    def _initialize_components(self):
        """Initialize all discovered components with enhanced error handling"""
        print("\n🚀 INITIALIZING AUTO-DISCOVERED COMPONENTS:")
        
        discovery_summary = self.component_registry.get_discovery_summary()
        print(f"   🔍 Pattern Detectors: {discovery_summary['pattern_detectors']}")
        print(f"   📊 Stage Analyzers: {discovery_summary['stage_analyzers']}")
        print(f"   🔬 Pattern Analyzers: {discovery_summary['pattern_analyzers']}")
        print(f"   ⏰ Timeframe Analyzers: {discovery_summary['timeframe_analyzers']}")
        
        if discovery_summary['initialization_errors'] > 0:
            print(f"   ⚠️  Discovery Warnings: {discovery_summary['initialization_errors']}")
        
        # Initialize the main components with enhanced error handling
        self.stage_analyzer = None
        self.pattern_detector = None
        self.multi_tf_analyzer = None
        self.pattern_analyzer = None
        
        try:
            # Initialize Enhanced Stage Analyzer
            if 'EnhancedStageAnalyzer' in self.component_registry.stage_analyzers:
                analyzer_class = self.component_registry.stage_analyzers['EnhancedStageAnalyzer']
                self.stage_analyzer = analyzer_class(self.blacklist_manager)
                
                # CRITICAL FIX: Safe initialization after construction
                if hasattr(self.stage_analyzer, 'safe_initialize'):
                    init_success = self.stage_analyzer.safe_initialize()
                    self.initialization_status['stage_analyzer'] = 'SUCCESS' if init_success else 'PARTIAL'
                else:
                    self.initialization_status['stage_analyzer'] = 'SUCCESS'
                
                logger.info("✅ EnhancedStageAnalyzer initialized successfully")
            else:
                self.initialization_status['stage_analyzer'] = 'MISSING'
                logger.warning("⚠️  EnhancedStageAnalyzer not found")
            
            # Initialize Advanced Pattern Detector
            if 'AdvancedPatternDetector' in self.component_registry.pattern_detectors:
                detector_class = self.component_registry.pattern_detectors['AdvancedPatternDetector']
                self.pattern_detector = detector_class()
                self.initialization_status['pattern_detector'] = 'SUCCESS'
                logger.info("✅ AdvancedPatternDetector initialized successfully")
            else:
                self.initialization_status['pattern_detector'] = 'MISSING'
                logger.warning("⚠️  AdvancedPatternDetector not found")
            
            # Initialize Enhanced Multi-Timeframe Analyzer (optional)
            if 'EnhancedMultiTimeframeAnalyzer' in self.component_registry.timeframe_analyzers:
                tf_analyzer_class = self.component_registry.timeframe_analyzers['EnhancedMultiTimeframeAnalyzer']
                self.multi_tf_analyzer = tf_analyzer_class(self.learning_system, self.blacklist_manager)
                self.initialization_status['multi_tf_analyzer'] = 'SUCCESS'
                logger.info("✅ EnhancedMultiTimeframeAnalyzer initialized successfully")
            else:
                self.initialization_status['multi_tf_analyzer'] = 'MISSING'
                logger.debug("ℹ️  EnhancedMultiTimeframeAnalyzer not found (optional)")
            
            # Initialize Professional Pattern Analyzer
            if 'ProfessionalPatternAnalyzer' in self.component_registry.pattern_analyzers:
                analyzer_class = self.component_registry.pattern_analyzers['ProfessionalPatternAnalyzer']
                self.pattern_analyzer = analyzer_class(str(self.data_dir))
                self.initialization_status['pattern_analyzer'] = 'SUCCESS'
                logger.info("✅ ProfessionalPatternAnalyzer initialized successfully")
            else:
                self.initialization_status['pattern_analyzer'] = 'MISSING'
                logger.warning("⚠️  ProfessionalPatternAnalyzer not found")
        
        except Exception as e:
            error_msg = f"Error initializing components: {e}"
            logger.warning(error_msg)
            
            # Enhanced error reporting
            print("⚠️  Some components failed to initialize - using fallback mode")
            if logger.isEnabledFor(logging.DEBUG):
                print(f"   Debug info: {error_msg}")
                traceback.print_exc()
    
    def _find_crypto_path(self) -> Optional[Path]:
        """Find available crypto data path"""
        for path in self.crypto_paths:
            if path.exists():
                logger.info(f"Found crypto data at: {path}")
                return path
        logger.warning("No crypto data path found")
        return None
    
    def run_analysis(self, market_type: MarketType = MarketType.BOTH,
                     max_symbols: int = None, num_processes: int = None) -> List[MultiTimeframeAnalysis]:
        """Run the complete analysis using auto-discovered components with enhanced error handling"""
        
        print(f"\n🎮 {random.choice(GAME_MESSAGES)}")
        print(f"📊 Market: {market_type.value}")
        print(f"🧠 Learning from {len(self.learning_system.pattern_performance)} advanced patterns")
        
        # Check component initialization status
        failed_components = [name for name, status in self.initialization_status.items() 
                           if status in ['MISSING', 'FAILED']]
        
        if failed_components:
            print(f"⚠️  Warning: Some components not available: {', '.join(failed_components)}")
            print("   System will use fallback methods where possible")
        
        # Use the main pattern analyzer if available
        if hasattr(self, 'pattern_analyzer') and self.pattern_analyzer:
            try:
                return self.pattern_analyzer.run_analysis(market_type, max_symbols, num_processes)
            except Exception as e:
                logger.error(f"Pattern analyzer failed: {e}")
                print(f"❌ Pattern analysis failed: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    traceback.print_exc()
                return []
        else:
            print("❌ No pattern analyzer available - cannot run analysis")
            print("   Please ensure ProfessionalPatternAnalyzer is properly installed")
            return []
    
    def print_results(self):
        """Print results using available components"""
        if hasattr(self, 'pattern_analyzer') and self.pattern_analyzer:
            try:
                self.pattern_analyzer.print_results()
            except Exception as e:
                logger.error(f"Error printing results: {e}")
                print(f"❌ Error displaying results: {e}")
        else:
            print("❌ No results to display - pattern analyzer not available")
    
    def save_results(self, filename: str = None):
        """Save results using available components"""
        if hasattr(self, 'pattern_analyzer') and self.pattern_analyzer:
            try:
                self.pattern_analyzer.save_results(filename)
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                print(f"❌ Error saving results: {e}")
        else:
            print("❌ No results to save - pattern analyzer not available")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'component_discovery': self.component_registry.get_discovery_summary(),
            'initialization_status': self.initialization_status,
            'data_paths': {
                'crypto_path': str(self.crypto_path) if self.crypto_path else None,
                'stock_paths': {k: str(v) for k, v in self.stock_paths.items()}
            },
            'blacklist_stats': {
                'static_stocks': len(self.blacklist_manager.blacklisted_stocks),
                'static_crypto': len(self.blacklist_manager.blacklisted_crypto),
                'dynamic': len(self.blacklist_manager.dynamic_blacklist)
            },
            'learning_stats': {
                'patterns_tracked': len(self.learning_system.pattern_performance),
                'results_dir': str(self.learning_system.results_dir)
            }
        }


# ================== MAIN EXECUTION - ENHANCED ==================
def main():
    """Enhanced main execution with auto-discovery and better error handling"""
    system_info = detect_m1_optimization()

    print("\n" + "=" * 100)
    print("   🎮 PROFESSIONAL PATTERN ANALYZER v6.1")
    print("   🧠 Auto-Discovery Modular Trading Analysis")
    print("   💎 Unlimited Patterns | Enhanced Architecture")
    print("   🔌 Plugin System - Add Components Without Code Changes!")
    if system_info['is_m1']:
        print("   🚀 M1/M2 MacBook Optimization ACTIVE!")
    print("=" * 100)

    print(f"\n{random.choice(GAME_MESSAGES)}")

    # Initialize orchestrator with enhanced error handling
    try:
        orchestrator = ProfessionalTradingOrchestrator()
    except Exception as e:
        print(f"❌ Critical error initializing system: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        print("\nPlease check your installation and try again.")
        return

    # Show system info
    print(f"\n💻 System Detection:")
    print(f"   🖥️  Platform: {system_info['platform']}")
    print(f"   🔧 Processor: {system_info['processor'] or system_info['machine']}")
    print(f"   ⚙️  CPU Cores: {system_info['cpu_count']}")
    if system_info['is_m1']:
        print(f"   🚀 M1/M2 Detected: Optimal processes = {system_info['optimal_processes']}")
    else:
        print(f"   💻 Intel/AMD: Optimal processes = {system_info['optimal_processes']}")

    # Show system status
    try:
        status = orchestrator.get_system_status()
        if status['component_discovery']['initialization_errors'] > 0:
            print(f"\n⚠️  System Status: {status['component_discovery']['initialization_errors']} component warnings")
            print("   Analysis will continue with available components")
    except Exception as e:
        logger.debug(f"Error getting system status: {e}")

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

    if choice == '4':
        max_symbols = 300
        num_processes = max(4, system_info['optimal_processes'] // 2)
    elif choice == '5':
        max_symbols = None
        num_processes = system_info['optimal_processes']
    else:
        max_symbols = None
        num_processes = system_info['optimal_processes']

    print(f"\n🎮 Professional Configuration:")
    print(f"   🌍 Market: {market_type.value}")
    print(f"   🎯 Patterns: Auto-discovered from modules")
    print(f"   ⏱️ Timeframes: 8 levels (including Fibonacci)")
    print(f"   🏆 Stages: Auto-discovered analyzers")
    print(f"   🚫 Blacklist: Dynamic + Static")
    print(f"   🔥 Special Combos: {len(PROFESSIONAL_PATTERNS)} professional patterns")
    print(f"   🧠 AI Features: ML-inspired analysis")
    print(f"   🔌 Plugin System: Auto-discovery enabled")
    if system_info['is_m1']:
        print(f"   🚀 M1 Processing: {num_processes} threads (OPTIMIZED)")
    else:
        print(f"   ⚡ Processing: {num_processes} threads")

    # Performance warning
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
        results = orchestrator.run_analysis(market_type, max_symbols, num_processes)
        
        analysis_time = datetime.now() - start_time
        if system_info['is_m1']:
            print(f"🚀 M1 Performance: Analysis completed in {analysis_time}")
        else:
            print(f"💻 Performance: Analysis completed in {analysis_time}")

        if results:
            orchestrator.print_results()

            save_choice = input("\n💾 Save professional analysis? (y/n): ").strip().lower()
            if save_choice == 'y':
                orchestrator.save_results()

            print(f"\n🎮 MODULAR ANALYSIS COMPLETE!")
            print(f"🏆 Discovered {len(results)} professional-grade opportunities!")
            print(f"🔌 Plugin System: Ready for new components!")
        else:
            print(f"\n🔍 No opportunities found matching the criteria.")
            print(f"💡 Try adjusting analysis parameters or check data availability.")

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Analysis paused by user")
        print(f"💾 Partial results may be available")
    except Exception as e:
        print(f"\n❌ Analysis error: {e}")
        logger.error(f"Analysis failed: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        print("Please check logs for detailed error information.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🎮 Professional analysis paused! Your trading edge awaits!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Professional analyzer crashed: {e}")
        print(f"\n❌ Critical system error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        print("Please check your Python environment and dependencies.")
        sys.exit(1)
