"""
Professional Pattern Analyzer - COMPLETED
Main orchestrator for professional-grade trading analysis with auto-discovery system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import warnings
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import random
import multiprocessing as mp
from tabulate import tabulate

from .base_pattern_analyzer import (
    BasePatternAnalyzer, MultiTimeframeAnalysis, TimeFrame, 
    PatternType, ConsolidationType, PatternMaturity, StageResult, PatternCombination
)

# Import MarketType from main module to ensure enum consistency
try:
    from main import MarketType
except ImportError:
    # Fallback for when running as standalone
    from enum import Enum
    class MarketType(Enum):
        CHINESE_STOCK = "Chinese A-Share"
        CRYPTO = "Cryptocurrency"
        BOTH = "Both Markets"

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Enhanced blacklisted symbols
BLACKLISTED_STOCKS = {
    '002916', '002780', '002594', '002415', '002273',
    '000661', '000725', '000792', '000858', '000903',
    '600519', '600276', '600809', '600887', '600900',
    '603259', '603288', '603501', '603658', '603899',
}

BLACKLISTED_CRYPTO = {
    'SFUND_USDT', 'METIS_USDT', 'LUNA_USDT', 'UST_USDT',
    'FTT_USDT', 'CEL_USDT', 'LUNC_USDT', 'USTC_USDT',
}

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

# Game messages for user experience
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


class ProfessionalPatternAnalyzer(BasePatternAnalyzer):
    """
    Professional Pattern Analyzer - Main orchestrator for trading analysis
    Integrates all analysis components with auto-discovery support
    
    This is the main class that orchestrates the entire analysis process using
    auto-discovered components from different folders:
    - pattern_detectors/: For pattern detection algorithms
    - stage_analyzers/: For multi-stage analysis pipelines  
    - timeframe_analyzers/: For multi-timeframe analysis
    
    The auto-discovery system automatically finds and loads new components
    without requiring changes to this main code.
    """
    
    def __init__(self, data_dir: str = "Chinese_Market/data"):
        super().__init__(data_dir)
        self.version = "6.1"
        self.results = []  # Initialize results list
        
        # Initialize enhanced components
        self.blacklist_manager = EnhancedBlacklistManager()
        self.learning_system = ProfessionalLearningSystem()
        
        # Initialize auto-discovered components
        self._initialize_auto_discovered_components()
    
    def _initialize_auto_discovered_components(self):
        """Initialize auto-discovered components"""
        try:
            # Import components with auto-discovery
            from pattern_detectors import pattern_detectors
            from stage_analyzers import stage_analyzers
            
            # Try to import timeframe analyzers (may not exist yet)
            try:
                from timeframe_analyzers import timeframe_analyzers
            except ImportError:
                timeframe_analyzers = {}
                logger.warning("timeframe_analyzers module not found, using fallback methods")
            
            # Get the main components
            self.pattern_detector = None
            self.stage_analyzer = None
            self.multi_tf_analyzer = None
            
            # Initialize pattern detector
            if 'AdvancedPatternDetector' in pattern_detectors:
                self.pattern_detector = pattern_detectors['AdvancedPatternDetector']()
                logger.info("✅ AdvancedPatternDetector initialized")
            else:
                logger.warning("AdvancedPatternDetector not found in pattern_detectors")
            
            # Initialize stage analyzer
            if 'EnhancedStageAnalyzer' in stage_analyzers:
                self.stage_analyzer = stage_analyzers['EnhancedStageAnalyzer'](self.blacklist_manager)
                logger.info("✅ EnhancedStageAnalyzer initialized")
            else:
                logger.warning("EnhancedStageAnalyzer not found in stage_analyzers")
            
            # Initialize timeframe analyzer
            if 'EnhancedMultiTimeframeAnalyzer' in timeframe_analyzers:
                self.multi_tf_analyzer = timeframe_analyzers['EnhancedMultiTimeframeAnalyzer'](
                    self.learning_system, self.blacklist_manager)
                logger.info("✅ EnhancedMultiTimeframeAnalyzer initialized")
            else:
                logger.warning("EnhancedMultiTimeframeAnalyzer not found, using fallback methods")
            
            # Log discovered components
            logger.info(f"🔍 Pattern Detectors: {len(pattern_detectors)}")
            logger.info(f"📊 Stage Analyzers: {len(stage_analyzers)}")
            logger.info(f"⏰ Timeframe Analyzers: {len(timeframe_analyzers)}")
            
        except ImportError as e:
            logger.warning(f"Could not import auto-discovered components: {e}")
            logger.warning("Falling back to standalone mode")
    
    def get_all_symbols(self, market_type: MarketType) -> List[str]:
        """Get all available symbols for analysis with enhanced error handling"""
        symbols = []
        total_files_found = 0
        
        # Process Chinese stocks - Enhanced enum comparison for robustness
        is_chinese_market = (market_type in [MarketType.CHINESE_STOCK, MarketType.BOTH] or 
                           market_type.value in ["Chinese A-Share", "Both Markets"])
        
        if is_chinese_market:
            logger.info(f"🔍 Searching for Chinese stock data in: {self.data_dir}")
            
            for exchange, folder in self.stock_paths.items():
                logger.info(f"📁 Checking {exchange} path: {folder}")
                
                if folder.exists():
                    logger.info(f"✅ Found {exchange} directory: {folder}")
                    csv_files = list(folder.glob("*.csv"))
                    total_files_found += len(csv_files)
                    logger.info(f"📄 Found {len(csv_files)} CSV files in {exchange}")
                    
                    if len(csv_files) == 0:
                        logger.warning(f"⚠️  No CSV files found in {folder}")
                        # List what files are actually there
                        all_files = list(folder.glob("*"))
                        if all_files:
                            logger.info(f"📋 Files present: {[f.name for f in all_files[:5]]}")
                    else:
                        # Sample CSV files found
                        sample_files = [f.name for f in csv_files[:3]]
                        logger.info(f"📋 Sample CSV files: {sample_files}")
                    
                    for file_path in csv_files:
                        symbol = file_path.stem
                        is_blacklisted = self.blacklist_manager.is_blacklisted(symbol, MarketType.CHINESE_STOCK)
                        
                        if not is_blacklisted:
                            symbols.append(symbol)
                        else:
                            logger.debug(f"🚫 Blacklisted symbol: {symbol}")
                else:
                    logger.warning(f"❌ Directory not found: {folder}")
                    
                    # Check parent directory
                    parent = folder.parent
                    if parent.exists():
                        subdirs = [d.name for d in parent.iterdir() if d.is_dir()]
                        logger.info(f"📂 Available subdirectories in {parent}: {subdirs}")
                    else:
                        logger.warning(f"❌ Parent directory also not found: {parent}")
        
        # Process Crypto - Enhanced enum comparison for robustness  
        is_crypto_market = (market_type in [MarketType.CRYPTO, MarketType.BOTH] or 
                          market_type.value in ["Cryptocurrency", "Both Markets"])
        
        if is_crypto_market:
            if self.crypto_path and self.crypto_path.exists():
                logger.info(f"🪙 Loading crypto symbols from: {self.crypto_path}")
                csv_files = list(self.crypto_path.glob("*.csv"))
                total_files_found += len(csv_files)
                logger.info(f"📄 Found {len(csv_files)} crypto files")
                
                for file_path in csv_files:
                    symbol = file_path.stem
                    check_symbol = symbol.replace('_USDT', '') if '_USDT' in symbol else symbol
                    if not self.blacklist_manager.is_blacklisted(check_symbol, MarketType.CRYPTO):
                        symbols.append(symbol)
            else:
                logger.warning(f"❌ Crypto path not found or doesn't exist: {self.crypto_path}")
        
        logger.info(f"📊 Total files found: {total_files_found}")
        logger.info(f"📈 Total symbols collected: {len(symbols)}")
        
        # Enhanced diagnostics when no symbols found
        if len(symbols) == 0:
            logger.error("🚨 NO SYMBOLS FOUND - DIAGNOSTIC INFORMATION:")
            logger.error(f"   Data directory: {self.data_dir}")
            logger.error(f"   Data directory exists: {self.data_dir.exists()}")
            
            if self.data_dir.exists():
                contents = list(self.data_dir.iterdir())
                logger.error(f"   Contents: {[item.name for item in contents]}")
            
            logger.error("   Expected structure:")
            logger.error("   Chinese_Market/data/shanghai_6xx/*.csv")
            logger.error("   Chinese_Market/data/shenzhen_0xx/*.csv")
            logger.error("   Chinese_Market/data/huobi/spot_usdt/1d/*.csv")
            
            # Provide helpful suggestions
            logger.error("💡 SUGGESTIONS:")
            logger.error("   1. Check if data files exist in the expected directories")
            logger.error("   2. Verify CSV file format and naming")
            logger.error("   3. Ensure proper file permissions")
            logger.error("   4. Try running from the correct working directory")
        
        return list(set(symbols))
    
    def analyze_symbol(self, symbol: str, market_type: MarketType) -> Optional[MultiTimeframeAnalysis]:
        """Analyze a single symbol with comprehensive analysis"""
        try:
            is_crypto = market_type == MarketType.CRYPTO
            
            # Blacklist check
            check_symbol = symbol.replace('_USDT', '') if '_USDT' in symbol else symbol
            if self.blacklist_manager.is_blacklisted(check_symbol, market_type):
                return None
            
            df = self.load_data(symbol, market_type)
            if df is None:
                return None
            
            # Enhanced minimum data requirements
            min_len = 60 if is_crypto else 120
            if len(df) < min_len:
                return None
            
            # Run enhanced stage analysis if available
            if self.stage_analyzer:
                stage_results = self.stage_analyzer.run_all_stages(df, symbol, is_crypto)
            else:
                # Fallback basic analysis
                stage_results = self._basic_stage_analysis(df, symbol, is_crypto)
            
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
            if self.multi_tf_analyzer:
                timeframe_patterns = {}
                for tf in [TimeFrame.D1, TimeFrame.D3, TimeFrame.D6, TimeFrame.D11, 
                          TimeFrame.D21, TimeFrame.D33, TimeFrame.D55, TimeFrame.D89]:
                    patterns = self.multi_tf_analyzer.analyze_timeframe(df, tf, is_crypto)
                    if patterns:
                        timeframe_patterns[tf] = patterns
                
                best_combo = self.multi_tf_analyzer.find_best_combination(timeframe_patterns, is_crypto)
                consolidation = self.multi_tf_analyzer.determine_consolidation_type(df, is_crypto)
                maturity = self.multi_tf_analyzer.determine_maturity(stage_results, timeframe_patterns)
                entry_setup = self.multi_tf_analyzer.calculate_entry_setup(df, stage_results, is_crypto)
                visual = self.multi_tf_analyzer.create_visual_summary(stage_results)
                game_score, rarity = self.multi_tf_analyzer.calculate_game_score(stage_results, timeframe_patterns)
                
                # Enhanced special pattern detection
                all_patterns = []
                for patterns in timeframe_patterns.values():
                    all_patterns.extend(patterns)
                special_pattern = self.multi_tf_analyzer.detect_special_pattern(all_patterns)
            else:
                # Fallback analysis
                timeframe_patterns = {TimeFrame.D1: []}
                best_combo = PatternCombination([], 60, TimeFrame.D1, 60, 0.6)
                consolidation = ConsolidationType.MODERATE
                maturity = PatternMaturity.DEVELOPING
                entry_setup = self._basic_entry_setup(df, is_crypto)
                visual = "📊"
                game_score, rarity = 300, "⚪ COMMON"
                special_pattern = ""
            
            # Ensure minimum pattern requirements
            total_patterns = sum(len(patterns) for patterns in timeframe_patterns.values())
            if total_patterns < (2 if is_crypto else 3):
                return None
            
            # Enhanced scoring - IMPROVED CONFIDENCE CALCULATION
            stage_scores = [r.score for r in stage_results]
            confidence_scores = [r.confidence for r in stage_results]
            
            # IMPROVED: Better scoring weights and bonuses
            base_score = np.mean(stage_scores) * 0.5 + np.mean(confidence_scores) * 0.3
            
            # Pattern count bonus
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
                prediction_confidence=prediction_confidence,
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
    
    def run_analysis(self, market_type: MarketType = MarketType.BOTH,
                     max_symbols: int = None, num_processes: int = None) -> List[MultiTimeframeAnalysis]:
        """Run comprehensive analysis with M1-optimized parallel processing and enhanced error handling"""
        # Try to import from main module for system optimization
        try:
            from main import detect_m1_optimization
        except ImportError:
            # Fallback system info
            def detect_m1_optimization():
                return {
                    'is_m1': False,
                    'optimal_processes': min(4, mp.cpu_count()),
                    'chunk_multiplier': 1
                }
        
        print(f"\n🎮 {random.choice(GAME_MESSAGES)}")
        print(f"📊 Market: {market_type.value}")
        print(f"🧠 Learning from {len(self.learning_system.pattern_performance)} advanced patterns")
        
        symbols = self.get_all_symbols(market_type)
        
        # CRITICAL FIX: Handle zero symbols case
        if len(symbols) == 0:
            print(f"❌ No symbols found for analysis!")
            print(f"🔍 Market type: {market_type.value}")
            print(f"📁 Data directory: {self.data_dir}")
            print(f"💡 Please check that data files exist in the expected directory structure.")
            return []
        
        # IMPROVED: Better limit handling
        if max_symbols and max_symbols < len(symbols):
            print(f"⚠️  Limiting analysis to {max_symbols} symbols (from {len(symbols)} available)")
            symbols = symbols[:max_symbols]
        else:
            print(f"📈 Analyzing ALL {len(symbols)} symbols")
        
        print(f"🚫 Blacklisted: {len(self.blacklist_manager.blacklisted_stocks)} stocks, {len(self.blacklist_manager.blacklisted_crypto)} crypto")
        
        results = []
        successful_analyses = 0
        failed_analyses = 0
        
        # M1-OPTIMIZED: Enhanced parallel processing
        system_info = detect_m1_optimization()
        
        if num_processes is None:
            num_processes = system_info['optimal_processes']
        
        print(f"🚀 M1-Optimized: Using {num_processes} parallel processes (CPU cores: {mp.cpu_count()})")
        
        # M1-OPTIMIZED: Use parallel processing for smaller datasets too
        if len(symbols) > 50 and num_processes > 1:
            
            base_chunk_size = max(5, len(symbols) // (num_processes * system_info['chunk_multiplier']))
            optimal_chunk_size = min(base_chunk_size, 50)
            
            if system_info['is_m1']:
                print(f"💪 M1 Turbo Mode: {num_processes} workers, {optimal_chunk_size} symbols per chunk")
            else:
                print(f"💻 Parallel Mode: {num_processes} workers, {optimal_chunk_size} symbols per chunk")
            
            # Create more evenly distributed chunks
            symbol_chunks = []
            for i in range(0, len(symbols), optimal_chunk_size):
                chunk = symbols[i:i + optimal_chunk_size]
                if chunk:
                    symbol_chunks.append(chunk)
            
            print(f"📦 Created {len(symbol_chunks)} processing chunks")
            
            # M1-OPTIMIZED: Use ThreadPoolExecutor for better M1 performance
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
                
                print(f"🔄 Submitted {len(futures)} analysis tasks to processor")
                
                # M1-OPTIMIZED: Collect results with enhanced progress tracking
                completed_count = 0
                batch_results = []
                
                for future, symbol, sym_market, chunk_idx in futures:
                    try:
                        timeout = 15 if system_info['is_m1'] else 20
                        analysis = future.result(timeout=timeout)
                        completed_count += 1
                        
                        if analysis:
                            batch_results.append(analysis)
                            successful_analyses += 1
                            
                            # M1-OPTIMIZED: Process in batches to manage memory
                            if len(batch_results) >= 100:
                                results.extend(batch_results)
                                batch_results = []
                        else:
                            failed_analyses += 1
                        
                        # Enhanced progress tracking
                        if completed_count % 50 == 0:
                            progress_pct = (completed_count / len(futures)) * 100
                            if system_info['is_m1']:
                                print(f"🚀 M1 Turbo: {completed_count}/{len(futures)} ({progress_pct:.1f}%) | ✅ {successful_analyses} | ❌ {failed_analyses}")
                            else:
                                print(f"💻 Progress: {completed_count}/{len(futures)} ({progress_pct:.1f}%) | ✅ {successful_analyses} | ❌ {failed_analyses}")
                    
                    except Exception as e:
                        failed_analyses += 1
                        completed_count += 1
                        if completed_count % 100 == 0:
                            logger.debug(f"Analysis failed for {symbol}: {e}")
                        continue
                
                # Add any remaining batch results
                if batch_results:
                    results.extend(batch_results)
        else:
            # Sequential processing for very small datasets
            print(f"🔄 Sequential processing for {len(symbols)} symbols")
            for i, symbol in enumerate(symbols):
                if i % 50 == 0:
                    progress_pct = (i / len(symbols)) * 100
                    print(f"Progress: {i}/{len(symbols)} ({progress_pct:.1f}%) | ✅ {successful_analyses} | ❌ {failed_analyses}")
                
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
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   📈 Total Processed: {len(symbols)}")
        print(f"   ✅ Successful: {successful_analyses}")
        print(f"   ❌ Failed/Filtered: {failed_analyses}")
        
        # CRITICAL FIX: Prevent division by zero
        if len(symbols) > 0:
            success_rate = (successful_analyses / len(symbols) * 100)
            print(f"   📊 Success Rate: {success_rate:.1f}%")
        else:
            print(f"   📊 Success Rate: N/A (no symbols processed)")
        
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
        """Enhanced prediction selection with multi-tier filtering"""
        if not self.results:
            return []
        
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
        """Enhanced results display with better error handling"""
        if not self.results:
            print("\n❌ No patterns found")
            self._print_troubleshooting_guide()
            return
        
        print("\n" + "=" * 140)
        print("🎮 PROFESSIONAL PATTERN ANALYZER v6.1 - ANALYSIS COMPLETE!")
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
                'version': '6.1',
                'analyzer': 'Professional Pattern Analyzer',
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
                }
            }
            data['market_analysis'].append(convert_numpy(result_data))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Professional analysis saved to: {output_path}")
        print(f"📊 Saved {len(data['market_analysis'])} detailed analyses")
    
    # ================== HELPER METHODS ==================
    
    def _basic_stage_analysis(self, df: pd.DataFrame, symbol: str, is_crypto: bool) -> List[StageResult]:
        """Basic fallback stage analysis when stage analyzer is not available"""
        results = []
        
        # Basic technical analysis
        score = 50
        confidence = 50
        
        # Simple momentum check
        if len(df) >= 20:
            ma_20 = df['Close'].rolling(20).mean()
            if df['Close'].iloc[-1] > ma_20.iloc[-1]:
                score += 15
                confidence += 10
        
        # Simple volume check
        if 'Volume' in df.columns:
            volume_trend = df['Volume'].tail(5).mean() / df['Volume'].mean()
            if volume_trend > 1.1:
                score += 10
                confidence += 5
        
        # Create basic stage results
        stage_names = ["Technical Analysis", "Volume Analysis", "Basic Patterns"]
        for stage in stage_names:
            passed = score >= 55
            results.append(StageResult(
                stage, passed, score, 
                {"basic_analysis": True, "score": score}, 
                "📊" if passed else "❌", [], confidence
            ))
        
        return results
    
    def _basic_entry_setup(self, df: pd.DataFrame, is_crypto: bool) -> Dict[str, float]:
        """Basic entry setup calculation"""
        current_price = df['Close'].iloc[-1]
        support = df['Low'].tail(20).min()
        resistance = df['High'].tail(20).max()
        
        return {
            'current': float(current_price),
            'entry': float(current_price * 0.99),
            'stop_loss': float(support * 0.98),
            'target1': float(resistance * 1.02),
            'target2': float(resistance * 1.05),
            'risk_reward_1': 2.0,
            'position_size_pct': 5.0
        }
    
    def _calculate_comprehensive_risk(self, df: pd.DataFrame, stage_results: List[StageResult], is_crypto: bool = False) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment"""
        try:
            volatility = df['Close'].pct_change().std() * np.sqrt(252)
            vol_threshold_high = 1.0 if is_crypto else 0.5
            vol_threshold_medium = 0.6 if is_crypto else 0.3
            
            if volatility < vol_threshold_medium:
                risk_level = 'LOW'
                overall_risk = 0.2
            elif volatility < vol_threshold_high:
                risk_level = 'MEDIUM'
                overall_risk = 0.4
            else:
                risk_level = 'HIGH'
                overall_risk = 0.7
            
            return {
                'overall_level': risk_level,
                'overall_score': overall_risk,
                'volatility': volatility
            }
        except:
            return {'overall_level': 'MEDIUM', 'overall_score': 0.4}
    
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
            
            return indicators
        except:
            return {'rsi': 50.0, 'macd': 0.0}
    
    def _print_troubleshooting_guide(self):
        """Print enhanced troubleshooting information"""
        print("\n💡 TROUBLESHOOTING GUIDE:")
        print("   • Check data file paths and formats")
        print(f"   • Crypto path: {self.crypto_path}")
        print(f"   • Stock paths: {list(self.stock_paths.values())}")
        print("   • Ensure minimum 60-120 days of data per symbol")
        print("   • Try lowering quality thresholds in settings")
        print("   • Check for data file corruption or format issues")
        print("\n🔍 DATA DIRECTORY VERIFICATION:")
        print(f"   • Working directory: {Path.cwd()}")
        print(f"   • Data directory: {self.data_dir}")
        print(f"   • Data directory exists: {self.data_dir.exists()}")
        if self.data_dir.exists():
            contents = list(self.data_dir.iterdir())
            print(f"   • Directory contents: {[item.name for item in contents[:10]]}")
    
    def _print_advanced_pattern_stats(self):
        """Print advanced pattern statistics"""
        print("\n🔬 ADVANCED PATTERN ANALYSIS:")
        
        # Count patterns
        pattern_counts = defaultdict(int)
        for result in self.results:
            for patterns in result.timeframe_patterns.values():
                for pattern in patterns:
                    pattern_counts[pattern.value] += 1
        
        if pattern_counts:
            print("   🧠 Top Patterns:")
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
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
        print(f"   💰 Entry: {pred.entry_setup['entry']} → Target: {pred.entry_setup['target1']} (R/R: {pred.entry_setup.get('risk_reward_1', 0)})")
        print(f"   ⚠️  Risk Level: {pred.risk_assessment.get('overall_level', 'UNKNOWN')}")
        print(f"   🎮 Visual: {pred.visual_summary}")
        
        # Show top patterns
        all_patterns = []
        for patterns in pred.timeframe_patterns.values():
            all_patterns.extend([p.value for p in patterns])
        
        if all_patterns:
            top_patterns = list(set(all_patterns))[:4]
            print(f"   📊 Key Patterns: {', '.join(top_patterns)}")
    
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


# Export the main class
__all__ = ['ProfessionalPatternAnalyzer']
