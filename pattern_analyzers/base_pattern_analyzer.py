"""
Base Pattern Analyzer
Defines the interface for all pattern analysis components
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Import shared data structures
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


class BasePatternAnalyzer(ABC):
    """
    Base class for all pattern analyzers
    
    All pattern analyzers must inherit from this class and implement:
    - analyze_symbol() method
    - get_all_symbols() method
    - run_analysis() method
    """
    
    def __init__(self, data_dir: str = "Chinese_Market/data"):
        self.name = self.__class__.__name__
        self.version = "1.0"
        self.data_dir = Path(data_dir)
        self.results = []
        self.used_symbols = set()
        
        # Setup data paths with enhanced validation
        self._setup_data_paths()
        
        # Find crypto path with better error handling
        self.crypto_path = self._find_crypto_path()
    
    def _setup_data_paths(self):
        """Setup and validate data paths with enhanced error handling"""
        # Stock paths
        self.stock_paths = {
            'shanghai': self.data_dir / 'shanghai_6xx',
            'shenzhen': self.data_dir / 'shenzhen_0xx',
            'beijing': self.data_dir / 'beijing_8xx'
        }
        
        # Crypto paths (in order of preference)
        self.crypto_paths = [
            self.data_dir / 'huobi' / 'spot_usdt' / '1d',
            self.data_dir / 'huobi' / 'csv' / 'spot',
            self.data_dir / 'csv' / 'huobi' / 'spot',
            self.data_dir / 'crypto' / 'spot',
            self.data_dir / 'binance' / 'spot'
        ]
        
        # Enhanced path validation and logging
        self._validate_data_paths()
    
    def _validate_data_paths(self):
        """Validate data paths and provide detailed diagnostics"""
        logger.info(f"🔍 Validating data paths from base directory: {self.data_dir}")
        
        # Check if base data directory exists
        if not self.data_dir.exists():
            logger.error(f"❌ Base data directory does not exist: {self.data_dir}")
            logger.error(f"   Current working directory: {Path.cwd()}")
            
            # Try to find alternative data directories
            possible_dirs = [
                Path("data"),
                Path("Chinese_Market") / "data",
                Path("..") / "Chinese_Market" / "data",
                Path(".") / "Chinese_Market" / "data"
            ]
            
            logger.info("🔍 Searching for alternative data directories:")
            for alt_dir in possible_dirs:
                if alt_dir.exists():
                    logger.info(f"   ✅ Found alternative: {alt_dir.absolute()}")
                    contents = list(alt_dir.iterdir())[:5]
                    logger.info(f"      Contents: {[item.name for item in contents]}")
                else:
                    logger.debug(f"   ❌ Not found: {alt_dir.absolute()}")
        else:
            logger.info(f"✅ Base data directory exists: {self.data_dir}")
            
            # List contents of base directory
            try:
                contents = list(self.data_dir.iterdir())
                logger.info(f"📁 Contents: {[item.name for item in contents]}")
            except PermissionError:
                logger.error(f"❌ Permission denied accessing: {self.data_dir}")
        
        # Validate stock paths
        logger.info("🏮 Checking Chinese stock directories:")
        for exchange, path in self.stock_paths.items():
            if path.exists():
                csv_count = len(list(path.glob("*.csv")))
                logger.info(f"   ✅ {exchange}: {path} ({csv_count} CSV files)")
                
                if csv_count == 0:
                    # Check what files are actually there
                    all_files = list(path.glob("*"))
                    if all_files:
                        file_types = {}
                        for file in all_files[:10]:  # Check first 10 files
                            ext = file.suffix.lower()
                            file_types[ext] = file_types.get(ext, 0) + 1
                        logger.warning(f"      📄 File types found: {file_types}")
                    else:
                        logger.warning(f"      📁 Directory is empty")
            else:
                logger.warning(f"   ❌ {exchange}: {path} (not found)")
                
                # Check if parent directory exists
                parent = path.parent
                if parent.exists():
                    subdirs = [d.name for d in parent.iterdir() if d.is_dir()]
                    logger.info(f"      📂 Available subdirectories in {parent}: {subdirs}")
        
        # Validate crypto paths
        logger.info("🪙 Checking cryptocurrency directories:")
        found_crypto = False
        for i, path in enumerate(self.crypto_paths):
            if path.exists():
                csv_count = len(list(path.glob("*.csv")))
                logger.info(f"   ✅ Path {i+1}: {path} ({csv_count} CSV files)")
                found_crypto = True
                
                if csv_count == 0:
                    # Check what files are actually there
                    all_files = list(path.glob("*"))
                    if all_files:
                        file_types = {}
                        for file in all_files[:10]:
                            ext = file.suffix.lower()
                            file_types[ext] = file_types.get(ext, 0) + 1
                        logger.info(f"      📄 File types found: {file_types}")
            else:
                logger.debug(f"   ❌ Path {i+1}: {path} (not found)")
        
        if not found_crypto:
            logger.warning("⚠️  No cryptocurrency data directories found")
    
    @abstractmethod
    def analyze_symbol(self, symbol: str, market_type: MarketType) -> Optional[MultiTimeframeAnalysis]:
        """
        Analyze a single symbol and return detailed analysis
        
        Args:
            symbol: Symbol to analyze
            market_type: Type of market (crypto/stock)
            
        Returns:
            MultiTimeframeAnalysis object or None if analysis fails
        """
        pass
    
    @abstractmethod
    def get_all_symbols(self, market_type: MarketType) -> List[str]:
        """
        Get all available symbols for the given market type
        
        Args:
            market_type: Type of market to get symbols for
            
        Returns:
            List of available symbols
        """
        pass
    
    @abstractmethod
    def run_analysis(self, market_type: MarketType = MarketType.BOTH,
                     max_symbols: int = None, num_processes: int = None) -> List[MultiTimeframeAnalysis]:
        """
        Run analysis on multiple symbols
        
        Args:
            market_type: Type of market to analyze
            max_symbols: Maximum number of symbols to analyze
            num_processes: Number of parallel processes to use
            
        Returns:
            List of MultiTimeframeAnalysis results
        """
        pass
    
    def load_data(self, symbol: str, market_type: MarketType) -> Optional[pd.DataFrame]:
        """
        Load data for a given symbol with enhanced error handling
        
        Args:
            symbol: Symbol to load data for
            market_type: Type of market
            
        Returns:
            DataFrame with OHLCV data or None if loading fails
        """
        try:
            file_path = None
            
            if market_type == MarketType.CRYPTO:
                if not self.crypto_path:
                    logger.debug(f"No crypto path available for {symbol}")
                    return None
                
                # Multiple naming conventions
                possible_names = [
                    f"{symbol}.csv",
                    f"{symbol.upper()}.csv",
                    f"{symbol.lower()}.csv",
                    f"{symbol}_USDT.csv",
                    f"{symbol.replace('_USDT', '')}_USDT.csv"
                ]
                
                for name in possible_names:
                    test_path = self.crypto_path / name
                    if test_path.exists():
                        file_path = test_path
                        break
                
                if not file_path:
                    logger.debug(f"Crypto file not found for {symbol} in {self.crypto_path}")
                    return None
            else:
                # Chinese stock data
                for exchange, folder in self.stock_paths.items():
                    if not folder.exists():
                        continue
                    
                    test_path = folder / f"{symbol}.csv"
                    if test_path.exists():
                        file_path = test_path
                        break
                
                if not file_path:
                    logger.debug(f"Stock file not found for {symbol}")
                    return None
            
            # Load and clean data
            logger.debug(f"Loading data from: {file_path}")
            
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            except Exception as e:
                logger.debug(f"Error reading CSV {file_path}: {e}")
                # Try without parsing dates
                try:
                    df = pd.read_csv(file_path, index_col=0)
                except Exception as e2:
                    logger.debug(f"Error reading CSV without date parsing {file_path}: {e2}")
                    return None
            
            # Enhanced column standardization - CRITICAL FIX for crypto data
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['open', '开盘价', 'open_price']:
                    column_mapping[col] = 'Open'
                elif col_lower in ['high', '最高价', 'high_price']:
                    column_mapping[col] = 'High'
                elif col_lower in ['low', '最低价', 'low_price']:
                    column_mapping[col] = 'Low'
                elif col_lower in ['close', '收盘价', 'close_price']:
                    column_mapping[col] = 'Close'
                elif col_lower in ['volume', '成交量', 'vol', 'volume_quote']:
                    column_mapping[col] = 'Volume'
            
            # Apply column mapping
            if column_mapping:
                df = df.rename(columns=column_mapping)
                logger.debug(f"Column mapping applied for {symbol}: {column_mapping}")
            
            # CRITICAL FIX: Ensure required columns exist after mapping
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.debug(f"Missing required columns for {symbol} after mapping: {missing_columns}")
                logger.debug(f"Available columns after mapping: {list(df.columns)}")
                logger.debug(f"Original columns: {list(pd.read_csv(file_path, nrows=0).columns)}")
                
                # Try alternative column detection for crypto data
                if market_type == MarketType.CRYPTO:
                    logger.debug(f"Attempting crypto-specific column detection for {symbol}")
                    
                    # Re-read original data to try alternative mapping
                    df_orig = pd.read_csv(file_path, index_col=0) if file_path else df
                    crypto_mapping = {}
                    
                    for col in df_orig.columns:
                        col_clean = col.lower().strip()
                        if col_clean == 'open' and 'Open' not in df.columns:
                            crypto_mapping[col] = 'Open'
                        elif col_clean == 'high' and 'High' not in df.columns:
                            crypto_mapping[col] = 'High'
                        elif col_clean == 'low' and 'Low' not in df.columns:
                            crypto_mapping[col] = 'Low'
                        elif col_clean == 'close' and 'Close' not in df.columns:
                            crypto_mapping[col] = 'Close'
                        elif col_clean in ['volume', 'vol'] and 'Volume' not in df.columns:
                            crypto_mapping[col] = 'Volume'
                    
                    if crypto_mapping:
                        df = df_orig.rename(columns=crypto_mapping)
                        logger.debug(f"Applied crypto-specific mapping for {symbol}: {crypto_mapping}")
                        
                        # Re-check missing columns
                        missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.debug(f"Still missing columns for {symbol}: {missing_columns}")
                    return None
            
            # Enhanced data cleaning
            original_length = len(df)
            
            # Sort by index (date)
            try:
                df = df.sort_index()
            except Exception:
                logger.debug(f"Could not sort by index for {symbol}")
            
            # Remove rows with NaN values in required columns
            df = df.dropna(subset=required_columns)
            
            # Remove rows with non-positive values
            df = df[(df[required_columns] > 0).all(axis=1)]
            
            # Remove extreme outliers (more than 3 standard deviations)
            for col in required_columns:
                if len(df) > 10:  # Only if we have enough data
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        lower_bound = mean_val - 3 * std_val
                        upper_bound = mean_val + 3 * std_val
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Clean volume data if available
            if 'Volume' in df.columns:
                df = df[df['Volume'] >= 0]  # Allow zero volume, but not negative
            
            final_length = len(df)
            if original_length > final_length:
                logger.debug(f"Cleaned data for {symbol}: {original_length} → {final_length} rows")
            
            # Check minimum data requirements
            min_required_length = 30
            if len(df) < min_required_length:
                logger.debug(f"Insufficient data for {symbol}: {len(df)} rows (minimum: {min_required_length})")
                return None
            
            # Validate data integrity
            if not self._validate_data_integrity(df, symbol):
                return None
            
            logger.debug(f"Successfully loaded {symbol}: {len(df)} rows, columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.debug(f"Unexpected error loading data for {symbol}: {e}")
            return None
    
    def _validate_data_integrity(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data integrity with enhanced checks"""
        try:
            # Check for basic data sanity
            required_columns = ['Open', 'High', 'Low', 'Close']
            
            for col in required_columns:
                if col not in df.columns:
                    logger.debug(f"Missing column {col} for {symbol}")
                    return False
                
                if df[col].isnull().any():
                    logger.debug(f"Null values in {col} for {symbol}")
                    return False
                
                if (df[col] <= 0).any():
                    logger.debug(f"Non-positive values in {col} for {symbol}")
                    return False
            
            # Check OHLC relationships
            invalid_ohlc = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close'])
            ).any()
            
            if invalid_ohlc:
                logger.debug(f"Invalid OHLC relationships for {symbol}")
                return False
            
            # Check for reasonable price variations
            price_changes = df['Close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
            
            if extreme_changes > len(df) * 0.1:  # More than 10% of data points
                logger.debug(f"Too many extreme price changes for {symbol}: {extreme_changes}")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating data integrity for {symbol}: {e}")
            return False
    
    def _find_crypto_path(self) -> Optional[Path]:
        """Find available crypto data path with enhanced detection"""
        logger.debug("🔍 Searching for crypto data directories...")
        
        for i, path in enumerate(self.crypto_paths):
            logger.debug(f"   Checking path {i+1}: {path}")
            if path.exists():
                csv_files = list(path.glob("*.csv"))
                logger.info(f"✅ Found crypto data at: {path} ({len(csv_files)} CSV files)")
                return path
            else:
                logger.debug(f"   ❌ Path does not exist: {path}")
        
        logger.warning("⚠️  No crypto data path found")
        return None
    
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
        
        if len(df) < 30:  # Minimum data points for pattern analysis
            return False
        
        if df[required_columns].isnull().any().any():
            return False
        
        if (df[required_columns] <= 0).any().any():
            return False
        
        return True
    
    def calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic metrics for a symbol
        
        Args:
            df: OHLCV data
            
        Returns:
            Dictionary of basic metrics
        """
        try:
            metrics = {}
            
            # Price metrics
            current_price = df['Close'].iloc[-1]
            metrics['current_price'] = current_price
            
            # Returns
            if len(df) >= 30:
                metrics['return_30d'] = (current_price - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
            if len(df) >= 60:
                metrics['return_60d'] = (current_price - df['Close'].iloc[-60]) / df['Close'].iloc[-60]
            
            # Volatility
            returns = df['Close'].pct_change().dropna()
            metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            
            # Volume metrics
            if 'Volume' in df.columns:
                metrics['avg_volume'] = df['Volume'].mean()
                metrics['volume_trend'] = df['Volume'].tail(10).mean() / df['Volume'].mean()
            
            # Technical levels
            metrics['high_52w'] = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
            metrics['low_52w'] = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()
            
            # Moving averages
            if len(df) >= 20:
                metrics['sma_20'] = df['Close'].rolling(20).mean().iloc[-1]
                metrics['price_vs_sma20'] = (current_price - metrics['sma_20']) / metrics['sma_20']
            
            if len(df) >= 50:
                metrics['sma_50'] = df['Close'].rolling(50).mean().iloc[-1]
                metrics['price_vs_sma50'] = (current_price - metrics['sma_50']) / metrics['sma_50']
            
            return metrics
        except Exception as e:
            logger.debug(f"Error calculating basic metrics: {e}")
            return {'error': str(e)}
    
    def print_results(self):
        """
        Print analysis results
        """
        if not self.results:
            print("\n❌ No patterns found")
            return
        
        print(f"\n📊 Analysis Results: {len(self.results)} opportunities found")
        
        # Print top results
        for i, result in enumerate(self.results[:10], 1):
            print(f"\n{i}. {result.symbol} - {result.market_type.value}")
            print(f"   Score: {result.overall_score:.1f} | Confidence: {result.prediction_confidence:.1f}%")
            print(f"   Patterns: {sum(len(p) for p in result.timeframe_patterns.values())}")
            if result.special_pattern:
                print(f"   Special: {result.special_pattern}")
    
    def save_results(self, filename: str = None):
        """
        Save analysis results to file
        
        Args:
            filename: Optional filename to save to
        """
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"pattern_analysis_{timestamp}.json"
        
        # Implementation would save results to JSON file
        print(f"💾 Results would be saved to: {filename}")
    
    def get_predictions(self, top_n: int = 50) -> List[MultiTimeframeAnalysis]:
        """
        Get top predictions from analysis results
        
        Args:
            top_n: Number of top predictions to return
            
        Returns:
            List of top MultiTimeframeAnalysis results
        """
        if not self.results:
            return []
        
        # Sort by overall score and confidence
        sorted_results = sorted(
            self.results,
            key=lambda x: (x.overall_score * 0.7 + x.prediction_confidence * 0.3),
            reverse=True
        )
        
        return sorted_results[:top_n]
    
    def get_analyzer_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this pattern analyzer
        
        Returns:
            Dictionary with analyzer information
        """
        return {
            "name": self.name,
            "version": self.version,
            "data_directory": str(self.data_dir),
            "crypto_path": str(self.crypto_path) if self.crypto_path else None,
            "stock_paths": {k: str(v) for k, v in self.stock_paths.items()},
            "results_count": len(self.results),
            "analyzed_symbols": len(self.used_symbols),
            "description": self.__doc__ or "No description available"
        }
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} - {len(self.results)} results"
    
    def __repr__(self) -> str:
        return f"<{self.name}(results={len(self.results)})>"
