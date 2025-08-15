"""
Base Pattern Analyzer - FIXED
Defines the interface for all pattern analysis components with improved data loading
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
                        logger.warning(f"      🔍 Directory is empty")
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
            logger.warning("⚠️ No cryptocurrency data directories found")

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
        Load data for a given symbol with SIGNIFICANTLY IMPROVED error handling and column mapping

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

                # Multiple naming conventions for crypto files
                possible_names = [
                    f"{symbol}.csv",
                    f"{symbol.upper()}.csv",
                    f"{symbol.lower()}.csv",
                    f"{symbol}_USDT.csv",
                    f"{symbol.replace('_USDT', '')}_USDT.csv",
                    f"{symbol.replace('-USDT', '_USDT')}.csv",
                    f"{symbol.replace('_', '-')}.csv"
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

            # Load and clean data with MULTIPLE fallback methods
            logger.debug(f"Loading data from: {file_path}")

            # Method 1: Try with date parsing
            df = None
            loading_method = None
            
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                loading_method = "with_date_parsing"
                logger.debug(f"Successfully loaded {symbol} with date parsing")
            except Exception as e1:
                logger.debug(f"Date parsing failed for {symbol}: {e1}")
                
                # Method 2: Try without date parsing
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    loading_method = "without_date_parsing"
                    logger.debug(f"Successfully loaded {symbol} without date parsing")
                except Exception as e2:
                    logger.debug(f"Standard loading failed for {symbol}: {e2}")
                    
                    # Method 3: Try without index_col
                    try:
                        df = pd.read_csv(file_path)
                        loading_method = "no_index"
                        logger.debug(f"Successfully loaded {symbol} without index column")
                    except Exception as e3:
                        logger.debug(f"All loading methods failed for {symbol}: {e3}")
                        return None

            if df is None or len(df) == 0:
                logger.debug(f"Empty dataframe for {symbol}")
                return None

            # CRITICAL FIX: ENHANCED COLUMN STANDARDIZATION
            logger.debug(f"Original columns for {symbol}: {list(df.columns)}")
            
            # Create comprehensive column mapping
            column_mapping = {}
            
            # Map all possible column variations to standard names
            for col in df.columns:
                col_clean = str(col).lower().strip()
                
                # Open price mapping
                if col_clean in ['open', '开盘价', 'open_price', 'o']:
                    column_mapping[col] = 'Open'
                # High price mapping  
                elif col_clean in ['high', '最高价', 'high_price', 'h']:
                    column_mapping[col] = 'High'
                # Low price mapping
                elif col_clean in ['low', '最低价', 'low_price', 'l']:
                    column_mapping[col] = 'Low'
                # Close price mapping
                elif col_clean in ['close', '收盘价', 'close_price', 'c']:
                    column_mapping[col] = 'Close'
                # Volume mapping
                elif col_clean in ['volume', '成交量', 'vol', 'volume_quote', 'v']:
                    column_mapping[col] = 'Volume'
                # Additional mappings for Chinese data
                elif col_clean in ['振幅', 'amplitude']:
                    column_mapping[col] = 'Amplitude'
                elif col_clean in ['股票代码', 'stock_code', 'symbol']:
                    column_mapping[col] = 'Symbol'
                elif col_clean in ['股票名称', 'stock_name', 'name']:
                    column_mapping[col] = 'Name'
                elif col_clean in ['price_change', 'change']:
                    column_mapping[col] = 'Change'

            # Apply column mapping if any mappings were found
            if column_mapping:
                df = df.rename(columns=column_mapping)
                logger.debug(f"Applied column mapping for {symbol}: {column_mapping}")
                logger.debug(f"New columns for {symbol}: {list(df.columns)}")

            # CRITICAL FIX: Verify required columns exist after mapping
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.debug(f"Missing required columns for {symbol} after mapping: {missing_columns}")
                logger.debug(f"Available columns: {list(df.columns)}")
                
                # ENHANCED: Try to find columns by position for crypto data
                if market_type == MarketType.CRYPTO and len(df.columns) >= 4:
                    logger.debug(f"Attempting positional column mapping for crypto {symbol}")
                    
                    # Typical crypto CSV format: timestamp, open, high, low, close, volume, volume_quote, symbol, price_change
                    # Try to map by position if we have at least 4 numeric columns
                    numeric_columns = []
                    for col in df.columns:
                        try:
                            pd.to_numeric(df[col].iloc[0] if len(df) > 0 else 0)
                            numeric_columns.append(col)
                        except:
                            continue
                    
                    if len(numeric_columns) >= 4:
                        positional_mapping = {}
                        if 'Open' not in df.columns:
                            positional_mapping[numeric_columns[0]] = 'Open'
                        if 'High' not in df.columns and len(numeric_columns) > 1:
                            positional_mapping[numeric_columns[1]] = 'High'
                        if 'Low' not in df.columns and len(numeric_columns) > 2:
                            positional_mapping[numeric_columns[2]] = 'Low'
                        if 'Close' not in df.columns and len(numeric_columns) > 3:
                            positional_mapping[numeric_columns[3]] = 'Close'
                        if 'Volume' not in df.columns and len(numeric_columns) > 4:
                            positional_mapping[numeric_columns[4]] = 'Volume'
                        
                        if positional_mapping:
                            df = df.rename(columns=positional_mapping)
                            logger.debug(f"Applied positional mapping for {symbol}: {positional_mapping}")
                            
                            # Re-check missing columns
                            missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    logger.debug(f"Still missing critical columns for {symbol}: {missing_columns}")
                    return None

            # ENHANCED DATA CLEANING with more lenient criteria
            original_length = len(df)

            # Sort by index (date) if possible
            try:
                df = df.sort_index()
            except Exception:
                logger.debug(f"Could not sort by index for {symbol}")

            # Remove rows with NaN values in required columns
            df = df.dropna(subset=required_columns)

            # Remove rows with non-positive values (but be more lenient)
            for col in required_columns:
                # Only remove rows where ALL OHLC values are non-positive
                df = df[df[col] > 0]

            # IMPROVED: More lenient outlier removal (only extreme cases)
            for col in required_columns:
                if len(df) > 20:  # Only if we have sufficient data
                    # Use percentiles instead of standard deviation for crypto
                    if market_type == MarketType.CRYPTO:
                        # For crypto, use wider bounds (0.1% and 99.9% percentiles)
                        lower_bound = df[col].quantile(0.001)
                        upper_bound = df[col].quantile(0.999)
                    else:
                        # For stocks, use standard deviation but more lenient (4 instead of 3)
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val > 0:
                            lower_bound = mean_val - 4 * std_val
                            upper_bound = mean_val + 4 * std_val
                        else:
                            continue
                    
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            # Clean volume data if available (more lenient)
            if 'Volume' in df.columns:
                df = df[df['Volume'] >= 0]  # Allow zero volume

            final_length = len(df)
            if original_length > final_length:
                logger.debug(f"Cleaned data for {symbol}: {original_length} → {final_length} rows")

            # IMPROVED: More lenient minimum data requirements
            min_required_length = 20 if market_type == MarketType.CRYPTO else 30
            if len(df) < min_required_length:
                logger.debug(f"Insufficient data for {symbol}: {len(df)} rows (minimum: {min_required_length})")
                return None

            # Validate data integrity with improved checks
            if not self._validate_data_integrity(df, symbol, market_type):
                return None

            logger.debug(f"Successfully loaded {symbol}: {len(df)} rows, columns: {list(df.columns)}")
            return df

        except Exception as e:
            logger.debug(f"Unexpected error loading data for {symbol}: {e}")
            return None

    def _validate_data_integrity(self, df: pd.DataFrame, symbol: str, market_type: MarketType = MarketType.CRYPTO) -> bool:
        """Validate data integrity with enhanced checks and crypto-specific tolerance"""
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

            # Check OHLC relationships (more lenient for crypto)
            tolerance = 0.001 if market_type == MarketType.CRYPTO else 0.0001
            
            invalid_ohlc = (
                (df['High'] < df['Low'] - tolerance) |
                (df['High'] < df['Open'] - tolerance) |
                (df['High'] < df['Close'] - tolerance) |
                (df['Low'] > df['Open'] + tolerance) |
                (df['Low'] > df['Close'] + tolerance)
            ).any()

            if invalid_ohlc:
                logger.debug(f"Invalid OHLC relationships for {symbol}")
                return False

            # IMPROVED: More lenient price variation checks for crypto
            price_changes = df['Close'].pct_change().abs()
            
            if market_type == MarketType.CRYPTO:
                # Crypto can have extreme moves, so be more lenient
                extreme_threshold = 0.9  # 90% change
                max_extreme_ratio = 0.05  # Allow 5% of data points to be extreme
            else:
                # Stocks are more conservative
                extreme_threshold = 0.5  # 50% change
                max_extreme_ratio = 0.02  # Allow 2% of data points to be extreme
                
            extreme_changes = (price_changes > extreme_threshold).sum()
            extreme_ratio = extreme_changes / len(df) if len(df) > 0 else 0

            if extreme_ratio > max_extreme_ratio:
                logger.debug(f"Too many extreme price changes for {symbol}: {extreme_changes} ({extreme_ratio:.2%})")
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

        logger.warning("⚠️ No crypto data path found")
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

        if len(df) < 20:  # Reduced minimum for crypto
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
