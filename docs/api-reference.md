# 📚 API Reference

## Overview

This document provides comprehensive API documentation for the Early Pump Detection System (EPDS) v6.1. The system follows an auto-discovery modular architecture with clear interfaces for extensibility.

## 🏗️ Core Interfaces

### BasePatternAnalyzer

The abstract base class for all pattern analysis components.

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

class BasePatternAnalyzer(ABC):
    """Abstract base class for pattern analyzers"""
```

#### Constructor
```python
def __init__(self, data_dir: str = "Chinese_Market/data"):
    """
    Initialize the pattern analyzer
    
    Args:
        data_dir (str): Directory containing market data
    """
```

#### Abstract Methods

##### `analyze_symbol()`
```python
@abstractmethod
def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> MultiTimeframeAnalysis:
    """
    Analyze a single symbol across multiple timeframes
    
    Args:
        symbol (str): Symbol identifier (e.g., '600519', 'BTC_USDT')
        df (pd.DataFrame): OHLCV data with columns [Date, Open, High, Low, Close, Volume]
    
    Returns:
        MultiTimeframeAnalysis: Complete analysis result
        
    Raises:
        ValueError: If data validation fails
        AnalysisError: If analysis cannot be completed
    """
```

##### `validate_data()`
```python
@abstractmethod
def validate_data(self, df: pd.DataFrame) -> bool:
    """
    Validate input data quality and format
    
    Args:
        df (pd.DataFrame): Input OHLCV data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
```

#### Concrete Methods

##### `calculate_basic_metrics()`
```python
def calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate basic technical indicators
    
    Args:
        df (pd.DataFrame): OHLCV data
        
    Returns:
        Dict[str, float]: Basic metrics including:
            - current_price: Latest closing price
            - price_change_pct: Daily price change percentage
            - volume_ratio: Current vs average volume
            - volatility: Price volatility measure
            - rsi: Relative Strength Index
            - sma_20: 20-period simple moving average
            - sma_50: 50-period simple moving average
    """
```

##### `print_results()`
```python
def print_results(self):
    """Print formatted analysis results to console"""
```

##### `save_results()`
```python
def save_results(self, filename: str = None):
    """
    Save analysis results to file
    
    Args:
        filename (str, optional): Output filename. Auto-generated if None.
    """
```

##### `get_predictions()`
```python
def get_predictions(self, top_n: int = 50) -> List[MultiTimeframeAnalysis]:
    """
    Get top predictions sorted by score
    
    Args:
        top_n (int): Number of top predictions to return
        
    Returns:
        List[MultiTimeframeAnalysis]: Top-ranked analysis results
    """
```

### BasePatternDetector

Abstract base class for pattern detection algorithms.

```python
class BasePatternDetector(ABC):
    """Abstract base for pattern detection components"""
```

#### Abstract Methods

##### `detect_patterns()`
```python
@abstractmethod
def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
    """
    Detect patterns in price data
    
    Args:
        df (pd.DataFrame): OHLCV data
        
    Returns:
        List[PatternResult]: Detected patterns with confidence scores
    """
```

#### Concrete Methods

##### `validate_pattern()`
```python
def validate_pattern(self, pattern_result: PatternResult) -> bool:
    """
    Validate detected pattern meets quality criteria
    
    Args:
        pattern_result (PatternResult): Pattern to validate
        
    Returns:
        bool: True if pattern is valid
    """
```

### BaseStageAnalyzer

Abstract base class for stage analysis components.

```python
class BaseStageAnalyzer(ABC):
    """Abstract base for stage analysis components"""
```

#### Abstract Methods

##### `analyze_stage()`
```python
@abstractmethod
def analyze_stage(self, df: pd.DataFrame, stage: str) -> StageResult:
    """
    Analyze specific market stage
    
    Args:
        df (pd.DataFrame): OHLCV data
        stage (str): Stage name ('accumulation', 'markup', 'distribution', 'decline')
        
    Returns:
        StageResult: Analysis result for the stage
    """
```

#### Concrete Methods

##### `calculate_support_resistance()`
```python
def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """
    Calculate support and resistance levels
    
    Args:
        df (pd.DataFrame): OHLCV data
        window (int): Analysis window size
        
    Returns:
        Dict[str, float]: Support/resistance levels:
            - primary_support: Main support level
            - primary_resistance: Main resistance level
            - support_levels: List of support levels
            - resistance_levels: List of resistance levels
    """
```

##### `calculate_volatility_metrics()`
```python
def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volatility-related metrics
    
    Args:
        df (pd.DataFrame): OHLCV data
        
    Returns:
        Dict[str, float]: Volatility metrics:
            - atr: Average True Range
            - volatility: Historical volatility
            - bb_width: Bollinger Band width
    """
```

##### `calculate_momentum_indicators()`
```python
def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate momentum indicators
    
    Args:
        df (pd.DataFrame): OHLCV data
        
    Returns:
        Dict[str, float]: Momentum indicators:
            - rsi: Relative Strength Index
            - macd: MACD line
            - macd_signal: MACD signal line
            - stoch_k: Stochastic %K
            - stoch_d: Stochastic %D
    """
```

##### `calculate_volume_analysis()`
```python
def calculate_volume_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze volume patterns
    
    Args:
        df (pd.DataFrame): OHLCV data
        
    Returns:
        Dict[str, Any]: Volume analysis:
            - avg_volume: Average volume
            - volume_trend: Volume trend direction
            - volume_spikes: Significant volume events
            - accumulation_score: Accumulation strength
    """
```

## 📊 Data Structures

### MultiTimeframeAnalysis

Complete analysis result for a symbol.

```python
@dataclass
class MultiTimeframeAnalysis:
    """Complete multi-timeframe analysis result"""
    
    symbol: str                              # Symbol identifier
    market_type: MarketType                  # CRYPTO or CHINESE_STOCK
    overall_score: float                     # Composite score (0-100)
    stage_results: List[StageResult]         # Stage analysis results
    pattern_combinations: List[PatternCombination]  # Detected pattern groups
    prediction_confidence: float             # Prediction confidence (0-1)
    timeframe_scores: Dict[str, float]       # Scores by timeframe
    risk_metrics: Dict[str, float]           # Risk assessment
    timestamp: datetime                      # Analysis timestamp
```

### StageResult

Result from single stage analysis.

```python
@dataclass
class StageResult:
    """Result from a single stage analysis"""
    
    stage_name: str                         # Stage identifier
    score: float                            # Stage score (0-100)
    confidence: float                       # Confidence level (0-1)
    indicators: Dict[str, float]            # Technical indicators
    support_resistance: Dict[str, float]    # S/R levels
    volume_profile: Dict[str, Any]          # Volume analysis
    momentum_signals: Dict[str, float]      # Momentum indicators
    timestamp: datetime                     # Analysis timestamp
```

### PatternCombination

Professional pattern combination result.

```python
@dataclass
class PatternCombination:
    """Professional pattern combination"""
    
    name: str                               # Pattern group name
    patterns: List[str]                     # Individual patterns
    score: float                            # Combination score
    confidence: float                       # Detection confidence
    timeframe: str                          # Primary timeframe
    market_regime: str                      # Market condition
    entry_signals: List[str]                # Entry indicators
    risk_factors: List[str]                 # Risk considerations
```

### PatternResult

Individual pattern detection result.

```python
@dataclass
class PatternResult:
    """Individual pattern detection result"""
    
    pattern_type: PatternType               # Pattern classification
    confidence: float                       # Detection confidence (0-1)
    start_index: int                        # Pattern start position
    end_index: int                          # Pattern end position
    key_levels: Dict[str, float]            # Important price levels
    volume_confirmation: bool               # Volume supports pattern
    strength: float                         # Pattern strength (0-100)
    reliability: float                      # Historical reliability
```

## 🔧 Core Systems

### ProfessionalTradingOrchestrator

Main orchestrator that coordinates all components.

```python
class ProfessionalTradingOrchestrator:
    """Main trading analysis orchestrator"""
```

#### Constructor
```python
def __init__(self, data_dir: str = "Chinese_Market/data"):
    """
    Initialize the orchestrator
    
    Args:
        data_dir (str): Data directory path
    """
```

#### Methods

##### `run_analysis()`
```python
def run_analysis(self, 
                market_type: MarketType = MarketType.BOTH,
                max_symbols: int = None,
                num_processes: int = None) -> List[MultiTimeframeAnalysis]:
    """
    Run complete analysis pipeline
    
    Args:
        market_type (MarketType): Markets to analyze
        max_symbols (int, optional): Limit symbol count
        num_processes (int, optional): Process count override
        
    Returns:
        List[MultiTimeframeAnalysis]: Analysis results
    """
```

##### `print_results()`
```python
def print_results(self):
    """Print formatted results to console"""
```

##### `save_results()`
```python
def save_results(self, filename: str = None):
    """
    Save results to file
    
    Args:
        filename (str, optional): Output filename
    """
```

### ComponentRegistry

Auto-discovery registry for components.

```python
class ComponentRegistry:
    """Auto-discovery registry for all components"""
```

#### Methods

##### `discover_components()`
```python
def discover_components(self):
    """
    Automatically discover and register all components
    
    Scans module directories and loads compatible components
    """
```

##### `get_pattern_analyzers()`
```python
def get_pattern_analyzers(self) -> Dict[str, type]:
    """
    Get all discovered pattern analyzers
    
    Returns:
        Dict[str, type]: Analyzer name to class mapping
    """
```

##### `get_pattern_detectors()`
```python
def get_pattern_detectors(self) -> Dict[str, type]:
    """
    Get all discovered pattern detectors
    
    Returns:
        Dict[str, type]: Detector name to class mapping
    """
```

##### `get_stage_analyzers()`
```python
def get_stage_analyzers(self) -> Dict[str, type]:
    """
    Get all discovered stage analyzers
    
    Returns:
        Dict[str, type]: Analyzer name to class mapping
    """
```

### EnhancedBlacklistManager

Dynamic blacklist management system.

```python
class EnhancedBlacklistManager:
    """Enhanced blacklist management with dynamic scoring"""
```

#### Methods

##### `is_blacklisted()`
```python
def is_blacklisted(self, symbol: str) -> bool:
    """
    Check if symbol is blacklisted
    
    Args:
        symbol (str): Symbol to check
        
    Returns:
        bool: True if blacklisted
    """
```

##### `add_to_blacklist()`
```python
def add_to_blacklist(self, symbol: str, reason: str = "Manual"):
    """
    Add symbol to dynamic blacklist
    
    Args:
        symbol (str): Symbol to blacklist
        reason (str): Reason for blacklisting
    """
```

##### `update_performance_tracking()`
```python
def update_performance_tracking(self, results: List[AnalysisResult]):
    """
    Update performance tracking for dynamic blacklist
    
    Args:
        results (List[AnalysisResult]): Analysis results with performance data
    """
```

### ProfessionalLearningSystem

Advanced learning system with pattern tracking.

```python
class ProfessionalLearningSystem:
    """Enhanced learning system with advanced pattern tracking"""
```

#### Methods

##### `learn_from_results()`
```python
def learn_from_results(self, results: List[AnalysisResult]):
    """
    Learn from analysis results and update scoring
    
    Args:
        results (List[AnalysisResult]): Historical analysis results
    """
```

##### `get_pattern_score()`
```python
def get_pattern_score(self, pattern: str, is_crypto: bool = False) -> Tuple[float, Dict]:
    """
    Get learned score for pattern
    
    Args:
        pattern (str): Pattern name
        is_crypto (bool): Whether analyzing crypto market
        
    Returns:
        Tuple[float, Dict]: Pattern score and metadata
    """
```

## 🎯 Utility Functions

### System Optimization

#### `detect_m1_optimization()`
```python
def detect_m1_optimization() -> Dict[str, Any]:
    """
    Detect M1 MacBook and return optimization settings
    
    Returns:
        Dict[str, Any]: System information and optimization settings:
            - platform: Operating system
            - processor: Processor information
            - is_m1: Whether M1/M2 detected
            - optimal_processes: Recommended process count
            - chunk_multiplier: Performance multiplier
    """
```

### Market Type Enum

```python
class MarketType(Enum):
    """Market type enumeration"""
    CHINESE_STOCK = "chinese_stock"
    CRYPTO = "crypto"
    BOTH = "both"
```

### Pattern Type Enum

```python
class PatternType(Enum):
    """Pattern type enumeration"""
    
    # Geometric Patterns
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    
    # Flag and Pennant Patterns
    BULL_FLAG = "Bull Flag"
    BEAR_FLAG = "Bear Flag"
    BULL_PENNANT = "Bull Pennant"
    BEAR_PENNANT = "Bear Pennant"
    
    # Wedge Patterns
    RISING_WEDGE = "Rising Wedge"
    FALLING_WEDGE = "Falling Wedge"
    
    # Advanced Patterns
    ACCUMULATION_PATTERN = "Accumulation Pattern"
    DISTRIBUTION_PATTERN = "Distribution Pattern"
    BREAKOUT_PATTERN = "Breakout Pattern"
    REVERSAL_PATTERN = "Reversal Pattern"
```

## 🔌 Extension Examples

### Custom Pattern Analyzer

```python
from pattern_analyzers.base_pattern_analyzer import BasePatternAnalyzer

class CustomAnalyzer(BasePatternAnalyzer):
    """Example custom analyzer implementation"""
    
    def __init__(self, data_dir: str = "Chinese_Market/data"):
        super().__init__(data_dir)
        self.name = "Custom Analyzer"
        self.version = "1.0"
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> MultiTimeframeAnalysis:
        """Custom analysis implementation"""
        
        # Validate data
        if not self.validate_data(df):
            raise ValueError(f"Invalid data for {symbol}")
        
        # Perform custom analysis
        basic_metrics = self.calculate_basic_metrics(df)
        stage_results = self._analyze_custom_stages(df)
        pattern_combinations = self._detect_custom_patterns(df)
        
        # Calculate overall score
        overall_score = self._calculate_custom_score(basic_metrics, stage_results)
        
        return MultiTimeframeAnalysis(
            symbol=symbol,
            market_type=self._determine_market_type(symbol),
            overall_score=overall_score,
            stage_results=stage_results,
            pattern_combinations=pattern_combinations,
            prediction_confidence=0.8,
            timeframe_scores={"1d": overall_score},
            risk_metrics=basic_metrics,
            timestamp=datetime.now()
        )
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Custom data validation"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in df.columns for col in required_columns) and len(df) >= 50
```

### Custom Pattern Detector

```python
from pattern_detectors.base_detector import BasePatternDetector

class CustomDetector(BasePatternDetector):
    """Example custom detector implementation"""
    
    def __init__(self):
        super().__init__()
        self.name = "Custom Detector"
        self.version = "1.0"
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Custom pattern detection"""
        
        patterns = []
        
        # Example: Simple momentum pattern
        momentum_pattern = self._detect_momentum_pattern(df)
        if momentum_pattern:
            patterns.append(momentum_pattern)
        
        # Example: Volume pattern
        volume_pattern = self._detect_volume_pattern(df)
        if volume_pattern:
            patterns.append(volume_pattern)
        
        return patterns
    
    def _detect_momentum_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect custom momentum pattern"""
        
        # Calculate momentum indicator
        momentum = df['Close'].pct_change(10)
        
        if momentum.iloc[-1] > 0.05:  # 5% momentum
            return PatternResult(
                pattern_type=PatternType.BREAKOUT_PATTERN,
                confidence=0.7,
                start_index=len(df) - 10,
                end_index=len(df) - 1,
                key_levels={"entry": df['Close'].iloc[-1]},
                volume_confirmation=df['Volume'].iloc[-1] > df['Volume'].mean(),
                strength=momentum.iloc[-1] * 100,
                reliability=0.6
            )
        
        return None
```

### BaseTimeframeAnalyzer

Abstract base class for multi-timeframe analysis components.

```python
class BaseTimeframeAnalyzer(ABC):
    """Abstract base for timeframe analysis components"""
```

#### Abstract Methods

##### `analyze_timeframe()`
```python
@abstractmethod
def analyze_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame, is_crypto: bool = False) -> List[PatternType]:
    """
    Analyze patterns for a specific timeframe
    
    Args:
        df (pd.DataFrame): OHLCV data
        timeframe (TimeFrame): Target timeframe for analysis
        is_crypto (bool): Whether analyzing cryptocurrency data
        
    Returns:
        List[PatternType]: Detected patterns for the timeframe
    """
```

##### `find_best_combination()`
```python
@abstractmethod
def find_best_combination(self, timeframe_patterns: Dict[TimeFrame, List[PatternType]], 
                         is_crypto: bool = False) -> PatternCombination:
    """
    Find optimal pattern combination across timeframes
    
    Args:
        timeframe_patterns (Dict[TimeFrame, List[PatternType]]): Patterns detected per timeframe
        is_crypto (bool): Whether analyzing cryptocurrency data
        
    Returns:
        PatternCombination: Best pattern combination with scoring
    """
```

##### `determine_consolidation_type()`
```python
@abstractmethod
def determine_consolidation_type(self, df: pd.DataFrame, is_crypto: bool = False) -> ConsolidationType:
    """
    Determine type of consolidation pattern
    
    Args:
        df (pd.DataFrame): OHLCV data
        is_crypto (bool): Whether analyzing cryptocurrency data
        
    Returns:
        ConsolidationType: Classification of consolidation pattern
    """
```

#### Concrete Methods

##### `get_supported_timeframes()`
```python
def get_supported_timeframes(self) -> List[TimeFrame]:
    """
    Return list of supported timeframe periods
    
    Returns:
        List[TimeFrame]: Fibonacci-based timeframe periods:
            - D1, D3, D6, D11, D21, D33, D55, D89
    """
```

##### `calculate_entry_setup()`
```python
def calculate_entry_setup(self, df: pd.DataFrame, stage_results: List, is_crypto: bool = False) -> Dict[str, float]:
    """
    Calculate optimal entry setup based on timeframe analysis
    
    Args:
        df (pd.DataFrame): OHLCV data
        stage_results (List): Results from stage analysis
        is_crypto (bool): Whether analyzing cryptocurrency data
        
    Returns:
        Dict[str, float]: Entry setup parameters:
            - entry_price: Recommended entry price
            - stop_loss: Stop loss level
            - target1: First profit target
            - target2: Second profit target
            - risk_reward_1: Risk/reward ratio
            - position_size_pct: Recommended position size percentage
    """
```

##### `validate_data()`
```python
def validate_data(self, df: pd.DataFrame) -> bool:
    """
    Validate input data for timeframe analysis
    
    Args:
        df (pd.DataFrame): OHLCV data to validate
        
    Returns:
        bool: True if data is suitable for analysis
    """
```

## 🚨 Error Handling

### Common Exceptions

#### `AnalysisError`
```python
class AnalysisError(Exception):
    """Raised when analysis cannot be completed"""
    pass
```

#### `DataValidationError`
```python
class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass
```

#### `ComponentLoadError`
```python
class ComponentLoadError(Exception):
    """Raised when component auto-discovery fails"""
    pass
```

### Error Handling Best Practices

```python
try:
    analyzer = ProfessionalPatternAnalyzer()
    results = analyzer.analyze_symbol("600519", df)
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
    # Handle invalid data
except AnalysisError as e:
    logger.error(f"Analysis failed: {e}")
    # Handle analysis failure
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## 📝 Usage Examples

### Basic Analysis
```python
# Initialize orchestrator
orchestrator = ProfessionalTradingOrchestrator()

# Run analysis
results = orchestrator.run_analysis(
    market_type=MarketType.CHINESE_STOCK,
    max_symbols=100
)

# Print results
orchestrator.print_results()

# Save results
orchestrator.save_results("analysis_results.json")
```

### Custom Component Usage
```python
# Use custom analyzer
analyzer = CustomAnalyzer()
result = analyzer.analyze_symbol("600519", stock_data)

# Use custom detector  
detector = CustomDetector()
patterns = detector.detect_patterns(stock_data)
```

### Batch Processing
```python
symbols = ["600519", "000858", "BTC_USDT"]
results = []

for symbol in symbols:
    try:
        df = load_symbol_data(symbol)
        result = analyzer.analyze_symbol(symbol, df)
        results.append(result)
    except Exception as e:
        logger.error(f"Failed to analyze {symbol}: {e}")
```

## EPD Data Collection Tools API Reference

### EPDScanner.py - Pattern Analyzer Game v6.0

#### Core Classes

##### ProfessionalTradingOrchestrator
Enhanced orchestrator for pattern analysis with auto-discovery architecture.

```python
class ProfessionalTradingOrchestrator:
    def __init__(self, data_dir: str = "Chinese_Market/data"):
        """Initialize orchestrator with auto-discovery"""
        
    def detect_m1_optimization(self) -> int:
        """Detect M1/M2 MacBook optimization and return process multiplier"""
        
    def run_analysis(self, symbols: List[str] = None) -> List[AnalysisResult]:
        """Run comprehensive pattern analysis"""
        
    def display_results(self, results: List[AnalysisResult]):
        """Display analysis results with professional grading"""
```

##### ComponentRegistry
Auto-discovery system for pattern analysis components.

```python
class ComponentRegistry:
    def auto_discover_all_components(self):
        """Automatically discover all pattern analysis components"""
        
    def get_pattern_analyzers(self) -> List[BasePatternAnalyzer]:
        """Get all discovered pattern analyzers"""
        
    def get_pattern_detectors(self) -> List[BasePatternDetector]:
        """Get all discovered pattern detectors"""
```

### EPDStocksUpdater.py - Chinese A-Share Data Manager v6.0

#### Core Classes

##### ChineseStockManager
Main class for Chinese stock data management and collection.

```python
class ChineseStockManager:
    def __init__(self, config: Config):
        """Initialize with production-ready configuration"""
        
    def update_all_stocks(self) -> Dict[str, Any]:
        """Update all Chinese stock data with intelligent retry"""
        
    def update_specific_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Update specific stock symbols"""
        
    def analyze_existing_files(self) -> Dict[str, Any]:
        """Analyze existing data files for quality assessment"""
        
    def migrate_to_organized_structure(self) -> Dict[str, int]:
        """Migrate files to organized directory structure"""
        
    def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market data overview"""
```

##### FileAnalyzer
Advanced file analysis and data quality assessment.

```python
class FileAnalyzer:
    def analyze_file(self, file_path: Path) -> FileAnalysisResult:
        """Analyze individual CSV file for data quality"""
        
    def batch_analyze(self, file_paths: List[Path]) -> List[FileAnalysisResult]:
        """Batch analyze multiple files with progress tracking"""
        
    def get_quality_summary(self) -> Dict[str, int]:
        """Get summary of data quality across all files"""
```

##### DataSourceManager
Multi-source data collection with intelligent fallback.

```python
class DataSourceManager:
    def __init__(self, config: Config):
        """Initialize with multiple data sources"""
        
    def get_stock_data(self, symbol: str, exchange: str) -> Optional[pd.DataFrame]:
        """Get stock data with automatic source fallback"""
        
    def test_all_sources(self) -> Dict[str, bool]:
        """Test connectivity to all data sources"""
```

### EPDHuobiUpdater.py - HTX Crypto Data Collector v5.0

#### Core Classes

##### HighSpeedDataCollector
Main class for high-performance cryptocurrency data collection.

```python
class HighSpeedDataCollector:
    def __init__(self, config: Config):
        """Initialize high-speed collector with parallel processing"""
        
    def collect_all_usdt_pairs(self) -> Dict[str, Any]:
        """Collect data for all USDT trading pairs"""
        
    def collect_symbol_data(self, symbol: str, interval: str) -> bool:
        """Collect data for specific symbol and interval"""
        
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about data collection progress"""
```

##### HTXAPIClient
HTX API client with authentication and rate limiting.

```python
class HTXAPIClient:
    def __init__(self, config: Config):
        """Initialize HTX client with API credentials"""
        
    def get_klines(self, symbol: str, interval: str, size: int = 1000) -> List[Dict]:
        """Get candlestick data with authentication"""
        
    def get_all_symbols(self) -> List[Dict]:
        """Get all available trading symbols"""
        
    def test_connection(self) -> bool:
        """Test API connection and authentication"""
```

##### CCXTDataCollector
Multi-exchange data collector using CCXT library.

```python
class CCXTDataCollector:
    def __init__(self, config: Config):
        """Initialize with multiple exchange support"""
        
    def collect_from_multiple_exchanges(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Collect data from multiple exchanges for comparison"""
        
    def get_supported_exchanges(self) -> List[str]:
        """Get list of supported exchanges"""
```

#### Configuration Classes

##### Config (EPDStocksUpdater)
Production-ready configuration for Chinese stock data collection.

```python
@dataclass
class Config:
    base_dir: str = "Chinese_Market"
    data_dir: str = ""
    enable_organized_structure: bool = True
    max_concurrent_downloads: int = 5
    enable_circuit_breaker: bool = True
    max_retries: int = 3
    retry_delay_base: float = 1.0
    
    def get_stock_file_path(self, symbol: str, exchange: str) -> Path:
        """Get organized file path for stock symbol"""
```

##### Config (EPDHuobiUpdater)
High-performance configuration for cryptocurrency data collection.

```python
@dataclass
class Config:
    htx_access_key: str = ""
    htx_secret_key: str = ""
    base_dir: str = "Market_Data"
    enable_crypto: bool = True
    crypto_intervals: List[str] = field(default_factory=list)
    min_volume_threshold: int = 10000
    
    def load_from_file(self, config_file: str = "htx_config.json"):
        """Load configuration from JSON file"""
```

### Usage Examples

#### Basic Data Collection Pipeline
```python
# Chinese stocks
stock_config = Config(enable_organized_structure=True)
stock_manager = ChineseStockManager(stock_config)
stock_results = stock_manager.update_all_stocks()

# Cryptocurrency
crypto_config = Config()
crypto_config.load_from_file("htx_config.json")
crypto_collector = HighSpeedDataCollector(crypto_config)
crypto_results = crypto_collector.collect_all_usdt_pairs()

# Pattern analysis
orchestrator = ProfessionalTradingOrchestrator()
analysis_results = orchestrator.run_analysis()
```

---

**This API reference provides comprehensive documentation for extending and using the Early Pump Detection System. For additional examples and tutorials, see the other documentation files.**