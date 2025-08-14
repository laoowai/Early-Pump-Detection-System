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

# Import shared data structures
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
        Load data for a given symbol
        
        Args:
            symbol: Symbol to load data for
            market_type: Type of market
            
        Returns:
            DataFrame with OHLCV data or None if loading fails
        """
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
            
            # Load and clean data
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
            return None
    
    def _find_crypto_path(self) -> Optional[Path]:
        """Find available crypto data path"""
        for path in self.crypto_paths:
            if path.exists():
                return path
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
