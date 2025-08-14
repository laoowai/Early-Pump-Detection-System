# Examples and Usage

This directory contains practical examples and usage scenarios for the Early Pump Detection System.

## 📁 Directory Structure

```
examples/
├── basic_usage/           # Simple usage examples
├── advanced_patterns/     # Advanced pattern detection examples
├── custom_components/     # Custom component development examples
├── data_samples/         # Sample data for testing
└── integration/          # Integration with other systems
```

## 🚀 Quick Start Examples

### Basic Analysis
```python
#!/usr/bin/env python3
"""Basic usage example for EPDS"""

from main import ProfessionalTradingOrchestrator, MarketType

def basic_analysis_example():
    """Run basic market analysis"""
    
    # Initialize the orchestrator
    orchestrator = ProfessionalTradingOrchestrator()
    
    # Run analysis on Chinese stocks only
    results = orchestrator.run_analysis(
        market_type=MarketType.CHINESE_STOCK,
        max_symbols=50  # Limit for quick testing
    )
    
    # Print top opportunities
    top_results = sorted(results, key=lambda x: x.overall_score, reverse=True)[:10]
    
    print("🏆 Top 10 Opportunities:")
    for i, result in enumerate(top_results, 1):
        print(f"{i:2d}. {result.symbol:10s} | Score: {result.overall_score:5.1f} | "
              f"Confidence: {result.prediction_confidence:.2f}")
    
    return results

if __name__ == "__main__":
    results = basic_analysis_example()
```

### Custom Pattern Detector Example
```python
#!/usr/bin/env python3
"""Example of creating a custom pattern detector"""

import pandas as pd
import numpy as np
from typing import List, Optional
from pattern_detectors.base_detector import BasePatternDetector, PatternResult, PatternType

class MomentumBreakoutDetector(BasePatternDetector):
    """
    Custom detector for momentum breakout patterns
    
    Detects stocks breaking out of consolidation with strong momentum
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Momentum Breakout Detector"
        self.version = "1.0"
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect momentum breakout patterns"""
        patterns = []
        
        # Detect consolidation breakout
        breakout = self._detect_consolidation_breakout(df)
        if breakout:
            patterns.append(breakout)
        
        # Detect volume breakout
        volume_breakout = self._detect_volume_breakout(df)
        if volume_breakout:
            patterns.append(volume_breakout)
        
        return patterns
    
    def _detect_consolidation_breakout(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect price breaking out of consolidation range"""
        
        if len(df) < 30:
            return None
        
        # Look for consolidation period (low volatility)
        lookback = 20
        recent_data = df.tail(lookback)
        
        # Calculate volatility (standard deviation of returns)
        returns = recent_data['Close'].pct_change()
        volatility = returns.std()
        
        # Calculate price range
        price_range = (recent_data['High'].max() - recent_data['Low'].min()) / recent_data['Close'].mean()
        
        # Conditions for consolidation breakout
        is_low_volatility = volatility < 0.02  # Less than 2% daily volatility
        is_tight_range = price_range < 0.15    # Price range less than 15%
        
        # Check for breakout (recent price above consolidation range)
        consolidation_high = recent_data['High'].max()
        current_price = df['Close'].iloc[-1]
        is_breakout = current_price > consolidation_high
        
        if is_low_volatility and is_tight_range and is_breakout:
            return PatternResult(
                pattern_type=PatternType.BREAKOUT_PATTERN,
                confidence=0.8,
                start_index=len(df) - lookback,
                end_index=len(df) - 1,
                key_levels={
                    'breakout_level': consolidation_high,
                    'support_level': recent_data['Low'].min(),
                    'target_level': consolidation_high * 1.1  # 10% target
                },
                volume_confirmation=self._check_volume_confirmation(df),
                strength=min(100, (current_price / consolidation_high - 1) * 500),
                reliability=0.75
            )
        
        return None
    
    def _detect_volume_breakout(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect volume breakout pattern"""
        
        if len(df) < 20:
            return None
        
        # Calculate average volume
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        
        # Volume breakout condition
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 2.0:  # Volume 2x above average
            return PatternResult(
                pattern_type=PatternType.BREAKOUT_PATTERN,
                confidence=0.6,
                start_index=len(df) - 5,
                end_index=len(df) - 1,
                key_levels={
                    'volume_level': current_volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': volume_ratio
                },
                volume_confirmation=True,
                strength=min(100, volume_ratio * 25),
                reliability=0.65
            )
        
        return None
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if recent volume supports the pattern"""
        if len(df) < 5:
            return False
        
        recent_volume = df['Volume'].tail(3).mean()
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        
        return recent_volume > avg_volume * 1.3  # 30% above average

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=50)
    prices = np.concatenate([
        np.full(30, 100) + np.random.normal(0, 1, 30),  # Consolidation
        np.linspace(100, 110, 20) + np.random.normal(0, 0.5, 20)  # Breakout
    ])
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(100000, 500000, 50)
    })
    
    # Test the detector
    detector = MomentumBreakoutDetector()
    patterns = detector.detect_patterns(sample_data)
    
    print(f"Detected {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"- {pattern.pattern_type.value}: Confidence {pattern.confidence:.2f}")
```

---

**For complete examples and usage scenarios, see the individual files in this directory. Each subdirectory contains specific examples for different use cases.**