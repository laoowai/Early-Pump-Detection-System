# 🤝 Contributing to Early Pump Detection System

Thank you for your interest in contributing to the Early Pump Detection System (EPDS)! This document provides guidelines for contributing to the project.

## 🌟 How to Contribute

### Types of Contributions

1. **🐛 Bug Reports**: Help us identify and fix issues
2. **✨ Feature Requests**: Suggest new features and improvements
3. **📝 Documentation**: Improve or add documentation
4. **🔧 Code Contributions**: Add new features or fix bugs
5. **🧪 Testing**: Add tests and improve test coverage
6. **🎯 Pattern Development**: Create new pattern detection algorithms
7. **⚡ Performance Improvements**: Optimize existing code

## 🚀 Getting Started

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/Early-Pump-Detection-System.git
   cd Early-Pump-Detection-System
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install development dependencies (optional)
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Verify Installation**
   ```bash
   # Run system tests
   python tests/test_system.py
   
   # Test basic functionality
   python -c "import main; print('✅ Development setup complete')"
   ```

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run system tests
   python tests/test_system.py
   
   # Run specific tests (if using pytest)
   pytest tests/ -v
   
   # Test with actual data (quick scan)
   python main.py  # Choose option 4 for quick test
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   git push origin your-branch-name
   ```

5. **Create Pull Request**
   - Open a PR on GitHub
   - Use the PR template
   - Ensure all tests pass

## 📋 Coding Standards

### Python Style Guide

We follow PEP 8 with some project-specific guidelines:

#### Code Formatting
```python
# Use Black for automatic formatting
black main.py pattern_analyzers/ pattern_detectors/ stage_analyzers/

# Line length: 88 characters (Black default)
# Use double quotes for strings
# 4 spaces for indentation
```

#### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np

# Local imports
from pattern_analyzers.base_pattern_analyzer import BasePatternAnalyzer
```

#### Naming Conventions
```python
# Classes: PascalCase
class AdvancedPatternDetector:
    pass

# Functions and variables: snake_case
def analyze_market_data():
    pattern_results = []

# Constants: UPPER_SNAKE_CASE
PROFESSIONAL_PATTERNS = {}

# Private methods: _leading_underscore
def _internal_calculation(self):
    pass
```

### Documentation Standards

#### Docstrings
```python
def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> MultiTimeframeAnalysis:
    """
    Analyze a single symbol across multiple timeframes
    
    Args:
        symbol (str): Symbol identifier (e.g., '600519', 'BTC_USDT')
        df (pd.DataFrame): OHLCV data with columns [Date, Open, High, Low, Close, Volume]
    
    Returns:
        MultiTimeframeAnalysis: Complete analysis result with:
            - overall_score: Composite score (0-100)
            - stage_results: List of stage analysis results
            - pattern_combinations: Detected pattern groups
    
    Raises:
        ValueError: If data validation fails
        AnalysisError: If analysis cannot be completed
    
    Example:
        >>> analyzer = ProfessionalPatternAnalyzer()
        >>> result = analyzer.analyze_symbol("600519", stock_data)
        >>> print(f"Score: {result.overall_score}")
    """
```

#### Type Hints
```python
from typing import Dict, List, Optional, Tuple, Any, Union

def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
    """Type hints are required for all public methods"""
    pass

def process_results(self, results: List[AnalysisResult]) -> Optional[str]:
    """Use Optional for values that can be None"""
    pass
```

## 🏗️ Architecture Guidelines

### Adding New Components

#### New Pattern Analyzer
```python
# File: pattern_analyzers/my_custom_analyzer.py
from pattern_analyzers.base_pattern_analyzer import BasePatternAnalyzer

class MyCustomAnalyzer(BasePatternAnalyzer):
    """
    Custom analyzer implementing specific trading strategy
    
    This analyzer focuses on [describe your strategy]
    """
    
    def __init__(self, data_dir: str = "Chinese_Market/data"):
        super().__init__(data_dir)
        self.name = "My Custom Analyzer"
        self.version = "1.0"
        # Add your custom initialization
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> MultiTimeframeAnalysis:
        """Implement your custom analysis logic"""
        # Your implementation here
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Custom data validation if needed"""
        return super().validate_data(df)  # Use base validation or customize
```

#### New Pattern Detector
```python
# File: pattern_detectors/my_custom_detector.py
from pattern_detectors.base_detector import BasePatternDetector, PatternType

class MyCustomDetector(BasePatternDetector):
    """
    Custom pattern detector for [specific patterns]
    
    Detects: [list the patterns this detector finds]
    """
    
    def __init__(self):
        super().__init__()
        self.name = "My Custom Detector"
        self.version = "1.0"
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Implement your pattern detection logic"""
        patterns = []
        
        # Example pattern detection
        pattern = self._detect_my_pattern(df)
        if pattern:
            patterns.append(pattern)
        
        return patterns
    
    def _detect_my_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Detect specific pattern"""
        # Your detection logic here
        pass
```

### Auto-Discovery Compliance

Your components will be automatically discovered if they:

1. **Inherit from the correct base class**
2. **Are placed in the correct directory**
3. **Don't start with underscore**
4. **Can be instantiated without required parameters**

```python
# ✅ Good - Will be auto-discovered
class GoodAnalyzer(BasePatternAnalyzer):
    def __init__(self, data_dir: str = "Chinese_Market/data"):
        super().__init__(data_dir)

# ❌ Bad - Won't be auto-discovered (required parameter)
class BadAnalyzer(BasePatternAnalyzer):
    def __init__(self, required_param: str, data_dir: str = "Chinese_Market/data"):
        super().__init__(data_dir)
        self.required_param = required_param
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── test_system.py           # System integration tests
├── test_pattern_analyzers.py   # Pattern analyzer tests
├── test_pattern_detectors.py   # Pattern detector tests
├── test_stage_analyzers.py     # Stage analyzer tests
└── test_data/              # Test data files
```

### Writing Tests

#### Unit Tests
```python
import unittest
from unittest.mock import patch, MagicMock

class TestMyComponent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = create_sample_ohlcv_data()
        self.component = MyComponent()
    
    def test_basic_functionality(self):
        """Test basic component functionality"""
        result = self.component.analyze(self.sample_data)
        self.assertIsNotNone(result)
        self.assertGreater(result.score, 0)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        with self.assertRaises(ValueError):
            self.component.analyze(pd.DataFrame())  # Empty data
    
    @patch('external_api.get_data')
    def test_with_mocked_dependency(self, mock_get_data):
        """Test with mocked external dependencies"""
        mock_get_data.return_value = self.sample_data
        result = self.component.analyze_with_external_data("SYMBOL")
        self.assertIsNotNone(result)
```

#### Integration Tests
```python
class TestIntegration(unittest.TestCase):
    def test_end_to_end_analysis(self):
        """Test complete analysis pipeline"""
        orchestrator = ProfessionalTradingOrchestrator()
        
        # Use test data
        with patch.object(orchestrator, '_load_symbols') as mock_load:
            mock_load.return_value = ["TEST_SYMBOL"]
            results = orchestrator.run_analysis(max_symbols=1)
            
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], MultiTimeframeAnalysis)
```

### Test Data Creation
```python
def create_sample_ohlcv_data(days: int = 100, trend: str = "up") -> pd.DataFrame:
    """Create realistic test data for testing"""
    dates = pd.date_range('2023-01-01', periods=days)
    
    if trend == "up":
        base_prices = np.linspace(100, 150, days)
    elif trend == "down":
        base_prices = np.linspace(150, 100, days)
    else:  # sideways
        base_prices = np.full(days, 100) + np.random.normal(0, 5, days)
    
    # Add realistic noise and ensure OHLC consistency
    data = pd.DataFrame({
        'Date': dates,
        'Open': base_prices + np.random.normal(0, 1, days),
        'High': base_prices + np.abs(np.random.normal(2, 1, days)),
        'Low': base_prices - np.abs(np.random.normal(2, 1, days)),
        'Close': base_prices + np.random.normal(0, 0.5, days),
        'Volume': np.random.randint(100000, 1000000, days)
    })
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[i, 'High'] = max(data.loc[i, ['Open', 'High', 'Close']])
        data.loc[i, 'Low'] = min(data.loc[i, ['Open', 'Low', 'Close']])
    
    return data
```

## 📊 Performance Guidelines

### Optimization Principles

1. **Vectorized Operations**: Use pandas/numpy vectorized operations
2. **Memory Efficiency**: Avoid unnecessary data copies
3. **CPU Optimization**: Leverage multiprocessing for I/O bound tasks
4. **M1 Compatibility**: Ensure compatibility with Apple Silicon

```python
# ✅ Good - Vectorized operation
returns = df['Close'].pct_change()

# ❌ Bad - Loop-based calculation
returns = []
for i in range(1, len(df)):
    ret = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
    returns.append(ret)
```

### Memory Management
```python
# ✅ Good - Process data in chunks
def process_large_dataset(symbols: List[str]):
    chunk_size = 100
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        process_chunk(chunk)
        # Memory is freed between chunks

# ❌ Bad - Load all data at once
def process_large_dataset_bad(symbols: List[str]):
    all_data = [load_symbol(s) for s in symbols]  # Memory intensive
    return process_all(all_data)
```

## 📝 Documentation Guidelines

### Code Comments
```python
class AdvancedPatternDetector:
    def detect_accumulation_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """
        Detect institutional accumulation patterns
        
        This method identifies periods where large players are quietly
        accumulating positions, indicated by:
        - Consistent volume above average
        - Price stability despite volume
        - Narrow trading ranges
        """
        
        # Calculate volume metrics (explain why this calculation)
        avg_volume = df['Volume'].rolling(20).mean()
        volume_ratio = df['Volume'] / avg_volume
        
        # Identify accumulation periods (explain the thresholds)
        accumulation_mask = (
            (volume_ratio > 1.5) &  # 50% above average volume
            (df['Close'].rolling(10).std() < df['Close'].mean() * 0.02)  # Low volatility
        )
        
        # Rest of implementation...
```

### README Updates
When adding new features, update the README:

```markdown
### 🆕 New Features (v6.2)
- **Custom Pattern X**: Added detection for institutional rotation patterns
- **Enhanced M1 Optimization**: 25% performance improvement on Apple Silicon
- **New Market Support**: Added support for European markets
```

## 🔍 Review Process

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] **Code Quality**
  - [ ] Follows coding standards
  - [ ] Has appropriate type hints
  - [ ] Includes comprehensive docstrings
  - [ ] No unused imports or variables

- [ ] **Testing**
  - [ ] All existing tests pass
  - [ ] New tests for added functionality
  - [ ] Integration tests updated if needed
  - [ ] Performance impact assessed

- [ ] **Documentation**
  - [ ] README updated if needed
  - [ ] API documentation updated
  - [ ] Code comments explain complex logic
  - [ ] Changelog entry added

- [ ] **Compatibility**
  - [ ] Works on Python 3.8+
  - [ ] Compatible with M1/M2 MacBooks
  - [ ] No breaking changes to public API

### Code Review Criteria

Reviewers will check:

1. **Functionality**: Does the code work as intended?
2. **Architecture**: Does it fit well with existing design?
3. **Performance**: Any negative performance impact?
4. **Security**: No security vulnerabilities introduced?
5. **Maintainability**: Is the code easy to understand and modify?

## 🎯 Specific Contribution Areas

### High-Priority Areas

1. **Pattern Detection Algorithms**
   - Implement new technical patterns
   - Improve existing pattern accuracy
   - Add cryptocurrency-specific patterns

2. **Performance Optimization**
   - M1/M2 MacBook optimizations
   - Memory usage improvements
   - Parallel processing enhancements

3. **Market Support**
   - Additional exchange support
   - New market types (options, futures)
   - International markets

4. **Machine Learning Integration**
   - Pattern learning from historical data
   - Adaptive scoring algorithms
   - Market regime detection

### Pattern Development Guide

To add a new pattern:

1. **Research the Pattern**
   - Study the pattern characteristics
   - Identify mathematical formulation
   - Gather historical examples

2. **Implement Detection**
   ```python
   def detect_your_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
       """
       Detect [Pattern Name]
       
       Pattern characteristics:
       - [List key characteristics]
       - [Mathematical conditions]
       - [Volume requirements]
       """
       # Implementation
   ```

3. **Add Tests**
   ```python
   def test_your_pattern_detection(self):
       # Create data that should trigger pattern
       test_data = create_pattern_data("your_pattern")
       result = detector.detect_your_pattern(test_data)
       self.assertIsNotNone(result)
   ```

4. **Document the Pattern**
   - Add to pattern list in README
   - Include in API documentation
   - Add usage examples

## 🆘 Getting Help

### Community Resources

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check docs/ directory for detailed guides

### Contact Information

- **Maintainers**: See CODEOWNERS file
- **Security Issues**: Use GitHub Security Advisories
- **General Questions**: Open a Discussion on GitHub

### Development Environment Issues

If you encounter setup issues:

1. **Check Python Version**: Ensure Python 3.8+
2. **Virtual Environment**: Use a clean virtual environment
3. **Dependencies**: Verify all requirements are installed
4. **System Info**: Run `python tests/test_system.py` for diagnosis

---

**Thank you for contributing to the Early Pump Detection System! Your contributions help make professional-grade trading analysis accessible to everyone. 🚀**