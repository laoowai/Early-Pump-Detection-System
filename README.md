# 🚀 Early Pump Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-6.1-red.svg)](main.py)
[![M1 Optimized](https://img.shields.io/badge/M1%2FM2-Optimized-orange.svg)](main.py)

A **Professional-Grade Trading Analysis System** designed for detecting early pump opportunities in Chinese A-shares and cryptocurrency markets. Features game-like scoring, M1 MacBook optimization, and institutional-quality pattern recognition with an auto-discovery modular architecture.

#@Chinese-Market #@Cryptocurrency

## 🌟 Key Features

### 🎯 Core Capabilities
- **🧠 Auto-Discovery Modular System**: Plugin architecture that automatically discovers and loads components
- **💎 20+ Advanced Pattern Detection**: Sophisticated algorithms for institutional-grade analysis
- **🎮 Game-like Scoring System**: Engaging interface with professional-grade results
- **🚀 M1/M2 MacBook Optimization**: Optimized performance for Apple Silicon
- **🌍 Multi-Market Support**: Chinese A-shares and cryptocurrency markets
- **⚡ Multi-Timeframe Analysis**: Comprehensive analysis across different time periods
- **🔥 Professional Pattern Combinations**: 8+ specialized pattern groups

### 🎪 Professional Pattern Groups
- **🔥 ACCUMULATION ZONE**: Hidden accumulation, smart money flow detection
- **💎 BREAKOUT IMMINENT**: Coiled spring and pressure cooker patterns
- **🚀 ROCKET FUEL**: Fuel tank patterns and momentum vacuum detection
- **⚡ STEALTH MODE**: Silent accumulation and whale activity tracking
- **🌟 PERFECT STORM**: Confluence zones and technical nirvana
- **🏆 MASTER SETUP**: Professional and institutional quality setups
- **💰 MONEY MAGNET**: Cash flow positive and profit engine patterns
- **🎯 PRECISION ENTRY**: Surgical strike and sniper entry points

## 🛠 Installation

### Prerequisites
- **Python 3.8+** (Recommended: Python 3.9+ for best performance)
- **M1/M2 MacBook**: Optimized performance detection included

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System

# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

### Advanced Setup
```bash
# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage
```bash
python main.py
```

### 📈 EPDStocksUpdater.py - Chinese A-Stock Data Collection

**Purpose**: Downloads and updates Chinese A-share stock data from multiple sources.

**Data Sources**:
- **AkShare**: Free, no API key required (primary source)
- **Tushare**: Requires free API token (backup source)  
- **BaoStock**: Free backup source

**Usage**:
```bash
# Basic usage - update all stocks
python EPDStocksUpdater.py

# Check system dependencies
python EPDStocksUpdater.py --check-deps

# Update specific stocks (comma-separated)
python EPDStocksUpdater.py --symbols 000001,600519,000002
```

**Configuration**: 
- Automatically creates organized directory structure
- Stores data in `Chinese_Market/data/shanghai_6xx/`, `shenzhen_0xx/`, etc.

### 🪙 EPDHuobiUpdater.py - Cryptocurrency Data Collection

**Purpose**: Collects HTX (Huobi) spot USDT cryptocurrency data with optional API acceleration.

**Features**:
- Downloads OHLCV data for major USDT pairs
- Supports API keys for higher rate limits
- Multiple exchange fallback via CCXT

**Usage**:
```bash
# Basic usage - collect crypto data
python EPDHuobiUpdater.py

# First time setup: configure API keys (optional)
# Edit htx_config.json:
# {
#   "htx_access_key": "your_htx_access_key",
#   "htx_secret_key": "your_htx_secret_key"
# }
```

**Configuration**: Edit `htx_config.json` to add HTX API credentials for faster data collection.

### 🔍 EPDScanner.py - Pattern Analysis & Scanning

**Purpose**: Analyzes collected data for early pump detection patterns using 25+ algorithms.

**Features**:
- 10-stage professional analysis pipeline
- Game-like scoring system (Common to Legendary)
- Multi-timeframe pattern detection
- Smart money and accumulation detection

**Usage**:
```bash
# Run comprehensive analysis
python EPDScanner.py

# The scanner will automatically:
# 1. Load data from Chinese_Market/data/ directories
# 2. Apply 25+ pattern recognition algorithms
# 3. Score and rank potential pump opportunities
# 4. Display results with game-like rarity scoring
```

### Advanced Python API Usage
```python
# For advanced users - direct API access
from EPDStocksUpdater import ChineseStockManager
from EPDHuobiUpdater import CryptoDataCollector
from EPDScanner import ProfessionalPatternAnalyzer

# Update specific stocks
stock_manager = ChineseStockManager()
results = stock_manager.update_specific_stocks(['000001', '600519'])

# Collect specific crypto pairs
crypto_collector = CryptoDataCollector()
crypto_data = crypto_collector.collect_symbol('BTC_USDT')

# Run pattern analysis
analyzer = ProfessionalPatternAnalyzer()
analysis_results = analyzer.analyze_all_markets()
```

### Interactive Menu Options
1. **🏮 Chinese A-Shares Professional Analysis**: Focus on Chinese stock markets
2. **🪙 Cryptocurrency Advanced Scanning**: Crypto market analysis
3. **🌍 Global Market Domination (Both)**: Complete market coverage (default)
4. **🎯 Quick Scan (Limited Symbols)**: Fast analysis for testing
5. **🚀 Full Professional Scan (All Symbols)**: Comprehensive analysis

## 📁 Project Structure

```
Early-Pump-Detection-System/
├── 📄 README.md                    # Comprehensive project overview and documentation
├── 📦 requirements.txt             # Python dependencies and package requirements
├── 🐍 main.py                      # Main system orchestrator and entry point
│   ├── detect_m1_optimization()   # Apple Silicon performance optimization detection
│   ├── main()                     # Main entry point with interactive menu
│   ├── ProfessionalTradingOrchestrator  # Main coordination class
│   ├── ComponentRegistry          # Auto-discovery component registry
│   ├── EnhancedBlacklistManager   # Dynamic blacklist management
│   └── ProfessionalLearningSystem # Advanced pattern learning system
├── 🎭 demo.py                      # Demonstration script and system overview
│   ├── show_system_overview()     # Display system capabilities and features
│   ├── show_project_structure()   # Show project file structure
│   ├── show_quick_start()         # Display quick start instructions
│   ├── show_grading_system()      # Show professional grading system
│   ├── show_system_requirements() # Display system requirements
│   ├── show_available_documentation() # Check documentation status
│   ├── show_next_steps()          # Show recommended next steps
│   └── main()                     # Demo script entry point
├── 🧪 validate_setup.py            # System validation and setup verification
│   ├── test_project_structure()   # Validate required files and directories
│   ├── test_documentation()       # Validate documentation completeness
│   ├── test_python_syntax()       # Validate Python file syntax
│   ├── test_requirements()        # Validate requirements.txt content
│   ├── test_readme_content()      # Validate README completeness
│   └── run_validation()           # Execute all validation tests
├── 🎮 EPDScanner.py                # Pattern Analyzer Game v6.0 - Professional Trading Edition
│   ├── ProfessionalTradingOrchestrator # Enhanced pattern analysis orchestrator
│   ├── AdvancedPatternDetector    # 20+ advanced pattern detection algorithms
│   ├── EnhancedStageAnalyzer      # 7-stage analysis pipeline
│   ├── ProfessionalLearningSystem # ML-inspired pattern learning
│   ├── EnhancedBlacklistManager   # Dynamic blacklist management
│   └── ComponentRegistry          # Auto-discovery component registry
├── 📊 EPDStocksUpdater.py          # Chinese A-Share Data Manager v6.0 - Production Ready
│   ├── ChineseStockManager        # Main stock data management class
│   ├── FileAnalyzer               # Stock data file analysis and quality assessment
│   ├── DataSourceManager          # Multi-source data collection with fallback
│   ├── StockDataUpdater           # High-performance data updating with retry logic
│   ├── Config                     # Production-ready configuration management
│   └── migrate_to_organized_structure() # Directory organization and file migration
├── 🚀 EPDHuobiUpdater.py           # HTX Crypto Data Collector v5.0 - High-Speed Edition
│   ├── HighSpeedDataCollector     # Parallel cryptocurrency data collection
│   ├── HTXAPIClient               # HTX API client with authentication
│   ├── CCXTDataCollector          # Multi-exchange data collector with fallback
│   ├── HTXSigner                  # HTX API request authentication
│   └── Config                     # High-performance collection configuration
├── ⚙️ htx_config.json              # HTX/Huobi API configuration and settings
├── 🚫 .gitignore                   # Git ignore patterns for clean repository
├── 📚 docs/                        # Comprehensive documentation
│   ├── 📖 installation.md          # Detailed installation and setup guide
│   ├── 📚 user-guide.md            # Complete user manual and usage examples
│   ├── 🏗️ architecture.md          # System design and technical architecture
│   └── 📋 api-reference.md         # API documentation and component reference
├── 🔧 pattern_analyzers/           # High-level pattern analysis components
│   ├── __init__.py                 # Package initialization and auto-discovery
│   │   └── auto_discover_pattern_analyzers() # Auto-discovery function
│   ├── base_pattern_analyzer.py    # Abstract base class for pattern analyzers
│   │   ├── BasePatternAnalyzer    # Base class for all pattern analyzers
│   │   ├── analyze_symbol()       # Core symbol analysis method
│   │   ├── run_analysis()         # Batch analysis execution
│   │   ├── load_data()            # Data loading and validation
│   │   ├── calculate_basic_metrics() # Technical indicator calculations
│   │   ├── print_results()        # Formatted result display
│   │   └── save_results()         # Result persistence
│   └── professional_pattern_analyzer.py  # Enhanced analyzer with v6.1 features
│       ├── ProfessionalPatternAnalyzer # Advanced pattern analysis class
│       ├── EnhancedBlacklistManager    # Dynamic blacklist management
│       ├── ProfessionalLearningSystem  # ML-inspired pattern learning
│       └── _initialize_auto_discovered_components() # Component initialization
├── 🔍 pattern_detectors/           # Core pattern detection algorithms
│   ├── __init__.py                 # Package initialization and auto-discovery
│   │   └── auto_discover_detectors() # Auto-discovery function
│   ├── base_detector.py            # Abstract base class for pattern detectors
│   │   ├── BasePatternDetector    # Base class for all detectors
│   │   ├── detect_patterns()      # Main pattern detection method
│   │   ├── get_supported_patterns() # List supported pattern types
│   │   ├── detect_specific_pattern() # Single pattern detection
│   │   └── calculate_technical_indicators() # Technical analysis tools
│   └── advanced_pattern_detector.py # 20+ sophisticated detection algorithms
│       ├── AdvancedPatternDetector # Advanced pattern detection class
│       ├── detect_hidden_accumulation() # Smart money flow detection
│       ├── detect_smart_money_flow()    # Institutional activity detection
│       ├── detect_whale_accumulation()  # Large holder analysis
│       ├── detect_coiled_spring()       # Breakout preparation detection
│       ├── detect_momentum_vacuum()     # Momentum gap analysis
│       ├── detect_fibonacci_retracement() # Fibonacci level analysis
│       └── detect_elliott_wave_3()      # Elliott Wave pattern detection
├── 📊 stage_analyzers/             # Multi-stage market analysis pipeline
│   ├── __init__.py                 # Package initialization and auto-discovery
│   │   └── auto_discover_stage_analyzers() # Auto-discovery function
│   ├── base_stage_analyzer.py      # Abstract base class for stage analysis
│   │   ├── BaseStageAnalyzer      # Base class for all stage analyzers
│   │   ├── run_all_stages()       # Execute complete stage analysis
│   │   ├── run_specific_stage()   # Single stage execution
│   │   ├── calculate_support_resistance() # Support/resistance calculation
│   │   ├── calculate_volatility_metrics() # Volatility analysis
│   │   └── calculate_momentum_indicators() # Momentum indicator calculation
│   └── enhanced_stage_analyzer.py  # Advanced multi-stage analysis system
│       ├── EnhancedStageAnalyzer  # Enhanced stage analysis class
│       ├── stage_1_smart_money_detection() # Smart money flow analysis
│       ├── stage_2_accumulation_analysis() # Accumulation pattern analysis
│       ├── stage_3_technical_confluence()  # Technical indicator confluence
│       ├── stage_4_volume_profiling()      # Volume profile analysis
│       ├── stage_5_momentum_analysis()     # Momentum and trend analysis
│       ├── stage_6_pattern_recognition()   # Advanced pattern recognition
│       └── stage_7_risk_assessment()       # Risk evaluation and scoring
├── ⏰ timeframe_analyzers/         # Multi-timeframe analysis components
│   ├── __init__.py                 # Package initialization and auto-discovery
│   │   └── auto_discover_timeframe_analyzers() # Auto-discovery function
│   ├── base_timeframe_analyzer.py  # Abstract base class for timeframe analysis
│   │   ├── BaseTimeframeAnalyzer  # Base class for all timeframe analyzers
│   │   ├── analyze_timeframe()    # Core timeframe analysis method
│   │   ├── find_best_combination() # Pattern combination optimization
│   │   ├── determine_consolidation_type() # Consolidation pattern classification
│   │   ├── get_supported_timeframes() # Available timeframe periods
│   │   └── calculate_entry_setup() # Entry point calculation
│   └── enhanced_multi_timeframe_analyzer.py # Advanced multi-timeframe correlation
│       ├── EnhancedMultiTimeframeAnalyzer # Enhanced timeframe analysis class
│       ├── _detect_timeframe_specific_patterns() # Timeframe-specific pattern detection
│       ├── _detect_technical_patterns() # Technical indicator patterns
│       ├── _detect_volume_patterns() # Volume-based pattern analysis
│       ├── _calculate_combination_score() # Pattern combination scoring
│       └── _adjust_data_for_timeframe() # Data sampling adjustment
├── 🧪 tests/                       # Testing infrastructure and test cases
│   └── test_system.py              # System integration and component tests
├── 📝 examples/                    # Usage examples and sample implementations
│   └── README.md                   # Examples documentation and usage guide
├── 🤝 CONTRIBUTING.md              # Development guidelines and contribution guide
└── 📋 CHANGELOG.md                 # Version history and release notes
```

### 🗂️ File and Function Details

#### Core System Files
- **`main.py`**: Central orchestrator implementing the ProfessionalTradingOrchestrator with auto-discovery
  - **Key Functions**: `detect_m1_optimization()`, `main()` - Main entry and M1 optimization
  - **Key Classes**: `ProfessionalTradingOrchestrator`, `ComponentRegistry`, `EnhancedBlacklistManager`, `ProfessionalLearningSystem`
- **`demo.py`**: Interactive demonstration showcasing system capabilities without full dependencies  
  - **Key Functions**: `show_system_overview()`, `show_project_structure()`, `show_quick_start()`, `show_grading_system()`
- **`validate_setup.py`**: Comprehensive validation script for system setup verification
  - **Key Functions**: `test_project_structure()`, `test_documentation()`, `test_python_syntax()`, `run_validation()`

#### Data Collection Tools
- **`EPDScanner.py`**: Pattern Analyzer Game v6.0 - Professional Trading Edition with enhanced 20+ pattern detection
  - **Key Classes**: `ProfessionalTradingOrchestrator`, `AdvancedPatternDetector`, `EnhancedStageAnalyzer`, `ProfessionalLearningSystem`
  - **Key Features**: Auto-discovery architecture, M1/M2 optimization, professional grading system, multi-market support
- **`EPDStocksUpdater.py`**: Chinese A-Share Data Manager v6.0 - Production-ready stock data collection and management
  - **Key Classes**: `ChineseStockManager`, `FileAnalyzer`, `DataSourceManager`, `StockDataUpdater`
  - **Key Features**: Multi-source data collection, intelligent retry logic, organized directory structure, data quality assessment
- **`EPDHuobiUpdater.py`**: HTX Crypto Data Collector v5.0 - High-speed cryptocurrency data collection
  - **Key Classes**: `HighSpeedDataCollector`, `HTXAPIClient`, `CCXTDataCollector`, `HTXSigner`
  - **Key Features**: Parallel processing, API authentication, multi-exchange support, concurrent data fetching
- **`htx_config.json`**: HTX/Huobi API configuration file with credentials and collection settings
  - **Configuration**: API keys, data directories, intervals, currencies, volume thresholds

#### Analysis Components (Auto-Discovery Architecture)

##### Pattern Analyzers (`pattern_analyzers/`)
- **Base Classes**: `BasePatternAnalyzer` - Core interface for all pattern analyzers
  - **Core Methods**: `analyze_symbol()`, `run_analysis()`, `load_data()`, `calculate_basic_metrics()`, `print_results()`, `save_results()`
- **Professional Implementation**: `ProfessionalPatternAnalyzer` - Enhanced v6.1 analyzer with auto-discovery
  - **Advanced Features**: Component initialization, blacklist management, learning system integration
- **Auto-Discovery**: `auto_discover_pattern_analyzers()` - Dynamic component loading

##### Pattern Detectors (`pattern_detectors/`)
- **Base Classes**: `BasePatternDetector` - Core interface for pattern detection algorithms
  - **Core Methods**: `detect_patterns()`, `get_supported_patterns()`, `detect_specific_pattern()`
- **Advanced Implementation**: `AdvancedPatternDetector` - 20+ sophisticated detection algorithms
  - **Pattern Detection**: `detect_hidden_accumulation()`, `detect_smart_money_flow()`, `detect_whale_accumulation()`, `detect_coiled_spring()`, `detect_momentum_vacuum()`, `detect_fibonacci_retracement()`, `detect_elliott_wave_3()`
- **Auto-Discovery**: `auto_discover_detectors()` - Dynamic detector loading

##### Stage Analyzers (`stage_analyzers/`)
- **Base Classes**: `BaseStageAnalyzer` - Core interface for multi-stage analysis
  - **Core Methods**: `run_all_stages()`, `calculate_support_resistance()`, `calculate_volatility_metrics()`, `calculate_momentum_indicators()`
- **Enhanced Implementation**: `EnhancedStageAnalyzer` - Advanced 7-stage analysis pipeline
  - **Stage Analysis**: `stage_1_smart_money_detection()`, `stage_2_accumulation_analysis()`, `stage_3_technical_confluence()`, `stage_4_volume_profiling()`, `stage_5_momentum_analysis()`, `stage_6_pattern_recognition()`, `stage_7_risk_assessment()`
- **Auto-Discovery**: `auto_discover_stage_analyzers()` - Dynamic stage analyzer loading

##### Timeframe Analyzers (`timeframe_analyzers/`)
- **Base Classes**: `BaseTimeframeAnalyzer` - Core interface for multi-timeframe analysis
  - **Core Methods**: `analyze_timeframe()`, `find_best_combination()`, `determine_consolidation_type()`, `get_supported_timeframes()`, `calculate_entry_setup()`
- **Enhanced Implementation**: `EnhancedMultiTimeframeAnalyzer` - Advanced timeframe correlation system
  - **Timeframe Analysis**: `_detect_timeframe_specific_patterns()`, `_detect_technical_patterns()`, `_detect_volume_patterns()`, `_calculate_combination_score()`, `_adjust_data_for_timeframe()`
  - **Supported Timeframes**: D1, D3, D6, D11, D21, D33, D55, D89 (Fibonacci-based periods)
- **Auto-Discovery**: `auto_discover_timeframe_analyzers()` - Dynamic timeframe analyzer loading

#### Documentation & Support
- **`docs/`**: Complete documentation suite with installation, usage, and architecture guides
- **`examples/`**: Practical usage examples and integration patterns  
- **`tests/`**: Testing infrastructure for system validation (`test_system.py`)

### Market Data Structure

#### Directory Structure (Current Implementation)
```
Chinese_Market/data/
├── shanghai_6xx/          # Shanghai Stock Exchange (6xx codes)
│   └── 600000.csv         # Example: 浦发银行 (SPDB)
├── shenzhen_0xx/          # Shenzhen Stock Exchange (0xx codes)  
│   └── 000001.csv         # Example: 平安银行 (Ping An Bank)
└── huobi/                 # Cryptocurrency data
    └── spot_usdt/1d/      # Daily USDT trading pairs
        └── XEN-USDT.csv   # Example: XEN token daily data
```

#### CSV File Formats

**Chinese Stocks Format** (Shanghai & Shenzhen):
```csv
Date,Close,Low,Volume,振幅,Open,股票代码,High,股票名称
1999-11-10,-1.3300,-1.4400,1740850,-10.4500,-1.0600,600000,-1.0200,浦发银行
```
- **Columns**: Date, Close, Low, Volume, 振幅(Amplitude), Open, 股票代码(Stock Code), High, 股票名称(Stock Name)
- **Date Format**: YYYY-MM-DD
- **Chinese Headers**: Mixed Chinese/English column names
- **Price Values**: May include negative values (adjusted prices)

**Cryptocurrency Format** (Huobi):
```csv
timestamp,open,high,low,close,volume,volume_quote,symbol,price_change
2024-08-07,8e-08,8.1e-08,7.7e-08,7.7e-08,4830221976723.218,380291.87812148104,XEN-USDT,-4.9383
```
- **Columns**: timestamp, open, high, low, close, volume, volume_quote, symbol, price_change
- **Timestamp Format**: YYYY-MM-DD
- **Scientific Notation**: Prices in scientific notation for small values
- **Additional Data**: Includes volume_quote and price_change metrics

> **Note**: File structures may vary across different data sources and timeframes. This documentation serves as a reference for current implementation and maintenance purposes. Additional exchanges (beijing_8xx, binance) may be added with different formats.

## 📊 System Architecture

### 🔌 Auto-Discovery Components

#### Pattern Analyzers (`pattern_analyzers/`)
- **BasePatternAnalyzer**: Core interface for pattern analysis
- **ProfessionalPatternAnalyzer**: Enhanced analysis with v6.1 features
- Auto-discovery of custom analyzers

#### Pattern Detectors (`pattern_detectors/`)
- **BasePatternDetector**: Core pattern detection interface
- **AdvancedPatternDetector**: 20+ sophisticated pattern algorithms
- Auto-discovery of custom detectors

#### Stage Analyzers (`stage_analyzers/`)
- **BaseStageAnalyzer**: Core stage analysis interface
- **EnhancedStageAnalyzer**: Multi-stage analysis pipeline
- Auto-discovery of custom stage analyzers

#### Timeframe Analyzers (`timeframe_analyzers/`)
- **BaseTimeframeAnalyzer**: Core timeframe analysis interface
- **EnhancedMultiTimeframeAnalyzer**: Multi-timeframe correlation system
- Auto-discovery of custom timeframe analyzers

### 🧠 Core Systems
- **ComponentRegistry**: Auto-discovery system for all components
- **EnhancedBlacklistManager**: Dynamic blacklist management
- **ProfessionalLearningSystem**: Advanced pattern tracking and learning
- **ProfessionalTradingOrchestrator**: Main coordination system

## 🎯 Performance Optimization

### M1/M2 MacBook Optimization
- **Automatic Detection**: System automatically detects Apple Silicon
- **Optimal Process Count**: Dynamically adjusts based on hardware
- **Enhanced Performance**: 2x chunk multiplier for M1/M2 systems
- **Estimated Performance**: 5-15 minutes for full scan (vs 10-30 minutes on Intel)

### Processing Configuration
- **Multi-threading**: Optimized concurrent processing
- **Memory Management**: Efficient data handling for large datasets
- **Batch Processing**: Intelligent symbol batching

## 📈 Analysis Output

### Professional Grading System
- **👑 Institutional Grade**: 85+ score
- **🏆 Professional Grade**: 70-84 score
- **⭐ Intermediate Grade**: 55-69 score

### Result Categories
- **Market Analysis**: Comprehensive market overview
- **Special Patterns**: Detected pattern combinations
- **Professional Predictions**: Top-ranked opportunities
- **Performance Metrics**: System performance statistics

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set data directory
export EPDS_DATA_DIR="/path/to/your/data"

# Optional: Set number of processes
export EPDS_PROCESSES=8
```

### Blacklist Management
- **Static Blacklist**: Pre-configured low-quality symbols
- **Dynamic Blacklist**: Learning-based exclusion system
- **Custom Blacklist**: User-defined exclusions

## 🧪 Development

### Adding Custom Components

#### Custom Pattern Analyzer
```python
from pattern_analyzers.base_pattern_analyzer import BasePatternAnalyzer

class MyPatternAnalyzer(BasePatternAnalyzer):
    def analyze_symbol(self, symbol, df):
        # Your custom analysis logic
        pass
```

#### Custom Pattern Detector
```python
from pattern_detectors.base_detector import BasePatternDetector

class MyPatternDetector(BasePatternDetector):
    def detect_patterns(self, df):
        # Your custom detection logic
        pass
```

### Testing
```bash
# Run basic functionality test
python -c "import main; print('✅ All imports successful')"

# Test with limited symbols for quick validation
# Select option 4 (Quick Scan) when prompted
python main.py
```

## 📚 Documentation

- **[Installation Guide](docs/installation.md)**: Detailed setup instructions
- **[User Guide](docs/user-guide.md)**: Complete usage documentation
- **[API Reference](docs/api-reference.md)**: Component and class documentation
- **[Architecture Guide](docs/architecture.md)**: System design and components
- **[Contributing Guide](CONTRIBUTING.md)**: Development guidelines

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System
pip install -r requirements.txt

# Run tests (when available)
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Not financial advice. Trading involves risk of loss. Always do your own research and consult with financial professionals before making investment decisions.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/laoowai/Early-Pump-Detection-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/laoowai/Early-Pump-Detection-System/discussions)

## 🔄 Version History

- **v6.1**: Current version with auto-discovery modular architecture
- **Enhanced Architecture**: Plugin system and component auto-discovery
- **M1 Optimization**: Apple Silicon performance optimization
- **Professional Patterns**: 8+ pattern combination groups

---

**Built with ❤️ for traders and developers who demand institutional-quality analysis**
