# 📊 User Guide

## Overview

The Early Pump Detection System (EPDS) is a professional-grade trading analysis tool that helps identify early pump opportunities in Chinese A-shares and cryptocurrency markets. This guide covers everything you need to know to use the system effectively.

## 🚀 Getting Started

### First Run
```bash
python main.py
```

You'll see the Professional Pattern Analyzer interface:
```
====================================================================================================
   🎮 PROFESSIONAL PATTERN ANALYZER v6.1
   🧠 Auto-Discovery Modular Trading Analysis
   💎 Unlimited Patterns | Enhanced Architecture
   🔌 Plugin System - Add Components Without Code Changes!
   🚀 M1/M2 MacBook Optimization ACTIVE!  (if detected)
====================================================================================================
```

### Interactive Menu
```
📊 Select Your Professional Quest:
1. 🏮 Chinese A-Shares Professional Analysis
2. 🪙 Cryptocurrency Advanced Scanning  
3. 🌍 Global Market Domination (Both)
4. 🎯 Quick Scan (Limited Symbols)
5. 🚀 Full Professional Scan (All Symbols)

Choose your trading destiny (1-5, default=3):
```

## 📈 Analysis Options

### 1. 🏮 Chinese A-Shares Analysis
- **Focus**: Chinese stock markets only
- **Coverage**: Shanghai (6xx), Shenzhen (0xx), Beijing (8xx) exchanges
- **Best for**: A-share specialists, domestic market focus
- **Estimated time**: 3-10 minutes (depending on hardware)

### 2. 🪙 Cryptocurrency Scanning
- **Focus**: Crypto markets only
- **Coverage**: Huobi, Binance, and other major exchanges
- **Best for**: Crypto traders, DeFi opportunities
- **Estimated time**: 2-8 minutes

### 3. 🌍 Global Market Domination (Default)
- **Focus**: Both markets simultaneously
- **Coverage**: Complete analysis across all supported assets
- **Best for**: Diversified traders, maximum opportunities
- **Estimated time**: 5-15 minutes (M1) / 10-30 minutes (Intel)

### 4. 🎯 Quick Scan (Recommended for Testing)
- **Focus**: Limited symbol set for rapid analysis
- **Coverage**: Representative sample from each market
- **Best for**: Testing, quick market pulse check
- **Estimated time**: 1-3 minutes

### 5. 🚀 Full Professional Scan
- **Focus**: Comprehensive analysis of all available symbols
- **Coverage**: Maximum depth and breadth
- **Best for**: Professional traders, comprehensive research
- **Estimated time**: 10-45 minutes

## 🎯 Understanding Results

### Professional Grading System

#### 👑 Institutional Grade (85-100)
- **Characteristics**: Exceptional setups with multiple confirmation signals
- **Quality**: Institutional-level quality
- **Risk/Reward**: Optimal risk-adjusted returns
- **Action**: Prime candidates for position allocation

#### 🏆 Professional Grade (70-84)
- **Characteristics**: High-quality setups with strong technical foundation
- **Quality**: Professional trader standard
- **Risk/Reward**: Favorable risk-reward ratios
- **Action**: Strong consideration for trading

#### ⭐ Intermediate Grade (55-69)
- **Characteristics**: Decent setups requiring careful evaluation
- **Quality**: Intermediate trader level
- **Risk/Reward**: Moderate potential
- **Action**: Detailed analysis recommended

### Pattern Combination Groups

#### 🔥 ACCUMULATION ZONE
```
Patterns: Hidden Accumulation, Smart Money Flow, Institutional Absorption
Signals: Large players quietly accumulating positions
Strategy: Early entry before institutional demand becomes visible
```

#### 💎 BREAKOUT IMMINENT  
```
Patterns: Coiled Spring, Pressure Cooker, Volume Pocket
Signals: Technical setup ready for explosive move
Strategy: Position before breakout, tight stops
```

#### 🚀 ROCKET FUEL
```
Patterns: Fuel Tank Pattern, Ignition Sequence, Momentum Vacuum
Signals: All systems ready for sustained upward movement
Strategy: Momentum play with trailing stops
```

#### ⚡ STEALTH MODE
```
Patterns: Silent Accumulation, Whale Whispers, Dark Pool Activity
Signals: Large volume operations below the radar
Strategy: Follow smart money, patient accumulation
```

## 🔍 Reading Analysis Output

### Sample Output Structure
```
🏆 PROFESSIONAL GRADE ANALYSIS:
   📊 Average Professional Grade: 76.3
   🏆 Highest Professional Grade: 94.2
   👑 Institutional Grade: 12
   🏆 Professional Grade: 28
   ⭐ Intermediate Grade: 45

💎 TOP OPPORTUNITIES:
Symbol: 600519    Grade: 94.2    Pattern: 🌟 PERFECT STORM
Symbol: BTC_USDT  Grade: 91.8    Pattern: 🚀 ROCKET FUEL
Symbol: 000858    Grade: 88.7    Pattern: 🔥 ACCUMULATION ZONE
```

### Key Metrics Explained

#### **Grade Score (0-100)**
- Composite score combining multiple technical factors
- Higher scores indicate better risk-adjusted opportunities
- Considers volume, momentum, pattern strength, and market conditions

#### **Pattern Classification**
- Primary pattern group the symbol belongs to
- Indicates the type of opportunity and suggested strategy
- Multiple patterns may apply to single symbol

#### **Market Type**
- CRYPTO: Cryptocurrency markets
- CHINESE_STOCK: Chinese A-share markets
- Helps with position sizing and risk management

## ⚙️ Advanced Features

### System Optimization

#### M1/M2 MacBook Users
```
Automatic detection and optimization:
✅ Optimal process count calculation
✅ Memory efficiency improvements  
✅ 2x performance multiplier
✅ Battery optimization
```

#### Performance Monitoring
```
System Information Display:
💻 Platform: Darwin
🔧 Processor: arm64 (M1 detected)
⚙️ CPU Cores: 8
🚀 M1/M2 Detected: Optimal processes = 6
```

### Blacklist Management

#### Static Blacklist
```python
# Pre-configured low-quality symbols
BLACKLISTED_STOCKS = {
    '002916', '002780', '002594'  # Already pumped Chinese stocks
}

BLACKLISTED_CRYPTO = {
    'LUNA_USDT', 'FTT_USDT'      # Failed projects
}
```

#### Dynamic Blacklist
- Learning system automatically excludes poor performers
- Adapts to market conditions
- Reduces noise in results

### Learning System
- Tracks pattern performance over time
- Adjusts scoring based on historical success
- Improves accuracy with each analysis run

## 🛠 Customization

### Environment Variables
```bash
# Custom data directory
export EPDS_DATA_DIR="/your/custom/path"

# Override process count
export EPDS_PROCESSES=12

# Enable debug mode
export EPDS_DEBUG=1
```

### Configuration Options
Create `config.json` in project root:
```json
{
    "max_symbols_quick_scan": 100,
    "min_professional_grade": 70,
    "enable_crypto_analysis": true,
    "enable_stock_analysis": true,
    "save_results_automatically": true,
    "result_file_prefix": "analysis_"
}
```

## 📊 Data Management

### Data Format Requirements

#### Chinese Stocks
```csv
Date,Close,Low,Volume,振幅,Open,股票代码,High,股票名称
1999-11-10,-1.3300,-1.4400,1740850,-10.4500,-1.0600,600000,-1.0200,浦发银行
1999-11-11,-1.3300,-1.3600,294034,-9.7700,-1.3500,600000,-1.2300,浦发银行
```

#### Cryptocurrency
```csv
timestamp,open,high,low,close,volume,volume_quote,symbol,price_change
2024-08-07,8e-08,8.1e-08,7.7e-08,7.7e-08,4830221976723.218,380291.87812148104,XEN-USDT,-4.9383
2024-08-08,7.7e-08,8.6e-08,7.3e-08,8.5e-08,2851972249722.883,224493.3912633243,XEN-USDT,10.3896
```

### Data Sources
- **Chinese Stocks**: Major financial data providers
- **Cryptocurrency**: Exchange APIs (Huobi, Binance)
- **Format**: Daily OHLCV data minimum
- **History**: 200+ days recommended for accurate analysis

## 🎮 Game Elements

### Messages & Motivation
```
🎰 Rolling the dice for patterns...
🎲 Shuffling the deck of stocks...
🎯 Hunting for hidden treasures...
🔮 Crystal ball says...
💎 Mining for diamond patterns...
```

### Achievement System
- **Level Up**: Discover new pattern combinations
- **Treasure Hunt**: Find high-grade opportunities
- **Pattern Master**: Consistent high-quality results

## 🚨 Risk Management

### Important Disclaimers
⚠️ **Educational Purpose Only**: This tool is for analysis and education
⚠️ **Not Financial Advice**: Always do your own research
⚠️ **Risk Warning**: Trading involves substantial risk of loss
⚠️ **Professional Consultation**: Consult financial advisors

### Best Practices
1. **Start Small**: Begin with paper trading or small positions
2. **Diversify**: Don't rely on single pattern or symbol
3. **Risk Management**: Always use stop losses
4. **Position Sizing**: Risk only what you can afford to lose
5. **Continuous Learning**: Markets evolve, keep studying

## 🔧 Troubleshooting

### Common Issues

#### "No data found"
```bash
# Check data directory structure
ls -la Chinese_Market/data/
# Ensure CSV files are present and properly formatted
```

#### Slow performance
```bash
# Try quick scan first
# Choose option 4 in menu
# Consider upgrading to SSD storage
```

#### Memory errors
```bash
# Reduce number of processes
export EPDS_PROCESSES=4
# Or use quick scan mode
```

### Performance Tips
1. **Use SSD**: Significantly faster data loading
2. **Close Other Apps**: Free up system resources  
3. **Regular Updates**: Keep dependencies current
4. **Monitor Resources**: Watch CPU and memory usage

## 📈 Interpreting Market Conditions

### Bull Market Signals
- High number of Institutional Grade opportunities
- Multiple ROCKET FUEL patterns
- Strong across both markets

### Bear Market Signals  
- Increased ACCUMULATION ZONE patterns
- Lower overall grades
- Focus on quality over quantity

### Sideways Market
- Mixed pattern distribution
- STEALTH MODE patterns prominent
- Selective opportunities

## 📊 Data Collection Tools

### EPDScanner.py - Pattern Analyzer Game v6.0

The EPDScanner provides a game-like interface for pattern analysis with 20+ advanced algorithms.

#### Basic Usage
```bash
python EPDScanner.py
```

#### Key Features
- **Interactive Game Interface**: Engaging pattern analysis experience
- **Professional Grading**: Institutional/Professional/Intermediate classifications
- **M1/M2 Optimization**: Automatic Apple Silicon performance enhancement
- **Multi-Market Analysis**: Chinese stocks + Cryptocurrency support
- **Real-time Learning**: Adaptive pattern recognition system

#### Configuration Options
```python
# Configure analysis parameters
config = {
    'enable_m1_optimization': True,
    'min_grade_threshold': 75,
    'blacklist_enabled': True,
    'learning_mode': True
}
```

### EPDStocksUpdater.py - Chinese A-Share Data Manager

Comprehensive tool for managing Chinese stock market data collection.

#### Basic Usage
```bash
python EPDStocksUpdater.py
```

#### Interactive Menu Options
1. **📊 Analyze existing files** - Assess data quality and coverage
2. **🔄 Update all data** - Refresh stock data with intelligent retry
3. **🎯 Update specific symbols** - Target specific stocks for updates
4. **🗂️ Migrate to organized structure** - Organize files by exchange
5. **📈 Quick market overview** - Current market status and statistics

#### Advanced Features
```python
# Custom configuration
config = Config(
    base_dir="Chinese_Market",
    enable_organized_structure=True,
    max_concurrent_downloads=10,
    enable_circuit_breaker=True,
    max_retries=3
)

# Run programmatically
manager = ChineseStockManager(config)
manager.update_all_stocks()
```

### EPDHuobiUpdater.py - HTX Crypto Data Collector

High-speed cryptocurrency data collection from HTX/Huobi and other exchanges.

#### Setup Configuration
First, configure your HTX API credentials in `htx_config.json`:
```json
{
  "htx_access_key": "your_access_key",
  "htx_secret_key": "your_secret_key",
  "base_dir": "Market_Data",
  "enable_crypto": true,
  "crypto_intervals": ["1min", "60min", "1day"],
  "min_volume_threshold": 10000
}
```

#### Basic Usage
```bash
python EPDHuobiUpdater.py
```

#### Advanced Usage
```python
# High-speed parallel collection
collector = HighSpeedDataCollector(config)
collector.collect_all_usdt_pairs()

# Specific symbol collection
collector.collect_symbol_data("BTC-USDT", "1day")
```

#### Supported Features
- **Multiple Exchanges**: HTX, Binance, OKX (via CCXT)
- **Parallel Processing**: Concurrent data fetching
- **API Authentication**: HTX API key support for higher rate limits
- **Smart Caching**: Avoid re-downloading existing data
- **Automatic Retry**: Intelligent error handling and recovery

### Data Organization

All tools organize data in a consistent structure:
```
Chinese_Market/data/
├── shanghai_6xx/          # Shanghai Stock Exchange
├── shenzhen_0xx/          # Shenzhen Stock Exchange  
├── beijing_8xx/           # Beijing Stock Exchange
└── huobi/                 # Cryptocurrency data
    └── spot_usdt/1d/      # Daily USDT pairs
```

### Best Practices

1. **Regular Updates**: Run data collection tools daily for fresh data
2. **Monitor Logs**: Check logs directory for error analysis
3. **API Limits**: Configure appropriate rate limits for your API tier
4. **Data Quality**: Use file analysis features to assess data completeness
5. **Backup Strategy**: Regular backups of collected data

## 🎯 Trading Strategies by Pattern

### For ACCUMULATION ZONE 🔥
```
Entry: Early stages of accumulation
Stop: Below accumulation range
Target: Breakout from accumulation
Timeframe: Medium to long term
```

### For BREAKOUT IMMINENT 💎
```
Entry: Just before or at breakout
Stop: Back below breakout level
Target: Measured move from pattern
Timeframe: Short to medium term
```

### For ROCKET FUEL 🚀
```
Entry: Confirmation of momentum
Stop: Trailing stop loss
Target: Ride the trend
Timeframe: Variable, momentum-based
```

## 📞 Support & Community

### Getting Help
1. **Documentation**: Check all docs/ files
2. **GitHub Issues**: Report bugs and feature requests
3. **Discussions**: Community Q&A and sharing
4. **Email Support**: For critical issues

### Contributing
- Pattern suggestions
- Performance improvements
- Additional market support
- Documentation enhancements

---

**Ready to find your next trading opportunity? 🚀 Start with a Quick Scan (option 4) to get familiar with the system!**