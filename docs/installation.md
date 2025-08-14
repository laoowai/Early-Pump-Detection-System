# 🛠 Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.8+ (Recommended: 3.9+ for optimal performance)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB free space for system and data
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Setup
- **M1/M2 MacBook**: Optimized performance with automatic detection
- **Python 3.9+**: Best compatibility and performance
- **16GB RAM**: For large dataset analysis
- **SSD Storage**: Faster data processing

## Installation Methods

### 🚀 Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import main; print('✅ Installation successful!')"
```

### 🔧 Method 2: Virtual Environment (Production)

```bash
# Clone repository
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System

# Create virtual environment
python -m venv epds_env

# Activate virtual environment
# On macOS/Linux:
source epds_env/bin/activate
# On Windows:
epds_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python main.py
```

### 🐳 Method 3: Docker (Advanced)

```dockerfile
# Dockerfile (create this file)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

```bash
# Build and run
docker build -t epds .
docker run -it epds
```

## Dependencies Overview

### Core Dependencies
```txt
pandas>=1.5.0          # Data processing
numpy>=1.24.0           # Numerical computing
scipy>=1.10.0           # Scientific computing
scikit-learn>=1.2.0     # Machine learning
ta>=0.10.2              # Technical analysis
tabulate>=0.9.0         # Table formatting
```

### Optional Performance Enhancements
```bash
# For enhanced performance (optional)
pip install numba>=0.57.0   # JIT compilation
pip install cython>=0.29.0  # C extensions
```

## Data Directory Setup

### Required Directory Structure (Current Implementation)
```
Chinese_Market/data/
├── shanghai_6xx/          # Shanghai Stock Exchange
│   └── 600000.csv         # Example: 浦发银行 (SPDB)
├── shenzhen_0xx/          # Shenzhen Stock Exchange  
│   └── 000001.csv         # Example: 平安银行 (Ping An Bank)
└── huobi/                 # Cryptocurrency data
    └── spot_usdt/1d/      # Daily USDT trading pairs
        └── XEN-USDT.csv   # Example: XEN token
```

### CSV File Formats

**Chinese Stocks** (Shanghai & Shenzhen):
```csv
Date,Close,Low,Volume,振幅,Open,股票代码,High,股票名称
1999-11-10,-1.3300,-1.4400,1740850,-10.4500,-1.0600,600000,-1.0200,浦发银行
```

**Cryptocurrency** (Huobi):
```csv
timestamp,open,high,low,close,volume,volume_quote,symbol,price_change
2024-08-07,8e-08,8.1e-08,7.7e-08,7.7e-08,4830221976723.218,380291.87812148104,XEN-USDT,-4.9383
```

> **Note**: The actual CSV formats may differ from standard OHLCV format. Chinese stock data includes additional columns in Chinese, while crypto data includes volume_quote and price_change fields.

## Platform-Specific Installation

### 🍎 macOS

#### M1/M2 MacBooks (Recommended)
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Clone and setup
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System
pip3 install -r requirements.txt
```

#### Intel Macs
```bash
# Same as M1/M2, but performance will be automatically adjusted
brew install python@3.9
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System
pip3 install -r requirements.txt
```

### 🐧 Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3.9 python3.9-pip python3.9-venv -y

# Install development tools
sudo apt install build-essential -y

# Clone and setup
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System
python3.9 -m pip install -r requirements.txt
```

### 🪟 Windows

#### Using Python from python.org
```powershell
# Download Python 3.9+ from python.org
# Make sure to check "Add Python to PATH"

# Open Command Prompt or PowerShell
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System
pip install -r requirements.txt
```

#### Using Anaconda
```bash
# Download and install Anaconda
# Open Anaconda Prompt

conda create -n epds python=3.9
conda activate epds
git clone https://github.com/laoowai/Early-Pump-Detection-System.git
cd Early-Pump-Detection-System
pip install -r requirements.txt
```

## Verification & Testing

### Basic Verification
```bash
# Test Python imports
python -c "import pandas, numpy, scipy, sklearn, ta; print('✅ All dependencies imported successfully')"

# Test system detection
python -c "from main import detect_m1_optimization; print(detect_m1_optimization())"

# Test main application (will show interactive menu)
python main.py
```

### Performance Test
```bash
# Quick performance test (select option 4 for quick scan)
python main.py
# Choose option 4: "🎯 Quick Scan (Limited Symbols)"
```

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
```bash
# Solution: Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### 2. Permission Errors (macOS/Linux)
```bash
# Solution: Use user installation
pip install --user -r requirements.txt
```

#### 3. Build Errors (scipy/numpy)
```bash
# macOS: Install Xcode Command Line Tools
xcode-select --install

# Linux: Install build dependencies
sudo apt install build-essential gfortran libatlas-base-dev
```

#### 4. M1 MacBook Specific Issues
```bash
# If using conda, ensure you have the right architecture
conda config --env --set subdir osx-arm64

# Or use native pip with Python 3.9+
brew install python@3.9
pip3 install -r requirements.txt
```

### Performance Optimization

#### For M1/M2 MacBooks
```bash
# Verify M1 optimization is active
python -c "
from main import detect_m1_optimization
info = detect_m1_optimization()
print(f'M1 Detected: {info[\"is_m1\"]}')
print(f'Optimal Processes: {info[\"optimal_processes\"]}')
"
```

#### Memory Optimization
```bash
# For large datasets, consider increasing memory limits
export PYTHONHASHSEED=0
ulimit -v 8388608  # Limit virtual memory to 8GB
```

## Environment Configuration

### Environment Variables
```bash
# Optional: Set custom data directory
export EPDS_DATA_DIR="/path/to/your/Chinese_Market/data"

# Optional: Set process count override
export EPDS_PROCESSES=8

# Optional: Enable debug logging
export EPDS_DEBUG=1
```

### Configuration File (Optional)
Create `config.json` in the project root:
```json
{
    "data_directory": "Chinese_Market/data",
    "max_processes": 8,
    "enable_m1_optimization": true,
    "log_level": "INFO",
    "blacklist_updates": true
}
```

## Next Steps

1. **📊 Test Installation**: Run `python main.py` and try option 4 (Quick Scan)
2. **📚 Read User Guide**: Check [user-guide.md](user-guide.md) for usage instructions
3. **🏗️ Setup Data**: Prepare your market data in the required format
4. **🚀 Start Analysis**: Begin with small datasets and scale up

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Search [GitHub Issues](https://github.com/laoowai/Early-Pump-Detection-System/issues)
3. Create a new issue with your system info and error details

---

**Ready to start? 🚀 Run `python main.py` and choose your trading destiny!**