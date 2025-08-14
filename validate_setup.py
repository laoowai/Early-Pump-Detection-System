#!/usr/bin/env python3
"""
Professional Pattern Analyzer - Setup Validation
Validates system setup and identifies potential issues
"""

import sys
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Any
import importlib.util

def test_project_structure() -> Tuple[bool, List[str]]:
    """Test if all required project files exist"""
    issues = []
    
    required_files = [
        "main.py",
        "pattern_analyzers/__init__.py",
        "pattern_analyzers/base_pattern_analyzer.py", 
        "pattern_analyzers/professional_pattern_analyzer.py",
        "pattern_detectors/__init__.py",
        "pattern_detectors/base_detector.py",
        "pattern_detectors/advanced_pattern_detector.py",
        "stage_analyzers/__init__.py",
        "stage_analyzers/base_stage_analyzer.py",
        "stage_analyzers/enhanced_stage_analyzer.py",
        "timeframe_analyzers/__init__.py",
        "timeframe_analyzers/base_timeframe_analyzer.py",
        "timeframe_analyzers/enhanced_multi_timeframe_analyzer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        issues.append(f"Missing required files: {', '.join(missing_files)}")
    
    # Check optional files
    optional_files = [
        "demo.py",
        "validate_setup.py",
        "README.md",
        "requirements.txt",
        "EPDScanner.py",
        "EPDStocksUpdater.py", 
        "EPDHuobiUpdater.py",
        "htx_config.json"
    ]
    
    missing_optional = []
    for file_path in optional_files:
        if not Path(file_path).exists():
            missing_optional.append(file_path)
    
    if missing_optional:
        issues.append(f"Missing optional files: {', '.join(missing_optional)}")
    
    return len(missing_files) == 0, issues

def test_data_directory_structure() -> Tuple[bool, List[str]]:
    """Test data directory structure"""
    issues = []
    
    base_data_dir = Path("Chinese_Market/data")
    
    if not base_data_dir.exists():
        issues.append(f"Data directory not found: {base_data_dir}")
        return False, issues
    
    expected_subdirs = [
        "shanghai_6xx",
        "shenzhen_0xx", 
        "beijing_8xx",
        "huobi/spot_usdt/1d"
    ]
    
    missing_dirs = []
    empty_dirs = []
    
    for subdir in expected_subdirs:
        full_path = base_data_dir / subdir
        if not full_path.exists():
            missing_dirs.append(subdir)
        else:
            csv_files = list(full_path.glob("*.csv"))
            if not csv_files:
                empty_dirs.append(subdir)
    
    if missing_dirs:
        issues.append(f"Missing data directories: {', '.join(missing_dirs)}")
    
    if empty_dirs:
        issues.append(f"Empty data directories (no CSV files): {', '.join(empty_dirs)}")
    
    # Check for sample data
    total_files = 0
    for subdir in expected_subdirs:
        full_path = base_data_dir / subdir
        if full_path.exists():
            csv_files = list(full_path.glob("*.csv"))
            total_files += len(csv_files)
    
    if total_files == 0:
        issues.append("No CSV data files found in any directory")
    elif total_files < 10:
        issues.append(f"Very few data files found ({total_files}). Consider adding more data for better analysis.")
    
    return len(missing_dirs) == 0 and total_files > 0, issues

def test_python_syntax() -> Tuple[bool, List[str]]:
    """Test Python syntax of all Python files"""
    issues = []
    
    python_files = [
        "main.py",
        "pattern_analyzers/base_pattern_analyzer.py",
        "pattern_analyzers/professional_pattern_analyzer.py",
        "pattern_detectors/base_detector.py", 
        "pattern_detectors/advanced_pattern_detector.py",
        "stage_analyzers/base_stage_analyzer.py",
        "stage_analyzers/enhanced_stage_analyzer.py",
        "timeframe_analyzers/base_timeframe_analyzer.py",
        "timeframe_analyzers/enhanced_multi_timeframe_analyzer.py"
    ]
    
    syntax_errors = []
    
    for file_path in python_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: Line {e.lineno} - {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{file_path}: {str(e)}")
    
    if syntax_errors:
        issues.extend(syntax_errors)
    
    return len(syntax_errors) == 0, issues

def test_imports() -> Tuple[bool, List[str]]:
    """Test if required imports work"""
    issues = []
    
    required_packages = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("pathlib", "Path"),
        ("datetime", "datetime"),
        ("concurrent.futures", "ThreadPoolExecutor"),
        ("scipy.stats", "linregress"),
        ("collections", "defaultdict")
    ]
    
    import_errors = []
    
    for package, alias in required_packages:
        try:
            if "." in package:
                module_name, attr_name = package.rsplit(".", 1)
                module = importlib.import_module(module_name)
                getattr(module, attr_name)
            else:
                importlib.import_module(package)
        except ImportError:
            import_errors.append(package)
        except Exception as e:
            import_errors.append(f"{package}: {str(e)}")
    
    if import_errors:
        issues.append(f"Import errors: {', '.join(import_errors)}")
    
    return len(import_errors) == 0, issues

def test_data_format() -> Tuple[bool, List[str]]:
    """Test data file format"""
    issues = []
    
    base_data_dir = Path("Chinese_Market/data")
    
    if not base_data_dir.exists():
        return False, ["Data directory not found"]
    
    # Find sample files
    sample_files = []
    for subdir in ["shanghai_6xx", "shenzhen_0xx", "huobi/spot_usdt/1d"]:
        subdir_path = base_data_dir / subdir
        if subdir_path.exists():
            csv_files = list(subdir_path.glob("*.csv"))
            if csv_files:
                sample_files.append(csv_files[0])
    
    if not sample_files:
        return False, ["No CSV files found to validate format"]
    
    format_issues = []
    
    for file_path in sample_files[:3]:  # Check first 3 files
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                format_issues.append(f"{file_path.name}: Missing columns {missing_columns}")
            
            if len(df) < 30:
                format_issues.append(f"{file_path.name}: Too few rows ({len(df)}), need at least 30")
            
            # Check for valid data
            for col in required_columns:
                if col in df.columns:
                    if df[col].isnull().any():
                        format_issues.append(f"{file_path.name}: Column {col} has null values")
                    if (df[col] <= 0).any():
                        format_issues.append(f"{file_path.name}: Column {col} has non-positive values")
        
        except Exception as e:
            format_issues.append(f"{file_path.name}: Error reading file - {str(e)}")
    
    if format_issues:
        issues.extend(format_issues)
    
    return len(format_issues) == 0, issues

def test_system_resources() -> Tuple[bool, List[str]]:
    """Test system resources"""
    issues = []
    warnings = []
    
    import platform
    try:
        import psutil
    except ImportError:
        psutil = None
    
    # Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(f"Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+")
    elif python_version < (3, 9):
        warnings.append(f"Python {python_version.major}.{python_version.minor} works but 3.9+ recommended")
    
    # Memory check (if psutil available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 4 * 1024**3:  # 4GB
            warnings.append(f"Low memory: {memory.total // (1024**3)}GB, recommend 8GB+")
    except ImportError:
        warnings.append("psutil not available - cannot check memory")
    
    # CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cpu_count < 2:
        warnings.append(f"Only {cpu_count} CPU core detected, recommend 4+")
    elif cpu_count < 4:
        warnings.append(f"{cpu_count} CPU cores detected, 4+ recommended for better performance")
    
    # Platform optimization
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        warnings.append("M1/M2 Mac detected - system is optimized for your hardware!")
    
    if warnings:
        issues.extend([f"Warning: {w}" for w in warnings])
    
    return len([i for i in issues if not i.startswith("Warning:")]) == 0, issues

def run_validation() -> Dict[str, Any]:
    """Run all validation tests"""
    results = {}
    
    print("🧪 PROFESSIONAL PATTERN ANALYZER - SETUP VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Data Directory", test_data_directory_structure),
        ("Python Syntax", test_python_syntax),
        ("Import Dependencies", test_imports),
        ("Data Format", test_data_format),
        ("System Resources", test_system_resources)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n🔧 Testing {test_name}...")
        try:
            passed, issues = test_func()
            results[test_name] = {"passed": passed, "issues": issues}
            
            if passed:
                print(f"   ✅ {test_name}: PASSED")
            else:
                print(f"   âŒ {test_name}: FAILED")
                all_passed = False
            
            for issue in issues:
                if issue.startswith("Warning:"):
                    print(f"   âš ï¸  {issue}")
                else:
                    print(f"   • {issue}")
        
        except Exception as e:
            print(f"   âŒ {test_name}: ERROR - {str(e)}")
            results[test_name] = {"passed": False, "issues": [str(e)]}
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for analysis.")
        print("🚀 Run 'python main.py' to start pattern analysis.")
    else:
        print("âš ï¸  SOME TESTS FAILED. Please fix issues before running analysis.")
        print("💡 Check the issues listed above and run validation again.")
    
    print("\n🔧 Quick fixes:")
    print("   • Missing files: Re-download project files")
    print("   • Missing data: Create Chinese_Market/data/ and add CSV files") 
    print("   • Import errors: pip install pandas numpy scipy scikit-learn")
    print("   • Format errors: Ensure CSV files have Open,High,Low,Close columns")
    
    return results

def main():
    """Main validation function"""
    try:
        results = run_validation()
        
        # Save results if possible
        try:
            import json
            with open("validation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Validation results saved to validation_results.json")
        except Exception:
            pass
    
    except KeyboardInterrupt:
        print("\n\nâ¸ï¸  Validation interrupted by user")
    except Exception as e:
        print(f"\n\nâŒ Validation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
