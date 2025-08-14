#!/usr/bin/env python3
"""
Simple validation script for Early Pump Detection System
Tests basic functionality without requiring all dependencies
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """Test that all required files and directories exist"""
    print("🔍 Testing project structure...")
    
    required_files = [
        "main.py",
        "README.md", 
        "requirements.txt",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        ".gitignore"
    ]
    
    required_dirs = [
        "pattern_analyzers",
        "pattern_detectors", 
        "stage_analyzers",
        "docs",
        "tests",
        "examples"
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    # Check directories
    for dir in required_dirs:
        if not Path(dir).exists():
            missing_dirs.append(dir)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Project structure is complete")
    return True

def test_documentation():
    """Test that documentation files exist and have content"""
    print("📚 Testing documentation...")
    
    doc_files = [
        "docs/installation.md",
        "docs/user-guide.md", 
        "docs/architecture.md",
        "docs/api-reference.md"
    ]
    
    for doc_file in doc_files:
        if not Path(doc_file).exists():
            print(f"❌ Missing documentation: {doc_file}")
            return False
        
        # Check file has reasonable content
        with open(doc_file, 'r') as f:
            content = f.read()
            if len(content) < 1000:  # Minimum content length
                print(f"❌ Documentation too short: {doc_file}")
                return False
    
    print("✅ Documentation is complete")
    return True

def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("🐍 Testing Python syntax...")
    
    python_files = [
        "main.py",
        "pattern_analyzers/__init__.py",
        "pattern_detectors/__init__.py",
        "stage_analyzers/__init__.py",
        "tests/test_system.py"
    ]
    
    for py_file in python_files:
        if not Path(py_file).exists():
            print(f"❌ Missing Python file: {py_file}")
            return False
        
        try:
            with open(py_file, 'r') as f:
                code = f.read()
                compile(code, py_file, 'exec')
        except SyntaxError as e:
            print(f"❌ Syntax error in {py_file}: {e}")
            return False
        except Exception as e:
            print(f"⚠️  Warning in {py_file}: {e}")
    
    print("✅ Python syntax is valid")
    return True

def test_requirements():
    """Test that requirements.txt exists and has content"""
    print("📦 Testing requirements...")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt missing")
        return False
    
    with open(req_file, 'r') as f:
        content = f.read()
        
    # Check for essential packages
    essential_packages = ['pandas', 'numpy', 'scipy', 'scikit-learn', 'ta']
    
    for package in essential_packages:
        if package not in content:
            print(f"❌ Missing essential package: {package}")
            return False
    
    print("✅ Requirements file is complete")
    return True

def test_readme_content():
    """Test that README has comprehensive content"""
    print("📄 Testing README content...")
    
    with open("README.md", 'r') as f:
        readme_content = f.read()
    
    # Check for essential sections
    essential_sections = [
        "# 🚀 Early Pump Detection System",
        "## 🌟 Key Features", 
        "## 🛠 Installation",
        "## 🚀 Quick Start",
        "## 📊 System Architecture",
        "requirements.txt"
    ]
    
    for section in essential_sections:
        if section not in readme_content:
            print(f"❌ Missing README section: {section}")
            return False
    
    # Check minimum length
    if len(readme_content) < 5000:
        print("❌ README content too short")
        return False
    
    print("✅ README content is comprehensive")
    return True

def run_validation():
    """Run all validation tests"""
    print("🧪 Early Pump Detection System - Validation")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_documentation,
        test_python_syntax,
        test_requirements,
        test_readme_content
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"📊 Validation Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! Project setup is complete.")
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Read the documentation in docs/")
        print("3. Try the quick start: python main.py (option 4 for quick test)")
        return True
    else:
        print("❌ Some validation tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)