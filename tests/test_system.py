"""
Test Suite for Early Pump Detection System
System Integration and Validation Tests
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestSystemIntegration(unittest.TestCase):
    """System integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_data_dir = Path(__file__).parent / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create sample test data
        cls._create_sample_data()
    
    @classmethod
    def _create_sample_data(cls):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Sample stock data
        np.random.seed(42)
        base_price = 100.0
        prices = [base_price]
        
        for i in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        # Create OHLCV data
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(100000, 2000000) for _ in prices]
        })
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        sample_data['High'] = sample_data[['Open', 'High', 'Close']].max(axis=1)
        sample_data['Low'] = sample_data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Save test data
        stock_dir = cls.test_data_dir / "shanghai_6xx"
        stock_dir.mkdir(exist_ok=True)
        sample_data.to_csv(stock_dir / "600519.csv", index=False)
        
        crypto_dir = cls.test_data_dir / "crypto" / "spot"
        crypto_dir.mkdir(parents=True, exist_ok=True)
        sample_data.to_csv(crypto_dir / "BTC_USDT.csv", index=False)
    
    def test_import_main_module(self):
        """Test that main module can be imported"""
        try:
            import main
            self.assertTrue(hasattr(main, 'main'))
            self.assertTrue(hasattr(main, 'ProfessionalTradingOrchestrator'))
            print("✅ Main module imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import main module: {e}")
    
    def test_import_pattern_analyzers(self):
        """Test pattern analyzer imports"""
        try:
            from pattern_analyzers import BasePatternAnalyzer, pattern_analyzers
            self.assertTrue(len(pattern_analyzers) > 0)
            print(f"✅ Pattern analyzers imported: {list(pattern_analyzers.keys())}")
        except ImportError as e:
            self.fail(f"Failed to import pattern analyzers: {e}")
    
    def test_import_pattern_detectors(self):
        """Test pattern detector imports"""
        try:
            from pattern_detectors import BasePatternDetector, pattern_detectors
            self.assertTrue(len(pattern_detectors) > 0)
            print(f"✅ Pattern detectors imported: {list(pattern_detectors.keys())}")
        except ImportError as e:
            self.fail(f"Failed to import pattern detectors: {e}")
    
    def test_import_stage_analyzers(self):
        """Test stage analyzer imports"""
        try:
            from stage_analyzers import BaseStageAnalyzer, stage_analyzers
            self.assertTrue(len(stage_analyzers) > 0)
            print(f"✅ Stage analyzers imported: {list(stage_analyzers.keys())}")
        except ImportError as e:
            self.fail(f"Failed to import stage analyzers: {e}")


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Valid OHLCV data
        self.valid_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Open': np.random.uniform(90, 110, 100),
            'High': np.random.uniform(100, 120, 100),
            'Low': np.random.uniform(80, 100, 100),
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.randint(100000, 1000000, 100)
        })
        
        # Ensure OHLC consistency
        for i in range(len(self.valid_data)):
            high = max(self.valid_data.loc[i, 'Open'], 
                      self.valid_data.loc[i, 'Close'], 
                      self.valid_data.loc[i, 'High'])
            low = min(self.valid_data.loc[i, 'Open'], 
                     self.valid_data.loc[i, 'Close'], 
                     self.valid_data.loc[i, 'Low'])
            self.valid_data.loc[i, 'High'] = high
            self.valid_data.loc[i, 'Low'] = low
    
    def test_valid_data_validation(self):
        """Test validation of good data"""
        try:
            from pattern_analyzers.base_pattern_analyzer import BasePatternAnalyzer
            
            class TestAnalyzer(BasePatternAnalyzer):
                def analyze_symbol(self, symbol, df):
                    pass
            
            analyzer = TestAnalyzer()
            is_valid = analyzer.validate_data(self.valid_data)
            self.assertTrue(is_valid)
            print("✅ Valid data passes validation")
            
        except Exception as e:
            self.fail(f"Data validation test failed: {e}")
    
    def test_invalid_data_validation(self):
        """Test validation of bad data"""
        try:
            from pattern_analyzers.base_pattern_analyzer import BasePatternAnalyzer
            
            class TestAnalyzer(BasePatternAnalyzer):
                def analyze_symbol(self, symbol, df):
                    pass
            
            analyzer = TestAnalyzer()
            
            # Test empty data
            empty_data = pd.DataFrame()
            self.assertFalse(analyzer.validate_data(empty_data))
            
            # Test insufficient data
            short_data = self.valid_data.head(10)
            self.assertFalse(analyzer.validate_data(short_data))
            
            # Test missing columns
            bad_columns = self.valid_data.drop('Volume', axis=1)
            self.assertFalse(analyzer.validate_data(bad_columns))
            
            print("✅ Invalid data fails validation as expected")
            
        except Exception as e:
            self.fail(f"Invalid data validation test failed: {e}")


class TestSystemOptimization(unittest.TestCase):
    """Test system optimization features"""
    
    def test_m1_detection(self):
        """Test M1 MacBook detection"""
        try:
            from main import detect_m1_optimization
            
            system_info = detect_m1_optimization()
            
            # Verify required fields
            required_fields = ['platform', 'processor', 'machine', 'cpu_count', 
                             'is_m1', 'optimal_processes']
            for field in required_fields:
                self.assertIn(field, system_info)
            
            self.assertIsInstance(system_info['is_m1'], bool)
            self.assertIsInstance(system_info['optimal_processes'], int)
            self.assertGreater(system_info['optimal_processes'], 0)
            
            print(f"✅ System optimization detection works: {system_info}")
            
        except Exception as e:
            self.fail(f"M1 detection test failed: {e}")
    
    def test_blacklist_system(self):
        """Test blacklist management"""
        try:
            from main import EnhancedBlacklistManager
            
            manager = EnhancedBlacklistManager()
            
            # Test static blacklist
            self.assertTrue(manager.is_blacklisted('002916'))  # Known blacklisted stock
            self.assertFalse(manager.is_blacklisted('600519'))  # Should not be blacklisted
            
            # Test dynamic blacklist
            manager.add_to_dynamic_blacklist('TEST_SYMBOL', 'Testing')
            self.assertTrue(manager.is_blacklisted('TEST_SYMBOL'))
            
            print("✅ Blacklist system functioning correctly")
            
        except Exception as e:
            self.fail(f"Blacklist test failed: {e}")


class TestPatternDetection(unittest.TestCase):
    """Test pattern detection functionality"""
    
    def setUp(self):
        """Set up test data for pattern detection"""
        # Create trending data for pattern testing
        dates = pd.date_range('2023-01-01', periods=50)
        trend = np.linspace(100, 150, 50)  # Upward trend
        noise = np.random.normal(0, 2, 50)
        
        self.trending_data = pd.DataFrame({
            'Date': dates,
            'Open': trend + noise,
            'High': trend + noise + np.abs(np.random.normal(0, 1, 50)),
            'Low': trend + noise - np.abs(np.random.normal(0, 1, 50)),
            'Close': trend + noise * 0.5,
            'Volume': np.random.randint(100000, 1000000, 50)
        })
        
        # Ensure OHLC consistency
        for i in range(len(self.trending_data)):
            self.trending_data.loc[i, 'High'] = max(
                self.trending_data.loc[i, ['Open', 'High', 'Close']]
            )
            self.trending_data.loc[i, 'Low'] = min(
                self.trending_data.loc[i, ['Open', 'Low', 'Close']]
            )
    
    def test_basic_metrics_calculation(self):
        """Test basic metrics calculation"""
        try:
            from pattern_analyzers.base_pattern_analyzer import BasePatternAnalyzer
            
            class TestAnalyzer(BasePatternAnalyzer):
                def analyze_symbol(self, symbol, df):
                    pass
            
            analyzer = TestAnalyzer()
            metrics = analyzer.calculate_basic_metrics(self.trending_data)
            
            # Check required metrics are present
            required_metrics = ['current_price', 'price_change_pct', 'volume_ratio']
            for metric in required_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
            
            print(f"✅ Basic metrics calculated: {list(metrics.keys())}")
            
        except Exception as e:
            self.fail(f"Basic metrics test failed: {e}")


class TestPerformanceValidation(unittest.TestCase):
    """Test system performance characteristics"""
    
    def test_memory_efficiency(self):
        """Test memory usage stays reasonable"""
        import psutil
        import gc
        
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Import and instantiate main components
            from main import ProfessionalTradingOrchestrator
            orchestrator = ProfessionalTradingOrchestrator()
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 500MB for basic instantiation)
            self.assertLess(memory_increase, 500, 
                          f"Memory usage increased by {memory_increase:.1f}MB")
            
            print(f"✅ Memory usage acceptable: {memory_increase:.1f}MB increase")
            
        except Exception as e:
            self.fail(f"Memory efficiency test failed: {e}")
    
    def test_import_speed(self):
        """Test that imports complete in reasonable time"""
        import time
        
        start_time = time.time()
        
        try:
            # Import main components
            import main
            from pattern_analyzers import pattern_analyzers
            from pattern_detectors import pattern_detectors
            from stage_analyzers import stage_analyzers
            
            import_time = time.time() - start_time
            
            # Imports should complete in under 10 seconds
            self.assertLess(import_time, 10.0, 
                          f"Imports took {import_time:.2f}s")
            
            print(f"✅ Import speed acceptable: {import_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Import speed test failed: {e}")


def run_system_tests():
    """Run all system tests"""
    print("🧪 Early Pump Detection System - Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSystemIntegration,
        TestDataValidation,
        TestSystemOptimization,
        TestPatternDetection,
        TestPerformanceValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All tests passed! System is ready for use.")
    else:
        print("❌ Some tests failed. Check the output above.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_system_tests()