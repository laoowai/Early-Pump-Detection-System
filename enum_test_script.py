#!/usr/bin/env python3
"""
Quick test to verify enum consistency fixes
Run this after applying the fixes to verify the solution works
"""

def test_enum_consistency():
    """Test that MarketType enums are properly imported and consistent"""
    print("🔍 Testing MarketType enum consistency...")
    
    try:
        # Import from main
        from main import MarketType as MainMarketType
        print("✅ Successfully imported MarketType from main")
        
        # Test enum values
        print(f"   CHINESE_STOCK: {MainMarketType.CHINESE_STOCK}")
        print(f"   CRYPTO: {MainMarketType.CRYPTO}")
        print(f"   BOTH: {MainMarketType.BOTH}")
        
        # Test enum comparison
        test_type = MainMarketType.CHINESE_STOCK
        is_in_list = test_type in [MainMarketType.CHINESE_STOCK, MainMarketType.BOTH]
        print(f"   Enum comparison test: {is_in_list}")
        
        if is_in_list:
            print("✅ Enum comparison working correctly")
        else:
            print("❌ Enum comparison still failing")
            
        # Test professional pattern analyzer import
        try:
            from pattern_analyzers.professional_pattern_analyzer import ProfessionalPatternAnalyzer
            print("✅ Successfully imported ProfessionalPatternAnalyzer")
            
            # Test that it uses the same enum
            analyzer = ProfessionalPatternAnalyzer()
            print("✅ ProfessionalPatternAnalyzer initialized successfully")
            
            # Quick symbol discovery test
            symbols = analyzer.get_all_symbols(MainMarketType.CHINESE_STOCK)
            print(f"✅ Symbol discovery test: Found {len(symbols)} symbols")
            
            if len(symbols) > 0:
                print("🎉 SUCCESS: Enum consistency fix worked!")
                print(f"   Sample symbols: {symbols[:5]}")
                return True
            else:
                print("⚠️  Found 0 symbols - check data directory")
                return False
                
        except Exception as e:
            print(f"❌ Error testing ProfessionalPatternAnalyzer: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error importing or testing enums: {e}")
        return False

if __name__ == "__main__":
    success = test_enum_consistency()
    if success:
        print("\n🎯 Ready to run the main analysis!")
    else:
        print("\n⚠️  Additional fixes may be needed")
