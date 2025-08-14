"""
Pattern Analyzers Module
Auto-Discovery System for Pattern Analysis Components
"""

from .base_pattern_analyzer import BasePatternAnalyzer

# Auto-import all pattern analyzers
import os
import importlib
from pathlib import Path

def auto_discover_pattern_analyzers():
    """Auto-discover all pattern analyzer classes"""
    analyzers = {}
    current_dir = Path(__file__).parent
    
    for file_path in current_dir.glob("*.py"):
        if file_path.name.startswith("__") or file_path.name == "base_pattern_analyzer.py":
            continue
            
        module_name = file_path.stem
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            
            # Find analyzer classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePatternAnalyzer) and 
                    attr != BasePatternAnalyzer):
                    analyzers[attr_name] = attr
        except Exception as e:
            print(f"Warning: Could not load pattern analyzer from {file_path.name}: {e}")
    
    return analyzers

# Auto-discover at import time
pattern_analyzers = auto_discover_pattern_analyzers()

__all__ = ['BasePatternAnalyzer', 'pattern_analyzers']
