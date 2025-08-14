"""
Timeframe Analyzers Module
Auto-Discovery System for Multi-Timeframe Analysis Components
"""

from .base_timeframe_analyzer import BaseTimeframeAnalyzer

# Auto-import all timeframe analyzers
import os
import importlib
from pathlib import Path

def auto_discover_timeframe_analyzers():
    """Auto-discover all timeframe analyzer classes"""
    analyzers = {}
    current_dir = Path(__file__).parent
    
    for file_path in current_dir.glob("*.py"):
        if file_path.name.startswith("__") or file_path.name == "base_timeframe_analyzer.py":
            continue
            
        module_name = file_path.stem
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            
            # Find analyzer classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseTimeframeAnalyzer) and 
                    attr != BaseTimeframeAnalyzer):
                    analyzers[attr_name] = attr
        except Exception as e:
            print(f"Warning: Could not load timeframe analyzer from {file_path.name}: {e}")
    
    return analyzers

# Auto-discover at import time
timeframe_analyzers = auto_discover_timeframe_analyzers()

__all__ = ['BaseTimeframeAnalyzer', 'timeframe_analyzers']
