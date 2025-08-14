"""
Pattern Detectors Module
Auto-Discovery System for Pattern Detection Components
"""

from .base_detector import BasePatternDetector

# Auto-import all pattern detectors
import os
import importlib
from pathlib import Path

def auto_discover_detectors():
    """Auto-discover all pattern detector classes"""
    detectors = {}
    current_dir = Path(__file__).parent
    
    for file_path in current_dir.glob("*.py"):
        if file_path.name.startswith("__") or file_path.name == "base_detector.py":
            continue
            
        module_name = file_path.stem
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            
            # Find detector classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePatternDetector) and 
                    attr != BasePatternDetector):
                    detectors[attr_name] = attr
        except Exception as e:
            print(f"Warning: Could not load pattern detector from {file_path.name}: {e}")
    
    return detectors

# Auto-discover at import time
pattern_detectors = auto_discover_detectors()

__all__ = ['BasePatternDetector', 'pattern_detectors']
