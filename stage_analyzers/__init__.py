"""
Stage Analyzers Module
Auto-Discovery System for Stage Analysis Components
"""

from .base_stage_analyzer import BaseStageAnalyzer

# Auto-import all stage analyzers
import os
import importlib
from pathlib import Path

def auto_discover_stage_analyzers():
    """Auto-discover all stage analyzer classes"""
    analyzers = {}
    current_dir = Path(__file__).parent
    
    for file_path in current_dir.glob("*.py"):
        if file_path.name.startswith("__") or file_path.name == "base_stage_analyzer.py":
            continue
            
        module_name = file_path.stem
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            
            # Find analyzer classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseStageAnalyzer) and 
                    attr != BaseStageAnalyzer):
                    analyzers[attr_name] = attr
        except Exception as e:
            print(f"Warning: Could not load stage analyzer from {file_path.name}: {e}")
    
    return analyzers

# Auto-discover at import time
stage_analyzers = auto_discover_stage_analyzers()

__all__ = ['BaseStageAnalyzer', 'stage_analyzers']
