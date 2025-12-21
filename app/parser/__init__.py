"""
HIVE215 Document Parser Module
"""

from .universal_parser import UniversalParser, FILE_TYPES, ALLOWED_EXTENSIONS
from .data_manager import DataExtractionManager, ExtractedData
from .routes import parser_bp

__all__ = [
    'UniversalParser',
    'DataExtractionManager', 
    'ExtractedData',
    'parser_bp',
    'FILE_TYPES',
    'ALLOWED_EXTENSIONS'
]

print(f"📄 Parser loaded: {len(FILE_TYPES)} file types supported")
