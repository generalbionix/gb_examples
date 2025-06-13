"""
Caching module for GeneralBionix API calls.

This module provides intelligent caching using joblib.Memory to avoid
repeated expensive API calls during development and testing.
"""

from .cache_client import CachedGeneralBionixClient

__all__ = ['CachedGeneralBionixClient'] 