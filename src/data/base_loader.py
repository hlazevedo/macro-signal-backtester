from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._cache = {}
        
    @abstractmethod
    def fetch_data(self, identifier: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for a given identifier and date range."""
        pass
    
    def _validate_dates(self, start_date: str, end_date: str) -> tuple:
        """Validate and parse date strings."""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start >= end:
                raise ValueError("Start date must be before end date")
            return start, end
        except Exception as e:
            logger.error(f"Date validation error: {e}")
            raise
    
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache if available."""
        return self._cache.get(key)
    
    def cache_data(self, key: str, data: pd.DataFrame) -> None:
        """Store data in cache."""
        self._cache[key] = data.copy()