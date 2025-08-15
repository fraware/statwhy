"""
Caching system for StatWhy verification results.

Provides intelligent caching of verification results to improve performance
and reduce redundant computations.
"""

import json
import logging
import pickle
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

from .models import VerificationResult
from .exceptions import CacheError

logger = logging.getLogger(__name__)


class VerificationCache:
    """
    Cache system for StatWhy verification results.

    Provides both in-memory and persistent caching with automatic
    expiration and intelligent cache management.
    """

    def __init__(
        self,
        cache_dir: str = ".statwhy_cache",
        max_size: int = 1000,
        ttl_hours: int = 24,
    ):
        """
        Initialize the verification cache.

        Args:
            cache_dir: Directory for persistent cache storage
            max_size: Maximum number of cached items
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.ttl_hours = ttl_hours

        # In-memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

        # Initialize persistent storage
        self._init_persistent_cache()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True
        )
        self._cleanup_thread.start()

    def _init_persistent_cache(self) -> None:
        """Initialize the persistent cache database."""
        try:
            self.cache_dir.mkdir(exist_ok=True)
            db_path = self.cache_dir / "cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS verification_cache (
                        cache_key TEXT PRIMARY KEY,
                        result_data BLOB,
                        timestamp REAL,
                        test_type TEXT,
                        data_hash TEXT,
                        parameters TEXT,
                        size INTEGER
                    )
                """
                )

                # Create indexes for better performance
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON verification_cache(timestamp)
                """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_test_type 
                    ON verification_cache(test_type)
                """
                )

                conn.commit()

        except Exception as e:
            logger.warning(f"Failed to initialize persistent cache: {e}")
            # Fall back to memory-only cache

    def get(self, cache_key: str) -> Optional[VerificationResult]:
        """
        Get a cached verification result.

        Args:
            cache_key: Unique cache key

        Returns:
            Cached verification result or None if not found
        """
        # Check memory cache first
        with self._cache_lock:
            if cache_key in self._memory_cache:
                cache_entry = self._memory_cache[cache_key]
                if self._is_valid(cache_entry):
                    logger.debug(f"Cache hit in memory: {cache_key}")
                    return cache_entry["result"]
                else:
                    # Remove expired entry
                    del self._memory_cache[cache_key]

        # Check persistent cache
        result = self._get_from_persistent(cache_key)
        if result:
            # Add to memory cache
            self._add_to_memory(cache_key, result)
            logger.debug(f"Cache hit in persistent storage: {cache_key}")
            return result

        logger.debug(f"Cache miss: {cache_key}")
        return None

    def set(self, cache_key: str, result: VerificationResult) -> None:
        """
        Cache a verification result.

        Args:
            cache_key: Unique cache key
            result: Verification result to cache
        """
        try:
            # Add to memory cache
            self._add_to_memory(cache_key, result)

            # Add to persistent cache
            self._add_to_persistent(cache_key, result)

            logger.debug(f"Cached result: {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def _add_to_memory(self, cache_key: str, result: VerificationResult) -> None:
        """Add result to memory cache."""
        with self._cache_lock:
            # Check if we need to evict old entries
            if len(self._memory_cache) >= self.max_size:
                self._evict_oldest_memory()

            # Add new entry
            self._memory_cache[cache_key] = {
                "result": result,
                "timestamp": time.time(),
                "size": self._estimate_size(result),
            }

    def _add_to_persistent(self, cache_key: str, result: VerificationResult) -> None:
        """Add result to persistent cache."""
        try:
            db_path = self.cache_dir / "cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                # Serialize result
                result_data = pickle.dumps(result)

                # Get metadata
                test_type = result.test_type.value
                data_hash = result.metadata.get("cache_key", "")
                parameters = json.dumps(result.metadata.get("parameters", {}))
                size = len(result_data)

                # Insert or update
                conn.execute(
                    """
                    INSERT OR REPLACE INTO verification_cache 
                    (cache_key, result_data, timestamp, test_type, 
                     data_hash, parameters, size)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        cache_key,
                        result_data,
                        time.time(),
                        test_type,
                        data_hash,
                        parameters,
                        size,
                    ),
                )

                conn.commit()

        except Exception as e:
            logger.warning(f"Failed to add to persistent cache: {e}")

    def _get_from_persistent(self, cache_key: str) -> Optional[VerificationResult]:
        """Get result from persistent cache."""
        try:
            db_path = self.cache_dir / "cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT result_data, timestamp FROM verification_cache 
                    WHERE cache_key = ?
                """,
                    (cache_key,),
                )

                row = cursor.fetchone()
                if row:
                    result_data, timestamp = row

                    # Check if entry is still valid
                    if time.time() - timestamp < self.ttl_hours * 3600:
                        result = pickle.loads(result_data)
                        return result
                    else:
                        # Remove expired entry
                        self._remove_from_persistent(cache_key)

        except Exception as e:
            logger.warning(f"Failed to get from persistent cache: {e}")

        return None

    def _remove_from_persistent(self, cache_key: str) -> None:
        """Remove entry from persistent cache."""
        try:
            db_path = self.cache_dir / "cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    DELETE FROM verification_cache WHERE cache_key = ?
                """,
                    (cache_key,),
                )
                conn.commit()

        except Exception as e:
            logger.warning(f"Failed to remove from persistent cache: {e}")

    def _is_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid."""
        timestamp = cache_entry["timestamp"]
        return time.time() - timestamp < self.ttl_hours * 3600

    def _evict_oldest_memory(self) -> None:
        """Evict the oldest entries from memory cache."""
        # Sort by timestamp and remove oldest 10% of entries
        entries = sorted(self._memory_cache.items(), key=lambda x: x[1]["timestamp"])

        to_remove = max(1, len(entries) // 10)
        for i in range(to_remove):
            if entries:
                key, _ = entries.pop(0)
                del self._memory_cache[key]

    def _estimate_size(self, result: VerificationResult) -> int:
        """Estimate the memory size of a result object."""
        # Simple size estimation
        size = 0
        size += len(result.components) * 100  # Rough estimate per component
        size += len(result.warnings) * 50  # Rough estimate per warning
        size += len(result.metadata) * 20  # Rough estimate per metadata item
        return size

    def clear(self) -> None:
        """Clear all cached results."""
        with self._cache_lock:
            self._memory_cache.clear()

        try:
            db_path = self.cache_dir / "cache.db"
            if db_path.exists():
                db_path.unlink()
            self._init_persistent_cache()
        except Exception as e:
            logger.warning(f"Failed to clear persistent cache: {e}")

        logger.info("Cache cleared")

    def clear_expired(self) -> None:
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = []

        # Clear expired memory cache entries
        with self._cache_lock:
            for key, entry in self._memory_cache.items():
                if not self._is_valid(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                del self._memory_cache[key]

        # Clear expired persistent cache entries
        try:
            db_path = self.cache_dir / "cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    DELETE FROM verification_cache 
                    WHERE timestamp < ?
                """,
                    (current_time - self.ttl_hours * 3600,),
                )
                conn.commit()

        except Exception as e:
            logger.warning(f"Failed to clear expired persistent cache: {e}")

        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            memory_size = len(self._memory_cache)
            memory_usage = sum(entry["size"] for entry in self._memory_cache.values())

        persistent_stats = self._get_persistent_stats()

        return {
            "memory_cache": {
                "entries": memory_size,
                "estimated_size_bytes": memory_usage,
                "max_size": self.max_size,
            },
            "persistent_cache": persistent_stats,
            "ttl_hours": self.ttl_hours,
            "cache_directory": str(self.cache_dir),
        }

    def _get_persistent_stats(self) -> Dict[str, Any]:
        """Get persistent cache statistics."""
        try:
            db_path = self.cache_dir / "cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                # Total entries
                cursor = conn.execute("SELECT COUNT(*) FROM verification_cache")
                total_entries = cursor.fetchone()[0]

                # Total size
                cursor = conn.execute("SELECT SUM(size) FROM verification_cache")
                total_size = cursor.fetchone()[0] or 0

                # Oldest entry
                cursor = conn.execute(
                    """
                    SELECT MIN(timestamp) FROM verification_cache
                """
                )
                oldest_timestamp = cursor.fetchone()[0]

                # Newest entry
                cursor = conn.execute(
                    """
                    SELECT MAX(timestamp) FROM verification_cache
                """
                )
                newest_timestamp = cursor.fetchone()[0]

                return {
                    "entries": total_entries,
                    "total_size_bytes": total_size,
                    "oldest_entry": oldest_timestamp,
                    "newest_entry": newest_timestamp,
                }

        except Exception as e:
            logger.warning(f"Failed to get persistent cache stats: {e}")
            return {
                "entries": 0,
                "total_size_bytes": 0,
                "oldest_entry": None,
                "newest_entry": None,
            }

    def _cleanup_worker(self) -> None:
        """Background worker for cache cleanup."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self.clear_expired()
            except Exception as e:
                logger.error(f"Cache cleanup worker error: {e}")

    def search(
        self,
        test_type: str = None,
        min_timestamp: float = None,
        max_timestamp: float = None,
    ) -> List[str]:
        """
        Search for cache entries matching criteria.

        Args:
            test_type: Filter by test type
            min_timestamp: Minimum timestamp filter
            max_timestamp: Maximum timestamp filter

        Returns:
            List of matching cache keys
        """
        try:
            db_path = self.cache_dir / "cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                query = "SELECT cache_key FROM verification_cache WHERE 1=1"
                params = []

                if test_type:
                    query += " AND test_type = ?"
                    params.append(test_type)

                if min_timestamp:
                    query += " AND timestamp >= ?"
                    params.append(min_timestamp)

                if max_timestamp:
                    query += " AND timestamp <= ?"
                    params.append(max_timestamp)

                cursor = conn.execute(query, params)
                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.warning(f"Failed to search cache: {e}")
            return []

    def export_cache(self, export_path: str) -> None:
        """
        Export cache to a file.

        Args:
            export_path: Path to export file
        """
        try:
            export_data = {
                "metadata": {
                    "export_timestamp": time.time(),
                    "cache_stats": self.get_stats(),
                },
                "entries": {},
            }

            # Export memory cache
            with self._cache_lock:
                for key, entry in self._memory_cache.items():
                    if self._is_valid(entry):
                        export_data["entries"][key] = {
                            "result": entry["result"].dict(),
                            "timestamp": entry["timestamp"],
                            "size": entry["size"],
                        }

            # Export persistent cache
            db_path = self.cache_dir / "cache.db"
            if db_path.exists():
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute(
                        """
                        SELECT cache_key, result_data, timestamp, 
                               test_type, data_hash, parameters, size
                        FROM verification_cache
                    """
                    )

                    for row in cursor.fetchall():
                        (
                            key,
                            result_data,
                            timestamp,
                            test_type,
                            data_hash,
                            parameters,
                            size,
                        ) = row
                        result = pickle.loads(result_data)
                        export_data["entries"][key] = {
                            "result": result.dict(),
                            "timestamp": timestamp,
                            "test_type": test_type,
                            "data_hash": data_hash,
                            "parameters": parameters,
                            "size": size,
                        }

            # Write export file
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Cache exported to {export_path}")

        except Exception as e:
            raise CacheError(f"Failed to export cache: {e}")

    def import_cache(self, import_path: str) -> None:
        """
        Import cache from a file.

        Args:
            import_path: Path to import file
        """
        try:
            with open(import_path, "r") as f:
                import_data = json.load(f)

            # Clear existing cache
            self.clear()

            # Import entries
            for key, entry_data in import_data["entries"].items():
                try:
                    # Reconstruct result object
                    result = VerificationResult(**entry_data["result"])

                    # Add to cache
                    self.set(key, result)

                except Exception as e:
                    logger.warning(f"Failed to import cache entry {key}: {e}")

            logger.info(f"Cache imported from {import_path}")

        except Exception as e:
            raise CacheError(f"Failed to import cache: {e}")

    def __del__(self):
        """Cleanup when cache is destroyed."""
        try:
            self.clear_expired()
        except:
            pass
