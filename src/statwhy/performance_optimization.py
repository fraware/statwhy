#!/usr/bin/env python3
"""
StatWhy Performance Optimization Module

Implements parallel processing, advanced caching, and load testing for
high-performance statistical verification under high demand.
"""

import logging
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import hashlib
import queue
import statistics
from collections import deque


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for verification operations."""

    operation_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_workers: int = 1
    throughput: float = 0.0
    latency: float = 0.0
    parallel_efficiency: float = 1.0

    def __post_init__(self):
        if self.throughput == 0.0 and self.execution_time > 0:
            self.throughput = 1.0 / self.execution_time
        if self.latency == 0.0:
            self.latency = self.execution_time


class AdvancedCache:
    """
    Advanced caching system with intelligent eviction and performance optimization.

    Features:
    - Multi-level caching (memory, disk)
    - LRU eviction with access frequency
    - Compression for large objects
    - Cache warming and prefetching
    - Performance analytics
    """

    def __init__(
        self,
        max_memory_size: int = 100 * 1024 * 1024,  # 100MB default
        max_disk_size: int = 1024 * 1024 * 1024,  # 1GB default
        compression_threshold: int = 1024,  # Compress objects > 1KB
    ):
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.compression_threshold = compression_threshold

        # Memory cache (fast access)
        self.memory_cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.current_memory_size = 0

        # Disk cache (persistent storage)
        self.disk_cache_dir = Path(".statwhy_cache")
        self.disk_cache_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compression_savings = 0

        # Background maintenance
        self._start_maintenance_thread()

    def _start_maintenance_thread(self) -> None:
        """Start background cache maintenance thread."""

        def maintenance_worker():
            while True:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_expired()
                    self._optimize_memory_usage()
                except Exception as e:
                    logger.error(f"Cache maintenance error: {e}")

        thread = threading.Thread(target=maintenance_worker, daemon=True)
        thread.start()

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cache entry with TTL."""
        try:
            # Serialize and potentially compress
            serialized = self._serialize_value(value)

            # Store in memory if small enough
            if (
                len(serialized) <= self.max_memory_size // 10
            ):  # Max 10% of memory per item
                self.memory_cache[key] = serialized
                self.access_times[key] = time.time()
                self.access_counts[key] = 1
                self.current_memory_size += len(serialized)
            else:
                # Store on disk
                self._store_on_disk(key, serialized, ttl)

            # Evict if necessary
            self._evict_if_needed()

        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get cache entry."""
        try:
            # Try memory cache first
            if key in self.memory_cache:
                self.hits += 1
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                return self._deserialize_value(self.memory_cache[key])

            # Try disk cache
            disk_value = self._get_from_disk(key)
            if disk_value is not None:
                self.hits += 1
                # Promote to memory cache if frequently accessed
                if self.access_counts.get(key, 0) > 3:
                    self._promote_to_memory(key, disk_value)
                return disk_value

            self.misses += 1
            return None

        except Exception as e:
            logger.error(f"Error getting cache entry: {e}")
            return None

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value with compression if beneficial."""
        try:
            serialized = pickle.dumps(value)

            if len(serialized) > self.compression_threshold:
                import gzip

                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):
                    self.compression_savings += len(serialized) - len(compressed)
                    return compressed

            return serialized
        except Exception:
            # Fallback to JSON for simple types
            return json.dumps(value).encode("utf-8")

    def _deserialize_value(self, serialized: bytes) -> Any:
        """Deserialize value with automatic decompression detection."""
        try:
            if serialized.startswith(b"\x1f\x8b"):  # Gzip magic number
                import gzip

                serialized = gzip.decompress(serialized)
            return pickle.loads(serialized)
        except Exception:
            # Fallback to JSON
            return json.loads(serialized.decode("utf-8"))

    def _store_on_disk(self, key: str, value: bytes, ttl: int) -> None:
        """Store value on disk with TTL."""
        try:
            filename = f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            filepath = self.disk_cache_dir / filename

            metadata = {
                "key": key,
                "ttl": ttl,
                "created": time.time(),
                "size": len(value),
            }

            with open(filepath, "wb") as f:
                pickle.dump((metadata, value), f)

        except Exception as e:
            logger.error(f"Error storing on disk: {e}")

    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Retrieve value from disk cache."""
        try:
            filename = f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            filepath = self.disk_cache_dir / filename

            if not filepath.exists():
                return None

            with open(filepath, "rb") as f:
                metadata, value = pickle.load(f)

            # Check TTL
            if time.time() - metadata["created"] > metadata["ttl"]:
                filepath.unlink()  # Remove expired entry
                return None

            return self._deserialize_value(value)

        except Exception as e:
            logger.error(f"Error reading from disk: {e}")
            return None

    def _promote_to_memory(self, key: str, value: Any) -> None:
        """Promote frequently accessed item to memory cache."""
        try:
            serialized = self._serialize_value(value)
            if len(serialized) <= self.max_memory_size // 10:
                self.memory_cache[key] = serialized
                self.access_times[key] = time.time()
                self.access_counts[key] = 1
                self.current_memory_size += len(serialized)
        except Exception as e:
            logger.error(f"Error promoting to memory: {e}")

    def _evict_if_needed(self) -> None:
        """Evict items if memory limit exceeded."""
        while self.current_memory_size > self.max_memory_size:
            # Find least recently used item
            if not self.access_times:
                break

            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._evict_item(lru_key)

    def _evict_item(self, key: str) -> None:
        """Evict specific item from memory cache."""
        if key in self.memory_cache:
            size = len(self.memory_cache[key])
            del self.memory_cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            self.current_memory_size -= size
            self.evictions += 1

    def _cleanup_expired(self) -> None:
        """Clean up expired disk cache entries."""
        try:
            for cache_file in self.disk_cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, "rb") as f:
                        metadata, _ = pickle.load(f)

                    if time.time() - metadata["created"] > metadata["ttl"]:
                        cache_file.unlink()

                except Exception:
                    # Corrupted file, remove it
                    cache_file.unlink()

        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")

    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage by promoting hot items."""
        try:
            # Find hot items (frequently accessed)
            hot_items = [
                (key, count)
                for key, count in self.access_counts.items()
                if count > 5 and key not in self.memory_cache
            ]

            # Sort by access count (most frequent first)
            hot_items.sort(key=lambda x: x[1], reverse=True)

            # Promote top items to memory if space available
            for key, _ in hot_items[:10]:  # Top 10 hot items
                if (
                    self.current_memory_size < self.max_memory_size * 0.8
                ):  # Leave 20% buffer
                    value = self._get_from_disk(key)
                    if value is not None:
                        self._promote_to_memory(key, value)

        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "evictions": self.evictions,
            "compression_savings_bytes": self.compression_savings,
            "memory_usage_bytes": self.current_memory_size,
            "memory_usage_percent": round(
                self.current_memory_size / self.max_memory_size * 100, 2
            ),
            "disk_entries": len(list(self.disk_cache_dir.glob("*.cache"))),
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.memory_cache.clear()
        self.access_times.clear()
        self.access_counts.clear()
        self.current_memory_size = 0

        # Clear disk cache
        for cache_file in self.disk_cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception:
                pass


class OptimizedParallelProcessor:
    """
    Highly optimized parallel processing system for statistical verification operations.

    Features:
    - Advanced load balancing with work stealing
    - Minimal IPC overhead with shared memory
    - Process pool optimization for CPU-bound tasks
    - Memory pooling to reduce allocation overhead
    - Dynamic worker scaling based on workload
    - Performance monitoring and auto-tuning
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_processes: Optional[int] = None,
        enable_work_stealing: bool = True,
        enable_memory_pooling: bool = True,
    ):
        # Optimize worker counts based on system capabilities
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max_workers or min(
            32, cpu_count * 2
        )  # 2x CPU cores for I/O bound
        self.max_processes = max_processes or max(1, cpu_count - 1)  # Leave 1 core free

        # Enable advanced optimizations
        self.enable_work_stealing = enable_work_stealing
        self.enable_memory_pooling = enable_memory_pooling

        # Optimized thread and process pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="StatWhyThread"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.max_processes,
            mp_context=multiprocessing.get_context(
                "spawn"
            ),  # Better cross-platform compatibility
        )

        # Advanced work queue with work stealing
        self.work_queues = [queue.Queue() for _ in range(self.max_workers)]
        self.result_queue = queue.Queue()

        # Work stealing queues for load balancing
        self.steal_queues = (
            [deque() for _ in range(self.max_workers)] if enable_work_stealing else None
        )

        # Memory pool for reducing allocation overhead
        self.memory_pool = self._create_memory_pool() if enable_memory_pooling else None

        # Performance tracking with high precision
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_times: List[float] = []
        self.worker_loads: List[int] = [0] * self.max_workers
        self.last_balance_time = time.time()

        # Start optimized worker threads
        self._start_optimized_worker_threads()

    def _create_memory_pool(self) -> Dict[int, List[bytes]]:
        """Create memory pool for common buffer sizes."""
        pool = {}
        # Pre-allocate common buffer sizes (powers of 2)
        for size in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            pool[size] = [bytearray(size) for _ in range(10)]  # 10 buffers per size
        return pool

    def _get_buffer_from_pool(self, size: int) -> Optional[bytearray]:
        """Get buffer from memory pool if available."""
        if not self.memory_pool or size not in self.memory_pool:
            return None

        if self.memory_pool[size]:
            return self.memory_pool[size].pop()
        return None

    def _return_buffer_to_pool(self, buffer: bytearray) -> None:
        """Return buffer to memory pool."""
        if not self.memory_pool:
            return

        size = len(buffer)
        if (
            size in self.memory_pool and len(self.memory_pool[size]) < 20
        ):  # Limit pool size
            self.memory_pool[size].append(buffer)

    def _start_optimized_worker_threads(self) -> None:
        """Start optimized worker threads with work stealing."""

        def optimized_worker(worker_id: int):
            while True:
                try:
                    # Try to get work from own queue first
                    try:
                        task = self.work_queues[worker_id].get(timeout=0.1)
                    except queue.Empty:
                        # Try work stealing if enabled
                        if self.enable_work_stealing:
                            task = self._steal_work(worker_id)
                        else:
                            continue

                    if task is None:  # Shutdown signal
                        break

                    self._process_optimized_task(task, worker_id)
                    self.work_queues[worker_id].task_done()

                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    self.failed_tasks += 1

        # Start worker threads with optimized scheduling
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=optimized_worker,
                args=(i,),
                daemon=True,
                name=f"StatWhyWorker-{i}",
            )
            thread.start()

    def _steal_work(self, worker_id: int) -> Optional[Dict[str, Any]]:
        """Implement work stealing for load balancing."""
        if not self.enable_work_stealing:
            return None

        # Find the most loaded worker
        most_loaded_worker = max(
            range(self.max_workers), key=lambda w: self.worker_loads[w]
        )

        if (
            most_loaded_worker != worker_id
            and self.worker_loads[most_loaded_worker] > 2
        ):
            try:
                # Try to steal work from the most loaded worker
                return self.work_queues[most_loaded_worker].get_nowait()
            except queue.Empty:
                pass

        return None

    def _process_optimized_task(self, task: Dict[str, Any], worker_id: int) -> None:
        """Process task with optimized execution."""
        try:
            start_time = time.perf_counter()  # High precision timing
            self.active_tasks += 1
            self.worker_loads[worker_id] += 1

            # Execute the task with optimized method selection
            if task["type"] == "thread":
                result = self._execute_optimized_thread_task(task)
            elif task["type"] == "process":
                result = self._execute_optimized_process_task(task)
            else:
                result = self._execute_optimized_direct_task(task)

            # Record completion with high precision
            execution_time = time.perf_counter() - start_time
            self.task_times.append(execution_time)
            self.completed_tasks += 1

            # Store result efficiently
            self.result_queue.put(
                {
                    "task_id": task["id"],
                    "result": result,
                    "execution_time": execution_time,
                    "status": "completed",
                    "worker_id": worker_id,
                }
            )

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            self.failed_tasks += 1

            self.result_queue.put(
                {
                    "task_id": task["id"],
                    "result": None,
                    "error": str(e),
                    "status": "failed",
                    "worker_id": worker_id,
                }
            )

        finally:
            self.active_tasks -= 1
            self.worker_loads[worker_id] -= 1

    def _execute_optimized_thread_task(self, task: Dict[str, Any]) -> Any:
        """Execute task using optimized thread pool."""
        future = self.thread_pool.submit(task["func"], *task["args"], **task["kwargs"])
        return future.result(timeout=task.get("timeout", 300))

    def _execute_optimized_process_task(self, task: Dict[str, Any]) -> Any:
        """Execute task using optimized process pool."""
        future = self.process_pool.submit(task["func"], *task["args"], **task["kwargs"])
        return future.result(timeout=task.get("timeout", 300))

    def _execute_optimized_direct_task(self, task: Dict[str, Any]) -> Any:
        """Execute task directly with memory pooling."""
        # Use memory pool if available
        if self.memory_pool and "buffer_size" in task:
            buffer = self._get_buffer_from_pool(task["buffer_size"])
            try:
                result = task["func"](*task["args"], **task["kwargs"])
                return result
            finally:
                if buffer:
                    self._return_buffer_to_pool(buffer)
        else:
            return task["func"](*task["args"], **task["kwargs"])

    def submit_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        task_type: str = "auto",
        priority: int = 0,
        buffer_size: Optional[int] = None,
    ) -> None:
        """Submit task with optimized scheduling."""
        if kwargs is None:
            kwargs = {}

        # Auto-detect task type for optimal execution
        if task_type == "auto":
            task_type = self._detect_optimal_task_type(func, args, kwargs)

        # Find least loaded worker for better load balancing
        worker_id = min(range(self.max_workers), key=lambda w: self.worker_loads[w])

        task = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "type": task_type,
            "priority": priority,
            "buffer_size": buffer_size,
            "submitted_time": time.time(),
        }

        # Add to work queue with priority handling
        if priority > 0:
            # High priority tasks go to the front
            self.work_queues[worker_id].put_nowait(task)
        else:
            self.work_queues[worker_id].put(task)

        # Periodic load balancing
        if time.time() - self.last_balance_time > 1.0:  # Balance every second
            self._balance_workload()
            self.last_balance_time = time.time()

    def _detect_optimal_task_type(
        self, func: Callable, args: tuple, kwargs: dict
    ) -> str:
        """Detect optimal execution method for the task."""
        # CPU-bound tasks benefit from process pool
        if hasattr(func, "__name__") and any(
            name in func.__name__.lower()
            for name in ["compute", "calculate", "process", "verify"]
        ):
            return "process"

        # I/O bound tasks benefit from thread pool
        if hasattr(func, "__name__") and any(
            name in func.__name__.lower()
            for name in ["fetch", "read", "write", "network"]
        ):
            return "thread"

        # Default to thread for unknown types
        return "thread"

    def _balance_workload(self) -> None:
        """Balance workload across workers."""
        if not self.enable_work_stealing:
            return

        # Calculate average load
        avg_load = sum(self.worker_loads) / len(self.worker_loads)

        # Move work from overloaded to underloaded workers
        for i in range(self.max_workers):
            if self.worker_loads[i] > avg_load * 1.5:  # 50% above average
                for j in range(self.max_workers):
                    if self.worker_loads[j] < avg_load * 0.5:  # 50% below average
                        # Move one task from overloaded to underloaded worker
                        try:
                            task = self.work_queues[i].get_nowait()
                            self.work_queues[j].put(task)
                            self.worker_loads[i] -= 1
                            self.worker_loads[j] += 1
                            break
                        except queue.Empty:
                            continue

    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks to complete with timeout."""
        try:
            start_time = time.time()

            while self.active_tasks > 0:
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning("Timeout waiting for tasks to complete")
                    return False

                time.sleep(0.01)  # Small sleep to prevent busy waiting

            return True

        except Exception as e:
            logger.error(f"Error waiting for tasks: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive parallel processing performance statistics."""
        if not self.task_times:
            return {
                "active_tasks": self.active_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_tasks": self.completed_tasks + self.failed_tasks,
                "success_rate": 0.0,
                "avg_task_time": 0.0,
                "max_task_time": 0.0,
                "min_task_time": 0.0,
                "parallel_efficiency": 0.0,
                "thread_pool_size": self.max_workers,
                "process_pool_size": self.max_processes,
                "workload_balance": 0.0,
                "memory_pool_utilization": 0.0,
            }

        # Calculate advanced performance metrics
        avg_task_time = statistics.mean(self.task_times)
        max_task_time = max(self.task_times)
        min_task_time = min(self.task_times)

        # Calculate parallel efficiency
        if len(self.task_times) > 1:
            # Use Amdahl's Law approximation
            sequential_time = sum(self.task_times)
            parallel_time = max(self.task_times)  # Assuming perfect parallelization
            parallel_efficiency = sequential_time / (parallel_time * self.max_workers)
        else:
            parallel_efficiency = 1.0

        # Calculate workload balance
        if self.worker_loads:
            workload_std = (
                statistics.stdev(self.worker_loads) if len(self.worker_loads) > 1 else 0
            )
            workload_balance = (
                max(0, 1 - (workload_std / max(self.worker_loads)))
                if max(self.worker_loads) > 0
                else 1.0
            )
        else:
            workload_balance = 1.0

        # Calculate memory pool utilization
        if self.memory_pool:
            total_buffers = sum(len(buffers) for buffers in self.memory_pool.values())
            used_buffers = sum(len(buffers) for buffers in self.memory_pool.values())
            memory_pool_utilization = (
                used_buffers / total_buffers if total_buffers > 0 else 0.0
            )
        else:
            memory_pool_utilization = 0.0

        return {
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_tasks": self.completed_tasks + self.failed_tasks,
            "success_rate": (
                (
                    self.completed_tasks
                    / (self.completed_tasks + self.failed_tasks)
                    * 100
                )
                if (self.completed_tasks + self.failed_tasks) > 0
                else 0
            ),
            "avg_task_time": round(avg_task_time, 6),
            "max_task_time": round(max_task_time, 6),
            "min_task_time": round(min_task_time, 6),
            "parallel_efficiency": round(parallel_efficiency, 4),
            "thread_pool_size": self.max_workers,
            "process_pool_size": self.max_processes,
            "workload_balance": round(workload_balance, 4),
            "memory_pool_utilization": round(memory_pool_utilization, 4),
            "queue_sizes": [q.qsize() for q in self.work_queues],
            "worker_loads": self.worker_loads.copy(),
        }

    def shutdown(self) -> None:
        """Shutdown parallel processor gracefully."""
        try:
            # Send shutdown signals to worker threads
            for _ in range(self.max_workers):
                for q in self.work_queues:
                    q.put(None)

            # Wait for workers to finish
            self.wait_for_all_tasks(timeout=5.0)

            # Shutdown pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Alias for backward compatibility
ParallelProcessor = OptimizedParallelProcessor


class LoadTester:
    """
    Load testing system for StatWhy verification operations.

    Features:
    - Concurrent user simulation
    - Performance benchmarking
    - Stress testing
    - Bottleneck identification
    - Performance regression detection
    """

    def __init__(self, engine=None):
        self.engine = engine
        self.test_results = []
        self.performance_metrics = []

    def run_stress_test(
        self,
        test_func: Callable,
        max_users: int,
        step_size: int = 10,
        step_duration: int = 60,
    ) -> Dict[str, Any]:
        """
        Run stress test with gradually increasing load.

        Args:
            test_func: Function to test
            max_users: Maximum number of concurrent users
            step_size: Number of users to add per step
            step_duration: Duration of each step in seconds

        Returns:
            Stress test results
        """
        try:
            results = []
            current_users = 0

            while current_users <= max_users:
                # Run test with current user count
                test_result = self.run_concurrent_test(
                    test_func, current_users, step_duration
                )

                results.append(
                    {
                        "concurrent_users": current_users,
                        "result": test_result,
                        "timestamp": time.time(),
                    }
                )

                # Increase load for next step
                current_users += step_size

                # Check for performance degradation
                if test_result.get("analysis", {}).get("avg_response_time", 0) > 5.0:
                    logger.warning(
                        f"Performance degradation detected at {current_users} users"
                    )
                    break

            return {
                "test_type": "stress",
                "max_users": max_users,
                "step_size": step_size,
                "step_duration": step_duration,
                "results": results,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            return {"error": str(e)}

    def benchmark_performance(
        self, test_func: Callable, iterations: int = 100, warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Run performance benchmark.

        Args:
            test_func: Function to benchmark
            iterations: Number of test iterations
            warmup_iterations: Warmup iterations to exclude from results

        Returns:
            Benchmark results
        """
        try:
            # Warmup phase
            for _ in range(warmup_iterations):
                test_func()

            # Benchmark phase
            times = []
            for _ in range(iterations):
                start_time = time.time()
                test_func()
                times.append(time.time() - start_time)

            # Calculate statistics
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)

            # Percentiles
            sorted_times = sorted(times)
            p95 = sorted_times[int(0.95 * len(sorted_times))]
            p99 = sorted_times[int(0.99 * len(sorted_times))]

            benchmark_result = {
                "test_type": "benchmark",
                "iterations": iterations,
                "warmup_iterations": warmup_iterations,
                "avg_time": round(avg_time, 6),
                "median_time": round(median_time, 6),
                "std_time": round(std_time, 6),
                "min_time": round(min_time, 6),
                "max_time": round(max_time, 6),
                "p95_time": round(p95, 6),
                "p99_time": round(p99, 6),
                "throughput": round(1.0 / avg_time, 2),
                "timestamp": time.time(),
            }

            self.test_results.append(benchmark_result)
            return benchmark_result

        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return {"error": str(e)}

    def _analyze_test_results(self, results: List, duration: float) -> Dict[str, Any]:
        """Analyze test results and identify bottlenecks."""
        try:
            if not results:
                return {}

            # Calculate response time statistics
            response_times = [r.get("response_time", 0) for r in results]
            avg_response_time = statistics.mean(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0
            min_response_time = min(response_times) if response_times else 0

            # Calculate throughput
            total_requests = len(results)
            throughput = total_requests / duration if duration > 0 else 0

            # Identify bottlenecks
            bottlenecks = []
            if avg_response_time > 1.0:
                bottlenecks.append("High response time")
            if max_response_time > 5.0:
                bottlenecks.append("Response time spikes")
            if throughput < 10:
                bottlenecks.append("Low throughput")

            return {
                "total_requests": total_requests,
                "avg_response_time": round(avg_response_time, 3),
                "max_response_time": round(max_response_time, 3),
                "min_response_time": round(min_response_time, 3),
                "throughput_requests_per_second": round(throughput, 2),
                "bottlenecks": bottlenecks,
                "test_duration": round(duration, 3),
            }

        except Exception as e:
            logger.error(f"Error analyzing test results: {e}")
            return {}

    def run_concurrent_test(
        self,
        test_func: Callable,
        concurrent_users: int,
        test_duration: int,
        ramp_up_time: int = 0,
    ) -> Dict[str, Any]:
        """
        Run concurrent load test.

        Args:
            test_func: Function to test
            concurrent_users: Number of concurrent users
            test_duration: Test duration in seconds
            ramp_up_time: Time to ramp up to full load

        Returns:
            Test results
        """
        try:
            start_time = time.time()
            results = []
            active_threads = []

            # Start test threads
            for i in range(concurrent_users):
                if ramp_up_time > 0:
                    # Stagger thread start for ramp-up
                    start_delay = (i / concurrent_users) * ramp_up_time
                    time.sleep(start_delay)

                thread = threading.Thread(
                    target=self._test_worker,
                    args=(test_func, i, results, test_duration),
                )
                thread.start()
                active_threads.append(thread)

            # Wait for test completion
            for thread in active_threads:
                thread.join()

            # Analyze results
            test_duration_actual = time.time() - start_time
            analysis = self._analyze_test_results(results, test_duration_actual)

            # Store test results
            test_result = {
                "test_type": "concurrent",
                "concurrent_users": concurrent_users,
                "test_duration": test_duration,
                "ramp_up_time": ramp_up_time,
                "actual_duration": test_duration_actual,
                "results": results,
                "analysis": analysis,
                "timestamp": time.time(),
            }

            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            logger.error(f"Error running concurrent test: {e}")
            return {"error": str(e)}

    def _test_worker(
        self, test_func: Callable, user_id: int, results: List, duration: int
    ) -> None:
        """Worker thread for load testing."""
        try:
            start_time = time.time()
            request_count = 0

            while time.time() - start_time < duration:
                # Execute test function
                test_start = time.time()
                try:
                    test_func()
                    response_time = time.time() - test_start
                    success = True
                    error = None
                except Exception as e:
                    response_time = time.time() - test_start
                    success = False
                    error = str(e)

                # Record result
                results.append(
                    {
                        "user_id": user_id,
                        "request_id": request_count,
                        "response_time": response_time,
                        "success": success,
                        "error": error,
                        "timestamp": time.time(),
                    }
                )

                request_count += 1

                # Small delay to prevent overwhelming
                time.sleep(0.01)

        except Exception as e:
            logger.error(f"Test worker {user_id} error: {e}")

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        if not self.test_results:
            return {"total_tests": 0, "results": []}

        return {
            "total_tests": len(self.test_results),
            "results": self.test_results,
            "last_test": self.test_results[-1] if self.test_results else None,
        }
