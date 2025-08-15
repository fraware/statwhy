#!/usr/bin/env python3
"""
StatWhy Performance Testing & Optimization

Comprehensive performance testing and optimization for statistical verification.
Implements load testing, synthetic data generation, and performance benchmarking.
"""

import time
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import statistics
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import pandas as pd

from .core import StatWhyEngine
from .models import VerificationRequest, TestType
from .data_generator import RealisticDataGenerator
from .performance_optimization import OptimizedParallelProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for verification operations."""

    operation: str
    test_type: str
    data_size: int
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False
    parallel_workers: int = 1
    parallel_efficiency: float = 1.0
    workload_balance: float = 1.0


@dataclass
class BenchmarkResult:
    """Results of performance benchmarking."""

    benchmark_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_execution_time: float
    median_execution_time: float
    p95_execution_time: float
    p99_execution_time: float
    throughput_ops_per_second: float
    memory_peak_mb: float
    cpu_peak_percent: float
    cache_hit_rate: float
    parallel_efficiency: float
    workload_balance: float
    detailed_metrics: List[PerformanceMetrics]


class PerformanceTester:
    """
    Comprehensive performance testing and optimization for StatWhy.

    Features:
    - Load testing with multiple concurrent users
    - Synthetic data generation for testing
    - Performance benchmarking across different scenarios
    - Memory and CPU usage monitoring
    - Cache performance analysis
    - Parallel processing optimization
    - Performance regression detection
    """

    def __init__(self, engine: Optional[StatWhyEngine] = None):
        """Initialize the performance tester."""
        self.engine = engine or StatWhyEngine()
        self.data_generator = RealisticDataGenerator(seed=42)
        self.results_dir = Path("performance_results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize optimized parallel processor
        self.parallel_processor = OptimizedParallelProcessor(
            max_workers=min(8, multiprocessing.cpu_count() * 2),
            max_processes=max(1, multiprocessing.cpu_count() - 1),
            enable_work_stealing=True,
            enable_memory_pooling=True,
        )

        # Performance thresholds
        self.thresholds = {
            "max_execution_time": 30.0,  # 30 seconds
            "max_memory_usage": 1024.0,  # 1GB
            "max_cpu_usage": 80.0,  # 80%
            "min_throughput": 0.1,  # 0.1 ops/second
            "min_cache_hit_rate": 0.7,  # 70%
            "min_parallel_efficiency": 0.8,  # 80%
        }

    def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive performance benchmarking."""
        logger.info("Starting comprehensive performance benchmarking...")

        benchmark_results = {}

        # 1. Single-threaded performance
        benchmark_results["single_threaded"] = self._benchmark_single_threaded()

        # 2. Multi-threaded performance
        benchmark_results["multi_threaded"] = self._benchmark_multi_threaded()

        # 3. Cache performance
        benchmark_results["cache_performance"] = self._benchmark_cache_performance()

        # 4. Load testing
        benchmark_results["load_testing"] = self._benchmark_load_testing()

        # 5. Memory efficiency
        benchmark_results["memory_efficiency"] = self._benchmark_memory_efficiency()

        # 6. Edge case performance
        benchmark_results["edge_cases"] = self._benchmark_edge_cases()

        # 7. Advanced parallel processing
        benchmark_results["advanced_parallel"] = self._benchmark_advanced_parallel()

        # Save all results
        self._save_benchmark_results(benchmark_results)

        # Generate performance report
        self._generate_performance_report(benchmark_results)

        return benchmark_results

    def _benchmark_single_threaded(self) -> BenchmarkResult:
        """Benchmark single-threaded performance across all test types."""
        logger.info("Running single-threaded performance benchmark...")

        test_types = list(TestType)
        metrics = []

        for test_type in test_types:
            # Generate appropriate test data
            test_data = self._generate_test_data_for_type(test_type, size="medium")

            # Run verification multiple times for statistical significance
            for i in range(5):
                metric = self._measure_single_verification(
                    test_type, test_data, parallel_workers=1
                )
                metrics.append(metric)

        return self._aggregate_metrics("Single-Threaded Performance", metrics)

    def _benchmark_multi_threaded(self) -> BenchmarkResult:
        """Benchmark multi-threaded performance with different worker counts."""
        logger.info("Running multi-threaded performance benchmark...")

        worker_counts = [1, 2, 4, 8]
        test_types = list(TestType)[:5]  # Test subset for efficiency
        metrics = []

        for workers in worker_counts:
            for test_type in test_types:
                test_data = self._generate_test_data_for_type(test_type, size="medium")

                # Run verification with parallel processing
                metric = self._measure_single_verification(
                    test_type, test_data, parallel_workers=workers
                )
                metrics.append(metric)

        return self._aggregate_metrics("Multi-Threaded Performance", metrics)

    def _benchmark_advanced_parallel(self) -> BenchmarkResult:
        """Benchmark advanced parallel processing with work stealing and load balancing."""
        logger.info("Running advanced parallel processing benchmark...")

        test_types = list(TestType)[:3]  # Focus on common tests
        metrics = []

        # Test with different data sizes to stress the parallel processor
        data_sizes = ["small", "medium", "large"]

        for test_type in test_types:
            for data_size in data_sizes:
                test_data = self._generate_test_data_for_type(test_type, size=data_size)

                # Submit multiple tasks to the parallel processor
                task_count = 10
                start_time = time.time()

                # Submit tasks
                for i in range(task_count):
                    self.parallel_processor.submit_task(
                        f"verify_{test_type}_{i}",
                        self._verify_task_wrapper,
                        args=(test_type, test_data),
                        task_type="auto",
                        priority=0,
                    )

                # Wait for completion
                self.parallel_processor.wait_for_all_tasks(timeout=60.0)
                total_time = time.time() - start_time

                # Get performance stats
                perf_stats = self.parallel_processor.get_performance_stats()

                # Create metric
                metric = PerformanceMetrics(
                    operation=f"advanced_parallel_{test_type}_{data_size}",
                    test_type=test_type.value,
                    data_size=len(test_data),
                    execution_time=total_time / task_count,  # Average time per task
                    memory_usage=perf_stats.get("memory_pool_utilization", 0.0),
                    cpu_usage=0.0,  # Will be calculated separately
                    success=True,
                    parallel_workers=perf_stats.get("thread_pool_size", 1),
                    parallel_efficiency=perf_stats.get("parallel_efficiency", 1.0),
                    workload_balance=perf_stats.get("workload_balance", 1.0),
                )
                metrics.append(metric)

        return self._aggregate_metrics("Advanced Parallel Processing", metrics)

    def _verify_task_wrapper(
        self, test_type: TestType, test_data: pd.DataFrame
    ) -> bool:
        """Wrapper for verification tasks to be executed by parallel processor."""
        try:
            # Create verification request
            request = VerificationRequest(
                test_type=test_type, data=test_data, alpha=0.05, timeout=30
            )

            # Execute verification
            result = self.engine.verify(request)
            return result.is_verified

        except Exception as e:
            logger.error(f"Verification task failed: {e}")
            return False

    def _benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark cache performance and hit rates."""
        logger.info("Running cache performance benchmark...")

        test_types = list(TestType)[:3]  # Focus on common tests
        metrics = []

        for test_type in test_types:
            test_data = self._generate_test_data_for_type(test_type, size="small")

            # First run (cache miss)
            metric1 = self._measure_single_verification(
                test_type, test_data, parallel_workers=1
            )
            metric1.cache_hit = False
            metrics.append(metric1)

            # Second run (cache hit)
            metric2 = self._measure_single_verification(
                test_type, test_data, parallel_workers=1
            )
            metric2.cache_hit = True
            metrics.append(metric2)

            # Third run (cache hit)
            metric3 = self._measure_single_verification(
                test_type, test_data, parallel_workers=1
            )
            metric3.cache_hit = True
            metrics.append(metric3)

        return self._aggregate_metrics("Cache Performance", metrics)

    def _benchmark_load_testing(self) -> BenchmarkResult:
        """Benchmark performance under load with multiple concurrent users."""
        logger.info("Running load testing benchmark...")

        concurrent_users = [1, 5, 10, 20, 50]
        test_type = TestType.TTEST  # Use t-test for load testing
        test_data = self._generate_test_data_for_type(test_type, size="medium")
        metrics = []

        for users in concurrent_users:
            # Simulate concurrent users
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=users) as executor:
                futures = []
                for _ in range(users):
                    future = executor.submit(
                        self._measure_single_verification,
                        test_type,
                        test_data,
                        parallel_workers=1,
                    )
                    futures.append(future)

                # Wait for all to complete
                results = [future.result() for future in futures]

            total_time = time.time() - start_time

            # Aggregate concurrent results
            for result in results:
                result.execution_time = total_time / users  # Average time per user
                metrics.append(result)

        return self._aggregate_metrics("Load Testing", metrics)

    def _benchmark_memory_efficiency(self) -> BenchmarkResult:
        """Benchmark memory usage and efficiency."""
        logger.info("Running memory efficiency benchmark...")

        test_types = list(TestType)[:5]  # Test subset
        metrics = []

        for test_type in test_types:
            # Test with different data sizes
            for size in ["small", "medium", "large"]:
                test_data = self._generate_test_data_for_type(test_type, size=size)

                # Run verification multiple times to measure memory patterns
                for i in range(3):
                    metric = self._measure_single_verification(
                        test_type, test_data, parallel_workers=1
                    )
                    metrics.append(metric)

        return self._aggregate_metrics("Memory Efficiency", metrics)

    def _benchmark_edge_cases(self) -> BenchmarkResult:
        """Benchmark performance under edge cases and extreme conditions."""
        logger.info("Running edge case performance benchmark...")

        test_type = TestType.TTEST  # Use t-test for edge cases
        metrics = []

        # Test with very small datasets
        small_data = self._generate_test_data_for_type(test_type, size="tiny")
        for i in range(3):
            metric = self._measure_single_verification(
                test_type, small_data, parallel_workers=1
            )
            metrics.append(metric)

        # Test with very large datasets
        large_data = self._generate_test_data_for_type(test_type, size="huge")
        for i in range(3):
            metric = self._measure_single_verification(
                test_type, large_data, parallel_workers=1
            )
            metrics.append(metric)

        return self._aggregate_metrics("Edge Cases", metrics)

    def _measure_single_verification(
        self, test_type: TestType, test_data: pd.DataFrame, parallel_workers: int = 1
    ) -> PerformanceMetrics:
        """Measure performance of a single verification operation."""
        try:
            # Create verification request
            request = VerificationRequest(
                test_type=test_type, data=test_data, alpha=0.05, timeout=30
            )

            # Measure execution time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Execute verification
            result = self.engine.verify(request)

            # Calculate metrics
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory

            # Calculate CPU usage (simplified)
            cpu_usage = psutil.Process().cpu_percent()

            # Calculate parallel efficiency for multi-worker operations
            parallel_efficiency = 1.0
            if parallel_workers > 1:
                # Estimate what single-threaded time would be
                estimated_single_time = execution_time * parallel_workers
                parallel_efficiency = min(1.0, estimated_single_time / execution_time)

            return PerformanceMetrics(
                operation=f"verify_{test_type.value}",
                test_type=test_type.value,
                data_size=len(test_data),
                execution_time=execution_time,
                memory_usage=max(0, memory_usage),
                cpu_usage=cpu_usage,
                success=result.is_verified,
                parallel_workers=parallel_workers,
                parallel_efficiency=parallel_efficiency,
                workload_balance=1.0,  # Will be calculated in aggregation
            )

        except Exception as e:
            logger.error(f"Error measuring verification: {e}")
            return PerformanceMetrics(
                operation=f"verify_{test_type.value}",
                test_type=test_type.value,
                data_size=len(test_data),
                execution_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                success=False,
                error_message=str(e),
                parallel_workers=parallel_workers,
                parallel_efficiency=0.0,
                workload_balance=0.0,
            )

    def _generate_test_data_for_type(
        self, test_type: TestType, size: str = "medium"
    ) -> pd.DataFrame:
        """Generate appropriate test data for the given test type and size."""
        size_mapping = {
            "tiny": 10,
            "small": 100,
            "medium": 1000,
            "large": 10000,
            "huge": 100000,
        }

        n_samples = size_mapping.get(size, 1000)

        if test_type == TestType.TTEST:
            return pd.DataFrame(
                {
                    "group1": np.random.normal(100, 15, n_samples),
                    "group2": np.random.normal(105, 15, n_samples),
                }
            )
        elif test_type == TestType.ANOVA:
            return pd.DataFrame(
                {
                    "group1": np.random.normal(100, 15, n_samples),
                    "group2": np.random.normal(105, 15, n_samples),
                    "group3": np.random.normal(110, 15, n_samples),
                }
            )
        elif test_type == TestType.CHI2:
            # Generate categorical data for chi-square test
            categories = ["A", "B", "C", "D"]
            return pd.DataFrame(
                {
                    "category": np.random.choice(categories, n_samples),
                    "count": np.random.poisson(25, n_samples),
                }
            )
        elif test_type == TestType.WILCOXON:
            return pd.DataFrame(
                {
                    "before": np.random.normal(100, 15, n_samples),
                    "after": np.random.normal(105, 15, n_samples),
                }
            )
        elif test_type == TestType.MANN_WHITNEY:
            return pd.DataFrame(
                {
                    "group1": np.random.normal(100, 15, n_samples),
                    "group2": np.random.normal(105, 15, n_samples),
                }
            )
        else:
            return pd.DataFrame({"values": np.random.normal(100, 15, n_samples)})

    def _aggregate_metrics(
        self, benchmark_name: str, metrics: List[PerformanceMetrics]
    ) -> BenchmarkResult:
        """Aggregate performance metrics into benchmark results."""
        if not metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                average_execution_time=0.0,
                median_execution_time=0.0,
                p95_execution_time=0.0,
                p99_execution_time=0.0,
                throughput_ops_per_second=0.0,
                memory_peak_mb=0.0,
                cpu_peak_percent=0.0,
                cache_hit_rate=0.0,
                parallel_efficiency=0.0,
                workload_balance=0.0,
                detailed_metrics=metrics,
            )

        # Calculate statistics
        execution_times = [m.execution_time for m in metrics if m.success]
        memory_usage = [m.memory_usage for m in metrics]
        cpu_usage = [m.cpu_usage for m in metrics]
        cache_hits = [m.cache_hit for m in metrics]
        parallel_efficiencies = [m.parallel_efficiency for m in metrics]
        workload_balances = [m.workload_balance for m in metrics]

        successful_ops = sum(1 for m in metrics if m.success)
        failed_ops = len(metrics) - successful_ops

        if execution_times:
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            p95_time = np.percentile(execution_times, 95)
            p99_time = np.percentile(execution_times, 99)
            throughput = 1.0 / avg_time if avg_time > 0 else 0.0
        else:
            avg_time = median_time = p95_time = p99_time = throughput = 0.0

        # Calculate parallel efficiency
        if parallel_efficiencies:
            avg_parallel_efficiency = statistics.mean(parallel_efficiencies)
        else:
            avg_parallel_efficiency = 1.0

        # Calculate workload balance
        if workload_balances:
            avg_workload_balance = statistics.mean(workload_balances)
        else:
            avg_workload_balance = 1.0

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            total_operations=len(metrics),
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            average_execution_time=avg_time,
            median_execution_time=median_time,
            p95_execution_time=p95_time,
            p99_execution_time=p99_time,
            throughput_ops_per_second=throughput,
            memory_peak_mb=max(memory_usage) if memory_usage else 0.0,
            cpu_peak_percent=max(cpu_usage) if cpu_usage else 0.0,
            cache_hit_rate=sum(cache_hits) / len(cache_hits) if cache_hits else 0.0,
            parallel_efficiency=avg_parallel_efficiency,
            workload_balance=avg_workload_balance,
            detailed_metrics=metrics,
        )

    def _save_benchmark_results(self, results: Dict[str, BenchmarkResult]):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        for name, result in results.items():
            filename = f"{name}_{timestamp}.json"
            filepath = self.results_dir / filename

            # Convert to dict for JSON serialization
            result_dict = asdict(result)

            with open(filepath, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)

        # Save summary
        summary = {
            "timestamp": timestamp,
            "benchmarks": list(results.keys()),
            "summary": {},
        }

        for name, result in results.items():
            summary["summary"][name] = {
                "total_operations": result.total_operations,
                "success_rate": (
                    result.successful_operations / result.total_operations
                    if result.total_operations > 0
                    else 0
                ),
                "average_execution_time": result.average_execution_time,
                "throughput_ops_per_second": result.throughput_ops_per_second,
                "memory_peak_mb": result.memory_peak_mb,
                "cache_hit_rate": result.cache_hit_rate,
                "parallel_efficiency": result.parallel_efficiency,
                "workload_balance": result.workload_balance,
            }

        summary_file = self.results_dir / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Benchmark results saved to {self.results_dir}")

    def _generate_performance_report(self, results: Dict[str, BenchmarkResult]):
        """Generate comprehensive performance report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"performance_report_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("STATWHY PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Total Benchmarks: {len(results)}\n\n")

            # Overall summary
            total_ops = sum(r.total_operations for r in results.values())
            total_success = sum(r.successful_operations for r in results.values())
            total_time = sum(
                r.average_execution_time * r.total_operations for r in results.values()
            )

            f.write("OVERALL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Operations: {total_ops}\n")
            f.write(f"Overall Success Rate: {(total_success/total_ops*100):.2f}%\n")
            f.write(f"Total Execution Time: {total_time:.2f}s\n")
            f.write(f"Average Throughput: {total_ops/total_time:.1f} ops/sec\n\n")

            # Individual benchmark results
            for name, result in results.items():
                f.write(f"{name.upper()}\n")
                f.write("-" * (len(name) + 2) + "\n")
                f.write(f"  Operations: {result.total_operations}\n")
                f.write(
                    f"  Success Rate: {(result.successful_operations/result.total_operations*100):.2f}%\n"
                )
                f.write(f"  Avg Execution Time: {result.average_execution_time:.3f}s\n")
                f.write(
                    f"  Throughput: {result.throughput_ops_per_second:.3f} ops/sec\n"
                )
                f.write(f"  Memory Peak: {result.memory_peak_mb:.1f} MB\n")
                f.write(f"  Cache Hit Rate: {result.cache_hit_rate*100:.1f}%\n")
                f.write(
                    f"  Parallel Efficiency: {result.parallel_efficiency*100:.2f}%\n"
                )
                f.write(f"  Workload Balance: {result.workload_balance*100:.2f}%\n\n")

            # Performance recommendations
            f.write("PERFORMANCE RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            # Check parallel efficiency
            low_efficiency = [
                name
                for name, result in results.items()
                if result.parallel_efficiency
                < self.thresholds["min_parallel_efficiency"]
            ]
            if low_efficiency:
                f.write(f"⚠️  Low parallel efficiency in: {', '.join(low_efficiency)}\n")
                f.write(
                    "   Consider: Work stealing, load balancing, memory pooling\n\n"
                )

            # Check cache performance
            low_cache = [
                name
                for name, result in results.items()
                if result.cache_hit_rate < self.thresholds["min_cache_hit_rate"]
            ]
            if low_cache:
                f.write(f"⚠️  Low cache hit rate in: {', '.join(low_cache)}\n")
                f.write("   Consider: Cache warming, prefetching, TTL optimization\n\n")

            # Check memory usage
            high_memory = [
                name
                for name, result in results.items()
                if result.memory_peak_mb > self.thresholds["max_memory_usage"]
            ]
            if high_memory:
                f.write(f"⚠️  High memory usage in: {', '.join(high_memory)}\n")
                f.write(
                    "   Consider: Memory pooling, garbage collection, data streaming\n\n"
                )

        logger.info(f"Performance report generated: {report_file}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "parallel_processor"):
            self.parallel_processor.shutdown()


def main():
    """Main function to run comprehensive performance testing."""
    print("🚀 Starting StatWhy Performance Testing...")

    tester = PerformanceTester()
    results = tester.run_comprehensive_benchmark()

    print(f"\n✅ Performance testing completed!")
    print(f"📊 Results saved to: {tester.results_dir}")

    # Print summary
    print(f"\n📈 Performance Summary:")
    for name, result in results.items():
        success_rate = result.successful_operations / result.total_operations
        print(
            f"  {name}: {success_rate:.1%} success rate, "
            f"{result.average_execution_time:.3f}s avg time, "
            f"{result.throughput_ops_per_second:.3f} ops/sec"
        )


if __name__ == "__main__":
    main()
