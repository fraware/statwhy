"""
Core StatWhy engine for formal verification of statistical procedures.

This module provides the main engine that coordinates verification using Why3
and Cameleer, manages the verification pipeline, and handles result caching.
"""

import asyncio
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures

import pandas as pd
from pydantic import ValidationError

from .models import (
    VerificationRequest,
    VerificationResult,
    ComponentResult,
    VerificationStatus,
    TestType,
)
from .verifiers import (
    TTestVerifier,
    ANOVAVerifier,
    ChiSquareVerifier,
    WilcoxonVerifier,
    MannWhitneyVerifier,
    KruskalVerifier,
    BartlettVerifier,
    FlignerVerifier,
    DunnettVerifier,
    TukeyVerifier,
    SteelVerifier,
    SteelDwassVerifier,
    WilliamsVerifier,
    PoissonVerifier,
    BinomVerifier,
)
from .cache import VerificationCache
from .exceptions import (
    VerificationError,
    TimeoutError,
    ConfigurationError,
    DataValidationError,
    TestNotSupportedError,
)


logger = logging.getLogger(__name__)


class StatWhyEngine:
    """
    Main engine for StatWhy verification.

    Coordinates the verification process, manages verifiers for different
    statistical tests, and provides a unified interface for all verification
    operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StatWhy engine.

        Args:
            config: Configuration dictionary for the engine
        """
        self.config = config or {}
        self.cache = VerificationCache()
        self.verifiers = self._initialize_verifiers()
        
        # Only check system requirements if explicitly requested
        if config and config.get("check_system_requirements", True):
            self._verify_system_requirements()

    def _initialize_verifiers(self) -> Dict[TestType, Any]:
        """Initialize verifiers for all supported test types."""
        return {
            TestType.TTEST: TTestVerifier(),
            TestType.ANOVA: ANOVAVerifier(),
            TestType.CHI2: ChiSquareVerifier(),
            TestType.WILCOXON: WilcoxonVerifier(),
            TestType.MANN_WHITNEY: MannWhitneyVerifier(),
            TestType.KRUSKAL: KruskalVerifier(),
            TestType.BARTLETT: BartlettVerifier(),
            TestType.FLIGNER: FlignerVerifier(),
            TestType.DUNNETT: DunnettVerifier(),
            TestType.TUKEY: TukeyVerifier(),
            TestType.STEEL: SteelVerifier(),
            TestType.STEEL_DWASS: SteelDwassVerifier(),
            TestType.WILLIAMS: WilliamsVerifier(),
            TestType.POISSON: PoissonVerifier(),
            TestType.BINOM: BinomVerifier(),
        }

    def _verify_system_requirements(self) -> None:
        """Verify that all system requirements are met."""
        try:
            # Check Why3 installation
            self._check_why3_installation()

            # Check Cameleer installation
            self._check_cameleer_installation()

            # Check Python dependencies
            self._check_python_dependencies()

        except Exception as e:
            logger.error(f"System requirements check failed: {e}")
            raise ConfigurationError(f"System requirements not met: {e}")

    def _check_why3_installation(self) -> None:
        """Check if Why3 is properly installed."""
        try:
            result = subprocess.run(
                ["why3", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise ConfigurationError("Why3 is not properly installed")
            logger.info(f"Why3 version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise ConfigurationError("Why3 not found in PATH")

    def _check_cameleer_installation(self) -> None:
        """Check if Cameleer is properly installed."""
        try:
            result = subprocess.run(
                ["cameleer", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise ConfigurationError("Cameleer is not properly installed")
            logger.info(f"Cameleer version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise ConfigurationError("Cameleer not found in PATH")

    def _check_python_dependencies(self) -> None:
        """Check if all required Python dependencies are available."""
        required_deps = ["numpy", "pandas", "scipy", "matplotlib", "plotly"]

        missing_deps = []
        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            raise ConfigurationError(
                f"Missing Python dependencies: {', '.join(missing_deps)}"
            )

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """
        Verify a statistical procedure.

        Args:
            request: Verification request containing test type, data, and parameters

        Returns:
            VerificationResult with detailed verification outcomes

        Raises:
            TestNotSupportedError: If the test type is not supported
            DataValidationError: If the data fails validation
            VerificationError: If verification fails
            TimeoutError: If verification times out
        """
        start_time = datetime.now()

        try:
            # Validate request
            self._validate_request(request)

            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached verification result")
                return cached_result

            # Get appropriate verifier
            verifier = self._get_verifier(request.test_type)

            # Perform verification
            logger.info(f"Starting verification of {request.test_type} test")
            components = verifier.verify(request)

            # Determine overall success
            is_verified = all(c.verified for c in components)
            status = (
                VerificationStatus.SUCCESS if is_verified else VerificationStatus.FAILED
            )

            # Create result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = VerificationResult(
                test_type=request.test_type,
                is_verified=is_verified,
                status=status,
                components=components,
                execution_time=execution_time,
                metadata={"cache_key": cache_key},
            )

            # Cache result
            self.cache.set(cache_key, result)

            logger.info(
                f"Verification completed in {execution_time:.2f}s. "
                f"Success: {is_verified}"
            )

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Verification failed: {e}")

            # Create error result
            error_component = ComponentResult(
                name="verification_engine",
                verified=False,
                status=VerificationStatus.ERROR,
                details=f"Verification failed: {str(e)}",
                error_message=str(e),
                execution_time=execution_time,
            )

            return VerificationResult(
                test_type=request.test_type,
                is_verified=False,
                status=VerificationStatus.ERROR,
                components=[error_component],
                execution_time=execution_time,
                error_message=str(e),
            )

    async def verify_async(self, request: VerificationRequest) -> VerificationResult:
        """
        Asynchronously verify a statistical procedure.

        Args:
            request: Verification request

        Returns:
            VerificationResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.verify, request)

    def verify_batch(
        self, requests: List[VerificationRequest], max_workers: Optional[int] = None
    ) -> List[VerificationResult]:
        """
        Verify multiple statistical procedures in parallel.

        Args:
            requests: List of verification requests
            max_workers: Maximum number of parallel workers

        Returns:
            List of verification results in the same order as requests
        """
        max_workers = max_workers or self.config.get("parallel_workers", 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_request = {
                executor.submit(self.verify, req): req for req in requests
            }

            results = []
            for future in concurrent.futures.as_completed(future_to_request):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch verification failed: {e}")
                    # Create error result for failed verification
                    request = future_to_request[future]
                    error_result = VerificationResult(
                        test_type=request.test_type,
                        is_verified=False,
                        status=VerificationStatus.ERROR,
                        components=[],
                        execution_time=0.0,
                        error_message=str(e),
                    )
                    results.append(error_result)

        return results

    def _validate_request(self, request: VerificationRequest) -> None:
        """Validate the verification request."""
        try:
            # Validate data
            if request.data.empty:
                raise DataValidationError("Data cannot be empty")

            # Validate test type support
            if request.test_type not in self.verifiers:
                raise TestNotSupportedError(
                    f"Test type {request.test_type} is not supported"
                )

            # Validate parameters
            verifier = self._get_verifier(request.test_type)
            verifier.validate_parameters(request.parameters)

        except ValidationError as e:
            raise DataValidationError(f"Request validation failed: {e}")

    def _get_verifier(self, test_type: TestType) -> Any:
        """Get the verifier for a specific test type."""
        if test_type not in self.verifiers:
            raise TestNotSupportedError(f"Test type {test_type} not supported")
        return self.verifiers[test_type]

    def _generate_cache_key(self, request: VerificationRequest) -> str:
        """Generate a cache key for the verification request."""
        import hashlib

        # Create a hash of the request data
        data_hash = hashlib.md5(request.data.to_string().encode()).hexdigest()

        # Combine with other request parameters
        key_parts = [
            request.test_type.value,
            str(request.alpha),
            data_hash,
            str(sorted(request.parameters.items())),
        ]

        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()

    def get_supported_tests(self) -> List[TestType]:
        """Get list of supported statistical tests."""
        return list(self.verifiers.keys())

    def get_test_info(self, test_type: TestType) -> Dict[str, Any]:
        """Get information about a specific test type."""
        verifier = self._get_verifier(test_type)
        return verifier.get_test_info()

    def clear_cache(self) -> None:
        """Clear the verification cache."""
        self.cache.clear()
        logger.info("Verification cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def shutdown(self) -> None:
        """Shutdown the engine and cleanup resources."""
        logger.info("Shutting down StatWhy engine")
        self.cache.clear()
        # Additional cleanup as needed
