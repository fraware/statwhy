#!/usr/bin/env python3
"""
StatWhy - Statistical Verification and Validation Framework

A comprehensive framework for verifying statistical procedures with formal methods,
ensuring correctness, reproducibility, and regulatory compliance.
"""

__version__ = "2.0.0"
__author__ = "StatWhy Development Team"
__email__ = "dev@statwhy.org"

# Core modules
from . import core
from . import models
from . import verifiers
from . import exceptions
from . import utils
from . import cache
from . import cli
from . import web

# Clinical trial verification features
from . import clinical_trials

# Financial risk modeling features
from . import financial_risk

# Manufacturing quality control features
from . import manufacturing_quality

# Plugin system architecture
from . import plugin_system

# Performance optimization features
from . import performance_optimization

# Error message system
from . import error_messages

# Automated testing system
from . import automated_testing

# Data generation and integration
from . import data_generator
from . import python_integration
from . import performance_tester

# Main classes and functions
from .core import StatWhyEngine
from .models import (
    TestType,
    ApplicationCategory,
    VerificationRequest,
    VerificationResult,
    ComponentResult,
    VerificationStatus,
)
from .verifiers import BaseVerifier
from .exceptions import (
    StatWhyError,
    VerificationError,
    DataValidationError,
    PluginError,
    SystemResourceError,
    CacheError,
    TestNotSupportedError,
)

# Plugin system components
from .plugin_system import (
    PluginManager,
    PluginMarketplace,
    PluginIncentivization,
    PluginMetadata,
    PluginValidationResult,
)

# Performance optimization components
from .performance_optimization import (
    AdvancedCache,
    OptimizedParallelProcessor as ParallelProcessor,
    LoadTester,
    PerformanceMetrics,
)

# Error message components
from .error_messages import (
    ErrorMessageGenerator,
    ErrorMessageFormatter,
    create_user_friendly_error,
    log_error_with_context,
)

# Automated testing components
from .automated_testing import (
    ComprehensiveTestSuite,
    PerformanceBenchmarker,
    ContinuousIntegration,
    run_comprehensive_tests,
    benchmark_statistical_procedure,
    run_ci_pipeline,
)

# Clinical trial components
from .clinical_trials import (
    ClinicalTrialVerifier,
    FDAGuidelines,
)

# Financial risk components
from .financial_risk import (
    FinancialRiskVerifier,
    BaselRequirements,
)

# Manufacturing quality components
from .manufacturing_quality import (
    ManufacturingQualityVerifier,
    SixSigmaRequirements,
)


# Utility functions
def get_version():
    """Get the current StatWhy version."""
    return __version__


def get_supported_tests():
    """Get list of supported statistical tests."""
    return [test.value for test in TestType]


def get_supported_categories():
    """Get list of supported application categories."""
    return [category.value for category in ApplicationCategory]


def create_verification_request(test_type: str, data: any, metadata: dict = None):
    """Create a verification request."""
    from .models import VerificationRequest

    return VerificationRequest(
        test_type=TestType(test_type), data=data, metadata=metadata or {}
    )


def verify_statistical_procedure(request):
    """Verify a statistical procedure."""
    statwhy = StatWhy()
    return statwhy.verify(request)


def run_comprehensive_verification(test_type: str, data: any, options: dict = None):
    """Run comprehensive verification with all checks."""
    if options is None:
        options = {}

    # Create verification request
    request = create_verification_request(test_type, data, options.get("metadata"))

    # Run verification
    result = verify_statistical_procedure(request)

    # Add additional checks based on test type
    if test_type in ["t_test", "anova", "regression"]:
        # Clinical trial verification
        clinical_verifier = clinical_trials.ClinicalTrialVerifier()
        clinical_result = clinical_verifier.verify_clinical_trial(request, result)
        result.metadata["clinical_trial_verification"] = clinical_result

    if test_type in ["risk_model", "var_calculation", "stress_test"]:
        # Financial risk verification
        risk_modeler = financial_risk.FinancialRiskModeler()
        risk_result = risk_modeler.verify_risk_model(request, result)
        result.metadata["financial_risk_verification"] = risk_result

    if test_type in ["quality_control", "process_control", "defect_analysis"]:
        # Manufacturing quality verification
        quality_controller = manufacturing_quality.ManufacturingQualityController()
        quality_result = quality_controller.verify_quality_control(request, result)
        result.metadata["manufacturing_quality_verification"] = quality_result

    return result


def install_plugin(plugin_path: str):
    """Install a StatWhy plugin."""
    plugin_manager = plugin_system.PluginManager()
    return plugin_manager.install_plugin(plugin_path)


def list_plugins():
    """List installed plugins."""
    plugin_manager = plugin_system.PluginManager()
    return plugin_manager.list_installed_plugins()


def run_performance_benchmark(func, *args, **kwargs):
    """Run performance benchmark on a function."""
    benchmarker = performance_optimization.PerformanceBenchmarker()
    return benchmarker.benchmark_function(func, args, kwargs)


def get_user_friendly_error(error: Exception, context: dict = None):
    """Get user-friendly error message."""
    return create_user_friendly_error(error, context)


def run_automated_tests():
    """Run comprehensive automated test suite."""
    return run_comprehensive_tests()


def run_ci_pipeline():
    """Run continuous integration pipeline."""
    return run_ci_pipeline()


# Package information
__all__ = [
    # Core
    "StatWhyEngine",
    "TestType",
    "ApplicationCategory",
    "VerificationRequest",
    "VerificationResult",
    "ComponentResult",
    "VerificationStatus",
    "BaseVerifier",
    # Clinical trials
    "ClinicalTrialVerifier",
    "FDAGuidelines",
    # Financial risk
    "FinancialRiskVerifier",
    "BaselRequirements",
    # Manufacturing quality
    "ManufacturingQualityVerifier",
    "SixSigmaRequirements",
    # Plugin system
    "PluginManager",
    "PluginMarketplace",
    "PluginIncentivization",
    "PluginMetadata",
    "PluginValidationResult",
    # Performance optimization
    "AdvancedCache",
    "ParallelProcessor",
    "LoadTester",
    "PerformanceMetrics",
    "performance_monitor",
    "cache_result",
    # Error messages
    "ErrorMessageGenerator",
    "ErrorMessageFormatter",
    "create_user_friendly_error",
    "log_error_with_context",
    # Automated testing
    "ComprehensiveTestSuite",
    "PerformanceBenchmarker",
    "ContinuousIntegration",
    "run_comprehensive_tests",
    "benchmark_statistical_procedure",
    "run_ci_pipeline",
    # Utility functions
    "get_version",
    "get_supported_tests",
    "get_supported_categories",
    "create_verification_request",
    "verify_statistical_procedure",
    "run_comprehensive_verification",
    "install_plugin",
    "list_plugins",
    "run_performance_benchmark",
    "get_user_friendly_error",
    "run_automated_tests",
    # Exceptions
    "StatWhyError",
    "VerificationError",
    "DataValidationError",
    "PluginError",
    "SystemResourceError",
    "CacheError",
    "TestNotSupportedError",
]
