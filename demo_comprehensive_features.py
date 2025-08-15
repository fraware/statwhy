#!/usr/bin/env python3
"""
StatWhy Comprehensive Features Demo

This script demonstrates all the new features implemented in StatWhy 2.0:
- Clinical Trial Verification (FDA compliance)
- Financial Risk Modeling (Basel compliance)
- Manufacturing Quality Control (Six Sigma)
- Plugin System Architecture
- Performance Optimization
- Better Error Messages
- Automated Testing
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import statwhy
    from statwhy import (
        # Core functionality
        StatWhy,
        TestType,
        ApplicationCategory,
        VerificationRequest,
        VerificationResult,
        # Clinical trials
        ClinicalTrialVerifier,
        FDAGuidelines,
        PatientSafetyChecker,
        StatisticalAssumptionValidator,
        # Financial risk
        FinancialRiskModeler,
        BaselComplianceChecker,
        RiskAssessmentValidator,
        RegulatoryRequirementChecker,
        # Manufacturing quality
        ManufacturingQualityController,
        SixSigmaValidator,
        ProcessOptimizer,
        DefectAnalyzer,
        # Plugin system
        PluginManager,
        PluginMarketplace,
        PluginIncentivization,
        # Performance optimization
        AdvancedCache,
        ParallelProcessor,
        LoadTester,
        performance_monitor,
        cache_result,
        # Error messages
        ErrorMessageGenerator,
        create_user_friendly_error,
        # Automated testing
        ComprehensiveTestSuite,
        PerformanceBenchmarker,
        ContinuousIntegration,
    )
except ImportError as e:
    print(f"Error importing StatWhy: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_clinical_trial_verification():
    """Demonstrate clinical trial verification features."""
    print("\n" + "=" * 60)
    print("🏥 CLINICAL TRIAL VERIFICATION FEATURES")
    print("=" * 60)

    try:
        # Create clinical trial verifier
        clinical_verifier = ClinicalTrialVerifier()

        # Sample clinical trial data
        clinical_data = {
            "treatment_group": [25.3, 26.1, 24.8, 25.9, 26.5, 25.2, 25.7, 25.1],
            "control_group": [23.1, 23.8, 22.9, 23.5, 23.2, 23.9, 23.4, 23.7],
            "patient_ids": [
                "P001",
                "P002",
                "P003",
                "P004",
                "P005",
                "P006",
                "P007",
                "P008",
            ],
            "trial_phase": "Phase II",
            "primary_endpoint": "Blood pressure reduction",
            "statistical_power": 0.8,
            "significance_level": 0.05,
        }

        # Verify clinical trial
        print("Verifying clinical trial data...")
        verification_result = clinical_verifier.verify_clinical_trial(
            clinical_data, {"test_type": "t_test", "data": clinical_data}
        )

        print(f"✅ Clinical trial verification completed")
        print(
            f"   - Patient safety checks: {verification_result.get('patient_safety', 'N/A')}"
        )
        print(
            f"   - FDA compliance: {verification_result.get('fda_compliance', 'N/A')}"
        )
        print(
            f"   - Statistical assumptions: {verification_result.get('statistical_assumptions', 'N/A')}"
        )

        # Check FDA guidelines
        fda_checker = FDAGuidelines()
        fda_compliance = fda_checker.check_compliance(clinical_data)
        print(f"   - FDA guideline compliance: {fda_compliance['overall_compliance']}")

        return True

    except Exception as e:
        print(f"❌ Clinical trial verification failed: {e}")
        return False


def demo_financial_risk_modeling():
    """Demonstrate financial risk modeling features."""
    print("\n" + "=" * 60)
    print("💰 FINANCIAL RISK MODELING FEATURES")
    print("=" * 60)

    try:
        # Create financial risk modeler
        risk_modeler = FinancialRiskModeler()

        # Sample financial data
        financial_data = {
            "portfolio_values": [
                1000000,
                1050000,
                1020000,
                1080000,
                1030000,
                1060000,
                1040000,
                1070000,
            ],
            "risk_factors": [
                "market_risk",
                "credit_risk",
                "liquidity_risk",
                "operational_risk",
            ],
            "confidence_level": 0.99,
            "time_horizon": 10,
            "regulatory_framework": "Basel III",
        }

        # Verify risk model
        print("Verifying financial risk model...")
        verification_result = risk_modeler.verify_risk_model(
            financial_data, {"test_type": "var_calculation", "data": financial_data}
        )

        print(f"✅ Financial risk model verification completed")
        print(
            f"   - Basel compliance: {verification_result.get('basel_compliance', 'N/A')}"
        )
        print(
            f"   - Risk assessment: {verification_result.get('risk_assessment', 'N/A')}"
        )
        print(
            f"   - Regulatory requirements: {verification_result.get('regulatory_requirements', 'N/A')}"
        )

        # Check Basel compliance
        basel_checker = BaselComplianceChecker()
        basel_compliance = basel_checker.check_compliance(financial_data)
        print(
            f"   - Basel III compliance score: {basel_compliance['compliance_score']:.2f}"
        )

        return True

    except Exception as e:
        print(f"❌ Financial risk modeling failed: {e}")
        return False


def demo_manufacturing_quality_control():
    """Demonstrate manufacturing quality control features."""
    print("\n" + "=" * 60)
    print("🏭 MANUFACTURING QUALITY CONTROL FEATURES")
    print("=" * 60)

    try:
        # Create manufacturing quality controller
        quality_controller = ManufacturingQualityController()

        # Sample manufacturing data
        manufacturing_data = {
            "product_measurements": [
                10.02,
                10.01,
                9.98,
                10.03,
                10.00,
                9.99,
                10.01,
                10.02,
            ],
            "specification_limits": {"lower": 9.95, "upper": 10.05},
            "target_value": 10.00,
            "process_capability_required": 1.33,
            "quality_standard": "Six Sigma",
        }

        # Verify quality control
        print("Verifying manufacturing quality control...")
        verification_result = quality_controller.verify_quality_control(
            manufacturing_data,
            {"test_type": "quality_control", "data": manufacturing_data},
        )

        print(f"✅ Manufacturing quality control verification completed")
        print(
            f"   - Six Sigma compliance: {verification_result.get('six_sigma_compliance', 'N/A')}"
        )
        print(
            f"   - Process capability: {verification_result.get('process_capability', 'N/A')}"
        )
        print(
            f"   - Quality metrics: {verification_result.get('quality_metrics', 'N/A')}"
        )

        # Check Six Sigma requirements
        six_sigma_validator = SixSigmaValidator()
        six_sigma_result = six_sigma_validator.validate_process(manufacturing_data)
        print(f"   - Six Sigma level: {six_sigma_result['sigma_level']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Manufacturing quality control failed: {e}")
        return False


def demo_plugin_system():
    """Demonstrate plugin system architecture."""
    print("\n" + "=" * 60)
    print("🔌 PLUGIN SYSTEM ARCHITECTURE")
    print("=" * 60)

    try:
        # Create plugin manager
        plugin_manager = PluginManager()

        # Generate plugin template
        print("Generating plugin template...")
        template = plugin_manager.generate_plugin_template(
            "Advanced Statistical Tests",
            "Dr. Jane Smith",
            "Advanced statistical procedures for research applications",
        )

        print(f"✅ Plugin template generated")
        print(f"   - Plugin directory: {template['plugin_dir']}")
        print(f"   - Files created: {len(template)}")

        # List installed plugins
        installed_plugins = plugin_manager.list_installed_plugins()
        print(f"   - Installed plugins: {len(installed_plugins)}")

        # Create marketplace
        marketplace = PluginMarketplace(plugin_manager)
        featured_plugins = marketplace.get_featured_plugins()
        print(f"   - Featured plugins: {len(featured_plugins)}")

        # Create incentivization system
        incentivization = PluginIncentivization(plugin_manager)
        contributor_score = incentivization.calculate_contributor_score(
            "Dr. Jane Smith"
        )
        print(f"   - Contributor score: {contributor_score:.3f}")

        return True

    except Exception as e:
        print(f"❌ Plugin system demo failed: {e}")
        return False


def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE OPTIMIZATION FEATURES")
    print("=" * 60)

    try:
        # Create advanced cache
        print("Testing advanced caching system...")
        cache = AdvancedCache(max_memory_size=10 * 1024 * 1024)  # 10MB

        # Test cache performance
        test_data = {"key": "value", "numbers": list(range(1000))}
        cache.set("test_key", test_data, ttl=3600)
        cached_data = cache.get("test_key")

        print(f"✅ Advanced caching system working")
        print(f"   - Cache hit: {cached_data is not None}")

        # Get cache stats
        cache_stats = cache.get_cache_stats()
        print(f"   - Cache hit rate: {cache_stats['hit_rate_percent']}%")

        # Test parallel processing
        print("Testing parallel processing...")
        processor = ParallelProcessor(max_workers=4)

        def sample_function(x):
            time.sleep(0.1)  # Simulate work
            return x * 2

        # Submit tasks
        for i in range(10):
            processor.submit_task(f"task_{i}", sample_function, args=(i,))

        # Wait for completion
        processor.wait_for_all_tasks(timeout=5.0)

        # Get performance stats
        perf_stats = processor.get_performance_stats()
        print(f"✅ Parallel processing completed")
        print(f"   - Completed tasks: {perf_stats['completed_tasks']}")
        print(f"   - Success rate: {perf_stats['success_rate']:.1f}%")

        # Test load testing
        print("Testing load testing system...")
        load_tester = LoadTester(None)

        def test_function(user_id):
            time.sleep(0.05)  # Simulate API call
            return {"user_id": user_id, "status": "success"}

        # Run concurrent test
        test_result = load_tester.run_concurrent_test(
            test_function, concurrent_users=5, test_duration=2
        )

        if "analysis" in test_result:
            analysis = test_result["analysis"]
            print(f"✅ Load testing completed")
            print(f"   - Total requests: {analysis['total_requests']}")
            print(f"   - Throughput: {analysis['throughput_rps']:.2f} req/s")
            print(f"   - Error rate: {analysis['error_rate_percent']:.2f}%")

        return True

    except Exception as e:
        print(f"❌ Performance optimization demo failed: {e}")
        return False


def demo_error_messages():
    """Demonstrate better error message system."""
    print("\n" + "=" * 60)
    print("💬 BETTER ERROR MESSAGES")
    print("=" * 60)

    try:
        # Create error message generator
        error_generator = ErrorMessageGenerator()

        # Test with a sample error
        print("Testing error message generation...")

        try:
            # Intentionally cause an error
            x = None
            x.append(1)  # This will cause an AttributeError
        except Exception as e:
            # Generate user-friendly error message
            error_context = error_generator.analyze_error(e)
            error_class = error_generator.classify_error(error_context)
            explanation = error_generator.generate_explanation(
                error_context, error_class
            )
            suggestions = error_generator.generate_fix_suggestions(error_class)

            print(f"✅ Error message generation working")
            print(f"   - Error type: {error_class}")
            print(f"   - Error title: {explanation.title}")
            print(f"   - Suggestions: {len(suggestions)}")

            # Show quick fix
            quick_fix = error_generator.generate_quick_fix(error_context)
            print(f"   - Quick fix: {quick_fix}")

        # Test error message formatting
        print("Testing error message formatting...")
        formatted_message = error_generator.format_error_message(
            error_context, explanation, suggestions
        )

        print(f"✅ Error message formatting completed")
        print(f"   - Message length: {len(formatted_message)} characters")

        return True

    except Exception as e:
        print(f"❌ Error message demo failed: {e}")
        return False


def demo_automated_testing():
    """Demonstrate automated testing system."""
    print("\n" + "=" * 60)
    print("🧪 AUTOMATED TESTING SYSTEM")
    print("=" * 60)

    try:
        # Create comprehensive test suite
        print("Setting up comprehensive test suite...")
        test_suite = ComprehensiveTestSuite()

        # Add a simple test suite
        import unittest

        class SampleTestSuite(unittest.TestCase):
            def test_addition(self):
                self.assertEqual(2 + 2, 4)

            def test_multiplication(self):
                self.assertEqual(3 * 4, 12)

            def test_string(self):
                self.assertEqual("hello" + " world", "hello world")

        # Create test suite
        sample_suite = unittest.TestLoader().loadTestsFromTestCase(SampleTestSuite)
        test_suite.add_test_suite("sample_tests", sample_suite)

        print(f"✅ Test suite setup completed")
        print(f"   - Test suites: {len(test_suite.test_suites)}")

        # Run tests
        print("Running comprehensive tests...")
        test_result = test_suite.run_all_tests(
            parallel=False,  # Disable parallel for demo
            coverage=True,
            performance=True,
            output_format="text",
        )

        print(f"✅ Comprehensive testing completed")
        print(f"   - Total tests: {test_result.total_tests}")
        print(f"   - Passed: {test_result.passed_tests}")
        print(f"   - Failed: {test_result.failed_tests}")
        print(f"   - Coverage: {test_result.coverage_percentage:.1f}%")
        print(f"   - Execution time: {test_result.execution_time:.2f}s")

        # Test performance benchmarking
        print("Testing performance benchmarking...")
        benchmarker = PerformanceBenchmarker()

        def benchmark_function(x):
            time.sleep(0.01)  # Simulate work
            return x * 2

        benchmark_result = benchmarker.benchmark_function(
            benchmark_function, args=(5,), iterations=10
        )

        print(f"✅ Performance benchmarking completed")
        print(f"   - Function: {benchmark_result.test_name}")
        print(f"   - Avg time: {benchmark_result.avg_execution_time:.6f}s")
        print(f"   - Throughput: {benchmark_result.throughput:.2f} ops/s")

        # Test continuous integration
        print("Testing continuous integration...")
        ci = ContinuousIntegration(test_suite, benchmarker)

        # Add custom quality gate
        ci.add_quality_gate("custom_gate", lambda result: result.total_tests > 0)

        ci_result = ci.run_ci_pipeline(
            run_tests=True, run_benchmarks=False, enforce_gates=True
        )

        print(f"✅ Continuous integration completed")
        print(f"   - Status: {ci_result['status']}")
        print(f"   - Quality gates: {len(ci_result['quality_gates'])}")

        return True

    except Exception as e:
        print(f"❌ Automated testing demo failed: {e}")
        return False


def demo_integration():
    """Demonstrate integration of all features."""
    print("\n" + "=" * 60)
    print("🔗 INTEGRATION DEMONSTRATION")
    print("=" * 60)

    try:
        # Create main StatWhy instance
        statwhy = StatWhy()

        # Sample data for comprehensive verification
        sample_data = {
            "group1": [25.3, 26.1, 24.8, 25.9, 26.5],
            "group2": [23.1, 23.8, 22.9, 23.5, 23.2],
            "metadata": {
                "application": "clinical_trial",
                "regulatory_framework": "FDA",
                "quality_standard": "Six Sigma",
            },
        }

        # Run comprehensive verification
        print("Running comprehensive verification...")
        result = statwhy.run_comprehensive_verification(
            "t_test", sample_data, {"metadata": sample_data["metadata"]}
        )

        print(f"✅ Comprehensive verification completed")
        print(f"   - Test type: {result.test_type}")
        print(f"   - Status: {result.status}")
        print(f"   - Components verified: {len(result.components)}")

        # Show integration features
        if "clinical_trial_verification" in result.metadata:
            print(f"   - Clinical trial verification: ✅")

        if "financial_risk_verification" in result.metadata:
            print(f"   - Financial risk verification: ✅")

        if "manufacturing_quality_verification" in result.metadata:
            print(f"   - Manufacturing quality verification: ✅")

        return True

    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False


def main():
    """Run the comprehensive features demo."""
    print("🚀 STATWHY COMPREHENSIVE FEATURES DEMO")
    print("=" * 60)
    print("This demo showcases all the new features in StatWhy 2.0")
    print("=" * 60)

    # Track demo results
    demo_results = {}

    # Run all demos
    demos = [
        ("Clinical Trial Verification", demo_clinical_trial_verification),
        ("Financial Risk Modeling", demo_financial_risk_modeling),
        ("Manufacturing Quality Control", demo_manufacturing_quality_control),
        ("Plugin System Architecture", demo_plugin_system),
        ("Performance Optimization", demo_performance_optimization),
        ("Better Error Messages", demo_error_messages),
        ("Automated Testing", demo_automated_testing),
        ("Integration", demo_integration),
    ]

    for demo_name, demo_func in demos:
        try:
            print(f"\n🔄 Running {demo_name} demo...")
            success = demo_func()
            demo_results[demo_name] = success
            time.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"❌ {demo_name} demo crashed: {e}")
            demo_results[demo_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)

    successful_demos = sum(demo_results.values())
    total_demos = len(demo_results)

    for demo_name, success in demo_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {demo_name}")

    print(f"\nOverall: {successful_demos}/{total_demos} demos successful")

    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("StatWhy 2.0 is ready for production use.")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) failed.")
        print("Please check the error messages above.")

    print("\n" + "=" * 60)
    print("Thank you for trying StatWhy 2.0!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Demo crashed with error: {e}")
        sys.exit(1)
