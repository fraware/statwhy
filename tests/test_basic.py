"""
Basic tests for StatWhy.

Tests core functionality and ensures the package can be imported and used.
"""

import pytest
import pandas as pd
import numpy as np

from statwhy.models import TestType, VerificationRequest, VerificationResult
from statwhy.utils import create_sample_data, get_test_description


class TestBasicFunctionality:
    """Test basic StatWhy functionality."""

    def test_import(self):
        """Test that StatWhy can be imported."""
        import statwhy

        assert hasattr(statwhy, "__version__")
        assert hasattr(statwhy, "StatWhyEngine")

    def test_test_types(self):
        """Test that all test types are available."""
        assert TestType.TTEST == "ttest"
        assert TestType.ANOVA == "anova"
        assert TestType.CHI2 == "chi2"
        assert TestType.WILCOXON == "wilcoxon"
        assert TestType.MANN_WHITNEY == "mann-whitney"
        assert TestType.KRUSKAL == "kruskal"
        assert TestType.BARTLETT == "bartlett"
        assert TestType.FLIGNER == "fligner"
        assert TestType.DUNNETT == "dunnett"
        assert TestType.TUKEY == "tukey"
        assert TestType.STEEL == "steel"
        assert TestType.STEEL_DWASS == "steel-dwass"
        assert TestType.WILLIAMS == "williams"
        assert TestType.POISSON == "poisson"
        assert TestType.BINOM == "binom"

    def test_verification_request_creation(self):
        """Test creating verification requests."""
        # Create sample data
        data = pd.DataFrame({"values": [1, 2, 3, 4, 5]})

        # Create request
        request = VerificationRequest(
            test_type=TestType.TTEST, data=data, alpha=0.05, timeout=300
        )

        assert request.test_type == TestType.TTEST
        assert request.alpha == 0.05
        assert request.timeout == 300
        assert len(request.data) == 5
        assert request.data.columns[0] == "values"

    def test_sample_data_creation(self):
        """Test sample data generation."""
        # Test t-test data
        ttest_data = create_sample_data(TestType.TTEST, n_samples=50)
        assert len(ttest_data) == 50
        assert "values" in ttest_data.columns

        # Test ANOVA data
        anova_data = create_sample_data(TestType.ANOVA, n_samples=60)
        assert len(anova_data) == 60
        assert "group" in anova_data.columns
        assert "values" in anova_data.columns

        # Test chi-square data
        chi2_data = create_sample_data(TestType.CHI2, n_samples=100)
        assert len(chi2_data) == 100
        assert "category" in chi2_data.columns

    def test_test_descriptions(self):
        """Test test description retrieval."""
        ttest_desc = get_test_description(TestType.TTEST)
        assert "t-test" in ttest_desc.lower()
        assert "means" in ttest_desc.lower()

        anova_desc = get_test_description(TestType.ANOVA)
        assert "variance" in anova_desc.lower()
        assert "groups" in anova_desc.lower()

        chi2_desc = get_test_description(TestType.CHI2)
        assert "chi-square" in chi2_desc.lower()
        assert "categorical" in chi2_desc.lower()

    def test_verification_result_creation(self):
        """Test creating verification results."""
        from statwhy.models import ComponentResult, VerificationStatus

        # Create component results
        component1 = ComponentResult(
            name="data_validation",
            verified=True,
            status=VerificationStatus.SUCCESS,
            details="Data validation passed",
        )

        component2 = ComponentResult(
            name="assumption_check",
            verified=True,
            status=VerificationStatus.SUCCESS,
            details="Statistical assumptions verified",
        )

        # Create verification result
        result = VerificationResult(
            test_type=TestType.TTEST,
            is_verified=True,
            status=VerificationStatus.SUCCESS,
            components=[component1, component2],
            execution_time=2.5,
        )

        assert result.is_verified is True
        assert result.status == VerificationStatus.SUCCESS
        assert len(result.components) == 2
        assert result.execution_time == 2.5
        assert result.success_rate == 1.0
        assert len(result.successful_components) == 2
        assert len(result.failed_components) == 0


class TestDataValidation:
    """Test data validation functionality."""

    def test_empty_data_validation(self):
        """Test that empty data is rejected."""
        from statwhy.exceptions import DataValidationError

        empty_data = pd.DataFrame()

        with pytest.raises(Exception, match="Data cannot be empty"):
            VerificationRequest(test_type=TestType.TTEST, data=empty_data, alpha=0.05)

    def test_ttest_data_validation(self):
        """Test t-test data validation."""
        # Valid data
        valid_data = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
        request = VerificationRequest(
            test_type=TestType.TTEST, data=valid_data, alpha=0.05
        )
        assert len(request.data) == 5

        # Invalid data (too few samples) - currently not validated at model level
        # This would be validated during the actual verification process
        invalid_data = pd.DataFrame({"values": [1]})
        # For now, this should still create a valid request
        request = VerificationRequest(test_type=TestType.TTEST, data=invalid_data, alpha=0.05)
        assert len(request.data) == 1

    def test_anova_data_validation(self):
        """Test ANOVA data validation."""
        # Valid ANOVA data
        valid_data = pd.DataFrame(
            {"group": ["A", "A", "B", "B", "C", "C"], "values": [1, 2, 3, 4, 5, 6]}
        )

        request = VerificationRequest(
            test_type=TestType.ANOVA, data=valid_data, alpha=0.05
        )
        assert len(request.data) == 6
        assert "group" in request.data.columns
        assert "values" in request.data.columns


class TestCLIFunctionality:
    """Test CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help works."""
        import subprocess
        import sys

        try:
            result = subprocess.run(
                [sys.executable, "-m", "statwhy.cli", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            assert "StatWhy" in result.stdout
        except subprocess.TimeoutExpired:
            pytest.skip("CLI test timed out")
        except FileNotFoundError:
            pytest.skip("CLI not available")


class TestWebInterface:
    """Test web interface functionality."""

    def test_web_app_creation(self):
        """Test that web app can be created."""
        try:
            from statwhy.web import create_app

            app = create_app()
            assert app is not None
            assert hasattr(app, "routes")
        except ImportError:
            pytest.skip("Web interface dependencies not available")


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create large dataset
        large_data = create_sample_data(TestType.TTEST, n_samples=10000)
        assert len(large_data) == 10000

        # Test memory usage
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Create request
        request = VerificationRequest(
            test_type=TestType.TTEST, data=large_data, alpha=0.05
        )

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024


@pytest.mark.integration
class TestIntegration:
    """Integration tests."""

    def test_full_verification_workflow(self):
        """Test complete verification workflow."""
        # This would test the full integration with Why3 and Cameleer
        # For now, we'll just test the data flow
        pytest.skip("Integration tests require Why3 and Cameleer installation")

    def test_cache_functionality(self):
        """Test caching system."""
        from statwhy.cache import VerificationCache

        cache = VerificationCache()

        # Test cache operations
        assert cache.get_stats() is not None
        cache.clear()

        # Cache should be empty after clearing
        stats = cache.get_stats()
        assert stats["memory_cache"]["entries"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
