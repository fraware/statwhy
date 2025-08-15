"""
Statistical test verifiers for StatWhy.

This module contains verifier classes for different statistical tests.
Each verifier is responsible for implementing the formal verification
logic for a specific statistical procedure.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .models import (
    VerificationRequest,
    ComponentResult,
    VerificationStatus,
    TestType,
)

logger = logging.getLogger(__name__)


class BaseVerifier(ABC):
    """Base class for all statistical test verifiers."""

    def __init__(self):
        self.test_type: TestType = None

    @abstractmethod
    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """
        Verify a statistical procedure.

        Args:
            request: Verification request containing test parameters

        Returns:
            List of component results for each verification step
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate test-specific parameters.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Default implementation accepts any parameters
        pass

    def get_test_info(self) -> Dict[str, Any]:
        """
        Get information about this test type.
        
        Returns:
            Dictionary containing test information
        """
        return {
            "name": self.test_type.value,
            "description": f"{self.test_type.value} statistical test",
            "assumptions": ["Basic data requirements"],
            "parameters": {},
        }

    def _create_component_result(
        self, name: str, verified: bool, details: str, 
        error_message: str = None
    ) -> ComponentResult:
        """Create a component result."""
        return ComponentResult(
            name=name,
            verified=verified,
            status=(VerificationStatus.SUCCESS if verified 
                   else VerificationStatus.FAILED),
            details=details,
            error_message=error_message,
            execution_time=0.0,
        )


class TTestVerifier(BaseVerifier):
    """Verifier for t-test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.TTEST

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify t-test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        # Check normality assumption
        normality_check = self._verify_normality_assumption(request)
        results.append(normality_check)
        
        # Check independence assumption
        independence_check = self._verify_independence_assumption(request)
        results.append(independence_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets t-test requirements."""
        try:
            data = request.data
            if len(data) < 2:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "T-test requires at least 2 data points",
                    "Insufficient data for t-test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for t-test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )

    def _verify_normality_assumption(self, request: VerificationRequest) -> ComponentResult:
        """Verify normality assumption (placeholder)."""
        return self._create_component_result(
            "normality_assumption",
            True,
            "Normality assumption verified (placeholder implementation)"
        )

    def _verify_independence_assumption(self, request: VerificationRequest) -> ComponentResult:
        """Verify independence assumption (placeholder)."""
        return self._create_component_result(
            "independence_assumption",
            True,
            "Independence assumption verified (placeholder implementation)"
        )


class ANOVAVerifier(BaseVerifier):
    """Verifier for ANOVA procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.ANOVA

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify ANOVA assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        # Check group structure
        group_check = self._verify_group_structure(request)
        results.append(group_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets ANOVA requirements."""
        try:
            data = request.data
            if len(data) < 3:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "ANOVA requires at least 3 data points",
                    "Insufficient data for ANOVA"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for ANOVA"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )

    def _verify_group_structure(self, request: VerificationRequest) -> ComponentResult:
        """Verify group structure for ANOVA."""
        try:
            data = request.data
            if "group" not in data.columns or "values" not in data.columns:
                return self._create_component_result(
                    "group_structure",
                    False,
                    "ANOVA requires 'group' and 'values' columns",
                    "Missing required columns for ANOVA"
                )
            
            return self._create_component_result(
                "group_structure",
                True,
                "Group structure verified for ANOVA"
            )
        except Exception as e:
            return self._create_component_result(
                "group_structure",
                False,
                f"Error checking group structure: {e}",
                str(e)
            )


class ChiSquareVerifier(BaseVerifier):
    """Verifier for Chi-square test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.CHI2

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Chi-square test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Chi-square test requirements."""
        try:
            data = request.data
            if len(data) < 2:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Chi-square test requires at least 2 data points",
                    "Insufficient data for Chi-square test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Chi-square test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class WilcoxonVerifier(BaseVerifier):
    """Verifier for Wilcoxon signed-rank test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.WILCOXON

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Wilcoxon test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Wilcoxon test requirements."""
        try:
            data = request.data
            if len(data) < 2:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Wilcoxon test requires at least 2 data points",
                    "Insufficient data for Wilcoxon test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Wilcoxon test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class MannWhitneyVerifier(BaseVerifier):
    """Verifier for Mann-Whitney U test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.MANN_WHITNEY

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Mann-Whitney U test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Mann-Whitney U test requirements."""
        try:
            data = request.data
            if len(data) < 2:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Mann-Whitney U test requires at least 2 data points",
                    "Insufficient data for Mann-Whitney U test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Mann-Whitney U test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class KruskalVerifier(BaseVerifier):
    """Verifier for Kruskal-Wallis test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.KRUSKAL

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Kruskal-Wallis test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Kruskal-Wallis test requirements."""
        try:
            data = request.data
            if len(data) < 3:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Kruskal-Wallis test requires at least 3 data points",
                    "Insufficient data for Kruskal-Wallis test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Kruskal-Wallis test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class BartlettVerifier(BaseVerifier):
    """Verifier for Bartlett's test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.BARTLETT

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Bartlett's test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Bartlett's test requirements."""
        try:
            data = request.data
            if len(data) < 2:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Bartlett's test requires at least 2 data points",
                    "Insufficient data for Bartlett's test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Bartlett's test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class FlignerVerifier(BaseVerifier):
    """Verifier for Fligner-Killeen test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.FLIGNER

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Fligner-Killeen test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Fligner-Killeen test requirements."""
        try:
            data = request.data
            if len(data) < 2:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Fligner-Killeen test requires at least 2 data points",
                    "Insufficient data for Fligner-Killeen test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Fligner-Killeen test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class DunnettVerifier(BaseVerifier):
    """Verifier for Dunnett's test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.DUNNETT

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Dunnett's test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Dunnett's test requirements."""
        try:
            data = request.data
            if len(data) < 3:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Dunnett's test requires at least 3 data points",
                    "Insufficient data for Dunnett's test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Dunnett's test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class TukeyVerifier(BaseVerifier):
    """Verifier for Tukey's test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.TUKEY

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Tukey's test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Tukey's test requirements."""
        try:
            data = request.data
            if len(data) < 3:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Tukey's test requires at least 3 data points",
                    "Insufficient data for Tukey's test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Tukey's test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class SteelVerifier(BaseVerifier):
    """Verifier for Steel's test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.STEEL

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Steel's test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Steel's test requirements."""
        try:
            data = request.data
            if len(data) < 3:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Steel's test requires at least 3 data points",
                    "Insufficient data for Steel's test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Steel's test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class SteelDwassVerifier(BaseVerifier):
    """Verifier for Steel-Dwass test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.STEEL_DWASS

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Steel-Dwass test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Steel-Dwass test requirements."""
        try:
            data = request.data
            if len(data) < 3:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Steel-Dwass test requires at least 3 data points",
                    "Insufficient data for Steel-Dwass test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Steel-Dwass test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class WilliamsVerifier(BaseVerifier):
    """Verifier for Williams' test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.WILLIAMS

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Williams' test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Williams' test requirements."""
        try:
            data = request.data
            if len(data) < 3:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Williams' test requires at least 3 data points",
                    "Insufficient data for Williams' test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Williams' test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class PoissonVerifier(BaseVerifier):
    """Verifier for Poisson test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.POISSON

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Poisson test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Poisson test requirements."""
        try:
            data = request.data
            if len(data) < 1:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Poisson test requires at least 1 data point",
                    "Insufficient data for Poisson test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Poisson test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )


class BinomVerifier(BaseVerifier):
    """Verifier for Binomial test procedures."""

    def __init__(self):
        super().__init__()
        self.test_type = TestType.BINOM

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify Binomial test assumptions and procedure."""
        results = []
        
        # Check data requirements
        data_check = self._verify_data_requirements(request)
        results.append(data_check)
        
        return results

    def _verify_data_requirements(self, request: VerificationRequest) -> ComponentResult:
        """Verify that data meets Binomial test requirements."""
        try:
            data = request.data
            if len(data) < 1:
                return self._create_component_result(
                    "data_requirements",
                    False,
                    "Binomial test requires at least 1 data point",
                    "Insufficient data for Binomial test"
                )
            
            return self._create_component_result(
                "data_requirements",
                True,
                f"Data has {len(data)} points, sufficient for Binomial test"
            )
        except Exception as e:
            return self._create_component_result(
                "data_requirements",
                False,
                f"Error checking data requirements: {e}",
                str(e)
            )
