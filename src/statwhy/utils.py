"""
Utility functions for StatWhy.

Provides helper functions for data loading, validation, result formatting,
and other common operations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from pydantic import ValidationError

from .models import TestType, VerificationResult
from .exceptions import DataValidationError, UnsupportedFormatError

logger = logging.getLogger(__name__)


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from various file formats.

    Args:
        file_path: Path to the data file

    Returns:
        Loaded data as pandas DataFrame

    Raises:
        UnsupportedFormatError: If file format is not supported
        DataValidationError: If data cannot be loaded
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise DataValidationError(f"File not found: {file_path}")

    try:
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == ".json":
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix.lower() == ".feather":
            return pd.read_feather(file_path)
        else:
            raise UnsupportedFormatError(f"Unsupported file format: {file_path.suffix}")
    except Exception as e:
        raise DataValidationError(f"Failed to load data: {e}")


def validate_data(data: pd.DataFrame, test_type: TestType) -> None:
    """
    Validate data for a specific statistical test.

    Args:
        data: Data to validate
        test_type: Type of statistical test

    Raises:
        DataValidationError: If data validation fails
    """
    if data.empty:
        raise DataValidationError("Data cannot be empty")

    # Basic data quality checks
    if data.isnull().all().all():
        raise DataValidationError("Data contains only null values")

    # Test-specific validation
    if test_type == TestType.TTEST:
        _validate_ttest_data(data)
    elif test_type == TestType.ANOVA:
        _validate_anova_data(data)
    elif test_type == TestType.CHI2:
        _validate_chisquare_data(data)
    elif test_type == TestType.WILCOXON:
        _validate_wilcoxon_data(data)
    elif test_type == TestType.MANN_WHITNEY:
        _validate_mannwhitney_data(data)
    elif test_type == TestType.KRUSKAL:
        _validate_kruskal_data(data)
    elif test_type == TestType.BARTLETT:
        _validate_bartlett_data(data)
    elif test_type == TestType.FLIGNER:
        _validate_fligner_data(data)
    elif test_type == TestType.DUNNETT:
        _validate_dunnett_data(data)
    elif test_type == TestType.TUKEY:
        _validate_tukey_data(data)
    elif test_type == TestType.STEEL:
        _validate_steel_data(data)
    elif test_type == TestType.STEEL_DWASS:
        _validate_steeldwass_data(data)
    elif test_type == TestType.WILLIAMS:
        _validate_williams_data(data)
    elif test_type == TestType.POISSON:
        _validate_poisson_data(data)
    elif test_type == TestType.BINOM:
        _validate_binom_data(data)
    else:
        raise DataValidationError(f"Unknown test type: {test_type}")


def _validate_ttest_data(data: pd.DataFrame) -> None:
    """Validate data for t-test."""
    if len(data.columns) < 1:
        raise DataValidationError("T-test requires at least one numeric column")

    # Check for numeric data
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise DataValidationError("T-test requires numeric data")

    # Check sample size
    if len(data) < 2:
        raise DataValidationError("T-test requires at least 2 observations")


def _validate_anova_data(data: pd.DataFrame) -> None:
    """Validate data for ANOVA."""
    if len(data.columns) < 2:
        raise DataValidationError("ANOVA requires at least 2 columns (group + value)")

    # Check for group column and value column
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise DataValidationError("ANOVA requires numeric data")

    # Check sample size per group
    if len(data) < 6:  # At least 2 groups with 3 observations each
        raise DataValidationError("ANOVA requires sufficient observations per group")


def _validate_chisquare_data(data: pd.DataFrame) -> None:
    """Validate data for chi-square test."""
    if len(data.columns) < 2:
        raise DataValidationError("Chi-square test requires at least 2 columns")

    # Check for categorical data
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) == 0:
        raise DataValidationError("Chi-square test requires categorical data")

    # Check expected frequencies
    if len(data) < 5:
        raise DataValidationError("Chi-square test requires at least 5 observations")


def _validate_wilcoxon_data(data: pd.DataFrame) -> None:
    """Validate data for Wilcoxon signed-rank test."""
    if len(data.columns) < 2:
        raise DataValidationError("Wilcoxon test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Wilcoxon test requires numeric data")

    if len(data) < 3:
        raise DataValidationError("Wilcoxon test requires at least 3 observations")


def _validate_mannwhitney_data(data: pd.DataFrame) -> None:
    """Validate data for Mann-Whitney U test."""
    if len(data.columns) < 2:
        raise DataValidationError("Mann-Whitney test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Mann-Whitney test requires numeric data")

    if len(data) < 4:
        raise DataValidationError("Mann-Whitney test requires at least 4 observations")


def _validate_kruskal_data(data: pd.DataFrame) -> None:
    """Validate data for Kruskal-Wallis test."""
    if len(data.columns) < 2:
        raise DataValidationError("Kruskal-Wallis test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Kruskal-Wallis test requires numeric data")

    if len(data) < 6:
        raise DataValidationError(
            "Kruskal-Wallis test requires sufficient observations"
        )


def _validate_bartlett_data(data: pd.DataFrame) -> None:
    """Validate data for Bartlett's test."""
    if len(data.columns) < 2:
        raise DataValidationError("Bartlett's test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Bartlett's test requires numeric data")

    if len(data) < 6:
        raise DataValidationError("Bartlett's test requires sufficient observations")


def _validate_fligner_data(data: pd.DataFrame) -> None:
    """Validate data for Fligner-Killeen test."""
    if len(data.columns) < 2:
        raise DataValidationError("Fligner-Killeen test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Fligner-Killeen test requires numeric data")

    if len(data) < 6:
        raise DataValidationError(
            "Fligner-Killeen test requires sufficient observations"
        )


def _validate_dunnett_data(data: pd.DataFrame) -> None:
    """Validate data for Dunnett's test."""
    if len(data.columns) < 2:
        raise DataValidationError("Dunnett's test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Dunnett's test requires numeric data")

    if len(data) < 6:
        raise DataValidationError("Dunnett's test requires sufficient observations")


def _validate_tukey_data(data: pd.DataFrame) -> None:
    """Validate data for Tukey's HSD test."""
    if len(data.columns) < 2:
        raise DataValidationError("Tukey's HSD test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Tukey's HSD test requires numeric data")

    if len(data) < 6:
        raise DataValidationError("Tukey's HSD test requires sufficient observations")


def _validate_steel_data(data: pd.DataFrame) -> None:
    """Validate data for Steel's test."""
    if len(data.columns) < 2:
        raise DataValidationError("Steel's test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Steel's test requires numeric data")

    if len(data) < 6:
        raise DataValidationError("Steel's test requires sufficient observations")


def _validate_steeldwass_data(data: pd.DataFrame) -> None:
    """Validate data for Steel-Dwass test."""
    if len(data.columns) < 2:
        raise DataValidationError("Steel-Dwass test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Steel-Dwass test requires numeric data")

    if len(data) < 6:
        raise DataValidationError("Steel-Dwass test requires sufficient observations")


def _validate_williams_data(data: pd.DataFrame) -> None:
    """Validate data for Williams' test."""
    if len(data.columns) < 2:
        raise DataValidationError("Williams' test requires at least 2 columns")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError("Williams' test requires numeric data")

    if len(data) < 6:
        raise DataValidationError("Williams' test requires sufficient observations")


def _validate_poisson_data(data: pd.DataFrame) -> None:
    """Validate data for Poisson test."""
    if len(data.columns) < 1:
        raise DataValidationError("Poisson test requires at least one column")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise DataValidationError("Poisson test requires numeric data")

    # Check for non-negative values
    if (data[numeric_cols] < 0).any().any():
        raise DataValidationError("Poisson test requires non-negative values")

    if len(data) < 1:
        raise DataValidationError("Poisson test requires at least 1 observation")


def _validate_binom_data(data: pd.DataFrame) -> None:
    """Validate data for binomial test."""
    if len(data.columns) < 1:
        raise DataValidationError("Binomial test requires at least one column")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise DataValidationError("Binomial test requires numeric data")

    # Check for binary values (0 or 1)
    if not ((data[numeric_cols] == 0) | (data[numeric_cols] == 1)).all().all():
        raise DataValidationError("Binomial test requires binary values (0 or 1)")

    if len(data) < 1:
        raise DataValidationError("Binomial test requires at least 1 observation")


def format_results(result: VerificationResult, format_type: str = "table") -> str:
    """
    Format verification results in various formats.

    Args:
        result: Verification result to format
        format_type: Output format type

    Returns:
        Formatted result string
    """
    if format_type == "json":
        return json.dumps(result.dict(), indent=2, default=str)
    elif format_type == "table":
        return _format_table(result)
    elif format_type == "summary":
        return _format_summary(result)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def _format_table(result: VerificationResult) -> str:
    """Format results as a table."""
    lines = []
    lines.append(f"StatWhy Verification Results")
    lines.append(f"=" * 50)
    lines.append(f"Test Type: {result.test_type.value.upper()}")
    lines.append(f"Status: {'✓ PASSED' if result.is_verified else '✗ FAILED'}")
    lines.append(f"Execution Time: {result.execution_time:.2f}s")
    lines.append(f"Timestamp: {result.timestamp}")
    lines.append("")

    lines.append("Component Results:")
    lines.append("-" * 30)

    for component in result.components:
        status = "✓" if component.verified else "✗"
        lines.append(f"{status} {component.name}: {component.details}")
        if component.error_message:
            lines.append(f"  Error: {component.error_message}")

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.append("-" * 10)
        for warning in result.warnings:
            lines.append(f"⚠ {warning}")

    return "\n".join(lines)


def _format_summary(result: VerificationResult) -> str:
    """Format results as a summary."""
    total_components = len(result.components)
    successful_components = len(result.successful_components)
    failed_components = len(result.failed_components)

    summary = (
        f"Verification {result.test_type.value.upper()}: "
        f"{'PASSED' if result.is_verified else 'FAILED'} "
        f"({successful_components}/{total_components} components) "
        f"in {result.execution_time:.2f}s"
    )

    return summary


def save_results(
    result: VerificationResult, file_path: Union[str, Path], format_type: str = "json"
) -> None:
    """
    Save verification results to a file.

    Args:
        result: Verification result to save
        file_path: Path to save the results
        format_type: Output format type
    """
    file_path = Path(file_path)

    if format_type == "json":
        with open(file_path, "w") as f:
            json.dump(result.dict(), f, indent=2, default=str)
    elif format_type == "txt":
        with open(file_path, "w") as f:
            f.write(format_results(result, "table"))
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def create_sample_data(test_type: TestType, n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample data for testing and examples.

    Args:
        test_type: Type of statistical test
        n_samples: Number of samples to generate

    Returns:
        Sample data as pandas DataFrame
    """
    np.random.seed(42)  # For reproducibility

    if test_type == TestType.TTEST:
        # One-sample t-test: normal distribution
        data = np.random.normal(0, 1, n_samples)
        return pd.DataFrame({"values": data})

    elif test_type == TestType.ANOVA:
        # ANOVA: three groups with different means
        group1 = np.random.normal(0, 1, n_samples // 3)
        group2 = np.random.normal(2, 1, n_samples // 3)
        group3 = np.random.normal(4, 1, n_samples // 3)

        data = np.concatenate([group1, group2, group3])
        groups = (
            ["A"] * (n_samples // 3)
            + ["B"] * (n_samples // 3)
            + ["C"] * (n_samples // 3)
        )

        return pd.DataFrame({"group": groups, "values": data})

    elif test_type == TestType.CHI2:
        # Chi-square: categorical data
        categories = ["A", "B", "C", "D"]
        counts = np.random.multinomial(n_samples, [0.25, 0.25, 0.25, 0.25])

        data = []
        for cat, count in zip(categories, counts):
            data.extend([cat] * count)

        return pd.DataFrame({"category": data})

    else:
        # Default: normal distribution
        data = np.random.normal(0, 1, n_samples)
        return pd.DataFrame({"values": data})


def get_test_description(test_type: TestType) -> str:
    """
    Get a description of a statistical test.

    Args:
        test_type: Type of statistical test

    Returns:
        Test description
    """
    descriptions = {
        TestType.TTEST: "Student's t-test for comparing means",
        TestType.ANOVA: "Analysis of Variance for comparing multiple groups",
        TestType.CHI2: "Chi-square test for categorical data",
        TestType.WILCOXON: "Wilcoxon signed-rank test for paired data",
        TestType.MANN_WHITNEY: "Mann-Whitney U test for independent samples",
        TestType.KRUSKAL: "Kruskal-Wallis H test for multiple groups",
        TestType.BARTLETT: "Bartlett's test for homogeneity of variances",
        TestType.FLIGNER: "Fligner-Killeen test for homogeneity of variances",
        TestType.DUNNETT: "Dunnett's test for multiple comparisons",
        TestType.TUKEY: "Tukey's HSD test for multiple comparisons",
        TestType.STEEL: "Steel's test for multiple comparisons",
        TestType.STEEL_DWASS: "Steel-Dwass test for multiple comparisons",
        TestType.WILLIAMS: "Williams' test for trend analysis",
        TestType.POISSON: "Poisson test for count data",
        TestType.BINOM: "Binomial test for proportions",
    }

    return descriptions.get(test_type, "Unknown test type")
