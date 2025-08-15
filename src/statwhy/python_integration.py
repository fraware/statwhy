#!/usr/bin/env python3
"""
StatWhy Python Integration Module

Provides seamless integration with the scipy/pandas ecosystem and Jupyter notebooks.
Enables users to use StatWhy within their existing Python workflows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .core import StatWhyEngine
from .models import VerificationRequest, VerificationResult, TestType
from .data_generator import RealisticDataGenerator


@dataclass
class StatWhyResult:
    """Enhanced result object for Python integration."""

    is_verified: bool
    test_type: str
    data_shape: Tuple[int, int]
    execution_time: float
    assumptions_checked: List[str]
    warnings: List[str]
    recommendations: List[str]
    statistical_summary: Dict[str, Any]
    visualization_data: Dict[str, Any]

    def __post_init__(self):
        """Initialize default values."""
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []
        if self.statistical_summary is None:
            self.statistical_summary = {}
        if self.visualization_data is None:
            self.visualization_data = {}


class StatWhyPython:
    """
    Main Python integration class for StatWhy.

    Provides pandas-like interface and Jupyter notebook integration.
    """

    def __init__(self, verbose: bool = True):
        """Initialize StatWhy Python integration."""
        self.verbose = verbose
        self.engine = StatWhyEngine()
        self.data_generator = RealisticDataGenerator()
        self._setup_plotting()

        if verbose:
            print("🚀 StatWhy Python Integration Ready!")
            print("📊 Available methods:")
            print("  • verify_dataframe() - Verify pandas DataFrame")
            print("  • generate_sample_data() - Create realistic datasets")
            print("  • analyze_assumptions() - Check test prerequisites")
            print("  • create_visualizations() - Generate plots")
            print("  • jupyter_tutorial() - Interactive tutorial")

    def _setup_plotting(self):
        """Setup plotting styles and configurations."""
        plt.style.use("default")
        sns.set_palette("husl")

        # Set default figure size
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12

    def verify_dataframe(
        self,
        df: pd.DataFrame,
        test_type: str,
        alpha: float = 0.05,
        timeout: int = 300,
        auto_clean: bool = True,
    ) -> StatWhyResult:
        """
        Verify a pandas DataFrame using StatWhy.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data to verify
        test_type : str
            Statistical test to perform
        alpha : float, default 0.05
            Significance level
        timeout : int, default 300
            Verification timeout in seconds
        auto_clean : bool, default True
            Automatically clean data before verification

        Returns:
        --------
        StatWhyResult
            Enhanced result object with verification details
        """
        if self.verbose:
            print(f"🔍 Verifying {test_type.upper()} test on DataFrame...")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")

        # Auto-clean data if requested
        if auto_clean:
            df_clean = self._auto_clean_dataframe(df)
            if self.verbose and len(df_clean) != len(df):
                print(f"   Cleaned: {len(df)} -> {len(df_clean)} rows")
        else:
            df_clean = df.copy()

        # Create verification request
        request = VerificationRequest(
            test_type=test_type, data=df_clean, alpha=alpha, timeout=timeout
        )

        # Perform verification
        start_time = pd.Timestamp.now()
        result = self.engine.verify(request)
        execution_time = (pd.Timestamp.now() - start_time).total_seconds()

        # Create enhanced result
        enhanced_result = StatWhyResult(
            is_verified=result.is_verified,
            test_type=test_type,
            data_shape=df_clean.shape,
            execution_time=execution_time,
            assumptions_checked=self._extract_assumptions(result),
            warnings=self._extract_warnings(result),
            recommendations=self._generate_recommendations(result, df_clean),
            statistical_summary=self._create_statistical_summary(df_clean, test_type),
            visualization_data=self._prepare_visualization_data(df_clean, test_type),
        )

        if self.verbose:
            self._display_verification_summary(enhanced_result)

        return enhanced_result

    def generate_sample_data(
        self,
        test_type: str,
        n_samples: int = 100,
        seed: Optional[int] = None,
        industry: str = "research",
    ) -> pd.DataFrame:
        """
        Generate realistic sample data for testing.

        Parameters:
        -----------
        test_type : str
            Type of statistical test
        n_samples : int, default 100
            Number of samples to generate
        seed : int, optional
            Random seed for reproducibility
        industry : str, default "research"
            Industry context (clinical, financial, manufacturing, research, education)

        Returns:
        --------
        pd.DataFrame
            Generated sample data
        """
        if self.verbose:
            print(f"🎲 Generating {test_type} sample data...")
            print(f"   Samples: {n_samples}")
            print(f"   Industry: {industry}")

        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Generate appropriate data based on test type
        if test_type == "ttest":
            df = self._generate_ttest_data(n_samples, industry)
        elif test_type == "anova":
            df = self._generate_anova_data(n_samples, industry)
        elif test_type == "chi2":
            df = self._generate_chi2_data(n_samples, industry)
        elif test_type == "wilcoxon":
            df = self._generate_wilcoxon_data(n_samples, industry)
        elif test_type == "mann-whitney":
            df = self._generate_mannwhitney_data(n_samples, industry)
        else:
            df = self._generate_generic_data(n_samples, industry)

        if self.verbose:
            print(f"✅ Generated data: {df.shape}")
            print(f"   Columns: {list(df.columns)}")

        return df

    def analyze_assumptions(self, df: pd.DataFrame, test_type: str) -> Dict[str, Any]:
        """
        Analyze data assumptions for statistical tests.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        test_type : str
            Statistical test type

        Returns:
        --------
        Dict[str, Any]
            Assumption analysis results
        """
        if self.verbose:
            print(f"🔍 Analyzing assumptions for {test_type}...")

        analysis = {
            "test_type": test_type,
            "data_shape": df.shape,
            "assumptions": {},
            "violations": [],
            "recommendations": [],
        }

        # Check common assumptions
        if test_type in ["ttest", "anova"]:
            analysis["assumptions"]["normality"] = self._check_normality(df)
            analysis["assumptions"]["independence"] = self._check_independence(df)

            if test_type == "anova":
                analysis["assumptions"]["homogeneity"] = self._check_homogeneity(df)

        elif test_type in ["wilcoxon", "mann-whitney"]:
            analysis["assumptions"]["symmetry"] = self._check_symmetry(df)
            analysis["assumptions"]["independence"] = self._check_independence(df)

        elif test_type == "chi2":
            analysis["assumptions"]["expected_frequencies"] = (
                self._check_expected_frequencies(df)
            )
            analysis["assumptions"]["independence"] = self._check_independence(df)

        # Generate recommendations
        analysis["recommendations"] = self._generate_assumption_recommendations(
            analysis
        )

        if self.verbose:
            self._display_assumption_analysis(analysis)

        return analysis

    def create_visualizations(
        self, df: pd.DataFrame, test_type: str, interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Create comprehensive visualizations for data analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        test_type : str
            Statistical test type
        interactive : bool, default True
            Use interactive Plotly plots if True, matplotlib if False

        Returns:
        --------
        Dict[str, Any]
            Visualization objects and data
        """
        if self.verbose:
            print(f"📊 Creating visualizations for {test_type}...")

        visualizations = {}

        if interactive:
            visualizations.update(self._create_plotly_plots(df, test_type))
        else:
            visualizations.update(self._create_matplotlib_plots(df, test_type))

        if self.verbose:
            print("✅ Visualizations created successfully!")

        return visualizations

    def jupyter_tutorial(self) -> None:
        """Launch interactive Jupyter notebook tutorial."""
        tutorial_html = """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
            <h1>🎓 StatWhy Interactive Tutorial</h1>
            <p>Welcome to the StatWhy Python integration tutorial!</p>
        </div>
        
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2>🚀 Quick Start Examples</h2>
            
            <h3>1. Generate Sample Data</h3>
            <pre><code>
# Generate clinical trial data
clinical_data = statwhy.generate_sample_data(
    test_type="ttest", 
    n_samples=150, 
    industry="clinical"
)
            </code></pre>
            
            <h3>2. Verify Statistical Test</h3>
            <pre><code>
# Verify t-test assumptions
result = statwhy.verify_dataframe(
    clinical_data, 
    test_type="ttest"
)
            </code></pre>
            
            <h3>3. Analyze Assumptions</h3>
            <pre><code>
# Check test prerequisites
assumptions = statwhy.analyze_assumptions(
    clinical_data, 
    test_type="ttest"
)
            </code></pre>
            
            <h3>4. Create Visualizations</h3>
            <pre><code>
# Generate interactive plots
plots = statwhy.create_visualizations(
    clinical_data, 
    test_type="ttest"
)
            </code></pre>
        </div>
        
        <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2>💡 Pro Tips</h2>
            <ul>
                <li>Use <code>auto_clean=True</code> for automatic data cleaning</li>
                <li>Set <code>seed</code> for reproducible sample generation</li>
                <li>Choose appropriate <code>industry</code> context for realistic data</li>
                <li>Use <code>interactive=True</code> for Plotly visualizations</li>
            </ul>
        </div>
        """

        display(HTML(tutorial_html))

        # Create sample data for demonstration
        print("\n🎯 Let's create some sample data to get started:")
        sample_data = self.generate_sample_data(
            "ttest", n_samples=50, industry="clinical"
        )
        print(f"📊 Sample data created: {sample_data.shape}")

        return sample_data

    def _auto_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically clean DataFrame for analysis."""
        df_clean = df.copy()

        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how="all").dropna(axis=1, how="all")

        # Handle missing values in numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use median for missing values
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Handle categorical missing values
        categorical_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use mode for missing values
                mode_value = (
                    df_clean[col].mode().iloc[0]
                    if len(df_clean[col].mode()) > 0
                    else "Unknown"
                )
                df_clean[col] = df_clean[col].fillna(mode_value)

        return df_clean

    def _extract_assumptions(self, result: VerificationResult) -> List[str]:
        """Extract assumptions from verification result."""
        assumptions = []
        for component in result.components:
            if component.verified:
                assumptions.append(f"✓ {component.name}: {component.details}")
            else:
                assumptions.append(f"✗ {component.name}: {component.details}")
        return assumptions

    def _extract_warnings(self, result: VerificationResult) -> List[str]:
        """Extract warnings from verification result."""
        warnings = []
        for component in result.components:
            if not component.verified:
                warnings.append(f"Assumption violation: {component.name}")
        return warnings

    def _generate_recommendations(
        self, result: VerificationResult, df: pd.DataFrame
    ) -> List[str]:
        """Generate recommendations based on verification results."""
        recommendations = []

        if result.is_verified:
            recommendations.append("✅ All assumptions met - test can proceed safely")
        else:
            recommendations.append(
                "⚠️ Some assumptions violated - consider alternatives"
            )

            # Data-specific recommendations
            if len(df) < 30:
                recommendations.append(
                    "📊 Sample size may be too small for parametric tests"
                )

            if df.select_dtypes(include=[np.number]).columns.size > 0:
                recommendations.append("🔍 Check for outliers and extreme values")

        return recommendations

    def _create_statistical_summary(
        self, df: pd.DataFrame, test_type: str
    ) -> Dict[str, Any]:
        """Create comprehensive statistical summary."""
        summary = {"descriptive_stats": {}, "test_specific": {}, "data_quality": {}}

        # Descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary["descriptive_stats"][col] = {
                "count": int(df[col].count()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "25%": float(df[col].quantile(0.25)),
                "50%": float(df[col].quantile(0.50)),
                "75%": float(df[col].quantile(0.75)),
                "max": float(df[col].max()),
            }

        # Test-specific statistics
        if test_type == "ttest":
            if len(numeric_cols) >= 1:
                col = numeric_cols[0]
                summary["test_specific"]["one_sample"] = {
                    "sample_mean": float(df[col].mean()),
                    "sample_std": float(df[col].std()),
                    "standard_error": float(df[col].std() / np.sqrt(len(df))),
                }

        elif test_type == "anova":
            if "group" in df.columns and len(numeric_cols) >= 1:
                col = numeric_cols[0]
                group_stats = df.groupby("group")[col].agg(["count", "mean", "std"])
                summary["test_specific"]["group_comparison"] = group_stats.to_dict()

        # Data quality metrics
        summary["data_quality"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "missing_percentage": float(df.isnull().sum().sum() / df.size * 100),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        return summary

    def _prepare_visualization_data(
        self, df: pd.DataFrame, test_type: str
    ) -> Dict[str, Any]:
        """Prepare data for visualizations."""
        viz_data = {
            "dataframe": df,
            "test_type": test_type,
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=["object"]).columns),
            "sample_size": len(df),
        }

        return viz_data

    def _generate_ttest_data(self, n_samples: int, industry: str) -> pd.DataFrame:
        """Generate t-test sample data."""
        if industry == "clinical":
            # Blood pressure data
            baseline_bp = np.random.normal(140, 15, n_samples)
            post_treatment_bp = baseline_bp - np.random.normal(8, 3, n_samples)

            df = pd.DataFrame(
                {
                    "patient_id": [f"P{i:03d}" for i in range(1, n_samples + 1)],
                    "baseline_bp": np.round(baseline_bp, 1),
                    "post_treatment_bp": np.round(post_treatment_bp, 1),
                    "age": np.random.normal(65, 12, n_samples),
                    "treatment_group": ["Active"] * n_samples,
                }
            )

        else:
            # Generic t-test data
            group_a = np.random.normal(50, 10, n_samples // 2)
            group_b = np.random.normal(55, 10, n_samples // 2)

            df = pd.DataFrame(
                {
                    "group": ["A"] * (n_samples // 2) + ["B"] * (n_samples // 2),
                    "value": np.concatenate([group_a, group_b]),
                    "sample_id": [f"S{i:03d}" for i in range(1, n_samples + 1)],
                }
            )

        return df

    def _generate_anova_data(self, n_samples: int, industry: str) -> pd.DataFrame:
        """Generate ANOVA sample data."""
        n_per_group = n_samples // 3

        if industry == "manufacturing":
            # Production line quality
            line_a = np.random.normal(95, 2, n_per_group)
            line_b = np.random.normal(92, 2.5, n_per_group)
            line_c = np.random.normal(88, 3, n_per_group)

            df = pd.DataFrame(
                {
                    "production_line": ["Line_A"] * n_per_group
                    + ["Line_B"] * n_per_group
                    + ["Line_C"] * n_per_group,
                    "quality_score": np.concatenate([line_a, line_b, line_c]),
                    "batch_number": list(range(1, n_per_group + 1)) * 3,
                }
            )

        else:
            # Generic ANOVA data
            group_1 = np.random.normal(45, 8, n_per_group)
            group_2 = np.random.normal(52, 8, n_per_group)
            group_3 = np.random.normal(38, 8, n_per_group)

            df = pd.DataFrame(
                {
                    "group": ["Group_1"] * n_per_group
                    + ["Group_2"] * n_per_group
                    + ["Group_3"] * n_per_group,
                    "value": np.concatenate([group_1, group_2, group_3]),
                    "sample_id": [f"S{i:03d}" for i in range(1, n_samples + 1)],
                }
            )

        return df

    def _generate_chi2_data(self, n_samples: int, industry: str) -> pd.DataFrame:
        """Generate chi-square test sample data."""
        if industry == "clinical":
            # Treatment response data
            treatments = np.random.choice(
                ["Drug_A", "Drug_B", "Placebo"], n_samples, p=[0.4, 0.4, 0.2]
            )
            responses = []

            for treatment in treatments:
                if treatment == "Drug_A":
                    response = np.random.choice(
                        ["Responder", "Non_Responder"], p=[0.75, 0.25]
                    )
                elif treatment == "Drug_B":
                    response = np.random.choice(
                        ["Responder", "Non_Responder"], p=[0.68, 0.32]
                    )
                else:  # Placebo
                    response = np.random.choice(
                        ["Responder", "Non_Responder"], p=[0.25, 0.75]
                    )
                responses.append(response)

            df = pd.DataFrame(
                {
                    "treatment": treatments,
                    "response": responses,
                    "patient_id": [f"P{i:03d}" for i in range(1, n_samples + 1)],
                }
            )

        else:
            # Generic chi-square data
            categories = np.random.choice(
                ["A", "B", "C", "D"], n_samples, p=[0.3, 0.3, 0.2, 0.2]
            )
            outcomes = np.random.choice(["Success", "Failure"], n_samples, p=[0.6, 0.4])

            df = pd.DataFrame(
                {
                    "category": categories,
                    "outcome": outcomes,
                    "sample_id": [f"S{i:03d}" for i in range(1, n_samples + 1)],
                }
            )

        return df

    def _generate_wilcoxon_data(self, n_samples: int, industry: str) -> pd.DataFrame:
        """Generate Wilcoxon test sample data."""
        if industry == "clinical":
            # Pain scores before/after
            baseline_pain = np.random.normal(7.5, 1.5, n_samples)
            treatment_effect = np.random.normal(2.5, 0.8, n_samples)
            post_pain = np.clip(baseline_pain - treatment_effect, 0, 10)

            df = pd.DataFrame(
                {
                    "patient_id": [f"P{i:03d}" for i in range(1, n_samples + 1)],
                    "baseline_pain": np.round(baseline_pain, 1),
                    "post_treatment_pain": np.round(post_pain, 1),
                    "pain_type": np.random.choice(
                        ["Chronic", "Acute", "Post-op"], n_samples
                    ),
                }
            )

        else:
            # Generic paired data
            before = np.random.normal(50, 10, n_samples)
            after = before + np.random.normal(5, 3, n_samples)

            df = pd.DataFrame(
                {
                    "before": np.round(before, 2),
                    "after": np.round(after, 2),
                    "sample_id": [f"S{i:03d}" for i in range(1, n_samples + 1)],
                }
            )

        return df

    def _generate_mannwhitney_data(self, n_samples: int, industry: str) -> pd.DataFrame:
        """Generate Mann-Whitney test sample data."""
        n_per_group = n_samples // 2

        if industry == "financial":
            # Investment strategy returns
            strategy_a = np.random.normal(0.12, 0.18, n_per_group)
            strategy_b = np.random.normal(0.15, 0.22, n_per_group)

            df = pd.DataFrame(
                {
                    "strategy": ["Strategy_A"] * n_per_group
                    + ["Strategy_B"] * n_per_group,
                    "annual_return": np.concatenate([strategy_a, strategy_b]),
                    "portfolio_id": [f"Port_{i:03d}" for i in range(1, n_samples + 1)],
                }
            )

        else:
            # Generic two-group data
            group_a = np.random.normal(45, 12, n_per_group)
            group_b = np.random.normal(52, 12, n_per_group)

            df = pd.DataFrame(
                {
                    "group": ["Group_A"] * n_per_group + ["Group_B"] * n_per_group,
                    "value": np.concatenate([group_a, group_b]),
                    "sample_id": [f"S{i:03d}" for i in range(1, n_samples + 1)],
                }
            )

        return df

    def _generate_generic_data(self, n_samples: int, industry: str) -> pd.DataFrame:
        """Generate generic sample data."""
        values = np.random.normal(50, 10, n_samples)

        df = pd.DataFrame(
            {
                "value": np.round(values, 2),
                "sample_id": [f"S{i:03d}" for i in range(1, n_samples + 1)],
                "category": np.random.choice(["A", "B", "C"], n_samples),
            }
        )

        return df

    def _check_normality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check normality assumption."""
        from scipy import stats

        normality_results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) >= 3:
                stat, p_value = stats.shapiro(data)
                normality_results[col] = {
                    "shapiro_stat": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05,
                    "sample_size": len(data),
                }

        return normality_results

    def _check_independence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check independence assumption."""
        # For now, assume independence (would need domain knowledge for real assessment)
        return {
            "assumption": "Independent observations assumed",
            "note": "Verify based on study design and data collection methods",
        }

    def _check_homogeneity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check homogeneity of variances."""
        from scipy import stats

        if "group" not in df.columns:
            return {"error": "No group column found for homogeneity test"}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"error": "No numeric columns found"}

        col = numeric_cols[0]
        groups = df.groupby("group")[col].apply(list).values

        if len(groups) < 2:
            return {"error": "Need at least 2 groups for homogeneity test"}

        try:
            stat, p_value = stats.levene(*groups)
            return {
                "levene_stat": float(stat),
                "p_value": float(p_value),
                "homogeneous": p_value > 0.05,
                "groups": len(groups),
            }
        except:
            return {"error": "Could not perform Levene's test"}

    def _check_symmetry(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check symmetry assumption for non-parametric tests."""
        # For paired data, check if differences are symmetric around zero
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            differences = df[col1] - df[col2]

            # Simple symmetry check: median of differences should be close to zero
            median_diff = np.median(differences)
            return {
                "median_difference": float(median_diff),
                "is_symmetric": abs(median_diff) < 0.1 * np.std(differences),
                "note": "Symmetry check based on median of differences",
            }

        return {"note": "Symmetry check requires paired numeric data"}

    def _check_expected_frequencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check expected frequencies for chi-square test."""
        if len(df.columns) < 2:
            return {"error": "Need at least 2 columns for chi-square test"}

        # Create contingency table
        contingency = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1])

        # Check if all expected frequencies are >= 5
        from scipy import stats

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        min_expected = np.min(expected)
        adequate_frequencies = min_expected >= 5

        return {
            "min_expected_frequency": float(min_expected),
            "adequate_frequencies": adequate_frequencies,
            "contingency_table_shape": contingency.shape,
            "note": "All expected frequencies should be >= 5",
        }

    def _generate_assumption_recommendations(
        self, analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on assumption analysis."""
        recommendations = []

        # Check normality violations
        if "normality" in analysis["assumptions"]:
            normality_results = analysis["assumptions"]["normality"]
            for col, result in normality_results.items():
                if not result["is_normal"]:
                    recommendations.append(
                        f"⚠️ {col} is not normally distributed - consider non-parametric alternatives"
                    )

        # Check homogeneity violations
        if "homogeneity" in analysis["assumptions"]:
            homogeneity_result = analysis["assumptions"]["homogeneity"]
            if (
                "homogeneous" in homogeneity_result
                and not homogeneity_result["homogeneous"]
            ):
                recommendations.append(
                    "⚠️ Variances are not homogeneous - consider Welch's ANOVA or non-parametric tests"
                )

        # Check expected frequencies
        if "expected_frequencies" in analysis["assumptions"]:
            ef_result = analysis["assumptions"]["expected_frequencies"]
            if (
                "adequate_frequencies" in ef_result
                and not ef_result["adequate_frequencies"]
            ):
                recommendations.append(
                    "⚠️ Expected frequencies too low - consider Fisher's exact test or combine categories"
                )

        # General recommendations
        if not recommendations:
            recommendations.append("✅ All assumptions appear to be met")

        return recommendations

    def _display_verification_summary(self, result: StatWhyResult) -> None:
        """Display verification summary."""
        print(f"\n📊 Verification Summary:")
        print(f"   Test Type: {result.test_type.upper()}")
        print(f"   Status: {'✅ VERIFIED' if result.is_verified else '❌ FAILED'}")
        print(f"   Execution Time: {result.execution_time:.3f}s")
        print(f"   Data Shape: {result.data_shape}")

        if result.warnings:
            print(f"   ⚠️  Warnings: {len(result.warnings)}")
            for warning in result.warnings[:3]:  # Show first 3 warnings
                print(f"      • {warning}")

        if result.recommendations:
            print(f"   💡 Recommendations: {len(result.recommendations)}")
            for rec in result.recommendations[:3]:  # Show first 3 recommendations
                print(f"      • {rec}")

    def _display_assumption_analysis(self, analysis: Dict[str, Any]) -> None:
        """Display assumption analysis results."""
        print(f"\n🔍 Assumption Analysis:")
        print(f"   Test Type: {analysis['test_type']}")
        print(f"   Data Shape: {analysis['data_shape']}")

        for assumption, result in analysis["assumptions"].items():
            if isinstance(result, dict) and "is_normal" in result:
                status = "✅" if result["is_normal"] else "❌"
                print(f"   {status} {assumption.title()}")
            elif isinstance(result, dict) and "homogeneous" in result:
                status = "✅" if result["homogeneous"] else "❌"
                print(f"   {status} {assumption.title()}")
            else:
                print(f"   ℹ️  {assumption.title()}")

        if analysis["recommendations"]:
            print(f"   💡 Recommendations: {len(analysis['recommendations'])}")
            for rec in analysis["recommendations"][:3]:
                print(f"      • {rec}")

    def _create_plotly_plots(self, df: pd.DataFrame, test_type: str) -> Dict[str, Any]:
        """Create interactive Plotly plots."""
        plots = {}

        # Basic distribution plots
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Histogram
            fig_hist = px.histogram(
                df,
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}",
                nbins=30,
            )
            plots["histogram"] = fig_hist

            # Box plot if groups exist
            if "group" in df.columns:
                fig_box = px.box(
                    df, x="group", y=numeric_cols[0], title=f"Box Plot by Group"
                )
                plots["boxplot"] = fig_box

        # Scatter plot for paired data
        if test_type == "ttest" and len(numeric_cols) >= 2:
            fig_scatter = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title="Before vs After Comparison",
            )
            plots["scatter"] = fig_scatter

        return plots

    def _create_matplotlib_plots(
        self, df: pd.DataFrame, test_type: str
    ) -> Dict[str, Any]:
        """Create matplotlib plots."""
        plots = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Histogram
            fig, ax = plt.subplots()
            ax.hist(df[numeric_cols[0]], bins=30, alpha=0.7, edgecolor="black")
            ax.set_title(f"Distribution of {numeric_cols[0]}")
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel("Frequency")
            plots["histogram"] = fig

            # Box plot if groups exist
            if "group" in df.columns:
                fig2, ax2 = plt.subplots()
                df.boxplot(column=numeric_cols[0], by="group", ax=ax2)
                ax2.set_title(f"Box Plot by Group")
                plots["boxplot"] = fig2

        return plots


# Convenience function for quick access
def statwhy(verbose: bool = True) -> StatWhyPython:
    """Quick access to StatWhy Python integration."""
    return StatWhyPython(verbose=verbose)


# Jupyter magic commands
def load_ipython_extension(ipython):
    """Load StatWhy as a Jupyter extension."""
    ipython.push(
        {
            "statwhy": StatWhyPython(),
            "statwhy_verify": lambda df, test: StatWhyPython().verify_dataframe(
                df, test
            ),
            "statwhy_generate": lambda test, n: StatWhyPython().generate_sample_data(
                test, n
            ),
            "statwhy_analyze": lambda df, test: StatWhyPython().analyze_assumptions(
                df, test
            ),
        }
    )

    print("🚀 StatWhy Jupyter extension loaded!")
    print("Available commands:")
    print("  • statwhy - Main StatWhy object")
    print("  • statwhy_verify(df, test) - Quick verification")
    print("  • statwhy_generate(test, n) - Quick data generation")
    print("  • statwhy_analyze(df, test) - Quick assumption analysis")


# Example usage and testing
if __name__ == "__main__":
    # Create StatWhy instance
    sw = StatWhyPython()

    # Generate sample data
    data = sw.generate_sample_data("ttest", n_samples=100, industry="clinical")

    # Verify the data
    result = sw.verify_dataframe(data, "ttest")

    # Analyze assumptions
    assumptions = sw.analyze_assumptions(data, "ttest")

    # Create visualizations
    plots = sw.create_visualizations(data, "ttest", interactive=False)

    print("✅ StatWhy Python integration test completed!")
