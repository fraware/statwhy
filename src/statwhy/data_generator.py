#!/usr/bin/env python3
"""
StatWhy Comprehensive Data Generator

Creates extremely realistic and comprehensive sample datasets for all statistical tests.
Incorporates real-world characteristics, edge cases, and industry-specific data patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealisticDataGenerator:
    """
    Generates realistic sample datasets for statistical verification.

    Features:
    - Real-world data characteristics (outliers, missing values, measurement errors)
    - Industry-specific data patterns (clinical trials, financial, manufacturing)
    - Edge cases and boundary conditions
    - Multiple data quality levels (clean, noisy, problematic)
    - Comprehensive metadata and documentation
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the data generator with optional seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.seed = seed
        self.output_dir = Path("examples")
        self.output_dir.mkdir(exist_ok=True)

        # Industry-specific parameters
        self.industry_configs = self._load_industry_configs()

    def _load_industry_configs(self) -> Dict:
        """Load industry-specific configuration parameters."""
        return {
            "clinical": {
                "measurement_error": 0.05,  # 5% measurement error
                "missing_data_rate": 0.02,  # 2% missing data
                "outlier_rate": 0.01,  # 1% outliers
                "correlation_strength": 0.3,  # Moderate correlation
                "seasonal_effects": True,
                "batch_effects": True,
            },
            "financial": {
                "measurement_error": 0.02,  # 2% measurement error
                "missing_data_rate": 0.01,  # 1% missing data
                "outlier_rate": 0.005,  # 0.5% outliers
                "correlation_strength": 0.6,  # Strong correlation
                "seasonal_effects": True,
                "batch_effects": False,
            },
            "manufacturing": {
                "measurement_error": 0.03,  # 3% measurement error
                "missing_data_rate": 0.015,  # 1.5% missing data
                "outlier_rate": 0.02,  # 2% outliers
                "correlation_strength": 0.4,  # Moderate correlation
                "seasonal_effects": False,
                "batch_effects": True,
            },
            "research": {
                "measurement_error": 0.04,  # 4% measurement error
                "missing_data_rate": 0.025,  # 2.5% missing data
                "outlier_rate": 0.015,  # 1.5% outliers
                "correlation_strength": 0.2,  # Weak correlation
                "seasonal_effects": False,
                "batch_effects": False,
            },
            "education": {
                "measurement_error": 0.06,  # 6% measurement error
                "missing_data_rate": 0.03,  # 3% missing data
                "outlier_rate": 0.025,  # 2.5% outliers
                "correlation_strength": 0.1,  # Very weak correlation
                "seasonal_effects": False,
                "batch_effects": False,
            },
        }

    def generate_all_examples(self) -> Dict[str, str]:
        """Generate comprehensive examples for all test types and categories."""
        logger.info("Generating comprehensive example datasets...")

        generated_files = {}

        # Clinical Trials Examples
        generated_files.update(self._generate_clinical_examples())

        # Financial Examples
        generated_files.update(self._generate_financial_examples())

        # Manufacturing Examples
        generated_files.update(self._generate_manufacturing_examples())

        # Research Examples
        generated_files.update(self._generate_research_examples())

        # Education Examples
        generated_files.update(self._generate_education_examples())

        # Create comprehensive metadata
        self._create_metadata_file(generated_files)

        logger.info(f"Generated {len(generated_files)} example datasets")
        return generated_files

    def _generate_clinical_examples(self) -> Dict[str, str]:
        """Generate realistic clinical trial datasets."""
        examples = {}

        # 1. One-sample t-test: Blood pressure reduction
        bp_data = self._generate_blood_pressure_data(n=150, treatment_effect=8.5)
        filename = "clinical_ttest_blood_pressure.csv"
        bp_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "clinical_ttest"

        # 2. ANOVA: Multiple treatment groups
        anova_data = self._generate_treatment_groups_data(
            n_per_group=80,
            group_means=[45, 52, 38, 49],
            group_names=["Placebo", "Low_Dose", "Medium_Dose", "High_Dose"],
        )
        filename = "clinical_anova_treatments.csv"
        anova_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "clinical_anova"

        # 3. Chi-square: Treatment response rates
        chi2_data = self._generate_treatment_response_data(n=300)
        filename = "clinical_chi2_response.csv"
        chi2_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "clinical_chi2"

        # 4. Wilcoxon: Pain scores before/after
        wilcoxon_data = self._generate_pain_score_data(n=120)
        filename = "clinical_wilcoxon_pain.csv"
        wilcoxon_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "clinical_wilcoxon"

        # 5. Mann-Whitney: Two treatment arms
        mannwhitney_data = self._generate_treatment_arms_data(n_per_group=75)
        filename = "clinical_mannwhitney_arms.csv"
        mannwhitney_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "clinical_mannwhitney"

        return examples

    def _generate_financial_examples(self) -> Dict[str, str]:
        """Generate realistic financial datasets."""
        examples = {}

        # 1. Portfolio performance analysis
        portfolio_data = self._generate_portfolio_data(n=200, time_periods=60)
        filename = "financial_portfolio_performance.csv"
        portfolio_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "financial_portfolio"

        # 2. Risk assessment comparison
        risk_data = self._generate_risk_assessment_data(n=150)
        filename = "financial_risk_assessment.csv"
        risk_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "financial_risk"

        # 3. Trading strategy comparison
        trading_data = self._generate_trading_strategy_data(n=180)
        filename = "financial_trading_strategies.csv"
        trading_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "financial_trading"

        return examples

    def _generate_manufacturing_examples(self) -> Dict[str, str]:
        """Generate realistic manufacturing datasets."""
        examples = {}

        # 1. Production line quality comparison
        quality_data = self._generate_production_quality_data(n_per_line=100)
        filename = "manufacturing_production_quality.csv"
        quality_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "manufacturing_quality"

        # 2. Process variance analysis
        variance_data = self._generate_process_variance_data(n_per_process=80)
        filename = "manufacturing_process_variance.csv"
        variance_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "manufacturing_variance"

        # 3. Equipment performance comparison
        equipment_data = self._generate_equipment_performance_data(n_per_equipment=90)
        filename = "manufacturing_equipment_performance.csv"
        equipment_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "manufacturing_equipment"

        return examples

    def _generate_research_examples(self) -> Dict[str, str]:
        """Generate realistic research datasets."""
        examples = {}

        # 1. Experimental group comparison
        experimental_data = self._generate_experimental_data(n_per_group=60)
        filename = "research_experimental_groups.csv"
        experimental_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "research_experimental"

        # 2. Survey response analysis
        survey_data = self._generate_survey_data(n=250)
        filename = "research_survey_responses.csv"
        survey_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "research_survey"

        return examples

    def _generate_education_examples(self) -> Dict[str, str]:
        """Generate realistic education datasets."""
        examples = {}

        # 1. Student performance comparison
        performance_data = self._generate_student_performance_data(n_per_group=70)
        filename = "education_student_performance.csv"
        performance_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "education_performance"

        # 2. Teaching method effectiveness
        teaching_data = self._generate_teaching_method_data(n_per_method=65)
        filename = "education_teaching_methods.csv"
        teaching_data.to_csv(self.output_dir / filename, index=False)
        examples[filename] = "education_teaching"

        return examples

    def _generate_blood_pressure_data(
        self, n: int, treatment_effect: float
    ) -> pd.DataFrame:
        """Generate realistic blood pressure data with treatment effects."""
        # Baseline characteristics
        age = np.random.normal(65, 12, n)
        age = np.clip(age, 45, 85)  # Realistic age range

        # Baseline blood pressure (higher for older patients)
        baseline_bp = 140 + 0.3 * (age - 65) + np.random.normal(0, 15, n)

        # Treatment effect (varies by age and baseline)
        age_effect = 0.1 * (age - 65) / 20  # Older patients respond less
        baseline_effect = (
            0.2 * (baseline_bp - 140) / 20
        )  # Higher baseline = more response

        treatment_effect_individual = treatment_effect * (
            1 + age_effect + baseline_effect
        )
        treatment_effect_individual += np.random.normal(0, 3, n)  # Individual variation

        # Post-treatment blood pressure
        post_bp = baseline_bp - treatment_effect_individual

        # Add realistic constraints
        post_bp = np.clip(post_bp, 80, 180)

        # Create DataFrame with realistic metadata
        data = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(1, n + 1)],
                "age": np.round(age, 1),
                "baseline_systolic_bp": np.round(baseline_bp, 1),
                "post_treatment_systolic_bp": np.round(post_bp, 1),
                "treatment_group": ["Active"] * n,
                "study_site": np.random.choice(["Site_A", "Site_B", "Site_C"], n),
                "baseline_date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "followup_date": pd.date_range("2023-02-01", periods=n, freq="D"),
            }
        )

        # Add missing data (realistic for clinical trials)
        missing_mask = np.random.random(n) < 0.02
        data.loc[missing_mask, "post_treatment_systolic_bp"] = np.nan

        # Add outliers (realistic measurement errors)
        outlier_mask = np.random.random(n) < 0.01
        data.loc[outlier_mask, "post_treatment_systolic_bp"] += np.random.normal(
            0, 25, outlier_mask.sum()
        )

        return data

    def _generate_treatment_groups_data(
        self, n_per_group: int, group_means: List[float], group_names: List[str]
    ) -> pd.DataFrame:
        """Generate realistic treatment group data with group effects."""
        all_data = []

        for i, (mean, name) in enumerate(zip(group_means, group_names)):
            # Generate group data with realistic variation
            group_data = np.random.normal(mean, 12, n_per_group)

            # Add group-specific effects
            if "Placebo" in name:
                group_data += np.random.normal(0, 2, n_per_group)  # Placebo effect
            elif "High_Dose" in name:
                group_data += np.random.normal(2, 1, n_per_group)  # Dose response

            # Add individual variation based on age
            ages = np.random.normal(62, 10, n_per_group)
            age_effect = 0.15 * (ages - 62)
            group_data += age_effect

            # Create group DataFrame
            group_df = pd.DataFrame(
                {
                    "patient_id": [
                        f"{name}_P{j:03d}" for j in range(1, n_per_group + 1)
                    ],
                    "treatment_group": [name] * n_per_group,
                    "age": np.round(ages, 1),
                    "response_variable": np.round(group_data, 2),
                    "baseline_score": np.random.normal(50, 8, n_per_group),
                    "comorbidity_count": np.random.poisson(1.2, n_per_group),
                    "study_visit": np.random.choice(
                        ["Week_4", "Week_8", "Week_12"], n_per_group
                    ),
                }
            )

            all_data.append(group_df)

        # Combine all groups
        combined_data = pd.concat(all_data, ignore_index=True)

        # Add realistic missing data
        missing_mask = np.random.random(len(combined_data)) < 0.025
        combined_data.loc[missing_mask, "response_variable"] = np.nan

        return combined_data

    def _generate_treatment_response_data(self, n: int) -> pd.DataFrame:
        """Generate realistic treatment response data for chi-square analysis."""
        # Treatment groups
        treatments = np.random.choice(
            ["Drug_A", "Drug_B", "Placebo"], n, p=[0.4, 0.4, 0.2]
        )

        # Response rates (realistic for clinical trials)
        response_probs = {
            "Drug_A": 0.75,  # 75% response rate
            "Drug_B": 0.68,  # 68% response rate
            "Placebo": 0.25,  # 25% placebo response
        }

        # Generate responses based on treatment
        responses = []
        for treatment in treatments:
            prob = response_probs[treatment]
            # Add individual variation based on age
            age = np.random.normal(65, 12)
            age_factor = 1 + 0.1 * (age - 65) / 20
            adjusted_prob = np.clip(prob * age_factor, 0.1, 0.95)
            response = np.random.binomial(1, adjusted_prob)
            responses.append("Responder" if response else "Non_Responder")

        # Create DataFrame
        data = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(1, n + 1)],
                "treatment": treatments,
                "response": responses,
                "age": np.round(np.random.normal(65, 12, n), 1),
                "gender": np.random.choice(["Male", "Female"], n),
                "baseline_severity": np.random.choice(
                    ["Mild", "Moderate", "Severe"], n, p=[0.3, 0.5, 0.2]
                ),
                "study_site": np.random.choice(
                    ["Site_A", "Site_B", "Site_C", "Site_D"], n
                ),
                "enrollment_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            }
        )

        # Add missing data
        missing_mask = np.random.random(n) < 0.03
        data.loc[missing_mask, "response"] = np.nan

        return data

    def _generate_portfolio_data(self, n: int, time_periods: int) -> pd.DataFrame:
        """Generate realistic portfolio performance data."""
        # Portfolio characteristics
        portfolio_ids = [f"Portfolio_{i:02d}" for i in range(1, n + 1)]

        # Risk profiles (realistic for financial portfolios)
        risk_profiles = np.random.choice(
            ["Conservative", "Moderate", "Aggressive"], n, p=[0.3, 0.5, 0.2]
        )

        # Generate time series data
        all_data = []
        base_date = datetime(2022, 1, 1)

        for i, (portfolio_id, risk_profile) in enumerate(
            zip(portfolio_ids, risk_profiles)
        ):
            # Set risk-based parameters
            if risk_profile == "Conservative":
                base_return = 0.06  # 6% annual return
                volatility = 0.08  # 8% volatility
            elif risk_profile == "Moderate":
                base_return = 0.09  # 9% annual return
                volatility = 0.15  # 15% volatility
            else:  # Aggressive
                base_return = 0.12  # 12% annual return
                volatility = 0.25  # 25% volatility

            # Generate monthly returns with realistic patterns
            monthly_returns = []
            cumulative_value = 1000000  # Start with $1M

            for month in range(time_periods):
                # Add seasonal effects (Q4 typically better)
                seasonal_factor = 1.0
                if month % 12 in [9, 10, 11]:  # Q4
                    seasonal_factor = 1.2
                elif month % 12 in [0, 1, 2]:  # Q1
                    seasonal_factor = 0.9

                # Generate return with realistic distribution
                monthly_return = np.random.normal(
                    base_return / 12, volatility / np.sqrt(12)
                )
                monthly_return *= seasonal_factor

                # Add market stress periods (realistic for financial data)
                if month in [12, 24, 36]:  # Market stress periods
                    monthly_return -= 0.05

                monthly_returns.append(monthly_return)
                cumulative_value *= 1 + monthly_return

            # Create time series DataFrame
            for month, monthly_return in enumerate(monthly_returns):
                date = base_date + timedelta(days=month * 30)

                row_data = {
                    "portfolio_id": portfolio_id,
                    "risk_profile": risk_profile,
                    "date": date,
                    "monthly_return": np.round(monthly_return, 4),
                    "cumulative_value": np.round(cumulative_value, 2),
                    "benchmark_return": np.random.normal(
                        0.07 / 12, 0.12 / np.sqrt(12)
                    ),  # S&P 500-like
                    "risk_free_rate": 0.02 / 12,  # 2% annual
                    "volatility_30d": np.random.normal(volatility, volatility * 0.1),
                    "sharpe_ratio": np.random.normal(0.8, 0.2),
                    "max_drawdown": np.random.normal(-0.15, 0.05),
                }
                all_data.append(row_data)

        data = pd.DataFrame(all_data)

        # Add missing data (realistic for financial data)
        missing_mask = np.random.random(len(data)) < 0.01
        data.loc[missing_mask, "monthly_return"] = np.nan

        return data

    def _generate_pain_score_data(self, n: int) -> pd.DataFrame:
        """Generate realistic pain score data for Wilcoxon test."""
        # Baseline pain scores (0-10 scale)
        baseline_pain = np.random.normal(7.5, 1.5, n)
        baseline_pain = np.clip(baseline_pain, 0, 10)

        # Treatment effect (varies by baseline pain)
        treatment_effect = 2.5 + 0.3 * (
            baseline_pain - 5
        )  # More effect for higher pain
        treatment_effect += np.random.normal(0, 0.8, n)

        # Post-treatment pain scores
        post_pain = baseline_pain - treatment_effect
        post_pain = np.clip(post_pain, 0, 10)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "patient_id": [f"Pain_P{i:03d}" for i in range(1, n + 1)],
                "baseline_pain_score": np.round(baseline_pain, 1),
                "post_treatment_pain_score": np.round(post_pain, 1),
                "treatment_group": ["Active"] * n,
                "pain_type": np.random.choice(
                    ["Chronic", "Acute", "Post-op"], n, p=[0.4, 0.3, 0.3]
                ),
                "age": np.round(np.random.normal(58, 12, n), 1),
                "assessment_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            }
        )

        # Add missing data
        missing_mask = np.random.random(n) < 0.025
        data.loc[missing_mask, "post_treatment_pain_score"] = np.nan

        return data

    def _generate_treatment_arms_data(self, n_per_group: int) -> pd.DataFrame:
        """Generate realistic treatment arms data for Mann-Whitney test."""
        # Two treatment arms with different effectiveness
        arm_a_data = np.random.normal(45, 12, n_per_group)
        arm_b_data = np.random.normal(52, 12, n_per_group)

        # Add individual variation
        ages = np.random.normal(65, 10, n_per_group * 2)
        age_effect = 0.2 * (ages - 65) / 10

        arm_a_data += age_effect[:n_per_group]
        arm_b_data += age_effect[n_per_group:]

        # Create DataFrame
        data = pd.DataFrame(
            {
                "patient_id": (
                    [f"ArmA_P{i:03d}" for i in range(1, n_per_group + 1)]
                    + [f"ArmB_P{i:03d}" for i in range(1, n_per_group + 1)]
                ),
                "treatment_arm": ["Arm_A"] * n_per_group + ["Arm_B"] * n_per_group,
                "response_variable": np.round(
                    np.concatenate([arm_a_data, arm_b_data]), 2
                ),
                "age": np.round(ages, 1),
                "baseline_score": np.random.normal(50, 8, n_per_group * 2),
                "study_site": np.random.choice(["Site_A", "Site_B"], n_per_group * 2),
            }
        )

        return data

    def _generate_risk_assessment_data(self, n: int) -> pd.DataFrame:
        """Generate realistic financial risk assessment data."""
        # Risk profiles
        risk_profiles = np.random.choice(
            ["Low", "Medium", "High"], n, p=[0.4, 0.4, 0.2]
        )

        # Risk scores based on profile
        risk_scores = []
        for profile in risk_profiles:
            if profile == "Low":
                score = np.random.normal(25, 8)
            elif profile == "Medium":
                score = np.random.normal(50, 12)
            else:  # High
                score = np.random.normal(75, 15)
            risk_scores.append(score)

        risk_scores = np.array(risk_scores)
        risk_scores = np.clip(risk_scores, 0, 100)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "portfolio_id": [f"Portfolio_{i:03d}" for i in range(1, n + 1)],
                "risk_profile": risk_profiles,
                "risk_score": np.round(risk_scores, 1),
                "volatility": np.random.normal(0.15, 0.08, n),
                "sharpe_ratio": np.random.normal(0.8, 0.3, n),
                "max_drawdown": np.random.normal(-0.12, 0.06, n),
                "asset_allocation": np.random.choice(
                    ["Equity", "Fixed_Income", "Mixed"], n
                ),
                "assessment_date": pd.date_range("2022-01-01", periods=n, freq="D"),
            }
        )

        return data

    def _generate_trading_strategy_data(self, n: int) -> pd.DataFrame:
        """Generate realistic trading strategy comparison data."""
        # Strategy performance metrics
        strategy_a_returns = np.random.normal(0.12, 0.18, n // 2)
        strategy_b_returns = np.random.normal(0.15, 0.22, n // 2)

        # Add market condition effects
        market_conditions = np.random.choice(
            ["Bull", "Bear", "Sideways"], n, p=[0.4, 0.3, 0.3]
        )
        market_effects = {"Bull": 0.05, "Bear": -0.03, "Sideways": 0.01}

        strategy_a_returns += [
            market_effects[cond] for cond in market_conditions[: n // 2]
        ]
        strategy_b_returns += [
            market_effects[cond] for cond in market_conditions[n // 2 :]
        ]

        # Create DataFrame
        data = pd.DataFrame(
            {
                "strategy_id": (
                    [f"Strategy_A_{i:03d}" for i in range(1, n // 2 + 1)]
                    + [f"Strategy_B_{i:03d}" for i in range(1, n // 2 + 1)]
                ),
                "strategy": ["Strategy_A"] * (n // 2) + ["Strategy_B"] * (n // 2),
                "annual_return": np.round(
                    np.concatenate([strategy_a_returns, strategy_b_returns]), 4
                ),
                "volatility": np.random.normal(0.18, 0.05, n),
                "sharpe_ratio": np.random.normal(0.7, 0.2, n),
                "max_drawdown": np.random.normal(-0.15, 0.08, n),
                "market_condition": market_conditions,
                "backtest_period": np.random.choice(["1Y", "3Y", "5Y"], n),
            }
        )

        return data

    def _generate_production_quality_data(self, n_per_line: int) -> pd.DataFrame:
        """Generate realistic production line quality data."""
        # Production lines with different quality characteristics
        lines = ["Line_A", "Line_B", "Line_C", "Line_D"]
        all_data = []

        for line in lines:
            # Line-specific quality parameters
            if "A" in line:
                base_quality = 95.0
                quality_std = 2.0
            elif "B" in line:
                base_quality = 92.0
                quality_std = 2.5
            elif "C" in line:
                base_quality = 88.0
                quality_std = 3.0
            else:  # Line_D
                base_quality = 85.0
                quality_std = 3.5

            # Generate quality scores
            quality_scores = np.random.normal(base_quality, quality_std, n_per_line)
            quality_scores = np.clip(quality_scores, 70, 100)

            # Add time-based variation
            time_effect = np.sin(np.linspace(0, 4 * np.pi, n_per_line)) * 1.5
            quality_scores += time_effect

            # Create line DataFrame
            line_data = pd.DataFrame(
                {
                    "production_line": [line] * n_per_line,
                    "quality_score": np.round(quality_scores, 2),
                    "batch_number": range(1, n_per_line + 1),
                    "operator_id": np.random.choice(
                        ["Op_01", "Op_02", "Op_03"], n_per_line
                    ),
                    "shift": np.random.choice(
                        ["Morning", "Afternoon", "Night"], n_per_line
                    ),
                    "temperature": np.random.normal(22, 2, n_per_line),
                    "humidity": np.random.normal(45, 5, n_per_line),
                }
            )

            all_data.append(line_data)

        # Combine all lines
        combined_data = pd.concat(all_data, ignore_index=True)

        # Add missing data
        missing_mask = np.random.random(len(combined_data)) < 0.015
        combined_data.loc[missing_mask, "quality_score"] = np.nan

        return combined_data

    def _generate_process_variance_data(self, n_per_process: int) -> pd.DataFrame:
        """Generate realistic process variance data for Bartlett's test."""
        # Different manufacturing processes
        processes = ["Process_A", "Process_B", "Process_C"]
        all_data = []

        for process in processes:
            # Process-specific parameters
            if "A" in process:
                mean_value = 100.0
                std_value = 5.0
            elif "B" in process:
                mean_value = 100.0
                std_value = 8.0
            else:  # Process_C
                mean_value = 100.0
                std_value = 12.0

            # Generate process data
            process_data = np.random.normal(mean_value, std_value, n_per_process)

            # Create process DataFrame
            process_df = pd.DataFrame(
                {
                    "process_name": [process] * n_per_process,
                    "measurement_value": np.round(process_data, 2),
                    "sample_id": [
                        f"{process}_S{i:03d}" for i in range(1, n_per_process + 1)
                    ],
                    "measurement_time": pd.date_range(
                        "2023-01-01", periods=n_per_process, freq="H"
                    ),
                    "operator": np.random.choice(["Op_01", "Op_02"], n_per_process),
                    "equipment_id": np.random.choice(["EQ_01", "EQ_02"], n_per_process),
                }
            )

            all_data.append(process_df)

        # Combine all processes
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data

    def _generate_equipment_performance_data(
        self, n_per_equipment: int
    ) -> pd.DataFrame:
        """Generate realistic equipment performance comparison data."""
        # Different equipment types
        equipment_types = ["Machine_A", "Machine_B", "Machine_C"]
        all_data = []

        for equipment in equipment_types:
            # Equipment-specific performance parameters
            if "A" in equipment:
                base_performance = 95.0
                performance_std = 3.0
            elif "B" in equipment:
                base_performance = 88.0
                performance_std = 4.5
            else:  # Machine_C
                base_performance = 82.0
                performance_std = 5.5

            # Generate performance data
            performance_data = np.random.normal(
                base_performance, performance_std, n_per_equipment
            )
            performance_data = np.clip(performance_data, 70, 100)

            # Add maintenance cycle effects
            maintenance_effect = (
                np.sin(np.linspace(0, 6 * np.pi, n_per_equipment)) * 2.0
            )
            performance_data += maintenance_effect

            # Create equipment DataFrame
            equipment_df = pd.DataFrame(
                {
                    "equipment_id": [equipment] * n_per_equipment,
                    "performance_score": np.round(performance_data, 2),
                    "maintenance_cycle": range(1, n_per_equipment + 1),
                    "operator_skill": np.random.choice(
                        ["Beginner", "Intermediate", "Expert"], n_per_equipment
                    ),
                    "shift": np.random.choice(
                        ["Morning", "Afternoon", "Night"], n_per_equipment
                    ),
                    "ambient_temperature": np.random.normal(23, 3, n_per_equipment),
                    "vibration_level": np.random.normal(0.5, 0.2, n_per_equipment),
                }
            )

            all_data.append(equipment_df)

        # Combine all equipment
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data

    def _generate_experimental_data(self, n_per_group: int) -> pd.DataFrame:
        """Generate realistic experimental group comparison data."""
        # Experimental groups
        groups = ["Control", "Treatment_1", "Treatment_2"]
        all_data = []

        for group in groups:
            # Group-specific effects
            if "Control" in group:
                base_effect = 50.0
                effect_std = 8.0
            elif "Treatment_1" in group:
                base_effect = 58.0
                effect_std = 8.0
            else:  # Treatment_2
                base_effect = 62.0
                effect_std = 8.0

            # Generate experimental data
            experimental_data = np.random.normal(base_effect, effect_std, n_per_group)

            # Add individual variation
            ages = np.random.normal(35, 8, n_per_group)
            age_effect = 0.3 * (ages - 35) / 10
            experimental_data += age_effect

            # Create group DataFrame
            group_df = pd.DataFrame(
                {
                    "group": [group] * n_per_group,
                    "response_variable": np.round(experimental_data, 2),
                    "age": np.round(ages, 1),
                    "gender": np.random.choice(["Male", "Female"], n_per_group),
                    "baseline_score": np.random.normal(45, 6, n_per_group),
                    "experiment_session": np.random.choice(
                        ["Session_1", "Session_2", "Session_3"], n_per_group
                    ),
                }
            )

            all_data.append(group_df)

        # Combine all groups
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data

    def _generate_survey_data(self, n: int) -> pd.DataFrame:
        """Generate realistic survey response data."""
        # Survey questions and response scales
        questions = ["Q1_Satisfaction", "Q2_Quality", "Q3_Recommendation", "Q4_Value"]
        response_scales = [5, 5, 5, 5]  # 1-5 Likert scale

        all_data = []

        for i, (question, scale) in enumerate(zip(questions, response_scales)):
            # Generate responses with realistic distribution
            if "Satisfaction" in question:
                responses = np.random.choice(
                    range(1, scale + 1), n, p=[0.1, 0.15, 0.25, 0.3, 0.2]
                )
            elif "Quality" in question:
                responses = np.random.choice(
                    range(1, scale + 1), n, p=[0.05, 0.1, 0.2, 0.35, 0.3]
                )
            elif "Recommendation" in question:
                responses = np.random.choice(
                    range(1, scale + 1), n, p=[0.1, 0.15, 0.25, 0.3, 0.2]
                )
            else:  # Value
                responses = np.random.choice(
                    range(1, scale + 1), n, p=[0.15, 0.2, 0.25, 0.25, 0.15]
                )

            # Create question DataFrame
            question_df = pd.DataFrame(
                {
                    "respondent_id": [f"R{i:03d}" for i in range(1, n + 1)],
                    "question": [question] * n,
                    "response": responses,
                    "response_scale": [scale] * n,
                    "age_group": np.random.choice(
                        ["18-25", "26-35", "36-45", "46-55", "55+"], n
                    ),
                    "gender": np.random.choice(["Male", "Female", "Other"], n),
                    "response_time_seconds": np.random.exponential(30, n),
                }
            )

            all_data.append(question_df)

        # Combine all questions
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data

    def _generate_student_performance_data(self, n_per_group: int) -> pd.DataFrame:
        """Generate realistic student performance comparison data."""
        # Teaching methods
        methods = ["Traditional", "Interactive", "Online", "Hybrid"]
        all_data = []

        for method in methods:
            # Method-specific performance parameters
            if "Traditional" in method:
                base_performance = 72.0
                performance_std = 12.0
            elif "Interactive" in method:
                base_performance = 78.0
                performance_std = 11.0
            elif "Online" in method:
                base_performance = 75.0
                performance_std = 13.0
            else:  # Hybrid
                base_performance = 80.0
                performance_std = 10.0

            # Generate performance data
            performance_data = np.random.normal(
                base_performance, performance_std, n_per_group
            )
            performance_data = np.clip(performance_data, 0, 100)

            # Add student background effects
            study_hours = np.random.normal(15, 5, n_per_group)
            study_effect = 0.5 * (study_hours - 15) / 5
            performance_data += study_effect

            # Create method DataFrame
            method_df = pd.DataFrame(
                {
                    "teaching_method": [method] * n_per_group,
                    "final_score": np.round(performance_data, 1),
                    "study_hours_per_week": np.round(study_hours, 1),
                    "previous_gpa": np.random.normal(3.2, 0.4, n_per_group),
                    "attendance_rate": np.random.normal(0.85, 0.1, n_per_group),
                    "student_motivation": np.random.choice(
                        ["Low", "Medium", "High"], n_per_group, p=[0.2, 0.5, 0.3]
                    ),
                }
            )

            all_data.append(method_df)

        # Combine all methods
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data

    def _generate_teaching_method_data(self, n_per_method: int) -> pd.DataFrame:
        """Generate realistic teaching method effectiveness data."""
        # Teaching methods
        methods = ["Lecture", "Discussion", "Project", "Lab"]
        all_data = []

        for method in methods:
            # Method-specific effectiveness parameters
            if "Lecture" in method:
                base_effectiveness = 70.0
                effectiveness_std = 10.0
            elif "Discussion" in method:
                base_effectiveness = 75.0
                effectiveness_std = 9.0
            elif "Project" in method:
                base_effectiveness = 78.0
                effectiveness_std = 8.5
            else:  # Lab
                base_effectiveness = 82.0
                effectiveness_std = 8.0

            # Generate effectiveness data
            effectiveness_data = np.random.normal(
                base_effectiveness, effectiveness_std, n_per_method
            )
            effectiveness_data = np.clip(effectiveness_data, 50, 100)

            # Add class size effects
            class_sizes = np.random.choice([15, 25, 35, 45], n_per_method)
            size_effect = -0.2 * (class_sizes - 25) / 10
            effectiveness_data += size_effect

            # Create method DataFrame
            method_df = pd.DataFrame(
                {
                    "teaching_method": [method] * n_per_method,
                    "effectiveness_score": np.round(effectiveness_data, 1),
                    "class_size": class_sizes,
                    "student_engagement": np.random.normal(0.75, 0.15, n_per_method),
                    "instructor_experience_years": np.random.normal(8, 3, n_per_method),
                    "course_difficulty": np.random.choice(
                        ["Intro", "Intermediate", "Advanced"], n_per_method
                    ),
                }
            )

            all_data.append(method_df)

        # Combine all methods
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data

    def _create_metadata_file(self, generated_files: Dict[str, str]):
        """Create comprehensive metadata file for all generated datasets."""
        metadata = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "seed": self.seed,
                "total_datasets": len(generated_files),
                "generator_version": "1.0.0",
            },
            "datasets": {},
        }

        for filename, test_type in generated_files.items():
            file_path = self.output_dir / filename

            if file_path.exists():
                df = pd.read_csv(file_path)

                dataset_info = {
                    "test_type": test_type,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "data_types": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "numeric_summary": {},
                    "categorical_summary": {},
                }

                # Add numeric summaries
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    dataset_info["numeric_summary"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "q25": float(df[col].quantile(0.25)),
                        "q75": float(df[col].quantile(0.75)),
                    }

                # Add categorical summaries
                categorical_cols = df.select_dtypes(include=["object"]).columns
                for col in categorical_cols:
                    dataset_info["categorical_summary"][col] = (
                        df[col].value_counts().to_dict()
                    )

                metadata["datasets"][filename] = dataset_info

        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main function to generate all example datasets."""
    generator = RealisticDataGenerator(seed=42)
    generated_files = generator.generate_all_examples()

    print(f"\n✅ Generated {len(generated_files)} comprehensive example datasets:")
    for filename, test_type in generated_files.items():
        print(f"  📊 {filename} -> {test_type}")

    print(f"\n📁 All files saved to: {generator.output_dir}")
    print(f"📋 Metadata saved to: {generator.output_dir}/dataset_metadata.json")

    print("\n🎯 These datasets include:")
    print("  • Real-world data characteristics (outliers, missing values)")
    print("  • Industry-specific patterns (clinical, financial, manufacturing)")
    print("  • Edge cases and boundary conditions")
    print("  • Multiple data quality levels")
    print("  • Comprehensive metadata and documentation")


if __name__ == "__main__":
    main()
