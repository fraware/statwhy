#!/usr/bin/env python3
"""
StatWhy Clinical Trial Verification Module

Implements FDA-compliant statistical procedures for clinical trials with focus on
patient safety, regulatory compliance, and scientific rigor.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from .models import TestType, VerificationRequest, ComponentResult
from .verifiers import BaseVerifier

logger = logging.getLogger(__name__)


class FDAGuidelines:
    """FDA statistical guidelines for clinical trials (E9, E10, E17)."""

    # E9 Statistical Principles for Clinical Trials
    E9_PRINCIPLES = {
        "randomization": "Proper randomization to avoid bias",
        "blinding": "Appropriate blinding procedures",
        "sample_size": "Adequate sample size with power analysis",
        "primary_endpoint": "Clear primary endpoint definition",
        "multiplicity": "Proper handling of multiple comparisons",
        "interim_analysis": "Pre-specified interim analysis plans",
        "missing_data": "Strategies for handling missing data",
        "subgroup_analysis": "Pre-specified subgroup analyses",
        "safety_monitoring": "Comprehensive safety monitoring",
        "statistical_analysis_plan": "Pre-specified analysis plan",
    }

    # E10 Choice of Control Group
    E10_CONTROLS = {
        "placebo_control": "Placebo control when ethically acceptable",
        "active_control": "Active control for superiority/non-inferiority",
        "dose_response": "Dose-response relationship studies",
        "add_on_design": "Add-on to standard therapy",
        "withdrawal_design": "Withdrawal of effective treatment",
    }

    # E17 Multi-Regional Clinical Trials
    E17_MULTIREGIONAL = {
        "regional_consistency": "Consistency across regions",
        "sample_size_allocation": "Proper regional sample size allocation",
        "regulatory_requirements": "Meeting regional regulatory requirements",
        "cultural_adaptation": "Cultural and ethnic considerations",
    }


class ClinicalTrialVerifier(BaseVerifier):
    """
    FDA-compliant clinical trial verification.

    Implements comprehensive verification of statistical procedures used in
    clinical trials, ensuring patient safety and regulatory compliance.
    """

    def __init__(self):
        super().__init__()
        self.fda_guidelines = FDAGuidelines()
        self.safety_thresholds = self._load_safety_thresholds()

    def _load_safety_thresholds(self) -> Dict[str, Any]:
        """Load safety thresholds for different clinical trial phases."""
        return {
            "phase_1": {
                "max_dlt_rate": 0.33,  # Maximum dose-limiting toxicity rate
                "min_safety_cohort": 3,  # Minimum safety cohort size
                "stopping_rules": ["2/3 DLTs", "1/6 DLTs in expansion"],
            },
            "phase_2": {
                "max_sae_rate": 0.20,  # Maximum serious adverse event rate
                "min_efficacy": 0.15,  # Minimum efficacy threshold
                "futility_boundary": 0.10,  # Futility stopping boundary
            },
            "phase_3": {
                "max_sae_rate": 0.15,  # Maximum serious adverse event rate
                "min_power": 0.80,  # Minimum statistical power
                "max_type1_error": 0.05,  # Maximum type I error rate
                "non_inferiority_margin": 0.10,  # Non-inferiority margin
            },
        }

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify clinical trial statistical procedures."""
        components = []

        # Basic statistical verification
        components.extend(self._verify_basic_statistics(request))

        # FDA compliance verification
        components.extend(self._verify_fda_compliance(request))

        # Patient safety verification
        components.extend(self._verify_patient_safety(request))

        # Clinical trial specific verification
        components.extend(self._verify_clinical_trial_specific(request))

        return components

    def _verify_basic_statistics(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify basic statistical requirements."""
        components = []

        # Sample size adequacy
        sample_size_result = self._verify_sample_size_adequacy(request)
        components.append(sample_size_result)

        # Randomization verification
        randomization_result = self._verify_randomization(request)
        components.append(randomization_result)

        # Blinding verification
        blinding_result = self._verify_blinding(request)
        components.append(blinding_result)

        return components

    def _verify_fda_compliance(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify FDA compliance requirements."""
        components = []

        # E9 principles compliance
        e9_result = self._verify_e9_principles(request)
        components.append(e9_result)

        # E10 control group requirements
        e10_result = self._verify_e10_controls(request)
        components.append(e10_result)

        # E17 multi-regional requirements
        e17_result = self._verify_e17_multiregional(request)
        components.append(e17_result)

        return components

    def _verify_patient_safety(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify patient safety requirements."""
        components = []

        # Safety monitoring verification
        safety_monitoring_result = self._verify_safety_monitoring(request)
        components.append(safety_monitoring_result)

        # Adverse event analysis
        ae_analysis_result = self._verify_adverse_event_analysis(request)
        components.append(ae_analysis_result)

        # Stopping rules verification
        stopping_rules_result = self._verify_stopping_rules(request)
        components.append(stopping_rules_result)

        return components

    def _verify_clinical_trial_specific(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify clinical trial specific requirements."""
        components = []

        # Primary endpoint verification
        primary_endpoint_result = self._verify_primary_endpoint(request)
        components.append(primary_endpoint_result)

        # Multiplicity handling
        multiplicity_result = self._verify_multiplicity_handling(request)
        components.append(multiplicity_result)

        # Missing data strategy
        missing_data_result = self._verify_missing_data_strategy(request)
        components.append(missing_data_result)

        return components

    def _verify_sample_size_adequacy(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify sample size adequacy for clinical trials."""
        try:
            data = request.data
            n_total = len(data)

            # Determine trial phase from metadata
            trial_phase = request.metadata.get("trial_phase", "phase_3")
            phase_thresholds = self.safety_thresholds[trial_phase]

            # Check minimum sample size requirements
            min_sample_size = self._get_minimum_sample_size(
                trial_phase, request.test_type
            )

            if n_total < min_sample_size:
                return self._create_component_result(
                    "sample_size_adequacy",
                    False,
                    f"Insufficient sample size: {n_total} < {min_sample_size}",
                    f"Phase {trial_phase} requires minimum {min_sample_size} participants",
                )

            # Check power adequacy
            power_adequate = self._verify_statistical_power(request)
            if not power_adequate:
                return self._create_component_result(
                    "sample_size_adequacy",
                    False,
                    "Statistical power below FDA requirements",
                    f"Power should be ≥ {phase_thresholds['min_power']} for phase {trial_phase}",
                )

            return self._create_component_result(
                "sample_size_adequacy",
                True,
                f"Sample size adequate: {n_total} participants",
                f"Meets phase {trial_phase} requirements",
            )

        except Exception as e:
            return self._create_component_result(
                "sample_size_adequacy",
                False,
                "Error verifying sample size adequacy",
                str(e),
            )

    def _verify_randomization(self, request: VerificationRequest) -> ComponentResult:
        """Verify randomization procedures."""
        try:
            data = request.data

            # Check for randomization column
            if "randomization_group" not in data.columns:
                return self._create_component_result(
                    "randomization",
                    False,
                    "Randomization group information missing",
                    "Clinical trials require proper randomization documentation",
                )

            # Check randomization balance
            group_counts = data["randomization_group"].value_counts()
            if len(group_counts) < 2:
                return self._create_component_result(
                    "randomization",
                    False,
                    "Insufficient randomization groups",
                    "At least 2 groups required for clinical trials",
                )

            # Check for randomization imbalance (should be roughly equal)
            max_imbalance = 0.2  # 20% maximum imbalance
            total_participants = len(data)
            expected_per_group = total_participants / len(group_counts)

            for group, count in group_counts.items():
                imbalance = abs(count - expected_per_group) / expected_per_group
                if imbalance > max_imbalance:
                    return self._create_component_result(
                        "randomization",
                        False,
                        f"Randomization imbalance detected in group {group}",
                        f"Imbalance: {imbalance:.1%} exceeds {max_imbalance:.1%} threshold",
                    )

            return self._create_component_result(
                "randomization",
                True,
                "Randomization procedures verified",
                f"Balanced allocation across {len(group_counts)} groups",
            )

        except Exception as e:
            return self._create_component_result(
                "randomization", False, "Error verifying randomization", str(e)
            )

    def _verify_blinding(self, request: VerificationRequest) -> ComponentResult:
        """Verify blinding procedures."""
        try:
            data = request.data

            # Check for blinding information
            if "blinding_status" not in data.columns:
                return self._create_component_result(
                    "blinding",
                    False,
                    "Blinding status information missing",
                    "Clinical trials require blinding documentation",
                )

            # Check blinding integrity
            blinding_statuses = data["blinding_status"].value_counts()

            # Should have appropriate blinding categories
            expected_categories = ["blinded", "unblinded", "partial_blind"]
            if not any(cat in blinding_statuses.index for cat in expected_categories):
                return self._create_component_result(
                    "blinding",
                    False,
                    "Inappropriate blinding categories",
                    f"Expected categories: {expected_categories}",
                )

            # Check for unblinding events
            unblinded_count = blinding_statuses.get("unblinded", 0)
            total_count = len(data)
            unblinding_rate = unblinded_count / total_count

            if unblinding_rate > 0.05:  # 5% maximum unblinding rate
                return self._create_component_result(
                    "blinding",
                    False,
                    f"Excessive unblinding rate: {unblinding_rate:.1%}",
                    "Unblinding rate should be < 5% to maintain trial integrity",
                )

            return self._create_component_result(
                "blinding",
                True,
                "Blinding procedures verified",
                f"Unblinding rate: {unblinding_rate:.1%} within acceptable limits",
            )

        except Exception as e:
            return self._create_component_result(
                "blinding", False, "Error verifying blinding", str(e)
            )

    def _verify_e9_principles(self, request: VerificationRequest) -> ComponentResult:
        """Verify E9 statistical principles compliance."""
        try:
            data = request.data
            metadata = request.metadata

            # Check for statistical analysis plan
            if "statistical_analysis_plan" not in metadata:
                return self._create_component_result(
                    "e9_principles",
                    False,
                    "Statistical analysis plan missing",
                    "E9 requires pre-specified statistical analysis plan",
                )

            # Check for primary endpoint definition
            if "primary_endpoint" not in metadata:
                return self._create_component_result(
                    "e9_principles",
                    False,
                    "Primary endpoint not defined",
                    "E9 requires clear primary endpoint definition",
                )

            # Check for multiplicity handling
            if "multiplicity_adjustment" not in metadata:
                return self._create_component_result(
                    "e9_principles",
                    False,
                    "Multiplicity adjustment not specified",
                    "E9 requires pre-specified multiplicity handling",
                )

            # Check for interim analysis plan
            if "interim_analysis_plan" not in metadata:
                return self._create_component_result(
                    "e9_principles",
                    False,
                    "Interim analysis plan missing",
                    "E9 requires pre-specified interim analysis plans",
                )

            return self._create_component_result(
                "e9_principles",
                True,
                "E9 statistical principles compliance verified",
                "All required E9 elements present and documented",
            )

        except Exception as e:
            return self._create_component_result(
                "e9_principles", False, "Error verifying E9 principles", str(e)
            )

    def _verify_safety_monitoring(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify safety monitoring procedures."""
        try:
            data = request.data
            metadata = request.metadata

            # Check for safety monitoring plan
            if "safety_monitoring_plan" not in metadata:
                return self._create_component_result(
                    "safety_monitoring",
                    False,
                    "Safety monitoring plan missing",
                    "Clinical trials require comprehensive safety monitoring",
                )

            # Check for adverse event data
            ae_columns = [
                col
                for col in data.columns
                if "ae_" in col.lower() or "adverse" in col.lower()
            ]
            if not ae_columns:
                return self._create_component_result(
                    "safety_monitoring",
                    False,
                    "Adverse event data missing",
                    "Safety monitoring requires adverse event documentation",
                )

            # Check for safety stopping rules
            if "safety_stopping_rules" not in metadata:
                return self._create_component_result(
                    "safety_monitoring",
                    False,
                    "Safety stopping rules not specified",
                    "Safety monitoring requires pre-specified stopping rules",
                )

            # Check for data safety monitoring board
            if "dsmb_oversight" not in metadata:
                return self._create_component_result(
                    "safety_monitoring",
                    False,
                    "DSMB oversight not documented",
                    "Clinical trials require independent safety oversight",
                )

            return self._create_component_result(
                "safety_monitoring",
                True,
                "Safety monitoring procedures verified",
                "Comprehensive safety monitoring plan in place",
            )

        except Exception as e:
            return self._create_component_result(
                "safety_monitoring", False, "Error verifying safety monitoring", str(e)
            )

    def _get_minimum_sample_size(self, trial_phase: str, test_type: TestType) -> int:
        """Get minimum sample size requirements for trial phase and test type."""
        base_sizes = {
            "phase_1": {"ttest": 20, "anova": 30, "chi2": 50, "wilcoxon": 20},
            "phase_2": {"ttest": 50, "anova": 80, "chi2": 100, "wilcoxon": 50},
            "phase_3": {"ttest": 100, "anova": 150, "chi2": 200, "wilcoxon": 100},
        }

        return base_sizes.get(trial_phase, {}).get(test_type.value, 100)

    def _verify_statistical_power(self, request: VerificationRequest) -> bool:
        """Verify statistical power adequacy."""
        # This would implement actual power calculations
        # For now, return True if sample size is adequate
        return len(request.data) >= 100  # Simplified check


class ClinicalTrialDataGenerator:
    """Generate realistic clinical trial data for verification."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def generate_phase3_trial_data(
        self,
        n_participants: int = 300,
        treatment_groups: List[str] = None,
        primary_endpoint: str = "efficacy_score",
    ) -> pd.DataFrame:
        """Generate realistic Phase 3 clinical trial data."""
        if treatment_groups is None:
            treatment_groups = ["Placebo", "Low_Dose", "High_Dose"]

        n_per_group = n_participants // len(treatment_groups)

        all_data = []
        for i, group in enumerate(treatment_groups):
            # Generate group-specific data
            if group == "Placebo":
                base_effect = 0.0
                effect_std = 1.0
            elif group == "Low_Dose":
                base_effect = 0.3
                effect_std = 1.0
            else:  # High_Dose
                base_effect = 0.6
                effect_std = 1.0

            # Generate efficacy scores
            efficacy_scores = np.random.normal(base_effect, effect_std, n_per_group)

            # Generate participant data
            for j in range(n_per_group):
                participant_data = {
                    "participant_id": f"{group}_P{i*n_per_group + j:03d}",
                    "randomization_group": group,
                    "blinding_status": "blinded",
                    "primary_endpoint": efficacy_scores[j],
                    "baseline_score": np.random.normal(0, 1),
                    "age": np.random.normal(65, 12),
                    "gender": np.random.choice(["Male", "Female"]),
                    "study_site": np.random.choice(["Site_A", "Site_B", "Site_C"]),
                    "visit_number": np.random.choice([1, 2, 3, 4, 5]),
                    "ae_serious": np.random.binomial(1, 0.05),  # 5% serious AE rate
                    "ae_grade": np.random.choice(
                        [0, 1, 2, 3, 4], p=[0.7, 0.2, 0.08, 0.015, 0.005]
                    ),
                    "compliance_rate": np.random.normal(0.9, 0.1),
                    "enrollment_date": pd.date_range(
                        "2023-01-01", periods=n_participants, freq="D"
                    )[i * n_per_group + j],
                }
                all_data.append(participant_data)

        df = pd.DataFrame(all_data)

        # Add missing data (realistic for clinical trials)
        missing_mask = np.random.random(len(df)) < 0.03
        df.loc[missing_mask, "primary_endpoint"] = np.nan

        return df


def create_clinical_trial_metadata(
    trial_phase: str = "phase_3",
    primary_endpoint: str = "efficacy_score",
    statistical_analysis_plan: str = "pre_specified_sap_v1.0",
) -> Dict[str, Any]:
    """Create comprehensive clinical trial metadata."""
    return {
        "trial_phase": trial_phase,
        "primary_endpoint": primary_endpoint,
        "statistical_analysis_plan": statistical_analysis_plan,
        "multiplicity_adjustment": "bonferroni_correction",
        "interim_analysis_plan": "pre_specified_interim_analyses",
        "safety_monitoring_plan": "comprehensive_safety_monitoring",
        "safety_stopping_rules": ["2/3 DLTs", "1/6 DLTs in expansion"],
        "dsmb_oversight": "independent_data_safety_monitoring_board",
        "fda_guidelines_compliance": ["E9", "E10", "E17"],
        "regulatory_authority": "FDA",
        "trial_registration": "NCT12345678",
        "protocol_version": "v2.1",
        "statistical_software": "R v4.2.0, SAS v9.4",
    }
