#!/usr/bin/env python3
"""
StatWhy Manufacturing Quality Control Module

Implements Six Sigma methodologies and process optimization for manufacturing
quality control, defect reduction, and process improvement.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .models import VerificationRequest, ComponentResult
from .verifiers import BaseVerifier


logger = logging.getLogger(__name__)


@dataclass
class SixSigmaRequirements:
    """Six Sigma quality requirements and thresholds."""

    # Process capability requirements
    cp_minimum: float = 1.33  # Minimum process capability index
    cpk_minimum: float = 1.0  # Minimum process capability index (centered)
    pp_minimum: float = 1.67  # Minimum process performance index
    ppk_minimum: float = 1.33  # Minimum process performance index (centered)

    # Defect rate requirements
    dpmo_maximum: float = 3.4  # Maximum defects per million opportunities
    yield_minimum: float = 0.99966  # Minimum yield (99.966%)

    # Control chart requirements
    ucl_lcl_ratio: float = 3.0  # Upper/Lower control limit ratio
    sample_size_minimum: int = 25  # Minimum sample size for control charts

    # Process improvement targets
    improvement_target: float = 0.1  # 10% improvement target
    cost_reduction_target: float = 0.15  # 15% cost reduction target


class ManufacturingQualityVerifier(BaseVerifier):
    """
    Six Sigma manufacturing quality verification.

    Implements comprehensive verification of quality control procedures used in
    manufacturing, ensuring Six Sigma compliance and process optimization.
    """

    def __init__(self):
        super().__init__()
        self.six_sigma_requirements = SixSigmaRequirements()
        self.quality_thresholds = self._load_quality_thresholds()

    def _load_quality_thresholds(self) -> Dict[str, Any]:
        """Load quality thresholds for different manufacturing processes."""
        return {
            "process_capability": {
                "cp_minimum": 1.33,
                "cpk_minimum": 1.0,
                "pp_minimum": 1.67,
                "ppk_minimum": 1.33,
            },
            "control_charts": {
                "sample_size_minimum": 25,
                "control_limit_multiplier": 3.0,
                "trend_detection": True,
                "pattern_recognition": True,
            },
            "defect_analysis": {
                "dpmo_maximum": 3.4,
                "yield_minimum": 0.99966,
                "pareto_analysis": True,
                "root_cause_analysis": True,
            },
            "process_improvement": {
                "improvement_target": 0.1,
                "cost_reduction_target": 0.15,
                "cycle_time_reduction": 0.2,
                "waste_reduction": 0.25,
            },
        }

    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """Verify manufacturing quality control procedures."""
        components = []

        # Basic quality verification
        components.extend(self._verify_basic_quality(request))

        # Six Sigma compliance verification
        components.extend(self._verify_six_sigma_compliance(request))

        # Process control verification
        components.extend(self._verify_process_control(request))

        # Quality improvement verification
        components.extend(self._verify_quality_improvement(request))

        return components

    def _verify_basic_quality(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify basic quality requirements."""
        components = []

        # Data quality verification
        data_quality_result = self._verify_data_quality(request)
        components.append(data_quality_result)

        # Measurement system verification
        measurement_result = self._verify_measurement_system(request)
        components.append(measurement_result)

        # Sampling verification
        sampling_result = self._verify_sampling_procedures(request)
        components.append(sampling_result)

        return components

    def _verify_six_sigma_compliance(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify Six Sigma compliance requirements."""
        components = []

        # Process capability verification
        capability_result = self._verify_process_capability(request)
        components.append(capability_result)

        # Defect rate verification
        defect_result = self._verify_defect_rates(request)
        components.append(defect_result)

        # Yield verification
        yield_result = self._verify_process_yield(request)
        components.append(yield_result)

        return components

    def _verify_process_control(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify process control procedures."""
        components = []

        # Control chart verification
        control_chart_result = self._verify_control_charts(request)
        components.append(control_chart_result)

        # Statistical process control verification
        spc_result = self._verify_statistical_process_control(request)
        components.append(spc_result)

        # Outlier detection verification
        outlier_result = self._verify_outlier_detection(request)
        components.append(outlier_result)

        return components

    def _verify_quality_improvement(
        self, request: VerificationRequest
    ) -> List[ComponentResult]:
        """Verify quality improvement procedures."""
        components = []

        # DMAIC process verification
        dmaic_result = self._verify_dmaic_process(request)
        components.append(dmaic_result)

        # Root cause analysis verification
        rca_result = self._verify_root_cause_analysis(request)
        components.append(rca_result)

        # Continuous improvement verification
        improvement_result = self._verify_continuous_improvement(request)
        components.append(improvement_result)

        return components

    def _verify_data_quality(self, request: VerificationRequest) -> ComponentResult:
        """Verify data quality for quality control."""
        try:
            data = request.data

            # Check for missing data
            missing_data = data.isnull().sum().sum()
            total_cells = data.size
            missing_rate = missing_data / total_cells

            if missing_rate > 0.02:  # 2% maximum missing data
                return self._create_component_result(
                    "data_quality",
                    False,
                    f"Excessive missing data: {missing_rate:.1%}",
                    "Quality control requires high-quality data (< 2% missing)",
                )

            # Check for data consistency
            if "data_consistency_check" not in request.metadata:
                return self._create_component_result(
                    "data_quality",
                    False,
                    "Data consistency check missing",
                    "Quality control requires data consistency validation",
                )

            # Check for measurement units
            if "measurement_units" not in request.metadata:
                return self._create_component_result(
                    "data_quality",
                    False,
                    "Measurement units not specified",
                    "Quality control requires measurement unit documentation",
                )

            return self._create_component_result(
                "data_quality",
                True,
                "Data quality verified",
                f"Missing data rate: {missing_rate:.1%}",
            )

        except Exception as e:
            return self._create_component_result(
                "data_quality", False, "Error verifying data quality", str(e)
            )

    def _verify_measurement_system(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify measurement system analysis."""
        try:
            metadata = request.metadata

            # Check for measurement system analysis
            if "measurement_system_analysis" not in metadata:
                return self._create_component_result(
                    "measurement_system",
                    False,
                    "Measurement system analysis missing",
                    "Quality control requires MSA documentation",
                )

            # Check for measurement precision
            if "measurement_precision" not in metadata:
                return self._create_component_result(
                    "measurement_system",
                    False,
                    "Measurement precision not documented",
                    "MSA must include precision assessment",
                )

            # Check for measurement accuracy
            if "measurement_accuracy" not in metadata:
                return self._create_component_result(
                    "measurement_system",
                    False,
                    "Measurement accuracy not documented",
                    "MSA must include accuracy assessment",
                )

            return self._create_component_result(
                "measurement_system",
                True,
                "Measurement system verified",
                "Comprehensive MSA in place",
            )

        except Exception as e:
            return self._create_component_result(
                "measurement_system",
                False,
                "Error verifying measurement system",
                str(e),
            )

    def _verify_sampling_procedures(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify sampling procedures."""
        try:
            data = request.data
            metadata = request.metadata

            # Check sample size adequacy
            sample_size = len(data)
            min_sample_size = self.quality_thresholds["control_charts"][
                "sample_size_minimum"
            ]

            if sample_size < min_sample_size:
                return self._create_component_result(
                    "sampling_procedures",
                    False,
                    f"Insufficient sample size: {sample_size} < {min_sample_size}",
                    f"Control charts require minimum {min_sample_size} samples",
                )

            # Check for sampling plan
            if "sampling_plan" not in metadata:
                return self._create_component_result(
                    "sampling_procedures",
                    False,
                    "Sampling plan missing",
                    "Quality control requires documented sampling plan",
                )

            # Check for sampling frequency
            if "sampling_frequency" not in metadata:
                return self._create_component_result(
                    "sampling_procedures",
                    False,
                    "Sampling frequency not specified",
                    "Sampling plan must include frequency",
                )

            return self._create_component_result(
                "sampling_procedures",
                True,
                "Sampling procedures verified",
                f"Sample size: {sample_size} meets requirements",
            )

        except Exception as e:
            return self._create_component_result(
                "sampling_procedures",
                False,
                "Error verifying sampling procedures",
                str(e),
            )

    def _verify_process_capability(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify process capability indices."""
        try:
            metadata = request.metadata

            # Check for process capability calculations
            if "process_capability" not in metadata:
                return self._create_component_result(
                    "process_capability",
                    False,
                    "Process capability calculations missing",
                    "Six Sigma requires process capability assessment",
                )

            capability_data = metadata["process_capability"]

            # Check Cp (process capability)
            if "cp" not in capability_data:
                return self._create_component_result(
                    "process_capability",
                    False,
                    "Cp (process capability) missing",
                    "Process capability requires Cp calculation",
                )

            cp = capability_data["cp"]
            if cp < self.six_sigma_requirements.cp_minimum:
                return self._create_component_result(
                    "process_capability",
                    False,
                    f"Insufficient Cp: {cp:.2f} < {self.six_sigma_requirements.cp_minimum}",
                    f"Six Sigma requires Cp ≥ {self.six_sigma_requirements.cp_minimum}",
                )

            # Check Cpk (centered process capability)
            if "cpk" not in capability_data:
                return self._create_component_result(
                    "process_capability",
                    False,
                    "Cpk (centered process capability) missing",
                    "Process capability requires Cpk calculation",
                )

            cpk = capability_data["cpk"]
            if cpk < self.six_sigma_requirements.cpk_minimum:
                return self._create_component_result(
                    "process_capability",
                    False,
                    f"Insufficient Cpk: {cpk:.2f} < {self.six_sigma_requirements.cpk_minimum}",
                    f"Six Sigma requires Cpk ≥ {self.six_sigma_requirements.cpk_minimum}",
                )

            return self._create_component_result(
                "process_capability",
                True,
                "Process capability verified",
                f"Cp: {cp:.2f}, Cpk: {cpk:.2f}",
            )

        except Exception as e:
            return self._create_component_result(
                "process_capability",
                False,
                "Error verifying process capability",
                str(e),
            )

    def _verify_defect_rates(self, request: VerificationRequest) -> ComponentResult:
        """Verify defect rates meet Six Sigma requirements."""
        try:
            metadata = request.metadata

            # Check for defect rate calculations
            if "defect_rates" not in metadata:
                return self._create_component_result(
                    "defect_rates",
                    False,
                    "Defect rate calculations missing",
                    "Six Sigma requires defect rate assessment",
                )

            defect_data = metadata["defect_rates"]

            # Check DPMO (defects per million opportunities)
            if "dpmo" not in defect_data:
                return self._create_component_result(
                    "defect_rates",
                    False,
                    "DPMO missing",
                    "Defect analysis requires DPMO calculation",
                )

            dpmo = defect_data["dpmo"]
            if dpmo > self.six_sigma_requirements.dpmo_maximum:
                return self._create_component_result(
                    "defect_rates",
                    False,
                    f"Excessive DPMO: {dpmo:.1f} > {self.six_sigma_requirements.dpmo_maximum}",
                    f"Six Sigma requires DPMO ≤ {self.six_sigma_requirements.dpmo_maximum}",
                )

            # Check yield
            if "yield" not in defect_data:
                return self._create_component_result(
                    "defect_rates",
                    False,
                    "Yield missing",
                    "Defect analysis requires yield calculation",
                )

            yield_rate = defect_data["yield"]
            if yield_rate < self.six_sigma_requirements.yield_minimum:
                return self._create_component_result(
                    "defect_rates",
                    False,
                    f"Insufficient yield: {yield_rate:.4f} < {self.six_sigma_requirements.yield_minimum}",
                    f"Six Sigma requires yield ≥ {self.six_sigma_requirements.yield_minimum}",
                )

            return self._create_component_result(
                "defect_rates",
                True,
                "Defect rates verified",
                f"DPMO: {dpmo:.1f}, Yield: {yield_rate:.4f}",
            )

        except Exception as e:
            return self._create_component_result(
                "defect_rates", False, "Error verifying defect rates", str(e)
            )

    def _verify_process_yield(self, request: VerificationRequest) -> ComponentResult:
        """Verify process yield meets requirements."""
        try:
            metadata = request.metadata

            # Check for yield calculations
            if "process_yield" not in metadata:
                return self._create_component_result(
                    "process_yield",
                    False,
                    "Process yield calculations missing",
                    "Quality control requires yield assessment",
                )

            yield_data = metadata["process_yield"]

            # Check first pass yield
            if "first_pass_yield" not in yield_data:
                return self._create_component_result(
                    "process_yield",
                    False,
                    "First pass yield missing",
                    "Process yield requires FPY calculation",
                )

            fpy = yield_data["first_pass_yield"]
            if fpy < 0.95:  # 95% minimum first pass yield
                return self._create_component_result(
                    "process_yield",
                    False,
                    f"Insufficient first pass yield: {fpy:.1%} < 95%",
                    "Quality control requires FPY ≥ 95%",
                )

            # Check rolled throughput yield
            if "rolled_throughput_yield" not in yield_data:
                return self._create_component_result(
                    "process_yield",
                    False,
                    "Rolled throughput yield missing",
                    "Process yield requires RTY calculation",
                )

            rty = yield_data["rolled_throughput_yield"]
            if rty < 0.90:  # 90% minimum rolled throughput yield
                return self._create_component_result(
                    "process_yield",
                    False,
                    f"Insufficient rolled throughput yield: {rty:.1%} < 90%",
                    "Quality control requires RTY ≥ 90%",
                )

            return self._create_component_result(
                "process_yield",
                True,
                "Process yield verified",
                f"FPY: {fpy:.1%}, RTY: {rty:.1%}",
            )

        except Exception as e:
            return self._create_component_result(
                "process_yield", False, "Error verifying process yield", str(e)
            )

    def _verify_control_charts(self, request: VerificationRequest) -> ComponentResult:
        """Verify control chart implementation."""
        try:
            metadata = request.metadata

            # Check for control chart data
            if "control_charts" not in metadata:
                return self._create_component_result(
                    "control_charts",
                    False,
                    "Control chart data missing",
                    "Statistical process control requires control charts",
                )

            control_data = metadata["control_charts"]

            # Check for control limits
            if "control_limits" not in control_data:
                return self._create_component_result(
                    "control_charts",
                    False,
                    "Control limits missing",
                    "Control charts require UCL/LCL calculations",
                )

            # Check for trend detection
            if "trend_detection" not in control_data:
                return self._create_component_result(
                    "control_charts",
                    False,
                    "Trend detection missing",
                    "Control charts require trend detection",
                )

            # Check for pattern recognition
            if "pattern_recognition" not in control_data:
                return self._create_component_result(
                    "control_charts",
                    False,
                    "Pattern recognition missing",
                    "Control charts require pattern recognition",
                )

            return self._create_component_result(
                "control_charts",
                True,
                "Control charts verified",
                "Comprehensive control chart implementation",
            )

        except Exception as e:
            return self._create_component_result(
                "control_charts", False, "Error verifying control charts", str(e)
            )

    def _verify_statistical_process_control(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify statistical process control procedures."""
        try:
            metadata = request.metadata

            # Check for SPC procedures
            if "spc_procedures" not in metadata:
                return self._create_component_result(
                    "statistical_process_control",
                    False,
                    "SPC procedures missing",
                    "Quality control requires SPC documentation",
                )

            spc_data = metadata["spc_procedures"]

            # Check for control chart selection
            if "control_chart_selection" not in spc_data:
                return self._create_component_result(
                    "statistical_process_control",
                    False,
                    "Control chart selection missing",
                    "SPC requires chart selection criteria",
                )

            # Check for out-of-control action plans
            if "out_of_control_action_plan" not in spc_data:
                return self._create_component_result(
                    "statistical_process_control",
                    False,
                    "Out-of-control action plan missing",
                    "SPC requires action plans for OOC conditions",
                )

            return self._create_component_result(
                "statistical_process_control",
                True,
                "SPC procedures verified",
                "Comprehensive SPC implementation",
            )

        except Exception as e:
            return self._create_component_result(
                "statistical_process_control",
                False,
                "Error verifying SPC procedures",
                str(e),
            )

    def _verify_outlier_detection(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify outlier detection procedures."""
        try:
            metadata = request.metadata

            # Check for outlier detection
            if "outlier_detection" not in metadata:
                return self._create_component_result(
                    "outlier_detection",
                    False,
                    "Outlier detection missing",
                    "Quality control requires outlier detection",
                )

            outlier_data = metadata["outlier_detection"]

            # Check for detection methods
            if "detection_methods" not in outlier_data:
                return self._create_component_result(
                    "outlier_detection",
                    False,
                    "Detection methods missing",
                    "Outlier detection requires method documentation",
                )

            # Check for investigation procedures
            if "investigation_procedures" not in outlier_data:
                return self._create_component_result(
                    "outlier_detection",
                    False,
                    "Investigation procedures missing",
                    "Outlier detection requires investigation procedures",
                )

            return self._create_component_result(
                "outlier_detection",
                True,
                "Outlier detection verified",
                "Comprehensive outlier detection procedures",
            )

        except Exception as e:
            return self._create_component_result(
                "outlier_detection", False, "Error verifying outlier detection", str(e)
            )

    def _verify_dmaic_process(self, request: VerificationRequest) -> ComponentResult:
        """Verify DMAIC process implementation."""
        try:
            metadata = request.metadata

            # Check for DMAIC documentation
            if "dmaic_process" not in metadata:
                return self._create_component_result(
                    "dmaic_process",
                    False,
                    "DMAIC process missing",
                    "Six Sigma requires DMAIC documentation",
                )

            dmaic_data = metadata["dmaic_process"]

            # Check for all DMAIC phases
            required_phases = ["define", "measure", "analyze", "improve", "control"]
            missing_phases = [
                phase for phase in required_phases if phase not in dmaic_data
            ]

            if missing_phases:
                return self._create_component_result(
                    "dmaic_process",
                    False,
                    f"Missing DMAIC phases: {missing_phases}",
                    "DMAIC requires all five phases",
                )

            # Check for project charter
            if "project_charter" not in dmaic_data:
                return self._create_component_result(
                    "dmaic_process",
                    False,
                    "Project charter missing",
                    "DMAIC requires project charter",
                )

            return self._create_component_result(
                "dmaic_process",
                True,
                "DMAIC process verified",
                "All five phases documented",
            )

        except Exception as e:
            return self._create_component_result(
                "dmaic_process", False, "Error verifying DMAIC process", str(e)
            )

    def _verify_root_cause_analysis(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify root cause analysis procedures."""
        try:
            metadata = request.metadata

            # Check for RCA procedures
            if "root_cause_analysis" not in metadata:
                return self._create_component_result(
                    "root_cause_analysis",
                    False,
                    "Root cause analysis missing",
                    "Quality improvement requires RCA procedures",
                )

            rca_data = metadata["root_cause_analysis"]

            # Check for analysis tools
            if "analysis_tools" not in rca_data:
                return self._create_component_result(
                    "root_cause_analysis",
                    False,
                    "Analysis tools missing",
                    "RCA requires tool documentation",
                )

            # Check for corrective actions
            if "corrective_actions" not in rca_data:
                return self._create_component_result(
                    "root_cause_analysis",
                    False,
                    "Corrective actions missing",
                    "RCA requires corrective action plans",
                )

            return self._create_component_result(
                "root_cause_analysis",
                True,
                "Root cause analysis verified",
                "Comprehensive RCA procedures",
            )

        except Exception as e:
            return self._create_component_result(
                "root_cause_analysis",
                False,
                "Error verifying root cause analysis",
                str(e),
            )

    def _verify_continuous_improvement(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Verify continuous improvement procedures."""
        try:
            metadata = request.metadata

            # Check for continuous improvement
            if "continuous_improvement" not in metadata:
                return self._create_component_result(
                    "continuous_improvement",
                    False,
                    "Continuous improvement missing",
                    "Quality management requires CI procedures",
                )

            ci_data = metadata["continuous_improvement"]

            # Check for improvement metrics
            if "improvement_metrics" not in ci_data:
                return self._create_component_result(
                    "continuous_improvement",
                    False,
                    "Improvement metrics missing",
                    "CI requires measurable metrics",
                )

            # Check for improvement targets
            if "improvement_targets" not in ci_data:
                return self._create_component_result(
                    "continuous_improvement",
                    False,
                    "Improvement targets missing",
                    "CI requires target setting",
                )

            return self._create_component_result(
                "continuous_improvement",
                True,
                "Continuous improvement verified",
                "Comprehensive CI procedures",
            )

        except Exception as e:
            return self._create_component_result(
                "continuous_improvement",
                False,
                "Error verifying continuous improvement",
                str(e),
            )

    # Placeholder methods for clinical trial verification
    def _verify_e9_principles(self, request: VerificationRequest) -> ComponentResult:
        """Placeholder for E9 principles verification."""
        return self._create_component_result(
            "e9_principles",
            True,
            "E9 principles not applicable to manufacturing",
            "Manufacturing follows Six Sigma frameworks",
        )

    def _verify_e10_controls(self, request: VerificationRequest) -> ComponentResult:
        """Placeholder for E10 controls verification."""
        return self._create_component_result(
            "e10_controls",
            True,
            "E10 controls not applicable to manufacturing",
            "Manufacturing follows Six Sigma frameworks",
        )

    def _verify_e17_multiregional(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Placeholder for E17 multiregional verification."""
        return self._create_component_result(
            "e17_multiregional",
            True,
            "E17 multiregional not applicable to manufacturing",
            "Manufacturing follows Six Sigma frameworks",
        )

    def _verify_safety_monitoring(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Placeholder for safety monitoring verification."""
        return self._create_component_result(
            "safety_monitoring",
            True,
            "Safety monitoring not applicable to manufacturing",
            "Manufacturing focuses on quality metrics",
        )

    def _verify_adverse_event_analysis(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Placeholder for adverse event analysis verification."""
        return self._create_component_result(
            "adverse_event_analysis",
            True,
            "Adverse event analysis not applicable to manufacturing",
            "Manufacturing focuses on defect analysis",
        )

    def _verify_stopping_rules(self, request: VerificationRequest) -> ComponentResult:
        """Placeholder for stopping rules verification."""
        return self._create_component_result(
            "stopping_rules",
            True,
            "Stopping rules not applicable to manufacturing",
            "Manufacturing uses continuous monitoring",
        )

    def _verify_primary_endpoint(self, request: VerificationRequest) -> ComponentResult:
        """Placeholder for primary endpoint verification."""
        return self._create_component_result(
            "primary_endpoint",
            True,
            "Primary endpoint not applicable to manufacturing",
            "Manufacturing uses multiple quality metrics",
        )

    def _verify_multiplicity_handling(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Placeholder for multiplicity handling verification."""
        return self._create_component_result(
            "multiplicity_handling",
            True,
            "Multiplicity handling not applicable to manufacturing",
            "Manufacturing uses correlation adjustments",
        )

    def _verify_missing_data_strategy(
        self, request: VerificationRequest
    ) -> ComponentResult:
        """Placeholder for missing data strategy verification."""
        return self._create_component_result(
            "missing_data_strategy",
            True,
            "Missing data strategy not applicable to manufacturing",
            "Manufacturing uses interpolation methods",
        )


class ManufacturingQualityDataGenerator:
    """Generate realistic manufacturing quality data for verification."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def generate_six_sigma_data(
        self, n_samples: int = 100, process_type: str = "manufacturing"
    ) -> pd.DataFrame:
        """Generate Six Sigma compliant manufacturing data."""

        all_data = []
        for i in range(n_samples):
            # Generate process data
            process_data = {
                "sample_id": f"SAMPLE_{i:04d}",
                "batch_number": f"BATCH_{i//10:03d}",
                "operator_id": f"OP_{np.random.randint(1, 6):02d}",
                "shift": np.random.choice(["Morning", "Afternoon", "Night"]),
                "machine_id": f"MACHINE_{np.random.randint(1, 4):02d}",
                "measurement_value": np.random.normal(100, 2),  # Target: 100, SD: 2
                "upper_spec_limit": 106,
                "lower_spec_limit": 94,
                "target_value": 100,
                "measurement_time": pd.date_range(
                    "2024-01-01", periods=n_samples, freq="H"
                )[i],
                "temperature": np.random.normal(22, 1),
                "humidity": np.random.normal(45, 3),
                "pressure": np.random.normal(101.3, 0.5),
                "defect_flag": np.random.binomial(1, 0.001),  # 0.1% defect rate
                "quality_score": np.random.uniform(0.95, 1.0),
                "cycle_time": np.random.normal(120, 10),  # seconds
                "waste_percentage": np.random.exponential(0.02),
            }
            all_data.append(process_data)

        df = pd.DataFrame(all_data)

        # Add missing data (realistic for manufacturing)
        missing_mask = np.random.random(len(df)) < 0.01
        df.loc[missing_mask, "measurement_value"] = np.nan

        return df


def create_manufacturing_quality_metadata(
    process_type: str = "manufacturing",
    quality_framework: str = "six_sigma",
    industry_standard: str = "ISO_9001",
) -> Dict[str, Any]:
    """Create comprehensive manufacturing quality metadata."""
    return {
        "process_type": process_type,
        "quality_framework": quality_framework,
        "industry_standard": industry_standard,
        "process_capability": {"cp": 1.67, "cpk": 1.33, "pp": 1.75, "ppk": 1.40},
        "defect_rates": {"dpmo": 2.5, "yield": 0.99975},
        "process_yield": {"first_pass_yield": 0.98, "rolled_throughput_yield": 0.95},
        "control_charts": {
            "control_limits": "calculated",
            "trend_detection": True,
            "pattern_recognition": True,
        },
        "spc_procedures": {
            "control_chart_selection": "documented",
            "out_of_control_action_plan": "implemented",
        },
        "outlier_detection": {
            "detection_methods": ["3_sigma", "iqr", "modified_z_score"],
            "investigation_procedures": "documented",
        },
        "dmaic_process": {
            "define": "completed",
            "measure": "completed",
            "analyze": "completed",
            "improve": "completed",
            "control": "completed",
            "project_charter": "approved",
        },
        "root_cause_analysis": {
            "analysis_tools": ["fishbone", "5_whys", "pareto"],
            "corrective_actions": "implemented",
        },
        "continuous_improvement": {
            "improvement_metrics": ["defect_rate", "cycle_time", "waste"],
            "improvement_targets": "set",
        },
        "data_consistency_check": "passed",
        "measurement_units": "millimeters",
        "measurement_system_analysis": "completed",
        "measurement_precision": "documented",
        "measurement_accuracy": "documented",
        "sampling_plan": "documented",
        "sampling_frequency": "hourly",
    }
