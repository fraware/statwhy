"""
Core data models for StatWhy.

Defines the structure of verification requests, results, and components
using Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, validator


class TestType(str, Enum):
    """Supported statistical test types."""

    TTEST = "ttest"
    ANOVA = "anova"
    CHI2 = "chi2"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann-whitney"
    KRUSKAL = "kruskal"
    BARTLETT = "bartlett"
    FLIGNER = "fligner"
    DUNNETT = "dunnett"
    TUKEY = "tukey"
    STEEL = "steel"
    STEEL_DWASS = "steel-dwass"
    WILLIAMS = "williams"
    POISSON = "poisson"
    BINOM = "binom"


class ApplicationCategory(str, Enum):
    """Application categories for statistical procedures."""

    CLINICAL = "clinical"
    FINANCIAL = "financial"
    MANUFACTURING = "manufacturing"
    RESEARCH = "research"
    EDUCATION = "education"


class VerificationStatus(str, Enum):
    """Verification status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class ComponentResult(BaseModel):
    """Result of a single verification component."""

    name: str = Field(..., description="Name of the verification component")
    verified: bool = Field(..., description="Whether verification succeeded")
    status: VerificationStatus = Field(..., description="Verification status")
    details: str = Field(..., description="Detailed description of the result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(
        None, description="Execution time in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class VerificationRequest(BaseModel):
    """Request for statistical procedure verification."""

    test_type: TestType = Field(..., description="Type of statistical test to verify")
    data: pd.DataFrame = Field(..., description="Data to be analyzed")
    alpha: float = Field(0.05, ge=0.0, le=1.0, description="Significance level")
    timeout: int = Field(
        300, ge=1, le=3600, description="Verification timeout in seconds"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Test-specific parameters"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("data")
    def validate_data_not_empty(cls, v):
        if v.empty:
            raise ValueError("Data cannot be empty")
        return v


class VerificationResult(BaseModel):
    """Result of a statistical procedure verification."""

    test_type: TestType = Field(..., description="Type of statistical test verified")
    is_verified: bool = Field(..., description="Overall verification success")
    status: VerificationStatus = Field(..., description="Overall verification status")
    components: List[ComponentResult] = Field(
        ..., description="Individual component results"
    )
    execution_time: float = Field(..., description="Total execution time in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Verification timestamp"
    )
    error_message: Optional[str] = Field(
        None, description="Overall error message if failed"
    )
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of verification components."""
        if not self.components:
            return 0.0
        successful = sum(1 for c in self.components if c.verified)
        return successful / len(self.components)

    @property
    def failed_components(self) -> List[ComponentResult]:
        """Get list of failed verification components."""
        return [c for c in self.components if not c.verified]

    @property
    def successful_components(self) -> List[ComponentResult]:
        """Get list of successful verification components."""
        return [c for c in self.components if c.verified]


class Example(BaseModel):
    """Statistical verification example."""

    id: str = Field(..., description="Unique identifier for the example")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed description")
    test_type: TestType = Field(..., description="Statistical test type")
    category: ApplicationCategory = Field(..., description="Application category")
    difficulty: str = Field(
        ..., description="Difficulty level (Beginner/Intermediate/Advanced)"
    )
    data_file: Optional[str] = Field(None, description="Path to example data file")
    code_file: Optional[str] = Field(None, description="Path to example code file")
    documentation: Optional[str] = Field(None, description="Path to documentation")
    tags: List[str] = Field(default_factory=list, description="Search tags")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    author: Optional[str] = Field(None, description="Example author")
    version: str = Field("1.0.0", description="Example version")


class Plugin(BaseModel):
    """StatWhy plugin definition."""

    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: str = Field(..., description="Plugin author")
    license: str = Field(..., description="Plugin license")
    homepage: Optional[str] = Field(None, description="Plugin homepage URL")
    repository: Optional[str] = Field(None, description="Plugin repository URL")
    dependencies: List[str] = Field(
        default_factory=list, description="Plugin dependencies"
    )
    entry_point: str = Field(..., description="Plugin entry point")
    test_types: List[TestType] = Field(..., description="Supported test types")
    categories: List[ApplicationCategory] = Field(
        ..., description="Supported categories"
    )
    installed: bool = Field(False, description="Whether plugin is installed")
    enabled: bool = Field(False, description="Whether plugin is enabled")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Configuration(BaseModel):
    """StatWhy configuration."""

    why3_path: Optional[str] = Field(None, description="Path to Why3 executable")
    cameleer_path: Optional[str] = Field(
        None, description="Path to Cameleer executable"
    )
    default_timeout: int = Field(300, description="Default verification timeout")
    max_memory: int = Field(8192, description="Maximum memory usage in MB")
    parallel_workers: int = Field(4, description="Number of parallel workers")
    cache_enabled: bool = Field(True, description="Enable result caching")
    cache_directory: str = Field(".statwhy_cache", description="Cache directory")
    log_level: str = Field("INFO", description="Logging level")
    output_format: str = Field("table", description="Default output format")
    theme: str = Field("default", description="CLI theme")
    plugins_directory: str = Field("plugins", description="Plugins directory")
    examples_directory: str = Field("examples", description="Examples directory")
    documentation_url: str = Field(
        "https://statwhy.readthedocs.io", description="Documentation URL"
    )


class SystemStatus(BaseModel):
    """System status information."""

    python_version: str = Field(..., description="Python version")
    statwhy_version: str = Field(..., description="StatWhy version")
    why3_installed: bool = Field(..., description="Whether Why3 is installed")
    why3_version: Optional[str] = Field(None, description="Why3 version")
    cameleer_installed: bool = Field(..., description="Whether Cameleer is installed")
    cameleer_version: Optional[str] = Field(None, description="Cameleer version")
    dependencies_ok: bool = Field(
        ..., description="Whether all dependencies are available"
    )
    system_resources: Dict[str, Any] = Field(
        ..., description="System resource information"
    )
    plugins_loaded: List[str] = Field(
        default_factory=list, description="Loaded plugins"
    )
    configuration_valid: bool = Field(..., description="Whether configuration is valid")
    last_check: datetime = Field(
        default_factory=datetime.utcnow, description="Last status check"
    )
