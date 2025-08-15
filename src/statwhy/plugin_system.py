#!/usr/bin/env python3
"""
StatWhy Plugin System Architecture

Implements a comprehensive plugin system for researchers to contribute verified
statistical procedures, including marketplace and incentivization features.
"""

import logging
import json
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import importlib.util
import inspect

from .models import TestType, ApplicationCategory
from .exceptions import PluginError, DataValidationError


logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a StatWhy plugin."""

    name: str
    version: str
    description: str
    author: str
    author_email: str
    license: str
    homepage: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None

    # Technical specifications
    python_version: str = ">=3.8"
    dependencies: List[str] = None
    entry_point: str = "main"

    # Statistical capabilities
    supported_tests: List[TestType] = None
    supported_categories: List[ApplicationCategory] = None

    # Quality metrics
    verification_status: str = "pending"
    test_coverage: float = 0.0
    documentation_score: float = 0.0
    code_quality_score: float = 0.0

    # Marketplace information
    downloads: int = 0
    rating: float = 0.0
    reviews: int = 0
    last_updated: datetime = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.supported_tests is None:
            self.supported_tests = []
        if self.supported_categories is None:
            self.supported_categories = []
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


@dataclass
class PluginValidationResult:
    """Result of plugin validation."""

    plugin_name: str
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    verification_score: float = 0.0
    test_results: Dict[str, bool] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.test_results is None:
            self.test_results = {}


class PluginManager:
    """
    Manages StatWhy plugins and the plugin marketplace.

    Handles plugin installation, validation, execution, and marketplace
    operations including incentivization and recognition.
    """

    def __init__(self, plugins_directory: str = "plugins"):
        self.plugins_directory = Path(plugins_directory)
        self.plugins_directory.mkdir(exist_ok=True)

        self.installed_plugins: Dict[str, Any] = {}
        self.plugin_registry: Dict[str, PluginMetadata] = {}
        self.marketplace_plugins: Dict[str, PluginMetadata] = {}

        self._load_installed_plugins()
        self._load_marketplace_plugins()

    def _load_installed_plugins(self) -> None:
        """Load all installed plugins from the plugins directory."""
        try:
            for plugin_dir in self.plugins_directory.iterdir():
                if plugin_dir.is_dir() and (plugin_dir / "__init__.py").exists():
                    self._load_plugin(plugin_dir)
        except Exception as e:
            logger.error(f"Error loading installed plugins: {e}")

    def _load_plugin(self, plugin_dir: Path) -> None:
        """Load a single plugin from directory."""
        try:
            # Load plugin metadata
            metadata_file = plugin_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                plugin_metadata = PluginMetadata(**metadata)
                self.plugin_registry[plugin_metadata.name] = plugin_metadata

                # Load plugin module
                spec = importlib.util.spec_from_file_location(
                    plugin_metadata.name,
                    plugin_dir / f"{plugin_metadata.entry_point}.py",
                )

                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.installed_plugins[plugin_metadata.name] = module

                    logger.info(
                        f"Loaded plugin: {plugin_metadata.name} v{plugin_metadata.version}"
                    )

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_dir}: {e}")

    def _load_marketplace_plugins(self) -> None:
        """Load marketplace plugin information."""
        try:
            marketplace_file = self.plugins_directory / "marketplace.json"
            if marketplace_file.exists():
                with open(marketplace_file, "r") as f:
                    marketplace_data = json.load(f)

                for plugin_data in marketplace_data.get("plugins", []):
                    metadata = PluginMetadata(**plugin_data)
                    self.marketplace_plugins[metadata.name] = metadata

        except Exception as e:
            logger.error(f"Error loading marketplace plugins: {e}")

    def install_plugin(self, plugin_path: Union[str, Path]) -> bool:
        """
        Install a plugin from a file or directory.

        Args:
            plugin_path: Path to plugin file or directory

        Returns:
            True if installation successful, False otherwise
        """
        try:
            plugin_path = Path(plugin_path)

            if plugin_path.is_file() and plugin_path.suffix == ".zip":
                return self._install_from_zip(plugin_path)
            elif plugin_path.is_dir():
                return self._install_from_directory(plugin_path)
            else:
                raise PluginError(f"Unsupported plugin format: {plugin_path}")

        except Exception as e:
            logger.error(f"Error installing plugin: {e}")
            return False

    def _install_from_zip(self, zip_path: Path) -> bool:
        """Install plugin from ZIP file."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                # Extract to temporary directory
                temp_dir = self.plugins_directory / f"temp_{zip_path.stem}"
                temp_dir.mkdir(exist_ok=True)

                zip_file.extractall(temp_dir)

                # Find plugin directory
                plugin_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
                if not plugin_dirs:
                    raise PluginError("No plugin directory found in ZIP")

                plugin_dir = plugin_dirs[0]

                # Validate plugin
                if not self._validate_plugin_structure(plugin_dir):
                    raise PluginError("Invalid plugin structure")

                # Install plugin
                plugin_name = plugin_dir.name
                target_dir = self.plugins_directory / plugin_name

                if target_dir.exists():
                    import shutil

                    shutil.rmtree(target_dir)

                import shutil

                shutil.move(str(plugin_dir), str(target_dir))

                # Clean up
                import shutil

                shutil.rmtree(temp_dir)

                # Reload plugins
                self._load_installed_plugins()

                logger.info(f"Plugin installed successfully: {plugin_name}")
                return True

        except Exception as e:
            logger.error(f"Error installing from ZIP: {e}")
            return False

    def _install_from_directory(self, plugin_dir: Path) -> bool:
        """Install plugin from directory."""
        try:
            # Validate plugin structure
            if not self._validate_plugin_structure(plugin_dir):
                raise PluginError("Invalid plugin structure")

            # Copy to plugins directory
            plugin_name = plugin_dir.name
            target_dir = self.plugins_directory / plugin_name

            if target_dir.exists():
                import shutil

                shutil.rmtree(target_dir)

            import shutil

            shutil.copytree(plugin_dir, target_dir)

            # Reload plugins
            self._load_installed_plugins()

            logger.info(f"Plugin installed successfully: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error installing from directory: {e}")
            return False

    def _validate_plugin_structure(self, plugin_dir: Path) -> bool:
        """Validate plugin directory structure."""
        required_files = ["__init__.py", "metadata.json"]

        for file_name in required_files:
            if not (plugin_dir / file_name).exists():
                logger.error(f"Missing required file: {file_name}")
                return False

        # Validate metadata
        try:
            with open(plugin_dir / "metadata.json", "r") as f:
                metadata = json.load(f)

            required_fields = ["name", "version", "description", "author", "license"]
            for field in required_fields:
                if field not in metadata:
                    logger.error(f"Missing required metadata field: {field}")
                    return False

        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            return False

        return True

    def uninstall_plugin(self, plugin_name: str) -> bool:
        """
        Uninstall a plugin.

        Args:
            plugin_name: Name of plugin to uninstall

        Returns:
            True if uninstallation successful, False otherwise
        """
        try:
            if plugin_name not in self.installed_plugins:
                raise PluginError(f"Plugin not installed: {plugin_name}")

            # Remove plugin directory
            plugin_dir = self.plugins_directory / plugin_name
            if plugin_dir.exists():
                import shutil

                shutil.rmtree(plugin_dir)

            # Remove from registry
            if plugin_name in self.plugin_registry:
                del self.plugin_registry[plugin_name]

            if plugin_name in self.installed_plugins:
                del self.installed_plugins[plugin_name]

            logger.info(f"Plugin uninstalled: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error uninstalling plugin: {e}")
            return False

    def validate_plugin(self, plugin_name: str) -> PluginValidationResult:
        """
        Validate a plugin for quality and compliance.

        Args:
            plugin_name: Name of plugin to validate

        Returns:
            PluginValidationResult with validation details
        """
        try:
            if plugin_name not in self.installed_plugins:
                return PluginValidationResult(
                    plugin_name=plugin_name,
                    is_valid=False,
                    errors=[f"Plugin not installed: {plugin_name}"],
                )

            plugin_module = self.installed_plugins[plugin_name]
            plugin_metadata = self.plugin_registry[plugin_name]

            result = PluginValidationResult(plugin_name=plugin_name, is_valid=True)

            # Validate code structure
            self._validate_code_structure(plugin_module, result)

            # Validate statistical procedures
            self._validate_statistical_procedures(plugin_module, result)

            # Validate documentation
            self._validate_documentation(plugin_module, result)

            # Validate tests
            self._validate_tests(plugin_module, result)

            # Calculate verification score
            result.verification_score = self._calculate_verification_score(result)
            result.is_valid = result.verification_score >= 0.8  # 80% threshold

            # Update plugin metadata
            plugin_metadata.verification_status = (
                "verified" if result.is_valid else "failed"
            )
            plugin_metadata.test_coverage = result.verification_score
            plugin_metadata.code_quality_score = result.verification_score
            plugin_metadata.last_updated = datetime.utcnow()

            return result

        except Exception as e:
            return PluginValidationResult(
                plugin_name=plugin_name,
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
            )

    def _validate_code_structure(
        self, plugin_module: Any, result: PluginValidationResult
    ) -> None:
        """Validate plugin code structure."""
        try:
            # Check for required functions
            required_functions = ["verify", "validate", "get_test_info"]
            for func_name in required_functions:
                if not hasattr(plugin_module, func_name):
                    result.errors.append(f"Missing required function: {func_name}")
                    result.is_valid = False
                elif not callable(getattr(plugin_module, func_name)):
                    result.errors.append(f"Required function not callable: {func_name}")
                    result.is_valid = False

            # Check for proper error handling
            if hasattr(plugin_module, "verify"):
                verify_func = getattr(plugin_module, "verify")
                source = inspect.getsource(verify_func)
                if "try:" in source and "except:" in source:
                    result.test_results["error_handling"] = True
                else:
                    result.warnings.append("Limited error handling detected")
                    result.test_results["error_handling"] = False

        except Exception as e:
            result.errors.append(f"Code structure validation error: {str(e)}")

    def _validate_statistical_procedures(
        self, plugin_module: Any, result: PluginValidationResult
    ) -> None:
        """Validate statistical procedures in plugin."""
        try:
            # Check for statistical test implementations
            if hasattr(plugin_module, "get_test_info"):
                test_info_func = getattr(plugin_module, "get_test_info")
                test_info = test_info_func()

                if isinstance(test_info, dict):
                    required_fields = ["test_type", "assumptions", "verification_steps"]
                    for field in required_fields:
                        if field in test_info:
                            result.test_results[f"test_info_{field}"] = True
                        else:
                            result.warnings.append(f"Missing test info field: {field}")
                            result.test_results[f"test_info_{field}"] = False
                else:
                    result.warnings.append("get_test_info should return a dictionary")
                    result.test_results["test_info_structure"] = False

        except Exception as e:
            result.errors.append(f"Statistical procedure validation error: {str(e)}")

    def _validate_documentation(
        self, plugin_module: Any, result: PluginValidationResult
    ) -> None:
        """Validate plugin documentation."""
        try:
            # Check module docstring
            if plugin_module.__doc__:
                doc_length = len(plugin_module.__doc__.strip())
                if doc_length > 100:
                    result.test_results["module_documentation"] = True
                else:
                    result.warnings.append(
                        "Module documentation could be more detailed"
                    )
                    result.test_results["module_documentation"] = False
            else:
                result.warnings.append("No module documentation found")
                result.test_results["module_documentation"] = False

            # Check function documentation
            if hasattr(plugin_module, "verify"):
                verify_func = getattr(plugin_module, "verify")
                if verify_func.__doc__:
                    result.test_results["function_documentation"] = True
                else:
                    result.warnings.append("verify function lacks documentation")
                    result.test_results["function_documentation"] = False

        except Exception as e:
            result.errors.append(f"Documentation validation error: {str(e)}")

    def _validate_tests(
        self, plugin_module: Any, result: PluginValidationResult
    ) -> None:
        """Validate plugin tests."""
        try:
            # Check for test files
            plugin_dir = self.plugins_directory / result.plugin_name
            test_files = list(plugin_dir.glob("test_*.py")) + list(
                plugin_dir.glob("*_test.py")
            )

            if test_files:
                result.test_results["test_files_present"] = True
                result.test_results["test_coverage"] = len(test_files)
            else:
                result.warnings.append("No test files found")
                result.test_results["test_files_present"] = False
                result.test_results["test_coverage"] = 0

        except Exception as e:
            result.errors.append(f"Test validation error: {str(e)}")

    def _calculate_verification_score(self, result: PluginValidationResult) -> float:
        """Calculate overall verification score."""
        if not result.test_results:
            return 0.0

        passed_tests = sum(1 for passed in result.test_results.values() if passed)
        total_tests = len(result.test_results)

        if total_tests == 0:
            return 0.0

        base_score = passed_tests / total_tests

        # Penalize for errors
        error_penalty = min(0.3, len(result.errors) * 0.1)

        # Penalize for warnings
        warning_penalty = min(0.1, len(result.warnings) * 0.02)

        final_score = max(0.0, base_score - error_penalty - warning_penalty)
        return round(final_score, 3)

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get information about a specific plugin."""
        return self.plugin_registry.get(plugin_name)

    def list_installed_plugins(self) -> List[PluginMetadata]:
        """List all installed plugins."""
        return list(self.plugin_registry.values())

    def list_marketplace_plugins(self) -> List[PluginMetadata]:
        """List all marketplace plugins."""
        return list(self.marketplace_plugins.values())

    def search_plugins(
        self, query: str, category: Optional[ApplicationCategory] = None
    ) -> List[PluginMetadata]:
        """Search for plugins by query and category."""
        results = []

        for plugin in self.marketplace_plugins.values():
            # Text search
            if (
                query.lower() in plugin.name.lower()
                or query.lower() in plugin.description.lower()
            ):
                if category is None or category in plugin.supported_categories:
                    results.append(plugin)

        # Sort by rating and downloads
        results.sort(key=lambda x: (x.rating, x.downloads), reverse=True)
        return results

    def rate_plugin(self, plugin_name: str, rating: float, review: str = "") -> bool:
        """
        Rate a plugin and provide review.

        Args:
            plugin_name: Name of plugin to rate
            rating: Rating from 1.0 to 5.0
            review: Optional review text

        Returns:
            True if rating successful, False otherwise
        """
        try:
            if plugin_name not in self.marketplace_plugins:
                raise PluginError(f"Plugin not found in marketplace: {plugin_name}")

            plugin = self.marketplace_plugins[plugin_name]

            # Update rating
            current_rating = plugin.rating
            current_reviews = plugin.reviews

            new_rating = ((current_rating * current_reviews) + rating) / (
                current_reviews + 1
            )
            plugin.rating = round(new_rating, 2)
            plugin.reviews += 1

            # Save to marketplace file
            self._save_marketplace_data()

            logger.info(f"Plugin rated: {plugin_name} - {rating}/5.0")
            return True

        except Exception as e:
            logger.error(f"Error rating plugin: {e}")
            return False

    def download_plugin(self, plugin_name: str) -> bool:
        """
        Record a plugin download.

        Args:
            plugin_name: Name of plugin downloaded

        Returns:
            True if download recorded, False otherwise
        """
        try:
            if plugin_name in self.marketplace_plugins:
                plugin = self.marketplace_plugins[plugin_name]
                plugin.downloads += 1
                self._save_marketplace_data()

            if plugin_name in self.plugin_registry:
                plugin = self.plugin_registry[plugin_name]
                plugin.downloads += 1

            logger.info(f"Plugin download recorded: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error recording download: {e}")
            return False

    def _save_marketplace_data(self) -> None:
        """Save marketplace data to file."""
        try:
            marketplace_data = {
                "last_updated": datetime.utcnow().isoformat(),
                "plugins": [
                    asdict(plugin) for plugin in self.marketplace_plugins.values()
                ],
            }

            marketplace_file = self.plugins_directory / "marketplace.json"
            with open(marketplace_file, "w") as f:
                json.dump(marketplace_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving marketplace data: {e}")

    def get_plugin_contributor_stats(self, author: str) -> Dict[str, Any]:
        """
        Get statistics for a plugin contributor.

        Args:
            author: Author name to get stats for

        Returns:
            Dictionary with contributor statistics
        """
        try:
            author_plugins = [
                plugin
                for plugin in self.marketplace_plugins.values()
                if plugin.author == author
            ]

            if not author_plugins:
                return {"error": "No plugins found for author"}

            total_downloads = sum(plugin.downloads for plugin in author_plugins)
            avg_rating = sum(plugin.rating for plugin in author_plugins) / len(
                author_plugins
            )
            total_reviews = sum(plugin.reviews for plugin in author_plugins)

            stats = {
                "author": author,
                "total_plugins": len(author_plugins),
                "total_downloads": total_downloads,
                "average_rating": round(avg_rating, 2),
                "total_reviews": total_reviews,
                "plugins": [
                    {
                        "name": plugin.name,
                        "version": plugin.version,
                        "downloads": plugin.downloads,
                        "rating": plugin.rating,
                        "reviews": plugin.reviews,
                    }
                    for plugin in author_plugins
                ],
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting contributor stats: {e}")
            return {"error": str(e)}

    def generate_plugin_template(
        self, plugin_name: str, author: str, description: str
    ) -> Dict[str, str]:
        """
        Generate a plugin template for new contributors.

        Args:
            plugin_name: Name for the new plugin
            author: Author name
            description: Plugin description

        Returns:
            Dictionary with template files
        """
        try:
            # Generate plugin directory structure
            plugin_dir = plugin_name.lower().replace(" ", "_")

            # Main plugin file
            main_plugin = f'''#!/usr/bin/env python3
"""
{plugin_name} - StatWhy Plugin

{description}

Author: {author}
"""

import logging
from typing import Dict, List, Any
from .models import TestType, VerificationRequest, ComponentResult

logger = logging.getLogger(__name__)


class {plugin_name.replace(" ", "")}Verifier:
    """Verifier for {plugin_name} statistical procedures."""
    
    def __init__(self):
        self.name = "{plugin_name}"
        self.version = "1.0.0"
    
    def verify(self, request: VerificationRequest) -> List[ComponentResult]:
        """
        Verify {plugin_name} statistical procedures.
        
        Args:
            request: Verification request
            
        Returns:
            List of component results
        """
        # TODO: Implement verification logic
        pass
    
    def validate(self, data: Any) -> bool:
        """
        Validate data for {plugin_name} procedures.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement validation logic
        pass
    
    def get_test_info(self) -> Dict[str, Any]:
        """
        Get information about supported tests.
        
        Returns:
            Dictionary with test information
        """
        return {{
            "test_type": "custom",
            "assumptions": ["Assumption 1", "Assumption 2"],
            "verification_steps": ["Step 1", "Step 2"],
            "description": "{description}"
        }}
'''

            # Init file
            init_file = f'''"""
{plugin_name} Plugin Package
"""

__version__ = "1.0.0"
__author__ = "{author}"
'''

            # Metadata file
            metadata = {
                {
                    "name": plugin_name,
                    "version": "1.0.0",
                    "description": description,
                    "author": author,
                    "author_email": "author@example.com",
                    "license": "MIT",
                    "python_version": ">=3.8",
                    "dependencies": [],
                    "entry_point": "main",
                    "supported_tests": [],
                    "supported_categories": ["research"],
                }
            }

            # Test file
            test_file = f'''#!/usr/bin/env python3
"""
Tests for {plugin_name} plugin.
"""

import unittest
from .main import {plugin_name.replace(" ", "")}Verifier


class Test{plugin_name.replace(" ", "")}Verifier(unittest.TestCase):
    """Test cases for {plugin_name}Verifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = {plugin_name.replace(" ", "")}Verifier()
    
    def test_verifier_initialization(self):
        """Test verifier initialization."""
        self.assertEqual(self.verifier.name, "{plugin_name}")
        self.assertEqual(self.verifier.version, "1.0.0")
    
    def test_get_test_info(self):
        """Test get_test_info method."""
        test_info = self.verifier.get_test_info()
        self.assertIsInstance(test_info, dict)
        self.assertIn("test_type", test_info)
        self.assertIn("assumptions", test_info)
        self.assertIn("verification_steps", test_info)


if __name__ == "__main__":
    unittest.main()
'''

            # README file
            readme_file = f"""# {plugin_name}

{description}

## Installation

1. Copy this plugin to your StatWhy plugins directory
2. Restart StatWhy
3. The plugin will be automatically loaded

## Usage

```python
from statwhy.plugins.{plugin_dir} import {plugin_name.replace(" ", "")}Verifier

verifier = {plugin_name.replace(" ", "")}Verifier()
# Use verifier methods...
```

## Development

This plugin template includes:
- Main verification logic
- Data validation
- Test information
- Unit tests
- Documentation

## License

{metadata["license"]}

## Author

{author}
"""

            return {
                {
                    "main_plugin": main_plugin,
                    "init_file": init_file,
                    "metadata": json.dumps(metadata, indent=2),
                    "test_file": test_file,
                    "readme": readme_file,
                    "plugin_dir": plugin_dir,
                }
            }

        except Exception as e:
            logger.error(f"Error generating plugin template: {e}")
            return {{"error": str(e)}}


class PluginMarketplace:
    """Manages the StatWhy plugin marketplace."""

    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.featured_plugins: List[str] = []
        self.categories: Dict[str, List[str]] = {}

    def get_featured_plugins(self) -> List[PluginMetadata]:
        """Get featured plugins for the marketplace."""
        featured = []
        for plugin_name in self.featured_plugins:
            if plugin_name in self.plugin_manager.marketplace_plugins:
                featured.append(self.plugin_manager.marketplace_plugins[plugin_name])
        return featured

    def get_plugins_by_category(
        self, category: ApplicationCategory
    ) -> List[PluginMetadata]:
        """Get plugins by application category."""
        return [
            plugin
            for plugin in self.plugin_manager.marketplace_plugins.values()
            if category in plugin.supported_categories
        ]

    def get_top_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """Get top-rated plugins."""
        plugins = list(self.plugin_manager.marketplace_plugins.values())
        plugins.sort(key=lambda x: (x.rating, x.downloads), reverse=True)
        return plugins[:limit]

    def get_newest_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """Get newest plugins."""
        plugins = list(self.plugin_manager.marketplace_plugins.values())
        plugins.sort(key=lambda x: x.last_updated, reverse=True)
        return plugins[:limit]

    def get_most_downloaded_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """Get most downloaded plugins."""
        plugins = list(self.plugin_manager.marketplace_plugins.values())
        plugins.sort(key=lambda x: x.downloads, reverse=True)
        return plugins[:limit]


class PluginIncentivization:
    """Manages plugin contributor incentivization and recognition."""

    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.contributor_rewards: Dict[str, Dict[str, Any]] = {}

    def calculate_contributor_score(self, author: str) -> float:
        """Calculate contributor score based on plugin quality and usage."""
        try:
            author_plugins = [
                plugin
                for plugin in self.plugin_manager.marketplace_plugins.values()
                if plugin.author == author
            ]

            if not author_plugins:
                return 0.0

            # Quality score (verification status, test coverage)
            quality_score = sum(
                plugin.test_coverage for plugin in author_plugins
            ) / len(author_plugins)

            # Usage score (downloads, ratings)
            total_downloads = sum(plugin.downloads for plugin in author_plugins)
            avg_rating = sum(plugin.rating for plugin in author_plugins) / len(
                author_plugins
            )

            usage_score = min(1.0, (total_downloads / 1000) * (avg_rating / 5.0))

            # Combined score
            combined_score = (quality_score * 0.7) + (usage_score * 0.3)
            return round(combined_score, 3)

        except Exception as e:
            logger.error(f"Error calculating contributor score: {e}")
            return 0.0

    def get_contributor_badges(self, author: str) -> List[str]:
        """Get badges for a contributor."""
        try:
            score = self.calculate_contributor_score(author)
            badges = []

            if score >= 0.9:
                badges.append("🏆 Master Contributor")
            elif score >= 0.8:
                badges.append("🥇 Gold Contributor")
            elif score >= 0.7:
                badges.append("🥈 Silver Contributor")
            elif score >= 0.6:
                badges.append("🥉 Bronze Contributor")
            elif score >= 0.5:
                badges.append("🌟 Rising Star")

            # Special badges
            author_plugins = [
                plugin
                for plugin in self.plugin_manager.marketplace_plugins.values()
                if plugin.author == author
            ]

            if len(author_plugins) >= 5:
                badges.append("📚 Prolific Author")

            total_downloads = sum(plugin.downloads for plugin in author_plugins)
            if total_downloads >= 10000:
                badges.append("🚀 High Impact")

            return badges

        except Exception as e:
            logger.error(f"Error getting contributor badges: {e}")
            return []

    def generate_citation(self, plugin_name: str, format_type: str = "bibtex") -> str:
        """Generate citation for a plugin."""
        try:
            if plugin_name not in self.plugin_manager.marketplace_plugins:
                raise PluginError(f"Plugin not found: {plugin_name}")

            plugin = self.plugin_manager.marketplace_plugins[plugin_name]

            if format_type == "bibtex":
                return f"""@software{{{plugin.author.replace(" ", "_").lower()}_{plugin.name.lower().replace(" ", "_")},
  title = {{{plugin.name} - StatWhy Plugin}},
  author = {{{plugin.author}}},
  year = {{{plugin.last_updated.year}}},
  url = {{{plugin.homepage or "https://statwhy.org"}}},
  note = {{StatWhy Plugin for {plugin.description.lower()}}}
}}"""

            elif format_type == "apa":
                return f"{plugin.author}. ({plugin.last_updated.year}). {plugin.name} - StatWhy Plugin. Retrieved from {plugin.homepage or 'https://statwhy.org'}"

            else:
                raise ValueError(f"Unsupported citation format: {format_type}")

        except Exception as e:
            logger.error(f"Error generating citation: {e}")
            return f"Error generating citation: {str(e)}"

    def get_contributor_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get contributor leaderboard."""
        try:
            contributors = {}

            for plugin in self.plugin_manager.marketplace_plugins.values():
                if plugin.author not in contributors:
                    contributors[plugin.author] = {
                        "author": plugin.author,
                        "plugins": [],
                        "total_downloads": 0,
                        "total_rating": 0,
                        "total_reviews": 0,
                    }

                contributors[plugin.author]["plugins"].append(plugin.name)
                contributors[plugin.author]["total_downloads"] += plugin.downloads
                contributors[plugin.author]["total_rating"] += plugin.rating
                contributors[plugin.author]["total_reviews"] += plugin.reviews

            # Calculate scores and add badges
            leaderboard = []
            for author, data in contributors.items():
                score = self.calculate_contributor_score(author)
                badges = self.get_contributor_badges(author)

                leaderboard.append(
                    {
                        **data,
                        "score": score,
                        "badges": badges,
                        "avg_rating": round(
                            data["total_rating"] / len(data["plugins"]), 2
                        ),
                    }
                )

            # Sort by score
            leaderboard.sort(key=lambda x: x["score"], reverse=True)
            return leaderboard[:limit]

        except Exception as e:
            logger.error(f"Error getting contributor leaderboard: {e}")
            return []
