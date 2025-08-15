"""
Custom exceptions for StatWhy.

Provides clear, user-friendly error messages and proper exception hierarchy
for different types of errors that can occur during verification.
"""


class StatWhyError(Exception):
    """Base exception for all StatWhy errors."""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigurationError(StatWhyError):
    """Raised when there's a configuration or setup error."""

    pass


class DataValidationError(StatWhyError):
    """Raised when data validation fails."""

    pass


class UnsupportedFormatError(StatWhyError):
    """Raised when an unsupported file format is encountered."""

    pass


class TestNotSupportedError(StatWhyError):
    """Raised when a statistical test is not supported."""

    pass


class VerificationError(StatWhyError):
    """Raised when verification fails."""

    pass


class TimeoutError(StatWhyError):
    """Raised when verification times out."""

    pass


class PluginError(StatWhyError):
    """Raised when there's an error with a plugin."""

    pass


class CacheError(StatWhyError):
    """Raised when there's an error with the cache system."""

    pass


class SystemResourceError(StatWhyError):
    """Raised when system resources are insufficient."""

    pass


class NetworkError(StatWhyError):
    """Raised when there's a network-related error."""

    pass


class AuthenticationError(StatWhyError):
    """Raised when authentication fails."""

    pass


class PermissionError(StatWhyError):
    """Raised when there are insufficient permissions."""

    pass


class CorruptedDataError(StatWhyError):
    """Raised when data appears to be corrupted."""

    pass


class InsufficientDataError(StatWhyError):
    """Raised when there's insufficient data for analysis."""

    pass


class AssumptionViolationError(StatWhyError):
    """Raised when statistical assumptions are violated."""

    pass


class NumericalError(StatWhyError):
    """Raised when there are numerical computation errors."""

    pass


class MemoryError(StatWhyError):
    """Raised when there's insufficient memory."""

    pass


class DependencyError(StatWhyError):
    """Raised when required dependencies are missing."""

    pass


class VersionCompatibilityError(StatWhyError):
    """Raised when there are version compatibility issues."""

    pass


class LicenseError(StatWhyError):
    """Raised when there are licensing issues."""

    pass


def format_error_message(error: Exception) -> str:
    """
    Format an error message in a user-friendly way.

    Args:
        error: The exception that occurred

    Returns:
        Formatted error message
    """
    if isinstance(error, StatWhyError):
        return str(error)

    # Handle common Python exceptions
    if isinstance(error, FileNotFoundError):
        return f"File not found: {error.filename}"
    elif isinstance(error, PermissionError):
        return f"Permission denied: {error.filename}"
    elif isinstance(error, ValueError):
        return f"Invalid value: {error}"
    elif isinstance(error, TypeError):
        return f"Type error: {error}"
    elif isinstance(error, ImportError):
        return f"Import error: {error}"
    elif isinstance(error, MemoryError):
        return "Insufficient memory to complete the operation"
    elif isinstance(error, KeyboardInterrupt):
        return "Operation cancelled by user"
    else:
        return f"Unexpected error: {error}"


def get_error_suggestions(error: Exception) -> list:
    """
    Get suggestions for fixing common errors.

    Args:
        error: The exception that occurred

    Returns:
        List of suggestion strings
    """
    suggestions = []

    if isinstance(error, ConfigurationError):
        suggestions.extend(
            [
                "Check that Why3 and Cameleer are properly installed",
                "Verify that all Python dependencies are installed",
                "Check your configuration file for syntax errors",
                "Ensure you have sufficient system permissions",
            ]
        )

    elif isinstance(error, DataValidationError):
        suggestions.extend(
            [
                "Check that your data file exists and is readable",
                "Verify that your data contains the required columns",
                "Ensure your data meets the minimum sample size requirements",
                "Check for missing or invalid values in your data",
            ]
        )

    elif isinstance(error, UnsupportedFormatError):
        suggestions.extend(
            [
                "Convert your data to a supported format (CSV, Excel, JSON)",
                "Use pandas to read and convert your data format",
                "Check the file extension and encoding",
            ]
        )

    elif isinstance(error, TestNotSupportedError):
        suggestions.extend(
            [
                "Check the list of supported statistical tests",
                "Consider using a different test that is supported",
                "Contact support if you need a specific test implemented",
            ]
        )

    elif isinstance(error, VerificationError):
        suggestions.extend(
            [
                "Check that your data meets the test assumptions",
                "Verify that your parameters are within valid ranges",
                "Try with a smaller dataset to isolate the issue",
                "Check the error logs for more details",
            ]
        )

    elif isinstance(error, TimeoutError):
        suggestions.extend(
            [
                "Try with a smaller dataset",
                "Increase the timeout value in your configuration",
                "Check your system resources (CPU, memory)",
                "Consider using a more powerful machine",
            ]
        )

    elif isinstance(error, MemoryError):
        suggestions.extend(
            [
                "Close other applications to free up memory",
                "Try with a smaller dataset",
                "Increase your system's virtual memory",
                "Use a machine with more RAM",
            ]
        )

    elif isinstance(error, DependencyError):
        suggestions.extend(
            [
                "Install the missing dependency: pip install <package>",
                "Check that you have the correct version installed",
                "Update your Python environment",
                "Use a virtual environment to avoid conflicts",
            ]
        )

    return suggestions


def get_error_category(error: Exception) -> str:
    """
    Categorize an error for better error handling.

    Args:
        error: The exception that occurred

    Returns:
        Error category string
    """
    if isinstance(error, (ConfigurationError, DependencyError)):
        return "setup"
    elif isinstance(
        error,
        (
            DataValidationError,
            UnsupportedFormatError,
            CorruptedDataError,
            InsufficientDataError,
        ),
    ):
        return "data"
    elif isinstance(error, (TestNotSupportedError, AssumptionViolationError)):
        return "test"
    elif isinstance(error, (VerificationError, NumericalError)):
        return "verification"
    elif isinstance(error, (TimeoutError, MemoryError, SystemResourceError)):
        return "system"
    elif isinstance(error, (NetworkError, AuthenticationError, PermissionError)):
        return "access"
    else:
        return "unknown"


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is recoverable.

    Args:
        error: The exception that occurred

    Returns:
        True if the error is recoverable, False otherwise
    """
    # Non-recoverable errors
    non_recoverable = (
        ConfigurationError,
        DependencyError,
        TestNotSupportedError,
        CorruptedDataError,
        LicenseError,
    )

    return not isinstance(error, non_recoverable)


def create_user_friendly_error(error: Exception) -> str:
    """
    Create a user-friendly error message with suggestions.

    Args:
        error: The exception that occurred

    Returns:
        User-friendly error message
    """
    message = format_error_message(error)
    suggestions = get_error_suggestions(error)
    category = get_error_category(error)

    # Create the full message
    full_message = f"Error: {message}\n"
    full_message += f"Category: {category}\n"

    if suggestions:
        full_message += "\nSuggestions to fix this error:\n"
        for i, suggestion in enumerate(suggestions, 1):
            full_message += f"  {i}. {suggestion}\n"

    if is_recoverable_error(error):
        full_message += "\nThis error may be recoverable. Try the suggestions above."
    else:
        full_message += "\nThis error requires manual intervention to resolve."

    return full_message
