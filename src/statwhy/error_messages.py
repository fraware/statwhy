#!/usr/bin/env python3
"""
StatWhy Error Message System

Provides user-friendly error explanations, suggestions for fixing common issues,
and links to relevant learning resources and documentation.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import traceback
import sys

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context information for error analysis."""

    error_type: str
    error_message: str
    function_name: str
    line_number: int
    file_path: str
    stack_trace: str
    user_input: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class ErrorExplanation:
    """User-friendly error explanation."""

    title: str
    description: str
    cause: str
    impact: str
    suggestions: List[str]
    examples: List[str]
    documentation_links: List[str]
    related_errors: List[str]
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class FixSuggestion:
    """Suggestion for fixing an error."""

    title: str
    description: str
    code_example: str
    difficulty: str  # easy, medium, hard
    estimated_time: str
    prerequisites: List[str]


class ErrorMessageGenerator:
    """
    Generates user-friendly error messages with explanations and solutions.

    Features:
    - Context-aware error analysis
    - User-friendly explanations
    - Step-by-step fix suggestions
    - Documentation links
    - Learning resources
    """

    def __init__(self, error_database_path: Optional[str] = None):
        self.error_database_path = (
            Path(error_database_path)
            if error_database_path
            else Path(__file__).parent / "error_database.json"
        )
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.error_solutions: Dict[str, List[FixSuggestion]] = {}
        self.documentation_links: Dict[str, List[str]] = {}

        self._load_error_database()
        self._initialize_default_patterns()

    def _load_error_database(self) -> None:
        """Load error database from file."""
        try:
            if self.error_database_path.exists():
                with open(self.error_database_path, "r") as f:
                    data = json.load(f)

                self.error_patterns = data.get("patterns", {})
                self.error_solutions = data.get("solutions", {})
                self.documentation_links = data.get("documentation", {})

                logger.info(
                    f"Loaded error database with {len(self.error_patterns)} patterns"
                )

        except Exception as e:
            logger.warning(f"Could not load error database: {e}")

    def _initialize_default_patterns(self) -> None:
        """Initialize default error patterns if database is empty."""
        if not self.error_patterns:
            self.error_patterns = {
                "syntax_error": {
                    "patterns": [
                        r"SyntaxError:",
                        r"invalid syntax",
                        r"unexpected token",
                        r"missing \w+",
                        r"unexpected \w+",
                    ],
                    "category": "syntax",
                    "severity": "high",
                },
                "type_error": {
                    "patterns": [
                        r"TypeError:",
                        r"unsupported operand type",
                        r"can't multiply sequence",
                        r"object is not callable",
                        r"argument must be",
                    ],
                    "category": "type",
                    "severity": "medium",
                },
                "value_error": {
                    "patterns": [
                        r"ValueError:",
                        r"invalid literal",
                        r"could not convert",
                        r"out of range",
                        r"invalid value",
                    ],
                    "category": "value",
                    "severity": "medium",
                },
                "attribute_error": {
                    "patterns": [
                        r"AttributeError:",
                        r"object has no attribute",
                        r"'NoneType' object has no attribute",
                    ],
                    "category": "attribute",
                    "severity": "medium",
                },
                "import_error": {
                    "patterns": [
                        r"ImportError:",
                        r"ModuleNotFoundError:",
                        r"No module named",
                        r"cannot import name",
                    ],
                    "category": "import",
                    "severity": "medium",
                },
                "file_not_found": {
                    "patterns": [
                        r"FileNotFoundError:",
                        r"No such file or directory",
                        r"file not found",
                    ],
                    "category": "file",
                    "severity": "low",
                },
                "permission_error": {
                    "patterns": [
                        r"PermissionError:",
                        r"permission denied",
                        r"access denied",
                    ],
                    "category": "permission",
                    "severity": "medium",
                },
                "timeout_error": {
                    "patterns": [r"TimeoutError:", r"timed out", r"timeout"],
                    "category": "timeout",
                    "severity": "medium",
                },
                "memory_error": {
                    "patterns": [
                        r"MemoryError:",
                        r"out of memory",
                        r"insufficient memory",
                    ],
                    "category": "memory",
                    "severity": "critical",
                },
                "verification_error": {
                    "patterns": [
                        r"VerificationError:",
                        r"verification failed",
                        r"statistical assumption violated",
                        r"data validation failed",
                    ],
                    "category": "verification",
                    "severity": "high",
                },
            }

    def analyze_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Analyze an error and extract context information.

        Args:
            error: The exception that occurred
            context: Additional context information

        Returns:
            ErrorContext with detailed error information
        """
        try:
            # Get exception information
            error_type = type(error).__name__
            error_message = str(error)

            # Get stack trace information
            exc_type, exc_value, exc_traceback = sys.exc_info()
            stack_trace = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )

            # Extract function and line information
            function_name = "unknown"
            line_number = 0
            file_path = "unknown"

            if exc_traceback:
                frame = exc_traceback.tb_frame
                function_name = frame.f_code.co_name
                line_number = exc_traceback.tb_lineno
                file_path = frame.f_code.co_filename

            # Create error context
            error_context = ErrorContext(
                error_type=error_type,
                error_message=error_message,
                function_name=function_name,
                line_number=line_number,
                file_path=file_path,
                stack_trace=stack_trace,
                user_input=context.get("user_input") if context else None,
                system_info=context.get("system_info") if context else None,
                timestamp=context.get("timestamp") if context else None,
            )

            return error_context

        except Exception as e:
            logger.error(f"Error analyzing error: {e}")
            return ErrorContext(
                error_type="AnalysisError",
                error_message=str(e),
                function_name="unknown",
                line_number=0,
                file_path="unknown",
                stack_trace="",
            )

    def classify_error(self, error_context: ErrorContext) -> str:
        """
        Classify error based on patterns and context.

        Args:
            error_context: Error context information

        Returns:
            Error classification string
        """
        try:
            error_text = f"{error_context.error_type}: {error_context.error_message}"

            # Check against known patterns
            for error_class, pattern_info in self.error_patterns.items():
                patterns = pattern_info.get("patterns", [])

                for pattern in patterns:
                    if re.search(pattern, error_text, re.IGNORECASE):
                        return error_class

            # Default classification based on error type
            if "Syntax" in error_context.error_type:
                return "syntax_error"
            elif "Type" in error_context.error_type:
                return "type_error"
            elif "Value" in error_context.error_type:
                return "value_error"
            elif "Attribute" in error_context.error_type:
                return "attribute_error"
            elif "Import" in error_context.error_type:
                return "import_error"
            elif "File" in error_context.error_type:
                return "file_not_found"
            elif "Permission" in error_context.error_type:
                return "permission_error"
            elif "Timeout" in error_context.error_type:
                return "timeout_error"
            elif "Memory" in error_context.error_type:
                return "memory_error"
            else:
                return "unknown_error"

        except Exception as e:
            logger.error(f"Error classifying error: {e}")
            return "unknown_error"

    def generate_explanation(
        self, error_context: ErrorContext, error_class: str
    ) -> ErrorExplanation:
        """
        Generate user-friendly error explanation.

        Args:
            error_context: Error context information
            error_class: Classified error type

        Returns:
            ErrorExplanation with user-friendly details
        """
        try:
            # Get base explanation from database
            base_explanation = self._get_base_explanation(error_class)

            # Customize based on context
            explanation = self._customize_explanation(base_explanation, error_context)

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._get_default_explanation(error_context)

    def _get_base_explanation(self, error_class: str) -> ErrorExplanation:
        """Get base explanation for error class."""
        explanations = {
            "syntax_error": ErrorExplanation(
                title="Syntax Error",
                description="There's a problem with the structure of your code.",
                cause="The code doesn't follow Python's syntax rules.",
                impact="The program cannot run until the syntax is fixed.",
                suggestions=[
                    "Check for missing colons, parentheses, or brackets",
                    "Verify indentation is consistent",
                    "Look for typos in keywords or function names",
                ],
                examples=[
                    "Missing colon: if x > 5 (should be if x > 5:)",
                    "Unmatched parentheses: print('hello' (should be print('hello'))",
                    "Incorrect indentation: if x > 5:\nprint(x) (should be properly indented)",
                ],
                documentation_links=[
                    "https://docs.python.org/3/reference/grammar.html",
                    "https://docs.python.org/3/tutorial/controlflow.html",
                ],
                related_errors=["indentation_error", "name_error"],
                severity="high",
            ),
            "type_error": ErrorExplanation(
                title="Type Error",
                description="You're trying to use a value of one type where another type is expected.",
                cause="Python is strict about data types and operations.",
                impact="The operation cannot be performed with the given types.",
                suggestions=[
                    "Check the types of your variables",
                    "Convert types using functions like int(), str(), float()",
                    "Verify function parameter types",
                ],
                examples=[
                    "Adding string and number: '5' + 3 (should be int('5') + 3)",
                    "Calling non-callable: x = 5; x() (x is not a function)",
                    "Wrong argument type: len(123) (len expects sequence, not number)",
                ],
                documentation_links=[
                    "https://docs.python.org/3/library/stdtypes.html",
                    "https://docs.python.org/3/library/functions.html",
                ],
                related_errors=["value_error", "attribute_error"],
                severity="medium",
            ),
            "value_error": ErrorExplanation(
                title="Value Error",
                description="A function received an argument with the right type but inappropriate value.",
                cause="The value is outside the acceptable range or format.",
                impact="The function cannot process the given value.",
                suggestions=[
                    "Check the range of acceptable values",
                    "Verify data format and structure",
                    "Use appropriate validation before calling functions",
                ],
                examples=[
                    "Invalid list index: [1, 2, 3][10] (index out of range)",
                    "Invalid conversion: int('abc') (cannot convert string to int)",
                    "Empty sequence: max([]) (max() arg is an empty sequence)",
                ],
                documentation_links=[
                    "https://docs.python.org/3/library/exceptions.html",
                    "https://docs.python.org/3/tutorial/errors.html",
                ],
                related_errors=["index_error", "type_error"],
                severity="medium",
            ),
            "attribute_error": ErrorExplanation(
                title="Attribute Error",
                description="You're trying to access an attribute or method that doesn't exist.",
                cause="The object doesn't have the attribute you're trying to use.",
                impact="The program cannot access the requested functionality.",
                suggestions=[
                    "Check if the object has the attribute using dir() or hasattr()",
                    "Verify the object type and available methods",
                    "Check for typos in attribute names",
                ],
                examples=[
                    "None object: x = None; x.append(1) (None has no append method)",
                    "Wrong method: 'hello'.append('x') (strings don't have append)",
                    "Typo: 'hello'.lenght (should be 'hello'.length)",
                ],
                documentation_links=[
                    "https://docs.python.org/3/library/functions.html#dir",
                    "https://docs.python.org/3/library/functions.html#hasattr",
                ],
                related_errors=["type_error", "name_error"],
                severity="medium",
            ),
            "import_error": ErrorExplanation(
                title="Import Error",
                description="Python cannot find or load the module you're trying to import.",
                cause="The module is not installed, not in the path, or has issues.",
                impact="The imported functionality is not available.",
                suggestions=[
                    "Install the required package using pip",
                    "Check if the module name is spelled correctly",
                    "Verify the module is in your Python path",
                ],
                examples=[
                    "Missing package: import numpy (install with: pip install numpy)",
                    "Wrong name: import numpi (should be import numpy)",
                    "Path issue: import mymodule (check file location and __init__.py)",
                ],
                documentation_links=[
                    "https://docs.python.org/3/installing/index.html",
                    "https://docs.python.org/3/tutorial/modules.html",
                ],
                related_errors=["module_not_found_error", "syntax_error"],
                severity="medium",
            ),
            "file_not_found": ErrorExplanation(
                title="File Not Found",
                description="The program cannot locate the specified file or directory.",
                cause="The file path is incorrect or the file doesn't exist.",
                impact="The program cannot read or write the required file.",
                suggestions=[
                    "Check the file path spelling and case sensitivity",
                    "Verify the file exists in the specified location",
                    "Use absolute paths or check relative path context",
                ],
                examples=[
                    "Wrong path: open('data.txt') (file not in current directory)",
                    "Case sensitive: open('Data.txt') vs 'data.txt'",
                    "Missing extension: open('data') vs 'data.csv'",
                ],
                documentation_links=[
                    "https://docs.python.org/3/library/pathlib.html",
                    "https://docs.python.org/3/tutorial/inputoutput.html",
                ],
                related_errors=["permission_error", "value_error"],
                severity="low",
            ),
            "verification_error": ErrorExplanation(
                title="Statistical Verification Error",
                description="The statistical procedure failed verification checks.",
                cause="Data or assumptions don't meet statistical requirements.",
                impact="The statistical analysis may be invalid or unreliable.",
                suggestions=[
                    "Check data quality and completeness",
                    "Verify statistical assumptions are met",
                    "Review the verification requirements",
                ],
                examples=[
                    "Insufficient data: Need at least 30 observations for t-test",
                    "Assumption violation: Data not normally distributed for parametric test",
                    "Missing values: Incomplete data prevents analysis",
                ],
                documentation_links=[
                    "https://statwhy.org/docs/verification",
                    "https://statwhy.org/docs/assumptions",
                ],
                related_errors=["data_error", "assumption_error"],
                severity="high",
            ),
        }

        return explanations.get(error_class, self._get_generic_explanation())

    def _get_generic_explanation(self) -> ErrorExplanation:
        """Get generic explanation for unknown errors."""
        return ErrorExplanation(
            title="Unknown Error",
            description="An unexpected error occurred during execution.",
            cause="The error type is not recognized by the system.",
            impact="The program cannot continue normal operation.",
            suggestions=[
                "Check the error message for clues",
                "Review the code around the error location",
                "Consult Python documentation for the error type",
            ],
            examples=[],
            documentation_links=[
                "https://docs.python.org/3/library/exceptions.html",
                "https://docs.python.org/3/tutorial/errors.html",
            ],
            related_errors=[],
            severity="medium",
        )

    def _customize_explanation(
        self, base: ErrorExplanation, context: ErrorContext
    ) -> ErrorExplanation:
        """Customize explanation based on error context."""
        # Add context-specific suggestions
        context_suggestions = []

        if context.function_name != "unknown":
            context_suggestions.append(f"Check the {context.function_name} function")

        if context.line_number > 0:
            context_suggestions.append(f"Review code around line {context.line_number}")

        if context.user_input:
            context_suggestions.append("Verify your input data format and values")

        # Combine suggestions
        all_suggestions = base.suggestions + context_suggestions

        return ErrorExplanation(
            title=base.title,
            description=base.description,
            cause=base.cause,
            impact=base.impact,
            suggestions=all_suggestions,
            examples=base.examples,
            documentation_links=base.documentation_links,
            related_errors=base.related_errors,
            severity=base.severity,
        )

    def _get_default_explanation(self, error_context: ErrorContext) -> ErrorExplanation:
        """Get default explanation when generation fails."""
        return ErrorExplanation(
            title=f"{error_context.error_type} Error",
            description=f"An error occurred: {error_context.error_message}",
            cause="Unknown cause",
            impact="Program execution stopped",
            suggestions=[
                "Check the error message above",
                "Review the code around the error",
                "Consult Python documentation",
            ],
            examples=[],
            documentation_links=["https://docs.python.org/3/tutorial/errors.html"],
            related_errors=[],
            severity="medium",
        )

    def generate_fix_suggestions(
        self, error_class: str, context: Optional[Dict[str, Any]] = None
    ) -> List[FixSuggestion]:
        """
        Generate fix suggestions for an error.

        Args:
            error_class: Classified error type
            context: Additional context information

        Returns:
            List of fix suggestions
        """
        try:
            # Get base suggestions from database
            base_suggestions = self._get_base_suggestions(error_class)

            # Customize based on context
            customized_suggestions = self._customize_suggestions(
                base_suggestions, context
            )

            return customized_suggestions

        except Exception as e:
            logger.error(f"Error generating fix suggestions: {e}")
            return []

    def _get_base_suggestions(self, error_class: str) -> List[FixSuggestion]:
        """Get base fix suggestions for error class."""
        suggestions = {
            "syntax_error": [
                FixSuggestion(
                    title="Fix Missing Colon",
                    description="Add missing colon after control flow statements",
                    code_example="if x > 5:\n    print('x is greater than 5')",
                    difficulty="easy",
                    estimated_time="1-2 minutes",
                    prerequisites=["Basic Python syntax"],
                ),
                FixSuggestion(
                    title="Check Parentheses Balance",
                    description="Ensure all parentheses, brackets, and braces are properly matched",
                    code_example="print('Hello', end='')\nresult = (a + b) * c",
                    difficulty="easy",
                    estimated_time="2-3 minutes",
                    prerequisites=["Basic Python syntax"],
                ),
                FixSuggestion(
                    title="Verify Indentation",
                    description="Check that indentation is consistent (4 spaces per level)",
                    code_example="if x > 5:\n    if y > 10:\n        print('Both conditions met')",
                    difficulty="medium",
                    estimated_time="3-5 minutes",
                    prerequisites=["Python indentation rules"],
                ),
            ],
            "type_error": [
                FixSuggestion(
                    title="Convert Data Types",
                    description="Convert data to the expected type before operations",
                    code_example="x = '5'\ny = int(x) + 3  # Convert string to int",
                    difficulty="easy",
                    estimated_time="2-3 minutes",
                    prerequisites=["Python data types"],
                ),
                FixSuggestion(
                    title="Check Variable Types",
                    description="Use type() function to verify variable types",
                    code_example="x = 5\nprint(f'x is {type(x).__name__}')\nif isinstance(x, int):\n    print('x is an integer')",
                    difficulty="medium",
                    estimated_time="3-5 minutes",
                    prerequisites=["Python type checking"],
                ),
                FixSuggestion(
                    title="Validate Function Parameters",
                    description="Ensure function parameters match expected types",
                    code_example="def process_numbers(numbers):\n    if not isinstance(numbers, list):\n        numbers = [numbers]\n    return sum(numbers)",
                    difficulty="medium",
                    estimated_time="5-10 minutes",
                    prerequisites=["Python functions and type checking"],
                ),
            ],
            "verification_error": [
                FixSuggestion(
                    title="Check Data Quality",
                    description="Verify data completeness and remove invalid entries",
                    code_example="import pandas as pd\n\n# Remove missing values\ndata = data.dropna()\n\n# Check data types\ndata.info()",
                    difficulty="easy",
                    estimated_time="3-5 minutes",
                    prerequisites=["Basic data manipulation"],
                ),
                FixSuggestion(
                    title="Verify Statistical Assumptions",
                    description="Test if data meets statistical test requirements",
                    code_example="from scipy import stats\n\n# Test normality\nstatistic, p_value = stats.normaltest(data)\nif p_value < 0.05:\n    print('Data not normally distributed')",
                    difficulty="medium",
                    estimated_time="5-10 minutes",
                    prerequisites=["Statistical testing", "SciPy library"],
                ),
                FixSuggestion(
                    title="Use Alternative Methods",
                    description="Apply non-parametric tests when assumptions aren't met",
                    code_example="# Instead of t-test, use Mann-Whitney U\ntest_stat, p_value = stats.mannwhitneyu(group1, group2)\nprint(f'Mann-Whitney U test: p = {p_value:.4f}')",
                    difficulty="hard",
                    estimated_time="10-15 minutes",
                    prerequisites=["Statistical testing", "Non-parametric methods"],
                ),
            ],
        }

        return suggestions.get(error_class, [])

    def _customize_suggestions(
        self,
        base_suggestions: List[FixSuggestion],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[FixSuggestion]:
        """Customize suggestions based on context."""
        if not context:
            return base_suggestions

        # Add context-specific suggestions
        customized = base_suggestions.copy()

        # Add suggestions based on user input context
        if context.get("user_input"):
            customized.append(
                FixSuggestion(
                    title="Review Input Data",
                    description="Check the format and values of your input data",
                    code_example="# Validate input data\nif not isinstance(data, (list, np.ndarray)):\n    raise ValueError('Data must be list or numpy array')",
                    difficulty="easy",
                    estimated_time="2-3 minutes",
                    prerequisites=["Basic Python validation"],
                )
            )

        # Add suggestions based on function context
        if context.get("function_name"):
            customized.append(
                FixSuggestion(
                    title="Debug Function",
                    description=f"Add debugging to the {context['function_name']} function",
                    code_example=f"def {context['function_name']}(*args, **kwargs):\n    print(f'Input: args={args}, kwargs={kwargs}')\n    # ... rest of function",
                    difficulty="medium",
                    estimated_time="3-5 minutes",
                    prerequisites=["Python debugging"],
                )
            )

        return customized

    def get_documentation_links(
        self, error_class: str, topic: Optional[str] = None
    ) -> List[str]:
        """
        Get relevant documentation links for an error.

        Args:
            error_class: Classified error type
            topic: Specific topic of interest

        Returns:
            List of documentation URLs
        """
        try:
            links = self.documentation_links.get(error_class, [])

            # Add topic-specific links
            if topic:
                topic_links = self.documentation_links.get(topic, [])
                links.extend(topic_links)

            # Add general Python documentation
            general_links = [
                "https://docs.python.org/3/tutorial/errors.html",
                "https://docs.python.org/3/library/exceptions.html",
                "https://docs.python.org/3/reference/simple_stmts.html",
            ]

            # Remove duplicates while preserving order
            seen = set()
            unique_links = []
            for link in links + general_links:
                if link not in seen:
                    unique_links.append(link)
                    seen.add(link)

            return unique_links

        except Exception as e:
            logger.error(f"Error getting documentation links: {e}")
            return ["https://docs.python.org/3/tutorial/errors.html"]

    def format_error_message(
        self,
        error_context: ErrorContext,
        explanation: ErrorExplanation,
        suggestions: List[FixSuggestion],
    ) -> str:
        """
        Format a complete, user-friendly error message.

        Args:
            error_context: Error context information
            explanation: Error explanation
            suggestions: Fix suggestions

        Returns:
            Formatted error message string
        """
        try:
            message = []

            # Header
            message.append("=" * 60)
            message.append(f"🚨 {explanation.title}")
            message.append("=" * 60)
            message.append("")

            # Error details
            message.append("📋 ERROR DETAILS")
            message.append("-" * 40)
            message.append(f"Error Type: {error_context.error_type}")
            message.append(f"Error Message: {error_context.error_message}")
            message.append(f"Function: {error_context.function_name}")
            message.append(f"File: {error_context.file_path}")
            message.append(f"Line: {error_context.line_number}")
            message.append("")

            # Explanation
            message.append("💡 EXPLANATION")
            message.append("-" * 40)
            message.append(f"Description: {explanation.description}")
            message.append(f"Cause: {explanation.cause}")
            message.append(f"Impact: {explanation.impact}")
            message.append("")

            # Suggestions
            if suggestions:
                message.append("🔧 HOW TO FIX")
                message.append("-" * 40)

                for i, suggestion in enumerate(suggestions, 1):
                    message.append(f"{i}. {suggestion.title}")
                    message.append(f"   {suggestion.description}")
                    message.append(f"   Difficulty: {suggestion.difficulty}")
                    message.append(f"   Estimated Time: {suggestion.estimated_time}")

                    if suggestion.code_example:
                        message.append(f"   Example:")
                        for line in suggestion.code_example.split("\n"):
                            message.append(f"      {line}")

                    message.append("")

            # Examples
            if explanation.examples:
                message.append("📝 EXAMPLES")
                message.append("-" * 40)
                for i, example in enumerate(explanation.examples, 1):
                    message.append(f"{i}. {example}")
                message.append("")

            # Documentation
            if explanation.documentation_links:
                message.append("📚 LEARN MORE")
                message.append("-" * 40)
                for i, link in enumerate(explanation.documentation_links, 1):
                    message.append(f"{i}. {link}")
                message.append("")

            # Footer
            message.append("=" * 60)
            message.append(
                "💡 Tip: Copy the error message above to search for solutions online"
            )
            message.append("=" * 60)

            return "\n".join(message)

        except Exception as e:
            logger.error(f"Error formatting error message: {e}")
            return f"Error formatting failed: {str(e)}"

    def generate_quick_fix(self, error_context: ErrorContext) -> str:
        """
        Generate a quick fix suggestion for common errors.

        Args:
            error_context: Error context information

        Returns:
            Quick fix suggestion string
        """
        try:
            error_class = self.classify_error(error_context)

            quick_fixes = {
                "syntax_error": "Check for missing colons, parentheses, or incorrect indentation",
                "type_error": "Convert data types or check variable types before operations",
                "value_error": "Verify data values are within acceptable ranges",
                "attribute_error": "Check if the object has the required attribute or method",
                "import_error": "Install missing packages with pip or check module names",
                "file_not_found": "Verify file path and check if file exists",
                "verification_error": "Check data quality and statistical assumptions",
            }

            return quick_fixes.get(
                error_class,
                "Review the error message and check the code around the error location",
            )

        except Exception as e:
            logger.error(f"Error generating quick fix: {e}")
            return (
                "Review the error message and check the code around the error location"
            )

    def save_error_report(
        self,
        error_context: ErrorContext,
        explanation: ErrorExplanation,
        suggestions: List[FixSuggestion],
        output_path: str,
    ) -> bool:
        """
        Save error report to file for later reference.

        Args:
            error_context: Error context information
            explanation: Error explanation
            suggestions: Fix suggestions
            output_path: Path to save the report

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            report = {
                "timestamp": error_context.timestamp or "unknown",
                "error_context": {
                    "error_type": error_context.error_type,
                    "error_message": error_context.error_message,
                    "function_name": error_context.function_name,
                    "line_number": error_context.line_number,
                    "file_path": error_context.file_path,
                },
                "explanation": {
                    "title": explanation.title,
                    "description": explanation.description,
                    "cause": explanation.cause,
                    "impact": explanation.impact,
                    "severity": explanation.severity,
                },
                "suggestions": [
                    {
                        "title": s.title,
                        "description": s.description,
                        "difficulty": s.difficulty,
                        "estimated_time": s.estimated_time,
                    }
                    for s in suggestions
                ],
                "documentation_links": explanation.documentation_links,
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Error report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving error report: {e}")
            return False


class ErrorMessageFormatter:
    """Utility class for formatting error messages in different styles."""

    @staticmethod
    def format_simple(error_message: str, suggestion: str) -> str:
        """Format a simple error message with suggestion."""
        return f"❌ Error: {error_message}\n💡 Suggestion: {suggestion}"

    @staticmethod
    def format_detailed(error_type: str, message: str, location: str, fix: str) -> str:
        """Format a detailed error message."""
        return f"""
🚨 {error_type}
   Message: {message}
   Location: {location}
   Fix: {fix}
"""

    @staticmethod
    def format_markdown(
        error_context: ErrorContext, explanation: ErrorExplanation
    ) -> str:
        """Format error message in Markdown format."""
        md = []
        md.append(f"# {explanation.title}")
        md.append("")
        md.append(f"**Error Type:** `{error_context.error_type}`")
        md.append(
            f"**Location:** `{error_context.file_path}:{error_context.line_number}`"
        )
        md.append(f"**Function:** `{error_context.function_name}`")
        md.append("")
        md.append("## Description")
        md.append(explanation.description)
        md.append("")
        md.append("## Cause")
        md.append(explanation.cause)
        md.append("")
        md.append("## Impact")
        md.append(explanation.impact)
        md.append("")

        if explanation.suggestions:
            md.append("## Suggestions")
            for i, suggestion in enumerate(explanation.suggestions, 1):
                md.append(f"{i}. {suggestion}")
            md.append("")

        if explanation.documentation_links:
            md.append("## Documentation")
            for link in explanation.documentation_links:
                md.append(f"- [{link}]({link})")

        return "\n".join(md)


def create_user_friendly_error(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a user-friendly error message for any exception.

    Args:
        error: The exception that occurred
        context: Additional context information

    Returns:
        User-friendly error message string
    """
    try:
        # Create error message generator
        generator = ErrorMessageGenerator()

        # Analyze the error
        error_context = generator.analyze_error(error, context)

        # Classify the error
        error_class = generator.classify_error(error_context)

        # Generate explanation
        explanation = generator.generate_explanation(error_context, error_class)

        # Generate fix suggestions
        suggestions = generator.generate_fix_suggestions(error_class, context)

        # Format the complete message
        return generator.format_error_message(error_context, explanation, suggestions)

    except Exception as e:
        logger.error(f"Error creating user-friendly message: {e}")
        return f"An error occurred: {str(error)}\n\nUnable to generate detailed explanation due to: {str(e)}"


def log_error_with_context(
    error: Exception, context: Optional[Dict[str, Any]] = None, log_level: str = "ERROR"
) -> None:
    """
    Log an error with context information for debugging.

    Args:
        error: The exception that occurred
        context: Additional context information
        log_level: Logging level to use
    """
    try:
        # Create error message generator
        generator = ErrorMessageGenerator()

        # Analyze the error
        error_context = generator.analyze_error(error, context)

        # Log with context
        log_message = f"Error in {error_context.function_name} at {error_context.file_path}:{error_context.line_number}"

        if log_level.upper() == "DEBUG":
            logger.debug(log_message, exc_info=True)
        elif log_level.upper() == "INFO":
            logger.info(log_message, exc_info=True)
        elif log_level.upper() == "WARNING":
            logger.warning(log_message, exc_info=True)
        else:
            logger.error(log_message, exc_info=True)

        # Log context information
        if context:
            logger.debug(f"Error context: {context}")

    except Exception as e:
        logger.error(f"Error logging error with context: {e}")
        logger.error(f"Original error: {error}", exc_info=True)
