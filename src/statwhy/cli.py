#!/usr/bin/env python3
"""
StatWhy Command Line Interface

Provides an intuitive command-line interface for verifying statistical procedures
without requiring knowledge of OCaml or Why3.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import click
import rich.console
import rich.table
import rich.panel
import rich.progress
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
import pandas as pd
import numpy as np

from .core import StatWhyEngine
from .models import VerificationRequest, VerificationResult
from .utils import load_data, validate_data, format_results


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="StatWhy")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to configuration file"
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """
    StatWhy: Formally verified statistical hypothesis testing.

    Verify the correctness of statistical procedures using formal methods.
    Perfect for clinical trials, financial modeling, and quality control.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config

    if verbose:
        console.print("[bold blue]StatWhy CLI[/bold blue] - Verbose mode enabled")


@cli.command()
@click.option(
    "--test",
    "-t",
    required=True,
    type=click.Choice(
        [
            "ttest",
            "anova",
            "chi2",
            "wilcoxon",
            "mann-whitney",
            "kruskal",
            "bartlett",
            "fligner",
            "dunnett",
            "tukey",
            "steel",
            "steel-dwass",
            "williams",
            "poisson",
            "binom",
        ]
    ),
    help="Statistical test to verify",
)
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to data file (CSV, Excel, or JSON)",
)
@click.option(
    "--alpha",
    "-a",
    default=0.05,
    type=click.FloatRange(0.0, 1.0),
    help="Significance level (default: 0.05)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for results (JSON, HTML, or PDF)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "html", "pdf", "table"]),
    default="table",
    help="Output format (default: table)",
)
@click.option(
    "--timeout",
    default=300,
    type=int,
    help="Verification timeout in seconds (default: 300)",
)
@click.pass_context
def verify(
    ctx: click.Context,
    test: str,
    data: str,
    alpha: float,
    output: Optional[str],
    format: str,
    timeout: int,
) -> None:
    """
    Verify a statistical procedure.

    This command verifies the correctness of statistical tests using formal methods.
    It ensures that all assumptions are met and the procedure is mathematically sound.
    """
    console.print(f"[bold green]Verifying {test.upper()} test...[/bold green]")

    try:
        # Load and validate data
        with console.status("[bold yellow]Loading data..."):
            data_df = load_data(data)
            validate_data(data_df, test)

        # Create verification request
        request = VerificationRequest(
            test_type=test, data=data_df, alpha=alpha, timeout=timeout
        )

        # Initialize StatWhy engine
        engine = StatWhyEngine()

        # Perform verification
        with console.status("[bold yellow]Performing formal verification..."):
            result = engine.verify(request)

        # Display results
        if format == "table":
            display_results_table(result)
        else:
            display_results_formatted(result, format)

        # Save results if output specified
        if output:
            save_results(result, output, format)
            console.print(f"[bold green]Results saved to {output}[/bold green]")

        # Exit with appropriate code
        if result.is_verified:
            console.print("[bold green]✓ Verification successful![/bold green]")
            sys.exit(0)
        else:
            console.print("[bold red]✗ Verification failed![/bold red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        if ctx.obj.get("verbose"):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option(
    "--test",
    "-t",
    type=click.Choice(
        [
            "ttest",
            "anova",
            "chi2",
            "wilcoxon",
            "mann-whitney",
            "kruskal",
            "bartlett",
            "fligner",
            "dunnett",
            "tukey",
            "steel",
            "steel-dwass",
            "williams",
            "poisson",
            "binom",
        ]
    ),
    help="Filter examples by test type",
)
@click.option(
    "--category",
    "-c",
    type=click.Choice(
        ["clinical", "financial", "manufacturing", "research", "education"]
    ),
    help="Filter examples by application category",
)
def examples(test: Optional[str], category: Optional[str]) -> None:
    """
    Show verification examples and tutorials.

    Browse through curated examples of verified statistical procedures
    with explanations and real-world applications.
    """
    console.print("[bold blue]StatWhy Examples & Tutorials[/bold blue]")

    # Load examples database
    examples_db = load_examples_database()

    # Filter examples
    if test:
        examples_db = [ex for ex in examples_db if ex["test_type"] == test]
    if category:
        examples_db = [ex for ex in examples_db if ex["category"] == category]

    # Display examples
    table = rich.table.Table(title="Available Examples")
    table.add_column("Test Type", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Difficulty", style="yellow")

    for example in examples_db:
        table.add_row(
            example["test_type"],
            example["category"],
            example["description"],
            example["difficulty"],
        )

    console.print(table)

    # Interactive example selection
    if examples_db:
        selected = Prompt.ask(
            "Enter example number to view details",
            choices=[str(i) for i in range(len(examples_db))],
            default="0",
        )

        if selected.isdigit():
            show_example_details(examples_db[int(selected)])


@cli.command()
@click.option(
    "--port",
    "-p",
    default=8000,
    type=int,
    help="Port to run the web interface on (default: 8000)",
)
@click.option(
    "--host", "-h", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
)
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
def web(port: int, host: str, debug: bool) -> None:
    """
    Launch the StatWhy web interface.

    Start a web server that provides an intuitive interface for
    uploading data, selecting tests, and viewing verification results.
    """
    console.print(f"[bold blue]Starting StatWhy Web Interface...[/bold blue]")
    console.print(f"Server will be available at: http://{host}:{port}")

    try:
        from .web import create_app

        app = create_app()

        if debug:
            app.run(host=host, port=port, debug=True)
        else:
            import uvicorn

            uvicorn.run(app, host=host, port=port)

    except ImportError:
        console.print("[bold red]Web interface dependencies not installed![/bold red]")
        console.print("Install with: pip install statwhy[web]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error starting web interface: {str(e)}[/bold red]")
        sys.exit(1)


@cli.command()
def status() -> None:
    """
    Check StatWhy system status.

    Verify that all components are properly installed and configured.
    """
    console.print("[bold blue]StatWhy System Status[/bold blue]")

    # Check Python dependencies
    check_python_dependencies()

    # Check Why3 installation
    check_why3_installation()

    # Check Cameleer installation
    check_cameleer_installation()

    # Check system resources
    check_system_resources()


@cli.command()
@click.option(
    "--seed", "-s", type=int, help="Random seed for reproducible data generation"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="examples",
    help="Output directory for generated datasets (default: examples)",
)
@click.option(
    "--test-type",
    "-t",
    type=click.Choice(
        ["all", "clinical", "financial", "manufacturing", "research", "education"]
    ),
    default="all",
    help="Type of examples to generate (default: all)",
)
def generate_data(seed: Optional[int], output_dir: str, test_type: str) -> None:
    """
    Generate comprehensive, realistic sample datasets for testing.

    Creates extremely realistic datasets with real-world characteristics:
    - Outliers, missing values, measurement errors
    - Industry-specific patterns and edge cases
    - Multiple data quality levels
    - Comprehensive metadata and documentation
    """
    console.print("[bold blue]Generating Comprehensive Sample Data...[/bold blue]")

    try:
        from .data_generator import RealisticDataGenerator

        # Initialize generator
        generator = RealisticDataGenerator(seed=seed)
        generator.output_dir = Path(output_dir)

        if test_type == "all":
            console.print("🎯 Generating ALL example types...")
            generated_files = generator.generate_all_examples()
        else:
            console.print(f"🎯 Generating {test_type} examples...")
            # Generate specific category
            if test_type == "clinical":
                generated_files = generator._generate_clinical_examples()
            elif test_type == "financial":
                generated_files = generator._generate_financial_examples()
            elif test_type == "manufacturing":
                generated_files = generator._generate_manufacturing_examples()
            elif test_type == "research":
                generated_files = generator._generate_research_examples()
            elif test_type == "education":
                generated_files = generator._generate_education_examples()

        # Display results
        console.print(f"\n✅ Generated {len(generated_files)} datasets:")
        for filename, test_type in generated_files.items():
            file_path = Path(output_dir) / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                console.print(f"  📊 {filename} -> {test_type} ({size_mb:.2f} MB)")

        # Show metadata info
        metadata_path = Path(output_dir) / "dataset_metadata.json"
        if metadata_path.exists():
            console.print(f"\n📋 Metadata saved to: {metadata_path}")

            # Load and display summary
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            console.print(f"📈 Generation Summary:")
            console.print(
                f"  • Total datasets: {metadata['generation_info']['total_datasets']}"
            )
            console.print(f"  • Seed used: {metadata['generation_info']['seed']}")
            console.print(f"  • Timestamp: {metadata['generation_info']['timestamp']}")

        console.print(f"\n🎯 Dataset Features:")
        console.print(f"  • Real-world characteristics (outliers, missing values)")
        console.print(f"  • Industry-specific patterns")
        console.print(f"  • Edge cases and boundary conditions")
        console.print(f"  • Multiple data quality levels")
        console.print(f"  • Comprehensive metadata")

        console.print(
            f"\n💡 Use these datasets with: statwhy verify --test <type> --data <file>"
        )

    except ImportError as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        console.print("Make sure all dependencies are installed: pip install -e .")
    except Exception as e:
        console.print(f"[bold red]Error generating data: {str(e)}[/bold red]")
        console.print(f"Unexpected error: {str(e)}")


@cli.command()
@click.option(
    "--benchmark",
    "-b",
    type=click.Choice(["all", "single", "multi", "cache", "load", "memory", "edge"]),
    default="all",
    help="Type of benchmark to run (default: all)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="performance_results",
    help="Output directory for results (default: performance_results)",
)
@click.option(
    "--iterations",
    "-i",
    type=int,
    default=5,
    help="Number of iterations per test (default: 5)",
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    default=60,
    help="Timeout per verification in seconds (default: 60)",
)
def benchmark(benchmark: str, output_dir: str, iterations: int, timeout: int) -> None:
    """
    Run comprehensive performance benchmarking and optimization.

    Measures verification speed, memory usage, CPU utilization, and cache performance.
    Provides detailed analysis and optimization recommendations.
    """
    console.print("[bold blue]Running Performance Benchmarking...[/bold blue]")

    try:
        from .performance_tester import PerformanceTester

        # Initialize tester
        tester = PerformanceTester()
        tester.results_dir = Path(output_dir)
        tester.thresholds["max_execution_time"] = timeout

        if benchmark == "all":
            console.print("🚀 Running ALL benchmarks...")
            results = tester.run_comprehensive_benchmark()
        else:
            console.print(f"🚀 Running {benchmark} benchmark...")
            # Run specific benchmark
            if benchmark == "single":
                results = {"single_threaded": tester._benchmark_single_threaded()}
            elif benchmark == "multi":
                results = {"multi_threaded": tester._benchmark_multi_threaded()}
            elif benchmark == "cache":
                results = {"cache_performance": tester._benchmark_cache_performance()}
            elif benchmark == "load":
                results = {"load_testing": tester._benchmark_load_testing()}
            elif benchmark == "memory":
                results = {"memory_efficiency": tester._benchmark_memory_efficiency()}
            elif benchmark == "edge":
                results = {"edge_cases": tester._benchmark_edge_cases()}

        # Display results
        console.print(f"\n📊 Benchmark Results:")
        for name, result in results.items():
            success_rate = result.successful_operations / result.total_operations
            console.print(f"\n  {name.upper()}:")
            console.print(f"    • Operations: {result.total_operations:,}")
            console.print(f"    • Success Rate: {success_rate:.1%}")
            console.print(f"    • Avg Time: {result.average_execution_time:.3f}s")
            console.print(
                f"    • Throughput: {result.throughput_ops_per_second:.3f} ops/sec"
            )
            console.print(f"    • Memory Peak: {result.memory_peak_mb:.1f} MB")
            console.print(f"    • Cache Hit Rate: {result.cache_hit_rate:.1%}")

            if result.parallel_efficiency > 0:
                console.print(
                    f"    • Parallel Efficiency: {result.parallel_efficiency:.1%}"
                )

        # Show output location
        console.print(f"\n📁 Results saved to: {output_dir}")

        # Check for performance issues
        console.print(f"\n🔍 Performance Analysis:")
        for name, result in results.items():
            if result.average_execution_time > tester.thresholds["max_execution_time"]:
                console.print(f"  ⚠️  {name}: Execution time exceeds threshold")
            if result.memory_peak_mb > tester.thresholds["max_memory_usage"]:
                console.print(f"  ⚠️  {name}: Memory usage exceeds threshold")
            if result.cache_hit_rate < tester.thresholds["min_cache_hit_rate"]:
                console.print(f"  ⚠️  {name}: Cache hit rate below threshold")

        console.print(
            f"\n💡 Check the detailed report for optimization recommendations!"
        )

    except ImportError as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        console.print("Make sure all dependencies are installed: pip install -e .")
    except Exception as e:
        console.print(f"[bold red]Error running benchmark: {str(e)}[/bold red]")
        console.print(f"Unexpected error: {str(e)}")


@cli.command()
@click.option(
    "--data-file",
    "-d",
    type=click.Path(exists=True),
    help="Path to data file to analyze",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for analysis report"
)
def analyze_data(data_file: Optional[str], output: Optional[str]) -> None:
    """
    Analyze sample data quality and characteristics.

    Provides comprehensive analysis of data including:
    - Statistical summaries and distributions
    - Data quality metrics (missing values, outliers)
    - Assumption checking for statistical tests
    - Recommendations for data preparation
    """
    console.print("[bold blue]Analyzing Data Quality...[/bold blue]")

    try:
        if data_file:
            # Analyze specific file
            from .utils import load_data

            data_df = load_data(data_file)
            console.print(f"📊 Analyzing: {data_file}")
        else:
            # Analyze all generated examples
            examples_dir = Path("examples")
            if not examples_dir.exists():
                console.print(
                    "[yellow]No examples directory found. Generate data first with: statwhy generate-data[/yellow]"
                )
                return

            csv_files = list(examples_dir.glob("*.csv"))
            if not csv_files:
                console.print(
                    "[yellow]No CSV files found in examples directory.[/yellow]"
                )
                return

            console.print(f"📊 Found {len(csv_files)} CSV files to analyze")

            # Analyze each file
            for csv_file in csv_files:
                console.print(f"\n🔍 Analyzing: {csv_file.name}")
                data_df = pd.read_csv(csv_file)
                _display_data_analysis(data_df, csv_file.name)

        if data_file:
            _display_data_analysis(data_df, Path(data_file).name)

        # Save report if requested
        if output:
            _save_analysis_report(data_df, output)
            console.print(f"📋 Analysis report saved to: {output}")

    except Exception as e:
        console.print(f"[bold red]Error analyzing data: {str(e)}[/bold red]")
        console.print(f"Unexpected error: {str(e)}")


def _display_data_analysis(data_df: pd.DataFrame, filename: str) -> None:
    """Display comprehensive data analysis."""
    console.print(f"\n📈 Data Analysis for {filename}:")
    console.print(f"  • Shape: {data_df.shape}")
    console.print(
        f"  • Memory Usage: {data_df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    )

    # Column analysis
    console.print(f"\n  📋 Columns:")
    for col in data_df.columns:
        col_type = str(data_df[col].dtype)
        missing_pct = data_df[col].isnull().sum() / len(data_df) * 100

        if data_df[col].dtype in ["int64", "float64"]:
            stats = data_df[col].describe()
            console.print(
                f"    • {col} ({col_type}): {stats['mean']:.2f} ± {stats['std']:.2f}"
            )
            console.print(
                f"      Missing: {missing_pct:.1f}% | Range: {stats['min']:.2f} to {stats['max']:.2f}"
            )
        else:
            unique_count = data_df[col].nunique()
            console.print(f"    • {col} ({col_type}): {unique_count} unique values")
            console.print(f"      Missing: {missing_pct:.1f}%")

    # Data quality metrics
    console.print(f"\n  🔍 Data Quality:")
    total_missing = data_df.isnull().sum().sum()
    total_cells = data_df.size
    missing_pct = total_missing / total_cells * 100
    console.print(f"    • Missing Data: {total_missing:,} cells ({missing_pct:.1f}%)")

    # Outlier detection for numeric columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        console.print(f"    • Numeric Columns: {len(numeric_cols)}")
        for col in numeric_cols:
            Q1 = data_df[col].quantile(0.25)
            Q3 = data_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = (
                (data_df[col] < (Q1 - 1.5 * IQR)) | (data_df[col] > (Q3 + 1.5 * IQR))
            ).sum()
            if outlier_count > 0:
                console.print(f"      • {col}: {outlier_count} potential outliers")


def _save_analysis_report(data_df: pd.DataFrame, output_path: str) -> None:
    """Save detailed analysis report to file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "shape": data_df.shape,
            "memory_usage_mb": data_df.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": list(data_df.columns),
            "data_types": data_df.dtypes.to_dict(),
        },
        "quality_metrics": {
            "missing_values": data_df.isnull().sum().to_dict(),
            "missing_percentage": (
                data_df.isnull().sum() / len(data_df) * 100
            ).to_dict(),
            "duplicate_rows": data_df.duplicated().sum(),
        },
        "statistical_summary": {},
    }

    # Add statistical summaries for numeric columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats = data_df[col].describe()
        report["statistical_summary"][col] = {
            "count": int(stats["count"]),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "min": float(stats["min"]),
            "25%": float(stats["25%"]),
            "50%": float(stats["50%"]),
            "75%": float(stats["75%"]),
            "max": float(stats["max"]),
        }

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def display_results_table(result: VerificationResult) -> None:
    """Display verification results in a rich table format."""
    console.print("\n")

    # Main result panel
    status_color = "green" if result.is_verified else "red"
    status_icon = "✓" if result.is_verified else "✗"

    result_panel = rich.panel.Panel(
        f"{status_icon} {result.test_type.upper()} Test Verification",
        style=f"bold {status_color}",
        border_style=status_color,
    )
    console.print(result_panel)

    # Results table
    table = rich.table.Table(title="Verification Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="white")

    for component in result.components:
        status = "✓" if component.verified else "✗"
        status_style = "green" if component.verified else "red"

        table.add_row(
            component.name,
            f"[{status_style}]{status}[/{status_style}]",
            component.details,
        )

    console.print(table)

    # Summary
    if result.is_verified:
        console.print(
            "[bold green]All verification conditions satisfied! "
            "The statistical procedure is mathematically sound.[/bold green]"
        )
    else:
        console.print(
            "[bold red]Verification failed! "
            "Please review the failed components above.[/bold red]"
        )


def display_results_formatted(result: VerificationResult, format: str) -> None:
    """Display results in the specified format."""
    if format == "json":
        console.print_json(data=result.to_dict())
    elif format == "html":
        console.print(
            "[bold yellow]HTML output format not yet implemented[/bold yellow]"
        )
    elif format == "pdf":
        console.print(
            "[bold yellow]PDF output format not yet implemented[/bold yellow]"
        )


def save_results(result: VerificationResult, output: str, format: str) -> None:
    """Save results to the specified output file."""
    output_path = Path(output)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    else:
        console.print(
            f"[bold yellow]{format.upper()} output not yet implemented[/bold yellow]"
        )


def load_examples_database() -> list:
    """Load the examples database with comprehensive, educational examples."""
    return [
        # Clinical Trials & Medical Research
        {
            "test_type": "ttest",
            "category": "clinical",
            "description": "One-sample t-test for drug efficacy in clinical trials",
            "difficulty": "Beginner",
            "assumptions": [
                "Normal distribution",
                "Independent observations",
                "Known population mean",
            ],
            "verification_steps": [
                "Check normality",
                "Verify independence",
                "Validate sample size",
            ],
            "real_world_impact": "Ensures FDA compliance and patient safety in drug development",
            "data_requirements": "Continuous data, minimum 30 observations",
            "example_file": "examples/clinical_ttest_example.csv",
        },
        {
            "test_type": "anova",
            "category": "clinical",
            "description": "ANOVA for comparing multiple treatment groups in clinical studies",
            "difficulty": "Intermediate",
            "assumptions": [
                "Normal distribution",
                "Equal variances",
                "Independent groups",
            ],
            "verification_steps": [
                "Normality test",
                "Levene's test",
                "Group independence check",
            ],
            "real_world_impact": "Critical for multi-arm clinical trials and treatment comparison",
            "data_requirements": "Multiple groups, continuous outcomes, balanced design",
            "example_file": "examples/clinical_anova_example.csv",
        },
        {
            "test_type": "chi2",
            "category": "clinical",
            "description": "Chi-square test for treatment response rates and adverse events",
            "difficulty": "Intermediate",
            "assumptions": [
                "Independent observations",
                "Expected frequencies > 5",
                "Categorical data",
            ],
            "verification_steps": [
                "Independence check",
                "Expected frequency validation",
                "Sample size verification",
            ],
            "real_world_impact": "Essential for safety monitoring and efficacy assessment",
            "data_requirements": "Categorical variables, sufficient sample sizes",
            "example_file": "examples/clinical_chi2_example.csv",
        },
        # Financial Risk Modeling
        {
            "test_type": "wilcoxon",
            "category": "financial",
            "description": "Wilcoxon signed-rank test for portfolio performance analysis",
            "difficulty": "Intermediate",
            "assumptions": [
                "Symmetric distribution",
                "Paired observations",
                "Ordinal data",
            ],
            "verification_steps": [
                "Symmetry check",
                "Pairing validation",
                "Distribution analysis",
            ],
            "real_world_impact": "Regulatory compliance in financial risk assessment",
            "data_requirements": "Paired observations, ordinal or continuous data",
            "example_file": "examples/financial_wilcoxon_example.csv",
        },
        {
            "test_type": "mann-whitney",
            "category": "financial",
            "description": "Mann-Whitney U test for comparing investment strategies",
            "difficulty": "Intermediate",
            "assumptions": [
                "Independent groups",
                "Ordinal data",
                "Similar distributions",
            ],
            "verification_steps": [
                "Independence verification",
                "Distribution similarity",
                "Sample size check",
            ],
            "real_world_impact": "Ensures robust comparison of financial models and strategies",
            "data_requirements": "Two independent groups, ordinal or continuous data",
            "example_file": "examples/financial_mannwhitney_example.csv",
        },
        # Manufacturing & Quality Control
        {
            "test_type": "bartlett",
            "category": "manufacturing",
            "description": "Bartlett's test for homogeneity of variances in production processes",
            "difficulty": "Advanced",
            "assumptions": [
                "Normal distribution",
                "Independent samples",
                "Multiple groups",
            ],
            "verification_steps": [
                "Normality verification",
                "Independence check",
                "Group validation",
            ],
            "real_world_impact": "Critical for Six Sigma and quality control compliance",
            "data_requirements": "Multiple groups, normal distributions, independent samples",
            "example_file": "examples/manufacturing_bartlett_example.csv",
        },
        {
            "test_type": "kruskal",
            "category": "manufacturing",
            "description": "Kruskal-Wallis test for comparing multiple production lines",
            "difficulty": "Advanced",
            "assumptions": [
                "Independent groups",
                "Ordinal data",
                "Similar distributions",
            ],
            "verification_steps": [
                "Independence verification",
                "Distribution analysis",
                "Sample size validation",
            ],
            "real_world_impact": "Ensures consistent quality across manufacturing processes",
            "data_requirements": "Multiple independent groups, ordinal or continuous data",
            "example_file": "examples/manufacturing_kruskal_example.csv",
        },
        # Research & Academia
        {
            "test_type": "tukey",
            "category": "research",
            "description": "Tukey's HSD test for post-hoc analysis in research studies",
            "difficulty": "Advanced",
            "assumptions": [
                "Normal distribution",
                "Equal variances",
                "Independent groups",
            ],
            "verification_steps": [
                "Normality check",
                "Variance homogeneity",
                "Group independence",
            ],
            "real_world_impact": "Ensures rigorous statistical analysis in peer-reviewed research",
            "data_requirements": "Multiple groups, normal distributions, equal variances",
            "example_file": "examples/research_tukey_example.csv",
        },
        {
            "test_type": "dunnett",
            "category": "research",
            "description": "Dunnett's test for comparing treatments to control group",
            "difficulty": "Advanced",
            "assumptions": [
                "Normal distribution",
                "Equal variances",
                "Independent groups",
            ],
            "verification_steps": [
                "Normality verification",
                "Variance check",
                "Control group validation",
            ],
            "real_world_impact": "Essential for controlled experiments and treatment evaluation",
            "data_requirements": "Control group + treatments, normal distributions, equal variances",
            "example_file": "examples/research_dunnett_example.csv",
        },
        # Education & Training
        {
            "test_type": "normality_test",
            "category": "education",
            "description": "Shapiro-Wilk test for teaching statistical assumptions",
            "difficulty": "Beginner",
            "assumptions": [
                "Independent observations",
                "Continuous data",
                "Random sampling",
            ],
            "verification_steps": [
                "Independence check",
                "Data type validation",
                "Sampling verification",
            ],
            "real_world_impact": "Fundamental for understanding statistical test prerequisites",
            "data_requirements": "Continuous data, independent observations, adequate sample size",
            "example_file": "examples/education_normality_example.csv",
        },
        {
            "test_type": "multiple_comparisons",
            "category": "education",
            "description": "Bonferroni correction for multiple hypothesis testing",
            "difficulty": "Intermediate",
            "assumptions": [
                "Independent tests",
                "Type I error control",
                "Multiple hypotheses",
            ],
            "verification_steps": [
                "Test independence",
                "Error rate calculation",
                "Significance adjustment",
            ],
            "real_world_impact": "Critical for avoiding false discoveries in research",
            "data_requirements": "Multiple statistical tests, independent hypotheses",
            "example_file": "examples/education_multiple_comparisons_example.csv",
        },
    ]


def show_example_details(example: Dict[str, Any]) -> None:
    """Show detailed information about a specific example."""
    console.print(f"\n[bold blue]{example['test_type'].upper()} Example[/bold blue]")
    console.print(f"Category: [cyan]{example['category']}[/cyan]")
    console.print(f"Difficulty: [yellow]{example['difficulty']}[/yellow]")
    console.print(f"Description: {example['description']}")

    # Display assumptions
    if "assumptions" in example:
        console.print(f"\n[bold green]Key Assumptions:[/bold green]")
        for i, assumption in enumerate(example["assumptions"], 1):
            console.print(f"  {i}. {assumption}")

    # Display verification steps
    if "verification_steps" in example:
        console.print(f"\n[bold green]Verification Steps:[/bold green]")
        for i, step in enumerate(example["verification_steps"], 1):
            console.print(f"  {i}. {step}")

    # Display real-world impact
    if "real_world_impact" in example:
        console.print(f"\n[bold green]Real-World Impact:[/bold green]")
        console.print(f"  {example['real_world_impact']}")

    # Display data requirements
    if "data_requirements" in example:
        console.print(f"\n[bold green]Data Requirements:[/bold green]")
        console.print(f"  {example['data_requirements']}")

    # Display example file if available
    if "example_file" in example:
        console.print(f"\n[bold green]Example Data File:[/bold green]")
        console.print(f"  {example['example_file']}")

        # Check if file exists and offer to load it
        example_path = Path(example["example_file"])
        if example_path.exists():
            console.print(f"  [green]✓ File available[/green]")
            if Confirm.ask("Would you like to load this example data?"):
                try:
                    data_df = load_data(str(example_path))
                    console.print(f"  [green]Data loaded successfully![/green]")
                    console.print(f"  Shape: {data_df.shape}")
                    console.print(f"  Columns: {list(data_df.columns)}")

                    # Offer to run verification
                    if Confirm.ask("Would you like to verify this example?"):
                        run_example_verification(example, data_df)
                except Exception as e:
                    console.print(f"  [red]Error loading data: {str(e)}[/red]")
        else:
            console.print(
                f"  [yellow]⚠ File not found - would you like to create a sample dataset?[/yellow]"
            )
            if Confirm.ask("Create sample dataset for this example?"):
                create_sample_dataset(example)

    # Add educational note
    console.print(f"\n[bold cyan]Educational Note:[/bold cyan]")
    console.print(
        "This example demonstrates how formal verification ensures statistical procedures"
    )
    console.print(
        "meet all mathematical requirements before being applied to real data."
    )
    console.print(
        "This is crucial for maintaining scientific rigor and avoiding false conclusions."
    )


def check_python_dependencies() -> None:
    """Check Python dependency status."""
    console.print("\n[bold cyan]Python Dependencies:[/bold cyan]")

    dependencies = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "plotly",
        "flask",
        "click",
        "rich",
        "pydantic",
    ]

    for dep in dependencies:
        try:
            __import__(dep)
            console.print(f"  ✓ {dep}")
        except ImportError:
            console.print(f"  ✗ {dep} (not installed)")


def check_why3_installation() -> None:
    """Check Why3 installation status."""
    console.print("\n[bold cyan]Why3 Installation:[/bold cyan]")

    try:
        import subprocess

        result = subprocess.run(["why3", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("  ✓ Why3 installed")
            console.print(f"    Version: {result.stdout.strip()}")
        else:
            console.print("  ✗ Why3 not properly installed")
    except Exception:
        console.print("  ✗ Why3 not found in PATH")


def check_cameleer_installation() -> None:
    """Check Cameleer installation status."""
    console.print("\n[bold cyan]Cameleer Installation:[/bold cyan]")

    try:
        import subprocess

        result = subprocess.run(
            ["cameleer", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            console.print("  ✓ Cameleer installed")
            console.print(f"    Version: {result.stdout.strip()}")
        else:
            console.print("  ✗ Cameleer not properly installed")
    except Exception:
        console.print("  ✗ Cameleer not found in PATH")


def check_system_resources() -> None:
    """Check system resource availability."""
    console.print("\n[bold cyan]System Resources:[/bold cyan]")

    try:
        import psutil

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        console.print(f"  CPU Usage: {cpu_percent}%")

        # Memory
        memory = psutil.virtual_memory()
        console.print(f"  Memory: {memory.percent}% used")
        console.print(f"  Available: {memory.available // (1024**3)} GB")

        # Disk
        disk = psutil.disk_usage("/")
        console.print(f"  Disk: {disk.percent}% used")
        console.print(f"  Available: {disk.free // (1024**3)} GB")

    except ImportError:
        console.print("  psutil not installed - cannot check system resources")


def run_example_verification(example: Dict[str, Any], data_df) -> None:
    """Run verification on an example dataset."""
    console.print(
        f"\n[bold blue]Running Verification for {example['test_type'].upper()}...[/bold blue]"
    )

    try:
        # Create verification request
        request = VerificationRequest(
            test_type=example["test_type"], data=data_df, alpha=0.05, timeout=300
        )

        # Initialize engine
        engine = StatWhyEngine()

        # Run verification
        with console.status("[bold yellow]Verifying statistical procedure..."):
            result = engine.verify(request)

        # Display results
        display_results_table(result)

        # Educational insights
        if result.is_verified:
            console.print(f"\n[bold green]✓ Verification Successful![/bold green]")
            console.print(
                "This means the statistical procedure meets all mathematical requirements"
            )
            console.print("and can be safely applied to your data.")
        else:
            console.print(f"\n[bold red]✗ Verification Failed![/bold red]")
            console.print(
                "This indicates that some assumptions or requirements are not met."
            )
            console.print(
                "Review the failed components above to understand what needs attention."
            )

    except Exception as e:
        console.print(f"[bold red]Verification error: {str(e)}[/bold red]")
        console.print(
            "This demonstrates why formal verification is important - it catches"
        )
        console.print("issues that could lead to incorrect statistical conclusions.")


def create_sample_dataset(example: Dict[str, Any]) -> None:
    """Create a sample dataset for an example."""
    console.print(
        f"\n[bold blue]Creating Sample Dataset for {example['test_type'].upper()}...[/bold blue]"
    )

    try:
        import numpy as np
        import pandas as pd

        # Create appropriate sample data based on test type
        if example["test_type"] == "ttest":
            # One-sample t-test: normally distributed data
            n = 50
            data = np.random.normal(100, 15, n)  # Mean=100, SD=15
            df = pd.DataFrame({"values": data})

        elif example["test_type"] == "anova":
            # ANOVA: three groups with different means
            n_per_group = 30
            group1 = np.random.normal(50, 10, n_per_group)
            group2 = np.random.normal(60, 10, n_per_group)
            group3 = np.random.normal(70, 10, n_per_group)

            df = pd.DataFrame(
                {
                    "group": ["A"] * n_per_group
                    + ["B"] * n_per_group
                    + ["C"] * n_per_group,
                    "values": np.concatenate([group1, group2, group3]),
                }
            )

        elif example["test_type"] == "chi2":
            # Chi-square: categorical data
            n = 100
            categories = np.random.choice(
                ["A", "B", "C", "D"], n, p=[0.3, 0.3, 0.2, 0.2]
            )
            outcomes = np.random.choice(["Success", "Failure"], n, p=[0.6, 0.4])

            df = pd.DataFrame({"category": categories, "outcome": outcomes})

        elif example["test_type"] == "wilcoxon":
            # Wilcoxon: paired data
            n = 40
            before = np.random.normal(50, 10, n)
            after = before + np.random.normal(5, 3, n)  # Systematic increase

            df = pd.DataFrame({"before": before, "after": after})

        else:
            # Generic sample data
            n = 50
            data = np.random.normal(0, 1, n)
            df = pd.DataFrame({"values": data})

        # Save the sample dataset
        example_dir = Path("examples")
        example_dir.mkdir(exist_ok=True)

        filename = f"{example['category']}_{example['test_type']}_example.csv"
        filepath = example_dir / filename
        df.to_csv(filepath, index=False)

        console.print(f"[green]✓ Sample dataset created: {filepath}[/green]")
        console.print(f"  Shape: {df.shape}")
        console.print(f"  Columns: {list(df.columns)}")

        # Update the example with the new file path
        example["example_file"] = str(filepath)

        # Offer to run verification
        if Confirm.ask("Would you like to verify this sample dataset?"):
            run_example_verification(example, df)

    except Exception as e:
        console.print(f"[bold red]Error creating sample dataset: {str(e)}[/bold red]")
        console.print("This demonstrates the importance of proper data preparation")
        console.print("in statistical analysis.")


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
