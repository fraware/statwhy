# StatWhy: Formal Verification for Statistical Analysis

StatWhy is a state-of-the-art formal verification tool for statistical analysis, built with modern software engineering practices. It provides mathematical proof of statistical procedures using Why3 and OCaml/Cameleer, while offering intuitive interfaces for statisticians and researchers.

## Key Benefits

- **Mathematical Rigor**: Formal verification using Why3 and OCaml/Cameleer
- **Python Integration**: Seamless integration with pandas, scipy, and Jupyter
- **Industry Applications**: Clinical trials, financial risk, manufacturing quality
- **Performance**: Optimized for speed with intelligent caching
- **Extensibility**: Plugin system for custom statistical procedures

## System Architecture

### Core Components
- **Formal Verification Engine**: Why3 + OCaml/Cameleer backend
- **Python Integration Layer**: Seamless scipy/pandas ecosystem integration
- **Web Dashboard**: Modern Bootstrap 5 interface
- **CLI Interface**: Command-line tool for automation and scripting
- **Plugin System**: Extensible architecture for researcher contributions

### Technology Stack
- **Backend**: OCaml, Why3, Cameleer
- **Python Integration**: Python 3.8+, pandas, scipy, plotly, matplotlib
- **Web Interface**: Flask, Bootstrap 5, JavaScript
- **Testing**: pytest, performance benchmarking
- **Deployment**: Docker, CI/CD ready

## Python Integration Module

### StatWhyPython Class
The main integration interface providing seamless access to all StatWhy features.

```python
from statwhy import StatWhyPython

# Initialize the integration
statwhy = StatWhyPython()

# Verify statistical procedures
result = statwhy.verify_test(
    test_type="ttest",
    data=df,
    alpha=0.05,
    assumptions=True
)
```

### Key Features
- **Pandas DataFrame Support**: Direct DataFrame verification
- **Scipy Ecosystem Compatibility**: Full statistical test integration
- **Auto-Cleaning**: Intelligent data preprocessing
- **Type Validation**: Smart data type detection
- **Assumption Analysis**: Comprehensive statistical assumption checking

### Jupyter Notebook Integration
```python
# Generate interactive tutorial
tutorial = statwhy.generate_jupyter_tutorial()
tutorial.save("statwhy_tutorial.ipynb")

# Use magic commands
%load_ext statwhy
%statwhy_verify --test ttest --data data.csv
```

## Advanced Data Generation

### Industry-Specific Datasets
StatWhy generates realistic, comprehensive datasets for various domains:

#### Clinical Trials
```python
clinical_data = statwhy.generate_clinical_data(
    n_patients=100,
    treatment_groups=2,
    effect_size=0.5,
    missing_rate=0.1
)
```

#### Financial Risk
```python
financial_data = statwhy.generate_financial_data(
    n_assets=50,
    time_periods=252,
    volatility=0.2,
    correlation_strength=0.3
)
```

#### Manufacturing Quality
```python
manufacturing_data = statwhy.generate_manufacturing_data(
    n_batches=30,
    measurements_per_batch=10,
    defect_rate=0.05,
    process_variation=0.1
)
```

### Data Characteristics
- **Realistic Patterns**: Outliers, missing values, correlations
- **Customizable Parameters**: Effect sizes, correlation strengths, missing rates
- **Reproducible Results**: Seed-based generation for consistent testing

## Comprehensive Assumption Analysis

### Statistical Assumption Verification
```python
assumptions = statwhy.analyze_assumptions(
    data=df,
    test_type="ttest",
    detailed=True
)

# Results include:
# - Normality testing (Shapiro-Wilk)
# - Independence verification
# - Homogeneity testing (Levene's test)
# - Symmetry checking
# - Expected frequencies validation
```

### Assumption Types
1. **Normality Testing**: Shapiro-Wilk test integration
2. **Independence Verification**: Observation independence checking
3. **Homogeneity Testing**: Levene's test for variance equality
4. **Symmetry Checking**: Distribution symmetry for non-parametric tests
5. **Expected Frequencies**: Chi-square test prerequisites

## Rich Visualization System

### Interactive Plotly Plots
```python
# Generate interactive plots
plot = statwhy.create_interactive_plot(
    data=df,
    plot_type="distribution",
    test_type="ttest"
)
plot.show()
```

### Static Matplotlib Plots
```python
# Publication-ready static plots
fig = statwhy.create_static_plot(
    data=df,
    plot_type="qq_plot",
    style="publication"
)
fig.savefig("publication_plot.png", dpi=300)
```

### Plot Types
- **Test-Specific Plots**: Appropriate visualizations for each test type
- **Customizable Styles**: Professional plotting configurations
- **Export Options**: High-resolution publication-ready outputs

## Educational Features

### Interactive Tutorials
- **Built-in Jupyter Notebooks**: Complete learning path
- **Real-World Examples**: Industry-specific applications
- **Best Practices**: Statistical methodology guidelines
- **Learning Resources**: Comprehensive educational content

### Tutorial Generation
```python
# Generate custom tutorials
tutorial = statwhy.generate_tutorial(
    topic="hypothesis_testing",
    difficulty="intermediate",
    examples=["clinical", "financial"]
)
```

## Web Interface

### Modern Dashboard
- **Bootstrap 5 UI**: Responsive, professional design
- **Data Upload**: Drag-and-drop file support
- **Test Selection**: Intuitive test type selection
- **Result Visualization**: Interactive result display
- **Export Options**: Multiple output formats

### Features
- **User Authentication**: Secure access control
- **Session Management**: Persistent user sessions
- **Real-time Updates**: Live verification progress
- **Mobile Responsive**: Works on all devices

## Command Line Interface

### Core Commands
```bash
# Verify statistical procedures
statwhy verify --test ttest --data data.csv --alpha 0.05

# Generate sample data
statwhy generate-data --type clinical --output clinical_data.csv

# Run performance benchmarks
statwhy benchmark --type comprehensive --output benchmark_results.json

# Analyze existing data
statwhy analyze-data --file data.csv --test ttest

# Start web interface
statwhy web --port 8000

# Check system status
statwhy status
```

### Advanced Options
- **Verbose Output**: Detailed execution information
- **Custom Configurations**: User-defined parameters
- **Batch Processing**: Multiple file processing
- **Output Formats**: JSON, CSV, HTML, PDF

## Clinical Trial Verification

### FDA Compliance Features
- **Statistical Procedure Validation**: Verified statistical methods
- **Patient Safety Checks**: Risk assessment and monitoring
- **Regulatory Compliance**: FDA guideline adherence
- **Audit Trail**: Complete verification history

### Clinical Data Generation
```python
clinical_data = statwhy.generate_clinical_data(
    n_patients=200,
    treatment_groups=3,
    primary_endpoint="survival_time",
    secondary_endpoints=["quality_of_life", "adverse_events"],
    follow_up_duration=365,
    dropout_rate=0.15
)
```

## Financial Risk Modeling

### Basel Compliance Features
- **Risk Assessment Validation**: Statistical model verification
- **Regulatory Requirement Checking**: Basel guideline compliance
- **Stress Testing**: Extreme scenario analysis
- **Capital Adequacy**: Risk-weighted asset calculations

### Financial Data Generation
```python
financial_data = statwhy.generate_financial_data(
    n_assets=100,
    time_periods=1000,
    risk_factors=["market", "credit", "operational"],
    correlation_matrix=corr_matrix,
    volatility_clustering=True
)
```

## Manufacturing Quality Control

### Six Sigma Features
- **Process Optimization**: Statistical process control
- **Defect Analysis**: Quality metric verification
- **Capability Studies**: Process capability indices
- **Control Charts**: Real-time quality monitoring

### Manufacturing Data Generation
```python
manufacturing_data = statwhy.generate_manufacturing_data(
    n_processes=5,
    n_measurements=1000,
    control_limits=[3.0, 3.5],
    process_capability=1.33,
    measurement_uncertainty=0.01
)
```

## Plugin System

### Architecture
- **Researcher Contributions**: Verified statistical procedures
- **Marketplace**: Repository of verified tests
- **Incentivization**: Recognition and citation system
- **Quality Control**: Automated verification pipeline

### Plugin Development
```python
from statwhy.plugin_system import PluginBase

class CustomTestPlugin(PluginBase):
    def __init__(self):
        super().__init__("custom_test", "Custom Statistical Test")
    
    def verify(self, data, parameters):
        # Implementation of custom verification logic
        pass
    
    def validate_assumptions(self, data):
        # Custom assumption validation
        pass
```

## Performance Optimization

### Parallel Processing
- **Multi-threaded Verification**: Concurrent test execution
- **Load Balancing**: Intelligent work distribution
- **Resource Management**: Optimal CPU/memory utilization

### Advanced Caching
- **Intelligent Result Caching**: Common parameter combinations
- **Cache Invalidation**: Smart cache management
- **Performance Monitoring**: Real-time optimization metrics

### Benchmarking
```python
# Comprehensive performance testing
benchmark_results = statwhy.run_comprehensive_benchmark(
    test_types=["ttest", "anova", "chi2"],
    data_sizes=[100, 1000, 10000],
    iterations=100
)
```

## Automated Testing

### Test Suite
- **Comprehensive Coverage**: All statistical procedures
- **Edge Case Testing**: Boundary condition validation
- **Performance Testing**: Speed and efficiency validation
- **Integration Testing**: End-to-end workflow validation

### Continuous Integration
- **Automated Quality Assurance**: CI/CD pipeline integration
- **Performance Regression Testing**: Continuous performance monitoring
- **Code Quality Checks**: Automated code review and validation

## Installation

### Prerequisites

- Python 3.8 or higher
- OCaml (for core verification engine)
- Why3 (formal verification framework)
- Cameleer (OCaml plugin for Why3)

### System Dependencies

#### Windows
```bash
# Install OCaml using OCaml for Windows
# Download from: https://fdopen.github.io/opam-repository-mingw/installation/

# Install Why3
opam install why3

# Install Cameleer
opam install cameleer
```

#### macOS
```bash
# Install OCaml using Homebrew
brew install ocaml opam

# Initialize opam
opam init

# Install Why3 and Cameleer
opam install why3 cameleer
```

#### Linux (Ubuntu/Debian)
```bash
# Install OCaml and opam
sudo apt-get update
sudo apt-get install ocaml opam

# Initialize opam
opam init

# Install Why3 and Cameleer
opam install why3 cameleer
```

### Python Installation

#### Using pip
```bash
pip install statwhy
```

#### From Source
```bash
git clone https://github.com/your-org/statwhy.git
cd statwhy
pip install -e .
```

#### Development Installation
```bash
git clone https://github.com/your-org/statwhy.git
cd statwhy
pip install -e .
pip install -r requirements-dev.txt
```

### Verification

After installation, verify that everything is working:

```bash
# Check StatWhy version
statwhy --version

# Check system status
statwhy status

# Test basic functionality
statwhy verify --test ttest --help
```

## Basic Usage

### Python Integration

#### Import and Initialize
```python
from statwhy import StatWhyPython
import pandas as pd

# Initialize StatWhy
statwhy = StatWhyPython()

# Check available features
print(f"Supported tests: {statwhy.get_supported_tests()}")
print(f"Available data types: {statwhy.get_available_data_types()}")
```

#### Verify a Statistical Test
```python
# Load your data
df = pd.read_csv("your_data.csv")

# Verify a t-test
result = statwhy.verify_test(
    test_type="ttest",
    data=df,
    alpha=0.05,
    assumptions=True
)

print(f"Verification successful: {result['success']}")
print(f"P-value: {result['p_value']}")
print(f"Assumptions met: {result['assumptions_met']}")
```

#### Comprehensive Verification
```python
# Run comprehensive verification with visualization
result = statwhy.verify_test_comprehensive(
    test_type="anova",
    data=df,
    alpha=0.05,
    assumptions=True,
    visualization=True,
    export_format="html"
)

# Access detailed results
print(f"Test statistic: {result['test_statistic']}")
print(f"Degrees of freedom: {result['degrees_of_freedom']}")
print(f"Effect size: {result['effect_size']}")
```

### Data Generation

#### Generate Sample Data
```python
# Generate clinical trial data
clinical_data = statwhy.generate_clinical_data(
    n_patients=200,
    treatment_groups=3,
    effect_size=0.6,
    missing_rate=0.1
)

# Generate financial data
financial_data = statwhy.generate_financial_data(
    n_assets=50,
    time_periods=252,
    volatility=0.25,
    correlation_strength=0.4
)

# Generate manufacturing data
manufacturing_data = statwhy.generate_manufacturing_data(
    n_batches=30,
    measurements_per_batch=15,
    defect_rate=0.03,
    process_variation=0.08
)
```

#### Custom Data Generation
```python
# Generate custom dataset with specific characteristics
custom_data = statwhy.generate_custom_data(
    n_samples=500,
    distributions=["normal", "exponential", "uniform"],
    correlations=[[1.0, 0.3, 0.1], 
                  [0.3, 1.0, 0.2], 
                  [0.1, 0.2, 1.0]],
    outliers_rate=0.05,
    missing_rate=0.08
)
```

### Assumption Analysis

#### Check Statistical Assumptions
```python
# Analyze assumptions for a specific test
assumptions = statwhy.analyze_assumptions(
    data=df,
    test_type="ttest",
    detailed=True
)

# Check specific assumptions
print(f"Normality: {assumptions['normality']['passed']}")
print(f"Independence: {assumptions['independence']['passed']}")
print(f"Homogeneity: {assumptions['homogeneity']['passed']}")

# Get detailed results
if not assumptions['normality']['passed']:
    print(f"Normality test details: {assumptions['normality']['details']}")
    print(f"Recommendations: {assumptions['normality']['recommendations']}")
```

#### Comprehensive Assumption Checking
```python
# Check all assumptions for multiple tests
all_assumptions = statwhy.check_all_assumptions(
    data=df,
    test_types=["ttest", "anova", "chi2"]
)

for test_type, test_assumptions in all_assumptions.items():
    print(f"\n{test_type.upper()} Assumptions:")
    for assumption, result in test_assumptions.items():
        status = "✓" if result['passed'] else "✗"
        print(f"  {status} {assumption}: {result['details']}")
```

### Visualization

#### Interactive Plots
```python
# Create interactive distribution plot
plot = statwhy.create_interactive_plot(
    data=df,
    plot_type="distribution",
    test_type="ttest",
    title="Data Distribution Analysis"
)

# Display plot
plot.show()

# Save as HTML
plot.write_html("distribution_plot.html")
```

#### Static Plots
```python
# Create publication-ready static plot
fig = statwhy.create_static_plot(
    data=df,
    plot_type="qq_plot",
    style="publication",
    figsize=(10, 8)
)

# Customize plot
fig.suptitle("Q-Q Plot for Normality Assessment", fontsize=16)
fig.tight_layout()

# Save high-resolution plot
fig.savefig("qq_plot.png", dpi=300, bbox_inches='tight')
```

#### Test-Specific Visualizations
```python
# Generate appropriate plots for ANOVA
anova_plots = statwhy.create_test_plots(
    data=df,
    test_type="anova",
    plot_types=["boxplot", "residuals", "qq_plot"]
)

# Display all plots
for plot_name, plot in anova_plots.items():
    print(f"\n{plot_name}:")
    plot.show()
```

## Advanced Usage

### Clinical Trial Applications

#### FDA Compliance Verification
```python
from statwhy import ClinicalTrialVerifier, FDAGuidelines

# Initialize clinical trial verifier
verifier = ClinicalTrialVerifier()

# Load clinical data
clinical_data = pd.read_csv("clinical_trial_data.csv")

# Verify statistical procedures
verification_result = verifier.verify_clinical_trial(
    data=clinical_data,
    primary_endpoint="survival_time",
    secondary_endpoints=["quality_of_life", "adverse_events"],
    alpha=0.05,
    power=0.8
)

# Check FDA compliance
fda_checker = FDAGuidelines()
compliance_result = fda_checker.check_compliance(
    clinical_data, 
    verification_result
)

print(f"FDA Compliant: {compliance_result.compliant}")
print(f"Compliance Score: {compliance_result.score}")
```

#### Patient Safety Monitoring
```python
# Monitor patient safety during trial
safety_result = verifier.monitor_patient_safety(
    data=clinical_data,
    safety_endpoints=["adverse_events", "serious_adverse_events"],
    monitoring_frequency="weekly"
)

# Generate safety report
safety_report = verifier.generate_safety_report(safety_result)
safety_report.save("safety_report.pdf")
```

### Financial Risk Applications

#### Basel Compliance Verification
```python
from statwhy import FinancialRiskVerifier, BaselRequirements

# Initialize financial risk verifier
verifier = FinancialRiskVerifier()

# Load financial data
financial_data = pd.read_csv("portfolio_data.csv")

# Verify risk model
verification_result = verifier.verify_risk_model(
    data=financial_data,
    risk_factors=["market", "credit", "operational"],
    confidence_level=0.99,
    time_horizon=10
)

# Check Basel compliance
basel_checker = BaselRequirements()
compliance_result = basel_checker.check_compliance(
    financial_data, 
    verification_result
)

print(f"Basel Compliant: {compliance_result.compliant}")
print(f"Capital Requirements: {compliance_result.capital_requirements}")
```

#### Stress Testing
```python
# Run comprehensive stress tests
stress_result = verifier.run_stress_tests(
    data=financial_data,
    scenarios=["2008_crisis", "covid_19", "interest_rate_shock"],
    confidence_level=0.99,
    time_horizon=10
)

# Analyze stress test results
stress_analysis = verifier.analyze_stress_test_results(stress_result)
print(f"Worst-case loss: {stress_analysis.worst_case_loss}")
print(f"Risk-adjusted return: {stress_analysis.risk_adjusted_return}")
```

### Manufacturing Quality Applications

#### Six Sigma Verification
```python
from statwhy import ManufacturingQualityVerifier, SixSigmaRequirements

# Initialize manufacturing quality verifier
verifier = ManufacturingQualityVerifier()

# Load manufacturing data
manufacturing_data = pd.read_csv("production_data.csv")

# Verify process capability
capability_result = verifier.verify_process_capability(
    data=manufacturing_data,
    specification_limits=[3.0, 3.5],
    target_value=3.25,
    process_stability=True
)

# Check Six Sigma compliance
six_sigma_checker = SixSigmaRequirements()
sigma_result = six_sigma_checker.check_sigma_level(
    manufacturing_data, 
    capability_result
)

print(f"Sigma Level: {sigma_result.sigma_level}")
print(f"Defects per Million: {sigma_result.dpm}")
print(f"Process Capability: {sigma_result.cp_cpk}")
```

#### Quality Control Charts
```python
# Generate quality control charts
control_charts = verifier.generate_control_charts(
    data=manufacturing_data,
    chart_types=["xbar_r", "individual_moving_range", "p_chart"],
    control_limits="3_sigma"
)

# Display control charts
for chart_name, chart in control_charts.items():
    print(f"\n{chart_name} Chart:")
    chart.show()
    
    # Check for out-of-control points
    ooc_points = verifier.check_out_of_control(chart)
    if ooc_points:
        print(f"Out-of-control points: {ooc_points}")
```

## Plugin Development

### Creating Custom Plugins

#### Basic Plugin Structure
```python
from statwhy.plugin_system import PluginBase
from statwhy.models import VerificationResult, AssumptionResult

class CustomTestPlugin(PluginBase):
    """Custom statistical test plugin for StatWhy."""
    
    def __init__(self):
        super().__init__(
            plugin_id="custom_test",
            name="Custom Statistical Test",
            version="1.0.0"
        )
    
    def verify(self, data, parameters):
        """
        Execute custom verification logic.
        
        Args:
            data: Input data for verification
            parameters: Verification parameters
            
        Returns:
            VerificationResult from plugin execution
        """
        try:
            # Your custom verification logic here
            test_statistic = self._calculate_test_statistic(data, parameters)
            p_value = self._calculate_p_value(test_statistic, data)
            
            return VerificationResult(
                success=True,
                test_statistic=test_statistic,
                p_value=p_value,
                assumptions_met=self._check_assumptions(data)
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                error=str(e)
            )
    
    def validate_assumptions(self, data):
        """
        Validate statistical assumptions for the custom test.
        
        Args:
            data: Input data for assumption validation
            
        Returns:
            AssumptionResult with validation outcomes
        """
        # Your custom assumption validation logic here
        normality = self._test_normality(data)
        independence = self._test_independence(data)
        
        return AssumptionResult(
            normality=normality,
            independence=independence,
            overall_passed=normality and independence
        )
    
    def _calculate_test_statistic(self, data, parameters):
        """Calculate custom test statistic."""
        # Implementation here
        pass
    
    def _calculate_p_value(self, test_statistic, data):
        """Calculate p-value for custom test."""
        # Implementation here
        pass
    
    def _check_assumptions(self, data):
        """Check if assumptions are met."""
        # Implementation here
        pass
```

#### Plugin Installation and Usage
```python
from statwhy.plugin_system import PluginManager

# Initialize plugin manager
plugin_manager = PluginManager()

# Install custom plugin
result = plugin_manager.install_plugin("path/to/custom_plugin.zip")
if result.success:
    print(f"Plugin installed: {result.plugin.name}")
else:
    print(f"Installation failed: {result.error}")

# List installed plugins
plugins = plugin_manager.list_installed_plugins()
for plugin in plugins:
    print(f"- {plugin.name} v{plugin.version}")

# Use custom plugin
custom_plugin = plugin_manager.get_plugin("custom_test")
if custom_plugin:
    result = custom_plugin.verify(data, parameters)
    print(f"Custom test result: {result}")
```

## Performance Optimization

### Caching and Optimization

#### Enable Caching
```python
# Initialize with caching enabled
statwhy = StatWhyPython(config={
    'cache_enabled': True,
    'cache_size': 1000,
    'cache_ttl': 3600
})

# Cache verification results
result = statwhy.verify_test(
    test_type="ttest",
    data=df,
    alpha=0.05,
    cache_result=True
)

# Check cache statistics
cache_stats = statwhy.get_cache_statistics()
print(f"Cache hit rate: {cache_stats.hit_rate:.2%}")
print(f"Cache size: {cache_stats.current_size}")
```

#### Parallel Processing
```python
# Enable parallel processing
statwhy = StatWhyPython(config={
    'parallel_processing': True,
    'max_workers': 4
})

# Run multiple verifications in parallel
verification_tasks = [
    ("ttest", df1, 0.05),
    ("anova", df2, 0.01),
    ("chi2", df3, 0.05)
]

results = statwhy.verify_multiple_tests(verification_tasks)
for test_type, result in results.items():
    print(f"{test_type}: {result['success']}")
```

### Benchmarking and Monitoring

#### Performance Monitoring
```python
# Monitor performance metrics
performance_metrics = statwhy.get_performance_metrics()
print(f"Average response time: {performance_metrics.avg_response_time:.3f}s")
print(f"Throughput: {performance_metrics.throughput:.1f} requests/s")
print(f"Memory usage: {performance_metrics.memory_usage:.1f}MB")

# Get performance recommendations
recommendations = statwhy.get_performance_recommendations()
for rec in recommendations:
    print(f"- {rec.description}: {rec.expected_improvement}")
```

#### Load Testing
```python
# Run load test
load_test_result = statwhy.run_load_test(
    test_type="ttest",
    concurrent_users=100,
    test_duration=300,
    data_size=1000
)

print(f"Load test completed:")
print(f"  Total requests: {load_test_result.total_requests}")
print(f"  Successful requests: {load_test_result.successful_requests}")
print(f"  Average response time: {load_test_result.avg_response_time:.3f}s")
print(f"  Throughput: {load_test_result.throughput:.1f} requests/s")
```

## Testing and Validation

### Running Tests

#### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test module
python -m pytest tests/unit/test_core.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=statwhy --cov-report=html
```

#### Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Run with specific markers
python -m pytest tests/integration/ -m "slow" -v
```

#### Performance Tests
```bash
# Run performance benchmarks
python -m pytest tests/performance/ --benchmark-only

# Run specific benchmark
python -m pytest tests/performance/test_verification_speed.py --benchmark-only
```

### Test Data Management

#### Generate Test Data
```python
# Generate test data for specific scenarios
test_data = statwhy.generate_test_data(
    scenario="edge_cases",
    test_type="ttest",
    data_sizes=[10, 100, 1000],
    include_outliers=True,
    include_missing=True
)

# Save test data
test_data.save("test_data_edge_cases.pkl")
```

#### Validate Test Results
```python
# Validate test results against known values
validation_result = statwhy.validate_test_results(
    actual_results=test_results,
    expected_results=expected_results,
    tolerance=0.001
)

print(f"Validation passed: {validation_result.passed}")
print(f"Accuracy: {validation_result.accuracy:.4f}")
```

## Deployment

### Production Deployment

#### Docker Deployment
```bash
# Build Docker image
docker build -t statwhy:latest .

# Run container
docker run -d \
    --name statwhy \
    -p 8000:8000 \
    -v /path/to/data:/app/data \
    -e STATWHY_ENVIRONMENT=production \
    statwhy:latest

# Check container status
docker ps
docker logs statwhy
```

#### Kubernetes Deployment
```yaml
# statwhy-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: statwhy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: statwhy
  template:
    metadata:
      labels:
        app: statwhy
    spec:
      containers:
      - name: statwhy
        image: statwhy:latest
        ports:
        - containerPort: 8000
        env:
        - name: STATWHY_ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Monitoring and Logging

#### Logging Configuration
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('statwhy.log'),
        logging.StreamHandler()
    ]
)

# Set specific logger levels
logging.getLogger('statwhy.core').setLevel(logging.DEBUG)
logging.getLogger('statwhy.web').setLevel(logging.INFO)
```

#### Health Checks
```python
# Check system health
health_status = statwhy.check_system_health()

print(f"System Status: {health_status.status}")
print(f"Database: {health_status.database}")
print(f"Cache: {health_status.cache}")
print(f"Verification Engine: {health_status.verification_engine}")

# Get detailed health information
if health_status.status != "healthy":
    for issue in health_status.issues:
        print(f"Issue: {issue.description}")
        print(f"Severity: {issue.severity}")
        print(f"Recommendation: {issue.recommendation}")
```

## Configuration

### Environment Variables

#### Basic Configuration
```bash
# Core settings
export STATWHY_LOG_LEVEL=INFO
export STATWHY_CACHE_SIZE=1000
export STATWHY_MAX_WORKERS=4
export STATWHY_TIMEOUT=300

# Database settings
export STATWHY_DB_HOST=localhost
export STATWHY_DB_PORT=5432
export STATWHY_DB_NAME=statwhy
export STATWHY_DB_USER=statwhy_user

# Security settings
export STATWHY_SECRET_KEY=your-secret-key
export STATWHY_SSL_ENABLED=true
```

#### Advanced Configuration
```bash
# Performance settings
export STATWHY_PARALLEL_PROCESSING=true
export STATWHY_CACHE_TTL=3600
export STATWHY_MAX_MEMORY=2GB

# Monitoring settings
export STATWHY_METRICS_ENABLED=true
export STATWHY_PROMETHEUS_PORT=9090
export STATWHY_GRAFANA_PORT=3000
```

### Configuration Files

#### YAML Configuration
```yaml
# config.yaml
verification:
  timeout: 300
  max_iterations: 1000
  cache_enabled: true
  parallel_processing: true

performance:
  max_workers: 4
  chunk_size: 1000
  memory_limit: "2GB"
  cache_size: 1000
  cache_ttl: 3600

web:
  host: "0.0.0.0"
  port: 8000
  debug: false
  ssl_enabled: false
  max_upload_size: "100MB"

database:
  host: "localhost"
  port: 5432
  name: "statwhy"
  user: "statwhy_user"
  password: ""

monitoring:
  metrics_enabled: true
  prometheus_port: 9090
  grafana_port: 3000
  health_check_interval: 30
```

#### Python Configuration
```python
# config.py
from statwhy.models import Configuration

config = Configuration(
    verification={
        'timeout': 300,
        'max_iterations': 1000,
        'cache_enabled': True,
        'parallel_processing': True
    },
    performance={
        'max_workers': 4,
        'chunk_size': 1000,
        'memory_limit': '2GB'
    },
    web={
        'host': '0.0.0.0',
        'port': 8000,
        'debug': False
    }
)

# Use configuration
statwhy = StatWhyPython(config=config)
```

## Troubleshooting

### Common Issues

#### Why3 Not Found
```bash
# Error: Why3 not found in PATH
# Solution: Install Why3 and add to PATH

# Install Why3
opam install why3

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.opam/default/bin:$PATH"

# Verify installation
why3 --version
```

#### Cameleer Not Found
```bash
# Error: Cameleer not found in PATH
# Solution: Install Cameleer plugin

# Install Cameleer
opam install cameleer

# Verify installation
why3 --list-provers | grep cameleer
```

#### Memory Issues
```bash
# Error: Memory allocation failed
# Solution: Adjust memory limits

# Set environment variables
export STATWHY_MAX_MEMORY=1GB
export STATWHY_CACHE_SIZE=500

# Or use configuration file
statwhy verify --test ttest --data data.csv --config low_memory_config.yaml
```

#### Performance Issues
```bash
# Slow verification times
# Solution: Enable optimization features

# Enable caching
statwhy verify --test ttest --data data.csv --cache

# Enable parallel processing
statwhy verify --test ttest --data data.csv --parallel

# Use optimized configuration
statwhy verify --test ttest --data data.csv --config optimized_config.yaml
```

### Debug Mode

#### Enable Debug Output
```bash
# Enable debug mode
statwhy --debug verify --test ttest --data data.csv

# Verbose output
statwhy verify --test ttest --data data.csv --verbose

# Debug specific components
statwhy verify --test ttest --data data.csv --debug-cache --debug-verification
```

#### Log Analysis
```python
import logging

# Set debug logging level
logging.getLogger("statwhy").setLevel(logging.DEBUG)

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)

# Run verification with debug logging
result = statwhy.verify_test(
    test_type="ttest",
    data=df,
    alpha=0.05
)
```

### Getting Help

#### Documentation
- **User Guide**: This document
- **Technical Documentation**: `TECHNICAL_DOCUMENTATION.md`
- **API Reference**: `README_COMPREHENSIVE_FEATURES.md`
- **Examples**: `examples/` directory

#### Support Channels
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Comprehensive guides and examples
- **Community**: Join the StatWhy community

#### Reporting Issues
When reporting issues, please include:
1. **Error message**: Complete error traceback
2. **Environment**: OS, Python version, StatWhy version
3. **Reproduction steps**: How to reproduce the issue
4. **Expected behavior**: What you expected to happen
5. **Actual behavior**: What actually happened
6. **Data sample**: Small sample of data causing the issue

## Advanced Features

### Custom Statistical Tests

#### Implementing Custom Tests
```python
class CustomStatisticalTest:
    """Custom statistical test implementation."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def calculate_statistic(self, data, parameters):
        """Calculate custom test statistic."""
        # Your implementation here
        pass
    
    def calculate_p_value(self, statistic, data, parameters):
        """Calculate p-value for custom test."""
        # Your implementation here
        pass
    
    def check_assumptions(self, data):
        """Check assumptions for custom test."""
        # Your implementation here
        pass

# Register custom test
statwhy.register_custom_test(CustomStatisticalTest("my_test", "My custom test"))
```

### Advanced Data Processing

#### Custom Data Transformations
```python
# Custom data preprocessing
def custom_preprocessor(data, test_type):
    """Custom data preprocessing function."""
    # Your preprocessing logic here
    processed_data = data.copy()
    
    # Example: Custom outlier detection
    if test_type == "ttest":
        processed_data = remove_custom_outliers(processed_data)
    
    # Example: Custom normalization
    if test_type == "anova":
        processed_data = custom_normalize(processed_data)
    
    return processed_data

# Register custom preprocessor
statwhy.register_data_preprocessor(custom_preprocessor)
```

### Integration with External Tools

#### Jupyter Notebook Integration
```python
# Load StatWhy extension
%load_ext statwhy

# Use magic commands
%statwhy_verify --test ttest --data data.csv --alpha 0.05

# Generate tutorial notebook
tutorial = statwhy.generate_jupyter_tutorial()
tutorial.save("statwhy_tutorial.ipynb")
```

#### API Integration
```python
import requests

# StatWhy REST API
api_url = "http://localhost:8000/api"

# Verify test via API
response = requests.post(f"{api_url}/verify", json={
    "test_type": "ttest",
    "data": df.to_dict(),
    "alpha": 0.05
})

result = response.json()
print(f"Verification result: {result}")
```

## Examples and Tutorials

### Basic Usage
```python
from statwhy import StatWhyPython
import pandas as pd

# Load data
df = pd.read_csv("clinical_data.csv")

# Initialize StatWhy
statwhy = StatWhyPython()

# Verify t-test
result = statwhy.verify_test(
    test_type="ttest",
    data=df,
    alpha=0.05
)

print(f"Verification successful: {result.success}")
print(f"P-value: {result.p_value}")
print(f"Assumptions met: {result.assumptions_met}")
```

### Advanced Usage
```python
# Comprehensive verification with assumption analysis
result = statwhy.verify_test_comprehensive(
    test_type="anova",
    data=df,
    alpha=0.05,
    assumptions=True,
    visualization=True,
    export_format="html"
)

# Generate custom dataset
custom_data = statwhy.generate_custom_data(
    n_samples=500,
    distributions=["normal", "exponential", "uniform"],
    correlations=[[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]
)
```

## Contributing

### Development Setup
```bash
git clone https://github.com/your-org/statwhy.git
cd statwhy
pip install -e .
pip install -r requirements-dev.txt
```

### Testing
```bash
pytest tests/ -v
pytest tests/ --benchmark-only
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Why3 team for the formal verification framework
- Cameleer developers for OCaml integration
- Statistical community for methodology validation
- Contributors and beta testers

---

**StatWhy**: Where Statistical Rigor Meets Mathematical Proof
