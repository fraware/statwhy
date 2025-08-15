/**
 * StatWhy Examples Page JavaScript
 * Handles example filtering, display, and interactive features
 */

class ExamplesManager {
    constructor() {
        this.examples = [];
        this.filteredExamples = [];
        this.initializeEventListeners();
        this.loadExamples();
    }

    initializeEventListeners() {
        // Filter change events
        const categoryFilter = document.getElementById('categoryFilter');
        const difficultyFilter = document.getElementById('difficultyFilter');
        const testFilter = document.getElementById('testFilter');

        if (categoryFilter) {
            categoryFilter.addEventListener('change', () => this.applyFilters());
        }
        if (difficultyFilter) {
            difficultyFilter.addEventListener('change', () => this.applyFilters());
        }
        if (testFilter) {
            testFilter.addEventListener('change', () => this.applyFilters());
        }

        // Example modal events
        const runExampleBtn = document.getElementById('runExampleBtn');
        if (runExampleBtn) {
            runExampleBtn.addEventListener('click', () => this.runExample());
        }
    }

    async loadExamples() {
        try {
            // In a real implementation, this would fetch from an API
            // For now, we'll use the examples from the CLI
            this.examples = this.getExampleData();
            this.filteredExamples = [...this.examples];
            this.renderExamples();
        } catch (error) {
            console.error('Error loading examples:', error);
            this.showError('Failed to load examples. Please try refreshing the page.');
        }
    }

    getExampleData() {
        // This would normally come from an API endpoint
        return [
            // Clinical Trials & Medical Research
            {
                id: 1,
                test_type: "ttest",
                category: "clinical",
                description: "One-sample t-test for drug efficacy in clinical trials",
                difficulty: "Beginner",
                assumptions: ["Normal distribution", "Independent observations", "Known population mean"],
                verification_steps: ["Check normality", "Verify independence", "Validate sample size"],
                real_world_impact: "Ensures FDA compliance and patient safety in drug development",
                data_requirements: "Continuous data, minimum 30 observations",
                example_file: "examples/clinical_ttest_example.csv",
                code_example: `
# Example: One-sample t-test for drug efficacy
import statwhy
import pandas as pd

# Load clinical trial data
data = pd.read_csv('clinical_data.csv')
blood_pressure = data['systolic_bp']

# Verify the statistical procedure
result = statwhy.verify(
    test_type='ttest',
    data=blood_pressure,
    alpha=0.05,
    population_mean=120  # Normal systolic BP
)

if result.is_verified:
    print("✓ t-test assumptions verified - safe to proceed")
    # Perform the actual test
    t_stat, p_value = scipy.stats.ttest_1samp(blood_pressure, 120)
else:
    print("✗ t-test assumptions not met - review data or choose alternative test")
                `
            },
            {
                id: 2,
                test_type: "anova",
                category: "clinical",
                description: "ANOVA for comparing multiple treatment groups in clinical studies",
                difficulty: "Intermediate",
                assumptions: ["Normal distribution", "Equal variances", "Independent groups"],
                verification_steps: ["Normality test", "Levene's test", "Group independence check"],
                real_world_impact: "Critical for multi-arm clinical trials and treatment comparison",
                data_requirements: "Multiple groups, continuous outcomes, balanced design",
                example_file: "examples/clinical_anova_example.csv",
                code_example: `
# Example: ANOVA for treatment group comparison
import statwhy
import pandas as pd

# Load clinical trial data with multiple treatment groups
data = pd.read_csv('treatment_data.csv')

# Verify ANOVA assumptions
result = statwhy.verify(
    test_type='anova',
    data=data,
    alpha=0.05,
    group_column='treatment_group',
    response_column='recovery_time'
)

if result.is_verified:
    print("✓ ANOVA assumptions verified - safe to proceed")
    # Perform the actual ANOVA
    f_stat, p_value = scipy.stats.f_oneway(
        data[data['treatment_group'] == 'A']['recovery_time'],
        data[data['treatment_group'] == 'B']['recovery_time'],
        data[data['treatment_group'] == 'C']['recovery_time']
    )
else:
    print("✗ ANOVA assumptions not met - consider non-parametric alternative")
                `
            },
            {
                id: 3,
                test_type: "chi2",
                category: "clinical",
                description: "Chi-square test for treatment response rates and adverse events",
                difficulty: "Intermediate",
                assumptions: ["Independent observations", "Expected frequencies > 5", "Categorical data"],
                verification_steps: ["Independence check", "Expected frequency validation", "Sample size verification"],
                real_world_impact: "Essential for safety monitoring and efficacy assessment",
                data_requirements: "Categorical variables, sufficient sample sizes",
                example_file: "examples/clinical_chi2_example.csv",
                code_example: `
# Example: Chi-square test for treatment response
import statwhy
import pandas as pd

# Load clinical trial data
data = pd.read_csv('response_data.csv')

# Verify chi-square assumptions
result = statwhy.verify(
    test_type='chi2',
    data=data,
    alpha=0.05,
    categorical_columns=['treatment', 'response']
)

if result.is_verified:
    print("✓ Chi-square assumptions verified - safe to proceed")
    # Create contingency table and perform test
    contingency_table = pd.crosstab(data['treatment'], data['response'])
    chi2_stat, p_value, dof, expected = scipy.stats.chi2_contingency(contingency_table)
else:
    print("✗ Chi-square assumptions not met - consider Fisher's exact test")
                `
            },
            
            // Financial Risk Modeling
            {
                id: 4,
                test_type: "wilcoxon",
                category: "financial",
                description: "Wilcoxon signed-rank test for portfolio performance analysis",
                difficulty: "Intermediate",
                assumptions: ["Symmetric distribution", "Paired observations", "Ordinal data"],
                verification_steps: ["Symmetry check", "Pairing validation", "Distribution analysis"],
                real_world_impact: "Regulatory compliance in financial risk assessment",
                data_requirements: "Paired observations, ordinal or continuous data",
                example_file: "examples/financial_wilcoxon_example.csv",
                code_example: `
# Example: Wilcoxon test for portfolio performance
import statwhy
import pandas as pd

# Load financial data
data = pd.read_csv('portfolio_data.csv')

# Verify Wilcoxon assumptions
result = statwhy.verify(
    test_type='wilcoxon',
    data=data,
    alpha=0.05,
    paired_columns=['before_crisis', 'after_crisis']
)

if result.is_verified:
    print("✓ Wilcoxon assumptions verified - safe to proceed")
    # Perform the actual test
    w_stat, p_value = scipy.stats.wilcoxon(
        data['before_crisis'], 
        data['after_crisis']
    )
else:
    print("✗ Wilcoxon assumptions not met - consider alternative paired test")
                `
            },
            
            // Manufacturing & Quality Control
            {
                id: 5,
                test_type: "bartlett",
                category: "manufacturing",
                description: "Bartlett's test for homogeneity of variances in production processes",
                difficulty: "Advanced",
                assumptions: ["Normal distribution", "Independent samples", "Multiple groups"],
                verification_steps: ["Normality verification", "Independence check", "Group validation"],
                real_world_impact: "Critical for Six Sigma and quality control compliance",
                data_requirements: "Multiple groups, normal distributions, independent samples",
                example_file: "examples/manufacturing_bartlett_example.csv",
                code_example: `
# Example: Bartlett's test for production line variances
import statwhy
import pandas as pd

# Load manufacturing data
data = pd.read_csv('production_data.csv')

# Verify Bartlett's test assumptions
result = statwhy.verify(
    test_type='bartlett',
    data=data,
    alpha=0.05,
    group_column='production_line',
    response_column='quality_score'
)

if result.is_verified:
    print("✓ Bartlett's test assumptions verified - safe to proceed")
    # Perform the actual test
    bartlett_stat, p_value = scipy.stats.bartlett(
        data[data['production_line'] == 'Line_A']['quality_score'],
        data[data['production_line'] == 'Line_B']['quality_score'],
        data[data['production_line'] == 'Line_C']['quality_score']
    )
else:
    print("✗ Bartlett's test assumptions not met - consider Levene's test")
                `
            }
        ];
    }

    applyFilters() {
        const categoryFilter = document.getElementById('categoryFilter');
        const difficultyFilter = document.getElementById('difficultyFilter');
        const testFilter = document.getElementById('testFilter');

        const selectedCategory = categoryFilter ? categoryFilter.value : '';
        const selectedDifficulty = difficultyFilter ? difficultyFilter.value : '';
        const selectedTest = testFilter ? testFilter.value : '';

        this.filteredExamples = this.examples.filter(example => {
            const categoryMatch = !selectedCategory || example.category === selectedCategory;
            const difficultyMatch = !selectedDifficulty || example.difficulty === selectedDifficulty;
            const testMatch = !selectedTest || example.test_type === selectedTest;

            return categoryMatch && difficultyMatch && testMatch;
        });

        this.renderExamples();
    }

    renderExamples() {
        const examplesGrid = document.getElementById('examplesGrid');
        if (!examplesGrid) return;

        if (this.filteredExamples.length === 0) {
            examplesGrid.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        No examples match the selected filters. Try adjusting your criteria.
                    </div>
                </div>
            `;
            return;
        }

        examplesGrid.innerHTML = this.filteredExamples.map(example => `
            <div class="col-lg-6 col-xl-4 mb-4">
                <div class="card h-100 example-card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span class="badge bg-primary">${example.test_type.toUpperCase()}</span>
                        <span class="badge bg-${this.getDifficultyColor(example.difficulty)}">${example.difficulty}</span>
                    </div>
                    <div class="card-body">
                        <h5 class="card-title">${example.description}</h5>
                        <p class="card-text text-muted">
                            <i class="fas fa-tag me-1"></i>${example.category}
                        </p>
                        
                        <div class="mb-3">
                            <strong>Key Assumptions:</strong>
                            <ul class="list-unstyled mt-2">
                                ${example.assumptions.slice(0, 2).map(assumption => 
                                    `<li><i class="fas fa-check text-success me-2"></i>${assumption}</li>`
                                ).join('')}
                                ${example.assumptions.length > 2 ? 
                                    `<li class="text-muted">... and ${example.assumptions.length - 2} more</li>` : 
                                    ''
                                }
                            </ul>
                        </div>
                        
                        <div class="mb-3">
                            <strong>Real-World Impact:</strong>
                            <p class="text-muted small">${example.real_world_impact}</p>
                        </div>
                    </div>
                    <div class="card-footer">
                        <button class="btn btn-outline-primary btn-sm" onclick="examplesManager.showExampleDetails(${example.id})">
                            <i class="fas fa-info-circle me-2"></i>View Details
                        </button>
                        <button class="btn btn-primary btn-sm" onclick="examplesManager.runExample(${example.id})">
                            <i class="fas fa-play me-2"></i>Run Example
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    getDifficultyColor(difficulty) {
        switch (difficulty) {
            case 'Beginner': return 'success';
            case 'Intermediate': return 'warning';
            case 'Advanced': return 'danger';
            default: return 'secondary';
        }
    }

    showExampleDetails(exampleId) {
        const example = this.examples.find(ex => ex.id === exampleId);
        if (!example) return;

        const modal = new bootstrap.Modal(document.getElementById('exampleModal'));
        const modalTitle = document.getElementById('exampleModalTitle');
        const modalBody = document.getElementById('exampleModalBody');

        if (modalTitle) {
            modalTitle.innerHTML = `
                <i class="fas fa-chart-bar me-2"></i>${example.test_type.toUpperCase()} Example
            `;
        }

        if (modalBody) {
            modalBody.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Description</h6>
                        <p>${example.description}</p>
                        
                        <h6>Category</h6>
                        <p><span class="badge bg-primary">${example.category}</span></p>
                        
                        <h6>Difficulty</h6>
                        <p><span class="badge bg-${this.getDifficultyColor(example.difficulty)}">${example.difficulty}</span></p>
                        
                        <h6>Data Requirements</h6>
                        <p>${example.data_requirements}</p>
                    </div>
                    <div class="col-md-6">
                        <h6>Key Assumptions</h6>
                        <ul>
                            ${example.assumptions.map(assumption => `<li>${assumption}</li>`).join('')}
                        </ul>
                        
                        <h6>Verification Steps</h6>
                        <ol>
                            ${example.verification_steps.map(step => `<li>${step}</li>`).join('')}
                        </ol>
                        
                        <h6>Real-World Impact</h6>
                        <p>${example.real_world_impact}</p>
                    </div>
                </div>
                
                <div class="mt-4">
                    <h6>Code Example</h6>
                    <pre><code class="language-python">${example.code_example}</code></pre>
                </div>
                
                <div class="mt-4">
                    <h6>Why This Matters</h6>
                    <div class="alert alert-info">
                        <i class="fas fa-lightbulb me-2"></i>
                        <strong>Educational Insight:</strong> This example demonstrates how formal verification 
                        catches potential errors that traditional statistical software cannot detect. 
                        By verifying assumptions before running tests, we ensure reliable and 
                        mathematically sound results.
                    </div>
                </div>
            `;
        }

        // Store current example for run button
        this.currentExample = example;
        modal.show();
    }

    async runExample(exampleId) {
        const example = this.examples.find(ex => ex.id === exampleId);
        if (!example) return;

        try {
            // Show loading state
            this.showLoading('Preparing example data...');

            // In a real implementation, this would create sample data and run verification
            // For now, we'll simulate the process
            await this.simulateExampleRun(example);

        } catch (error) {
            console.error('Error running example:', error);
            this.showError('Failed to run example. Please try again.');
        }
    }

    async simulateExampleRun(example) {
        // Simulate verification process
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Show results
        this.showResults(example);
    }

    showLoading(message) {
        const modal = new bootstrap.Modal(document.getElementById('exampleModal'));
        const modalBody = document.getElementById('exampleModalBody');

        if (modalBody) {
            modalBody.innerHTML = `
                <div class="text-center py-5">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5>${message}</h5>
                    <p class="text-muted">This may take a few moments...</p>
                </div>
            `;
        }

        modal.show();
    }

    showResults(example) {
        const modal = new bootstrap.Modal(document.getElementById('exampleModal'));
        const modalBody = document.getElementById('exampleModalBody');

        if (modalBody) {
            modalBody.innerHTML = `
                <div class="alert alert-success">
                    <h5><i class="fas fa-check-circle me-2"></i>Example Completed Successfully!</h5>
                    <p>This demonstrates how StatWhy verifies statistical procedures before they are applied to data.</p>
                </div>
                
                <h6>What Was Verified</h6>
                <ul>
                    ${example.verification_steps.map(step => `<li>${step}</li>`).join('')}
                </ul>
                
                <h6>Next Steps</h6>
                <p>Now that you've seen how verification works, you can:</p>
                <ul>
                    <li>Upload your own data for verification</li>
                    <li>Explore other statistical tests</li>
                    <li>Learn more about formal verification</li>
                </ul>
                
                <div class="text-center mt-4">
                    <a href="/" class="btn btn-primary">
                        <i class="fas fa-upload me-2"></i>Try Your Own Data
                    </a>
                </div>
            `;
        }

        modal.show();
    }

    showError(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
        }
    }
}

// Initialize the examples manager when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.examplesManager = new ExamplesManager();
});
