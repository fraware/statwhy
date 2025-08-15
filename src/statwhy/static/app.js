/**
 * StatWhy Web Interface - Main Application JavaScript
 * Handles form submission, file uploads, and verification results display
 */

class StatWhyApp {
    constructor() {
        this.initializeEventListeners();
        this.setupDragAndDrop();
        this.currentFile = null;
    }

    initializeEventListeners() {
        // Form submission
        const form = document.getElementById('verificationForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleFormSubmission(e));
        }

        // File input change
        const fileInput = document.getElementById('dataFile');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelection(e));
        }

        // About modal
        const aboutLinks = document.querySelectorAll('[onclick="showAbout()"]');
        aboutLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.showAbout();
            });
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');
        if (!uploadArea) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.unhighlight, false);
        });

        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(e) {
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.add('drag-over');
    }

    unhighlight(e) {
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.remove('drag-over');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        this.handleFiles(files);
    }

    handleFileSelection(e) {
        const files = e.target.files;
        this.handleFiles(files);
    }

    handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            this.currentFile = file;
            this.updateUploadArea(file);
        }
    }

    updateUploadArea(file) {
        const uploadArea = document.getElementById('uploadArea');
        if (!uploadArea) return;

        uploadArea.innerHTML = `
            <div class="upload-content">
                <i class="fas fa-file-alt fa-3x mb-3 text-success"></i>
                <h5>File Selected: ${file.name}</h5>
                <p class="text-muted mb-3">
                    Size: ${this.formatFileSize(file.size)} | Type: ${file.type || 'Unknown'}
                </p>
                <button type="button" class="btn btn-outline-secondary btn-sm" onclick="this.resetUploadArea()">
                    <i class="fas fa-times me-2"></i>Change File
                </button>
            </div>
        `;
    }

    resetUploadArea() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('dataFile');
        
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="upload-content">
                    <i class="fas fa-cloud-upload-alt fa-3x mb-3 text-muted"></i>
                    <h5>Drag & Drop Your Data</h5>
                    <p class="text-muted mb-3">
                        Supports CSV, Excel (.xlsx, .xls), and JSON formats
                    </p>
                    <input type="file" id="dataFile" name="data_file" 
                           accept=".csv,.xlsx,.xls,.json" style="display: none;" required>
                    <button type="button" class="btn btn-outline-primary" 
                            onclick="document.getElementById('dataFile').click()">
                        <i class="fas fa-folder-open me-2"></i>Choose File
                    </button>
                </div>
            `;
        }
        
        if (fileInput) {
            fileInput.value = '';
        }
        
        this.currentFile = null;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async handleFormSubmission(e) {
        e.preventDefault();
        
        if (!this.currentFile) {
            this.showAlert('Please select a data file first.', 'warning');
            return;
        }

        const formData = new FormData();
        formData.append('data_file', this.currentFile);
        formData.append('test_type', document.getElementById('testType').value);
        formData.append('alpha', document.getElementById('alpha').value);

        // Validate form
        if (!this.validateForm()) {
            return;
        }

        // Show loading state
        this.setLoadingState(true);

        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.displayResults(result);
            } else {
                this.showAlert(`Verification failed: ${result.message || 'Unknown error'}`, 'danger');
            }
        } catch (error) {
            console.error('Verification error:', error);
            this.showAlert('An error occurred during verification. Please try again.', 'danger');
        } finally {
            this.setLoadingState(false);
        }
    }

    validateForm() {
        const testType = document.getElementById('testType').value;
        const alpha = document.getElementById('alpha').value;

        if (!testType) {
            this.showAlert('Please select a statistical test.', 'warning');
            return false;
        }

        if (!alpha) {
            this.showAlert('Please select a significance level.', 'warning');
            return false;
        }

        return true;
    }

    setLoadingState(loading) {
        const verifyBtn = document.getElementById('verifyBtn');
        if (verifyBtn) {
            if (loading) {
                verifyBtn.disabled = true;
                verifyBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                    Verifying...
                `;
            } else {
                verifyBtn.disabled = false;
                verifyBtn.innerHTML = `
                    <i class="fas fa-check-circle me-2"></i>Verify Procedure
                `;
            }
        }
    }

    displayResults(result) {
        const modal = new bootstrap.Modal(document.getElementById('resultsModal'));
        const resultsContent = document.getElementById('resultsContent');
        
        if (!resultsContent) return;

        const statusClass = result.success ? 'success' : 'danger';
        const statusIcon = result.success ? 'check-circle' : 'times-circle';
        const statusText = result.success ? 'Verification Successful' : 'Verification Failed';

        resultsContent.innerHTML = `
            <div class="alert alert-${statusClass}">
                <h5 class="alert-heading">
                    <i class="fas fa-${statusIcon} me-2"></i>${statusText}
                </h5>
                <p class="mb-0">${result.message}</p>
            </div>

            ${result.result ? this.formatVerificationResult(result.result) : ''}
        `;

        modal.show();
    }

    formatVerificationResult(result) {
        if (!result || !result.components) {
            return '<p>No detailed results available.</p>';
        }

        let html = `
            <h6 class="mt-4 mb-3">Verification Details</h6>
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Component</th>
                            <th>Status</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        result.components.forEach(component => {
            const statusClass = component.verified ? 'success' : 'danger';
            const statusIcon = component.verified ? 'check' : 'times';
            const statusText = component.verified ? 'Verified' : 'Failed';
            
            html += `
                <tr>
                    <td><strong>${component.name}</strong></td>
                    <td>
                        <span class="badge bg-${statusClass}">
                            <i class="fas fa-${statusIcon} me-1"></i>${statusText}
                        </span>
                    </td>
                    <td>${component.details || 'No details provided'}</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        if (result.verification_time) {
            html += `
                <div class="mt-3">
                    <small class="text-muted">
                        <i class="fas fa-clock me-1"></i>
                        Verification completed in ${result.verification_time.toFixed(2)} seconds
                    </small>
                </div>
            `;
        }

        return html;
    }

    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
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

    showAbout() {
        const modal = new bootstrap.Modal(document.getElementById('aboutModal'));
        modal.show();
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.statWhyApp = new StatWhyApp();
});

// Global function for resetting upload area (called from HTML)
function resetUploadArea() {
    if (window.statWhyApp) {
        window.statWhyApp.resetUploadArea();
    }
}

// Global function for showing about modal (called from HTML)
function showAbout() {
    if (window.statWhyApp) {
        window.statWhyApp.showAbout();
    }
}
