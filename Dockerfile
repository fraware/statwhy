# Multi-stage Dockerfile for StatWhy
# Stage 1: Base image with system dependencies
FROM ubuntu:22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/opam/default/bin:$PATH"
ENV OPAMROOT="/opt/opam"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libcairo2-dev \
    libgtk-3-dev \
    libgtksourceview-3.0-dev \
    cvc5 \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install OCaml and OPAM
RUN wget https://github.com/ocaml/opam/releases/download/2.1.5/opam-2.1.5-x86_64-linux -O /usr/local/bin/opam \
    && chmod +x /usr/local/bin/opam

# Initialize OPAM and install OCaml
RUN opam init --disable-sandboxing --compiler=5.0.0 --shell-setup \
    && opam install -y dune ocamlfind

# Stage 2: Python environment
FROM base as python

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.9 /usr/bin/python \
    && ln -s /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Stage 3: Development environment
FROM python as development

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    flake8 \
    mypy \
    pre-commit \
    jupyter \
    ipykernel

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY src/ ./src/

# Install StatWhy in development mode
RUN pip install -e ".[dev]"

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 statwhy \
    && chown -R statwhy:statwhy /app

USER statwhy

# Expose ports
EXPOSE 8000

# Default command
CMD ["statwhy", "web", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Production environment
FROM python as production

# Install production dependencies only
RUN pip install \
    gunicorn \
    uvicorn[standard]

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY src/ ./src/

# Install StatWhy
RUN pip install .

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 statwhy \
    && chown -R statwhy:statwhy /app

USER statwhy

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default command
CMD ["gunicorn", "statwhy.web:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]

# Stage 5: Jupyter environment
FROM python as jupyter

# Install Jupyter dependencies
RUN pip install \
    jupyter \
    jupyterlab \
    ipykernel \
    notebook

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY src/ ./src/

# Install StatWhy with Jupyter support
RUN pip install -e ".[jupyter]"

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 statwhy \
    && chown -R statwhy:statwhy /app

USER statwhy

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]

# Stage 6: CLI-only environment
FROM python as cli

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY src/ ./src/

# Install StatWhy CLI only
RUN pip install -e ".[cli]"

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 statwhy \
    && chown -R statwhy:statwhy /app

USER statwhy

# Default command
CMD ["statwhy", "--help"]
