FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=en_GB.UTF-8 \
    LANG=en_GB.UTF-8 \
    PYTHONUNBUFFERED=1

# ============================================================
# STAGE 1: System dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    curl \
    gnupg \
    locales \
    git \
    build-essential \
    libxml2-dev \
    libglpk-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    python3 \
    python3-pip \
    && locale-gen en_GB.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# STAGE 2: Install Quarto
# ============================================================
RUN wget -qO /tmp/quarto.deb https://quarto.org/download/latest/quarto-linux-amd64.deb && \
    apt-get update && \
    apt-get install -y /tmp/quarto.deb && \
    rm /tmp/quarto.deb && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# STAGE 3: Install uv
# ============================================================
# This grabs the binary from the official image and puts it in your path
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Workspace Setup
WORKDIR /workspace

# Install Dependencies
# We copy only these first to leverage Docker layer caching
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-install-project --all-extras

# Copy Project and Install
COPY . .
RUN uv sync --frozen --all-extras

# Ensure the virtualenv is used by default
ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["/bin/bash"]
