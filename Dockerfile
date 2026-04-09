FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# STAGE 1: System dependencies
# ============================================================
RUN set -eux; \
    for i in 1 2 3; do \
      apt-get update && break; \
      echo "apt-get update failed, retrying ($i/3)..."; \
      sleep 5; \
    done; \
    apt-get install -y --no-install-recommends \
        wget ca-certificates gnupg software-properties-common \
        dirmngr locales git \
        build-essential gfortran \
        libxml2-dev \
        libglpk-dev \
        libgmp-dev \
        libblas-dev \
        liblapack-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        python3 python3-pip && \
    locale-gen en_GB.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# STAGE 2: Install Quarto
# ============================================================
RUN wget -qO /tmp/quarto.deb https://quarto.org/download/latest/quarto-linux-amd64.deb && \
    apt-get update && \
    apt-get install -y /tmp/quarto.deb && \
    rm /tmp/quarto.deb && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# STAGE 3: Install uv + Python deps
# ============================================================
WORKDIR /workspace

# Install uv
RUN pip install uv

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies into system Python
RUN uv sync --frozen --all-extras --dev

# ============================================================
# STAGE 4: Copy project
# ============================================================
COPY . .

# Install your package (editable)
RUN uv pip install -e .

CMD ["/bin/bash"]
