# it standardizes on Python 3.11 with a lightweight, CPU-focused base image.
FROM python:3.11-slim

WORKDIR /workspace/solution

# Enforces deterministic Python execution and redirects all runtime cache directories 
# to a subfolder within our project structure (avoiding write permission issues in read-only systems).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/workspace/solution/artifacts/.matplotlib \
    XDG_CACHE_HOME=/workspace/solution/artifacts/.cache \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

# installs LibGOMP (required by scikit-learn's CPU wheels) while keeping the image size 
# minimal. No CUDA or GPU compiler toolchains are needed here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cache the pip dependencies layer first to speed up future builds when code changes.
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
        torchvision==0.20.1

# Exclude raw datasets and generated artifacts via .dockerignore. 
# During evaluation, data splits are mounted read-only, and outputs are written to artifacts/.
COPY . .
RUN mkdir -p artifacts/.matplotlib artifacts/.cache

# runs a quick smoke test to catch any broken scientific packages or import errors before finishing the build.
RUN python -c "import matplotlib, numpy, pandas, PIL, pyarrow, sklearn, torch, torchvision; import clean, prepare, train, train_augmented, predict, predict_augmented, explain"
