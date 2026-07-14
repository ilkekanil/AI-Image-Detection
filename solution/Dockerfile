# Use Python 3.11 CPU-only base image
FROM python:3.11-slim

WORKDIR /workspace/solution

# Keep Python quiet/deterministic and route all temp caches to the writable artifacts 
# directory (the only folder we're guaranteed to have write access to during grading).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/workspace/solution/artifacts/.matplotlib \
    XDG_CACHE_HOME=/workspace/solution/artifacts/.cache \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

# Install libgomp1 (needed for scikit-learn's OpenMP runtime).
# No heavy compilers or GPU libraries are required since we use pre-built CPU wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to take advantage of Docker's layer caching
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
        torchvision==0.20.1

# Copy the remaining project files. Note: data/ and artifacts/ are ignored 
# via .dockerignore because the grading environment mounts data/ as read-only.
COPY . .
RUN mkdir -p artifacts/.matplotlib artifacts/.cache

# Sanity check the build to make sure imports and local scripts aren't broken
RUN python -c "import matplotlib, numpy, pandas, PIL, pyarrow, sklearn, torch, torchvision; import clean, prepare, train, train_augmented, predict, predict_augmented, explain"
