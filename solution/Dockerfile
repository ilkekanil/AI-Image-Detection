# Python 3.11 and requires a CPU-only image.
FROM python:3.11-slim

WORKDIR /workspace/solution

# Keep Python deterministic/non-interactive and place runtime caches below the
# only writable project directory required by the exercise specification.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/workspace/solution/artifacts/.matplotlib \
    XDG_CACHE_HOME=/workspace/solution/artifacts/.cache \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

# scikit-learn's Linux wheel uses the GNU OpenMP runtime. No compiler or GPU
# libraries are needed because all Python dependencies have binary CPU wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first so Docker can cache this expensive layer.
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
        torchvision==0.20.1

# data/ and generated artifacts/ are excluded by .dockerignore. During grading,
# data/ is mounted read-only and every script writes derived files to artifacts/.
COPY . .
RUN mkdir -p artifacts/.matplotlib artifacts/.cache

# Fail the build early if the scientific stack or project modules are broken.
RUN python -c "import matplotlib, numpy, pandas, PIL, pyarrow, sklearn, torch, torchvision; import clean, prepare, train, train_augmented, predict, predict_augmented, explain"
