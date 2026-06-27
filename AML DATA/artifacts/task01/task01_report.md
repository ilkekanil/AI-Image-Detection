# Task 1.1 Dataset Exploration and Cleaning

## Main findings

The training split contains 29688 images. The original source classes are balanced, but
after merging classes 1-5 into the AI-generated class, the binary task is imbalanced:
real=4948 and AI-generated=24740.
All readable training images are JPEG/RGB.

| source class | meaning | images | most common original dimension | mean JPEG bytes |
|---:|---|---:|---|---:|
| 0 | real | 4948 | 640x480 (1073) | 51023 |
| 1 | sd_2_1 | 4948 | 320x320 (4948) | 31898 |
| 2 | sdxl | 4948 | 320x320 (4948) | 27280 |
| 3 | sd_3 | 4948 | 320x320 (4948) | 30585 |
| 4 | dall_e_3 | 4948 | 270x270 (4922) | 17955 |
| 5 | midjourney | 4948 | 320x320 (4948) | 25715 |

## Shortcut characteristics

The most important dataset artifact is that original dimensions are highly predictive of the source
class in the training split:

- class 0 (real) has nan% images at 320x320
- class 1 (sd_2_1) has 100.0% images at 320x320
- class 2 (sdxl) has 100.0% images at 320x320
- class 3 (sd_3) has 100.0% images at 320x320
- class 4 (dall_e_3) has 0.5% images at 320x320
- class 5 (midjourney) has 100.0% images at 320x320
- class 4 (dall_e_3) has 4922/4948 images at 270x270

This is dangerous because calibration, validation, augmented validation, and prediction data are expected
to be standardized to 320x320. A model trained directly on original dimensions or byte-size artifacts could
perform well on the training data for the wrong reason and fail on held-out data.

## Deterministic cleaning pipeline

The cleaning script applies the same deterministic transformation to every training image:

1. Decode bytes with Pillow and apply EXIF orientation.
2. Convert to RGB.
3. Center-crop to a square using the shorter side.
4. Resize to 224x224 with bicubic interpolation.
5. Re-encode as JPEG with quality=90, progressive=False, and fixed chroma subsampling.
6. Store cleaned image bytes plus `source_class`, merged `binary_label`, and original metadata in parquet shards.

This is cleaning rather than augmentation: each source image maps to exactly one cleaned image. The purpose is
to remove obvious dimension leakage, keep preprocessing reproducible, reduce input size for CPU-friendly
training, and preserve enough visual content for downstream modeling.

## Generated artifacts

- `summary.json`: machine-readable statistics.
- `metadata.csv`: one row per image with original dimensions, byte length, format, and descriptive statistics.
- `cleaned_train/*.parquet`: cleaned training shards.
- `*.png`: visualizations for the final PDF report.
