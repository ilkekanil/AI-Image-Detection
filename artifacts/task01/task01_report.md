# Task 1.1 Dataset Exploration and Cleaning

## Dataset Summary

The training split contains 29688 images. 29688 images were decoded
successfully and 0 images were skipped as corrupted or unreadable.
The original six source classes are preserved in the metadata. For later binary modeling, class `0` remains
`0` (real) and classes `1` to `5` are mapped to binary label `1` (AI-generated).

Binary distribution:

- real (`0`): 4948
- AI-generated (`1`): 24740

| source_class | meaning | images | most common original size | mean bytes | mean brightness | mean contrast |
|---:|---|---:|---|---:|---:|---:|
| 0 | real | 4948 | 640x480 (1073) | 51023 | 113.25 | 59.68 |
| 1 | sd_2_1 | 4948 | 320x320 (4948) | 31898 | 117.20 | 54.32 |
| 2 | sdxl | 4948 | 320x320 (4948) | 27280 | 119.29 | 46.96 |
| 3 | sd_3 | 4948 | 320x320 (4948) | 30585 | 135.73 | 65.21 |
| 4 | dall_e_3 | 4948 | 270x270 (4922) | 17955 | 115.76 | 60.29 |
| 5 | midjourney | 4948 | 320x320 (4948) | 25715 | 106.58 | 53.30 |

## Characteristics and Shortcut Risks

The exploration computes original width, height, aspect ratio, JPEG byte size, brightness, contrast, RGB
channel means, and RGB channel standard deviations for every valid image. The most important shortcut risk
is that original image dimensions are strongly related to the source class:

- source_class 0 (real) has 0.0% images with original size 320x320
- source_class 1 (sd_2_1) has 100.0% images with original size 320x320
- source_class 2 (sdxl) has 100.0% images with original size 320x320
- source_class 3 (sd_3) has 100.0% images with original size 320x320
- source_class 4 (dall_e_3) has 0.5% images with original size 320x320
- source_class 5 (midjourney) has 100.0% images with original size 320x320
- source_class 4 (dall_e_3) has 4922/4948 images with original size 270x270

Because later splits are standardized differently, a model should not be allowed to rely on original training
dimensions as a class cue. This motivates deterministic resizing during cleaning.

## Cleaning Pipeline

The cleaning pipeline is deterministic and applies exactly one cleaned output to each valid training image:

1. Decode JPEG bytes using Pillow.
2. Apply EXIF orientation handling.
3. Convert to RGB.
4. Resize to 224x224 with bicubic interpolation.
5. Save the cleaned result as a JPEG image.
6. Save `labels.csv` with image path, original source class, binary label, original size, and cleaned size.

RGB conversion is used to give later CPU-friendly models a consistent three-channel input. Resizing to
224x224 removes image-size shortcuts and reduces downstream training cost. No random augmentation is applied
because Task 1.1 is limited to exploration and cleaning; robustness augmentation belongs to later tasks.

## Current Output Files

- `train_metadata.csv`
- `summary.json`
- `cleaning_notes.txt`
- `cleaned_train/images/*.jpg`
- `cleaned_train/labels.csv`

Plots:

- `plots/class_distribution_source_class.png`
- `plots/binary_class_distribution.png`
- `plots/width_distribution.png`
- `plots/height_distribution.png`
- `plots/aspect_ratio_distribution.png`
- `plots/byte_size_distribution.png`
- `plots/brightness_by_source_class.png`
- `plots/contrast_by_source_class.png`
