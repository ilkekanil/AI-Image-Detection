# Task 1.4 Explainability Summary

## Method and model

The final Task 1.3 `StrongerCNNDetector` checkpoint was explained with Grad-CAM at convolutional layer
`feature_extractor.3.0`. Grad-CAM is appropriate because this CNN retains spatial convolutional feature maps before
global average pooling. Each displayed map uses the aggregate binary AI-versus-real logit (combining source
classes 1--5), ReLU, per-image normalization, and bilinear resizing. The saved calibrated Task 1.3 threshold
was 0.834237.

## Evaluation

| split | n | accuracy | AI recall | real-image FPR | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| validation | 1124 | 0.871 | 0.866 | 0.106 | 20 | 125 |
| validation_augmented | 1124 | 0.732 | 0.710 | 0.155 | 29 | 272 |

`source_class == 0` was treated as real; source classes 1--5 were merged as AI-generated. Results marked
partial in `metrics_summary.json` should not be treated as full-split estimates.

The relatively conservative threshold protects real images: FPR was 10.6% on validation and 15.5% on validation_augmented, while precision remained 97.6% and 95.8%. The cost is a false-negative-heavy error profile. AI recall fell from 86.6% to 71.0%, with false negatives increasing from 125 to 272.

## Representative examples

The deterministic selection requested confident true positives and true negatives, available false positives
and false negatives, then predictions closest to the calibrated threshold. It alternated between validation
splits when both supplied a category. 11 examples were selected and 11 received
the complete Grad-CAM and perturbation analysis.

## Real versus AI-generated attention

For 20 correctly classified real and 20 correctly classified AI-generated examples, the median fraction of saliency mass in the most salient 20% of pixels was 0.765 for real images and 0.357 for AI-generated images. Median normalized entropy was 0.897 versus 0.983, and median center mass was 0.202 versus 0.273. These are descriptive differences, not causal proof.

The accompanying panels show comparatively diffuse predicted-class attention over correctly classified
AI-generated subjects/scenes, whereas the selected real-image maps are more localized and often include image
boundaries. This visual pattern agrees with the concentration and entropy summaries, but it does not show that
the highlighted content caused the prediction.

## False positives and false negatives

There were 49 false positives and 397 false negatives in the evaluated rows. False positives had median AI score 0.914; 59.2% of them came from validation_augmented. False negatives had median AI score 0.658; 68.5% came from validation_augmented. In the deterministic full-run panels, confident false positives include visually unusual or staged real scenes with strong saturation, blur, or striking composition. Selected false negatives include photorealistic synthetic architecture, interiors, and dense city scenes. Their maps often emphasize boundaries or isolated structures rather than an obvious synthetic defect. These observations are plausible shortcut hypotheses, not automatically verified semantic explanations.

## Perturbation sanity check

Mean absolute AI-score change was 0.021 for salient occlusion and 0.013 for the low-saliency control. The salient effect was larger on average and exceeded the control in 81.8% of 11 comparisons. This provides partial faithfulness evidence, but the small changes and inconsistent per-image advantage do not validate every highlighted region.

Occlusion used an equally sized low-saliency region as a control. Changes in AI probability can have opposite
directions for real-class and AI-class explanations, so absolute changes are used for the aggregate comparison.

## Plausibility, shortcuts, and dataset bias

An explanation is more plausible when highlighted regions are visually relevant and salient-region occlusion
has a larger effect than the control. Even then, the result is only a basic faithfulness check. Center-heavy or
highly concentrated maps may indicate reliance on dominant subjects, while diffuse or border-heavy maps may
indicate texture, compression, padding, or background cues. The attention statistics and panels should be read
together; ambiguous differences remain ambiguous.

A validation-versus-augmented performance gap or concentration of errors in `validation_augmented` is
consistent with sensitivity to distortions, although it does not identify the cause. Source classes also come
from different generators, and real and synthetic images may differ in content, compression, resolution history,
or editing pipeline. Consequently, the classifier may learn generator/dataset signatures instead of a general
concept of image authenticity. Resizing reduces original-dimension shortcuts but cannot remove all acquisition
or encoding bias.

## Limitations

Grad-CAM is low resolution at the final convolutional layer and can omit evidence, merge distinct regions, or
look convincing when it is not faithful. Per-image min-max normalization hides absolute attribution strength.
Occlusion creates out-of-distribution content and measures association with a region rather than human-like or
causal reasoning. Visual explanations therefore are not definitive proof of why the model decided, and this
descriptive subset cannot establish general behavior for all images or unseen generators.
