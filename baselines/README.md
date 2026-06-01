# Baseline Models

This folder contains baseline training and evaluation code for the vehicle Re-ID dataset.

The main benchmark setting should train or fine-tune each baseline model on a training identity split, then evaluate on held-out identities:

```text
train.csv   -> before + after crops from train identities
query.csv   -> after crops from held-out identities
gallery.csv -> before crops from held-out identities
```

Recommended initial baselines:

| Baseline | Torchreid model name |
| --- | --- |
| OSNet | `osnet_x1_0` |
| ResNet50 | `resnet50` |

The pretrained-only OSNet result can be kept as a sanity check, but the paper benchmark table should use fine-tuned/trained baselines.
