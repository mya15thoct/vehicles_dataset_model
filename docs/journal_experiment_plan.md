# Journal Experiment Plan (IEEE Access submission)

Master checklist mapping every paper table/figure to the command that
produces it. Priority order reflects what blocks the writing.

Environment: `conda activate recognition`, repo `~/mya/vehicles_dataset_model`.

```bash
SPLIT_ROOT=/mnt/recover/ngan/vehicles/reid_benchmark_identity_full
CROSS_ROOT=/mnt/recover/ngan/vehicles/reid_cross_condition
```

## Status snapshot (2026-07-20)

| Item | Status |
| --- | --- |
| WICV-Net full, osnet_x1_0 (test mAP 0.768) | done |
| WICV-Net full, resnet50 (test mAP 0.801) | done |
| Ablation 6 variants, osnet_x1_0 | done, but no_adv (0.815) > full (0.762) -> w_adv must be re-tuned |
| Baseline table (`results/baselines_full_e100`) | INCOMPLETE: only resnet101 finished |
| w_adv sensitivity sweep | pending (P1) |
| Cross-condition generalization | pending (P2) |
| Multi-seed mean±std | pending (P3) |
| Transformer backbone row | pending (P4) |
| Re-ranking rows | pending (cheap, after checkpoints exist) |
| Qualitative retrieval figures | pending |

## P1 — Fix the adversarial weight (decides the paper's story)

```bash
nohup python -u methods/wicv/run_sensitivity.py \
  --params w_adv w_cvpa \
  --results-root results/wicv_sensitivity \
  --skip-existing \
  > results/wicv_sensitivity.log 2>&1 &
```

Outcome A: some w_adv in {0.05, 0.1, 0.25} makes full beat no_adv -> keep the
three-component story; the sweep table goes in the paper as sensitivity
analysis, and the final model uses the best w_adv everywhere below.

Outcome B: w_adv=0 stays best -> reposition the method as CV-Tri + CVPA
(two components), report FCA honestly in a discussion subsection as a
negative result with the sweep as evidence. Update run_seeds / cross-condition
runs with `--w-adv <best>` accordingly.

## P1b — Complete the baseline table

Diagnose why only resnet101 finished, then rerun the remaining five:

```bash
tail -n 40 run_all_baselines_full_e100.log
ls results/baselines_full_e100/

nohup python -u baselines/torchreid/run_all.py \
  --manifest /mnt/recover/ngan/vehicles/reid_crops_full/manifest.csv \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --results-root results/baselines_full_e100 \
  --epochs 100 --eval-every 5 --patience 4 --batch-size 64 \
  --num-workers 4 --no-auto-split \
  > run_all_baselines_full_e100_resume.log 2>&1 &
```

The osnet_x1_0 row is mandatory: it is the same-backbone comparison for
WICV-Net. (The ablation `ce_only` row is an equivalent internal control.)

## P2 — Cross-condition generalization

Build the four protocols once (fast, CPU-only):

```bash
python scripts/build_cross_condition_splits.py \
  --split-root "$SPLIT_ROOT" \
  --output-root "$CROSS_ROOT"
```

Then train baseline + full per protocol (8 trainings; use best w_adv from P1):

```bash
nohup python -u methods/wicv/run_cross_condition.py \
  --protocol-root "$CROSS_ROOT" \
  --results-root results/wicv_cross_condition \
  --w-adv <BEST_W_ADV> \
  --skip-existing \
  > results/wicv_cross_condition.log 2>&1 &
```

Paper claim: WICV-Net degrades less than CE baseline on unseen domains.

## P3 — Multi-seed robustness (main table becomes mean±std)

```bash
nohup bash -c '
set -e
python -u methods/wicv/run_seeds.py --variant full --w-adv <BEST_W_ADV> --skip-existing
python -u methods/wicv/run_seeds.py --variant ce_only --skip-existing
' > results/wicv_seeds.log 2>&1 &
```

## P4 — Transformer backbone row (optional but strengthens tables)

```bash
nohup bash -c '
set -e
python -u methods/wicv/train.py \
  --model-name tv_swin_t --lr 1e-4 \
  --output-dir results/wicv/tv_swin_t_full \
  --epochs 60 --eval-every 5 --patience 4 --w-adv <BEST_W_ADV>
python -u methods/wicv/evaluate.py \
  --checkpoint results/wicv/tv_swin_t_full/model_best.pth
' > results/wicv_swin.log 2>&1 &
```

Notes: `tv_swin_t` works at the default 256x128. `tv_vit_b_16` requires
`--height 224 --width 224`. Use lr 1e-4 for transformer fine-tuning.

## Cheap add-ons once checkpoints exist

Re-ranked rows (adds "+ re-ranking" line to main table, needs ~6-8 GB RAM):

```bash
python -u methods/wicv/evaluate.py \
  --checkpoint results/wicv/osnet_x1_0_full/model_best.pth --rerank \
  --output results/wicv/osnet_x1_0_full/eval_rerank.json
```

Qualitative figures (success/failure retrieval strips per condition):

```bash
python -u methods/wicv/make_retrieval_figures.py \
  --checkpoint results/wicv/osnet_x1_0_full/model_best.pth \
  --output-root docs/figures/retrieval
```

## Paper table map

| Paper element | Source file |
| --- | --- |
| Table: main comparison (baselines vs WICV-Net, per backbone) | `results/baselines_full_e100/summary.csv` + `results/wicv/*/eval.json` + seeds summary |
| Table: component ablation | `results/wicv_ablation/summary.csv` |
| Table: per-condition breakdown | `eval.json` per_condition |
| Table: cross-condition generalization | `results/wicv_cross_condition/summary.csv` |
| Table/figure: loss-weight sensitivity | `results/wicv_sensitivity/summary.csv` |
| Rows: + re-ranking | `eval_rerank.json` |
| Figure: qualitative retrieval | `docs/figures/retrieval/*.jpg` |
| Figures: dataset overview/statistics | `scripts/make_paper_figures.py` outputs |

## Reviewer-proofing checklist

- [ ] Same-backbone baseline vs method comparison (osnet_x1_0)
- [ ] Ablation where full model is best or the story matches the numbers
- [ ] Sensitivity analysis for every introduced loss weight
- [ ] mean±std over >= 3 seeds for headline numbers
- [ ] Cross-domain generalization experiment
- [ ] A transformer-era comparison row
- [ ] Qualitative success AND failure cases
- [ ] Leakage audit reported (audit.json), identity-disjoint protocol described
- [ ] Code + dataset links in the paper (GitHub + Hugging Face)
