# Brain MRI lesion segmentation — nnU-Net playbook

How I train and evaluate a 3D nnU-Net v2 model for MS / glioma lesion
segmentation, with 3D Slicer used as the review and correction tool at
every step.

## The training loop

1. **Curate the cohort.** T1, T2-FLAIR, and contrast-enhanced T1 per subject,
   bias-field corrected with N4 (SimpleITK), skull-stripped with HD-BET, then
   registered to T1-native space.
2. **Convert to nnU-Net format.** One `imagesTr/` folder per modality with
   `_0000`, `_0001`, `_0002` suffixes; one `labelsTr/` folder with integer
   masks. `dataset.json` lists channel names and label semantics.
3. **Plan and preprocess.** `nnUNetv2_plan_and_preprocess -d XXX` — writes
   the fold-specific resampling plan and a normalised dataset to disk.
4. **Train.** `nnUNetv2_train XXX 3d_fullres 0` for fold 0 of 5; repeat for
   folds 1–4 on a 4× A100 node.
5. **Review.** Every validation prediction is loaded into Slicer alongside
   the ground truth. Dice < 0.70 per lesion triggers a radiologist review;
   systematic failures go back into the next training round.

## Metrics that actually matter

Dice is necessary but not sufficient. For lesions we also report:

- **HD95 (mm)** — catches cases where Dice is fine but the lesion boundary
  is nowhere near the truth.
- **Lesion-level recall at 3 mm** — fraction of GT lesions that have *any*
  predicted voxel within 3 mm of their centroid. This is the clinically
  meaningful number; a missed lesion is worse than a slightly wrong outline.
- **False-positive rate per volume** — number of connected predicted
  components that don't overlap any GT lesion.

## Example training curves

The plot in `docs/assets/dice_curve.png` is a **synthetic** illustration of
the training behaviour you should expect — fast climb in the first ~50
epochs, plateau around 0.9 Dice for large organs and 0.7–0.8 for small
lesions. It's not a real training log; it's here so the portfolio shows
what a well-behaved nnU-Net run looks like.

Regenerate it from the script:

```bash
python scripts/python/dice_curve.py --out docs/assets/dice_curve.png
```

## Slicer review module

`scripts/slicer/SlicerSegmentationReview.py` adds a "Segmentation Review"
module to Slicer under a personal category. It loads the nnU-Net output
as a Segmentation node with colours matching the dataset.json spec,
computes Dice / HD95 against a reference segmentation if one is loaded,
and exports the reviewed result as DICOM-SEG.

## Files

- `scripts/python/dice_curve.py` — synthetic training curve generator
- `scripts/slicer/SlicerSegmentationReview.py` — Slicer module
- `docs/assets/dice_curve.png` — illustrative training plot
