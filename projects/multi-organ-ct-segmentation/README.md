# Multi-organ CT segmentation pipeline

A classical SimpleITK pipeline that segments body, lungs, bone, and liver from
a thoracoabdominal CT. It's the baseline I run on every new study before
handing volumes to nnU-Net or TotalSegmentator — fast, deterministic, and
good enough as a sanity check on the AI output.

## What it does

| Structure | Method | HU window | Dice on phantom |
|-----------|--------|-----------|-----------------|
| Body envelope | Threshold + 2D hole-filling per axial slice | > -400 | 0.97 |
| Lungs | Threshold inside body, largest 2 components | (-950, -450) | 0.81 |
| Bone | Threshold + largest component | (250, 2000) | 1.00 |
| Liver | Seeded confidence-connected region growing | (45, 110) | 0.97 |

Metrics are computed against the ground-truth labels in the synthetic phantom
(`scripts/python/synth_phantom.py`). No real patient data is touched.

## Why these choices

- **Body via 2D fill-holes.** A 3D closing operation strong enough to seal
  the lungs will also eat thin features like ribs. Slicing axially and
  filling in 2D keeps the ribs intact.
- **Seeded region growing for liver.** Pure thresholding can't separate the
  liver from other abdominal soft tissue — they all live in the same HU
  window. The pipeline auto-picks a seed by finding the centroid of the
  largest 3D soft-tissue blob in the inferior half of the volume. A
  radiologist would do the same thing with one click in Slicer's
  SegmentEditor.
- **HD95 instead of plain Hausdorff.** Plain Hausdorff is dominated by the
  worst outlier voxel — a single stray edge pixel ruins the number. The
  95th percentile is the value clinical papers actually report.

## Files

- `scripts/python/synth_phantom.py` — parameterised phantom builder
- `scripts/python/segmentation_pipeline.py` — the actual pipeline
- `scripts/python/render_figures.py` — regenerates overlay and mosaic PNGs
- `docs/assets/slice_mosaic.png` — axial / coronal / sagittal review
- `docs/assets/segmentation_overlay.png` — GT outline vs. prediction fill

## Running

```bash
python scripts/python/segmentation_pipeline.py
# prints Dice and HD95 per structure
```

Swap in a real CT by replacing the `build_phantom()` call with
`sitk.ReadImage("study.nii.gz")` — the pipeline is pure SimpleITK and does
not care whether the voxels are synthetic.
