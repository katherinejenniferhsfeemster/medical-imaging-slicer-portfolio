# CT ↔ MRI registration toolkit

Rigid + B-spline deformable registration in SimpleITK, matching the pipeline
Slicer's *General Registration (BRAINS)* module runs under the hood.

## Pipeline

1. **Centre-of-mass initialisation.** `CenteredTransformInitializer` aligns
   the two volumes by their physical centres of gravity. This avoids the
   common failure where a rigid search starts hundreds of millimetres off
   and the optimiser gets stuck in a local minimum.
2. **Multi-resolution rigid.** `Euler3DTransform` optimised against Mattes
   Mutual Information — the correct metric for multi-modal registration
   (CT has HU, MRI has arbitrary intensity; MI couples them through their
   joint histogram). Three pyramid levels, shrink factors 4/2/1, gradient
   descent.
3. **B-spline deformable refinement.** 6×6×6 control-point mesh, cubic
   B-spline basis. Runs against the rigid-warped moving image so the
   deformable model only has to capture residual mismatch.

## Result on the synthetic phantom

```
mean |fixed - moving|, HU
  before         :   82.05
  after rigid    :   44.71  (+45.5 %)
  after B-spline :   16.12  (+80.4 %)
```

The script applies a known rigid deformation to the phantom, then recovers
it — so the test is self-contained. Swap in real DICOM paths at the two
commented lines in `registration.py` and the rest of the pipeline runs
unchanged.

## Why Mattes Mutual Information

Normalized Cross Correlation assumes a linear intensity relationship between
fixed and moving — fine for CT-to-CT, broken for CT-to-MRI. Mean Squares is
even more restrictive. Mattes MI works as long as there's a consistent
(but unknown) statistical relationship between intensities, which is the
realistic assumption for multi-modal registration.

## Files

- `scripts/python/registration.py` — full rigid + B-spline pipeline, runs
  end-to-end on the synthetic phantom

## Running

```bash
python scripts/python/registration.py
```

## Integration with Slicer

The same transforms can be loaded in Slicer via `vtkMRMLTransformNode` and
applied to a volume node with *Transforms → Apply*. Our pipeline writes
rigid transforms as `.tfm` and B-spline transforms as `.h5` — both formats
Slicer reads natively.
