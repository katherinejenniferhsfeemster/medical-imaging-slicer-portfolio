# 3D Slicer Python extension — Segmentation Review

A scripted module for 3D Slicer 5.x that wraps the same SimpleITK pipeline
used everywhere else in this repo, so a radiologist can:

1. Load a DICOM series into a Volume node (via Slicer's DICOM browser).
2. Run the baseline body / lungs / bone / liver pipeline.
3. See the result as a `vtkMRMLSegmentationNode` with sensible segment
   names and colours.
4. Compute Dice / HD95 against a reference segmentation that's already in
   the scene (e.g. an nnU-Net prediction or a manually drawn ground truth).
5. Export the approved segmentation as DICOM-SEG for PACS ingest.

## Installation

Drop the file into a folder on your Slicer module path and add the path
under *Settings → Modules → Additional module paths*:

```
<your-extensions-dir>/SlicerSegmentationReview/
  └── SlicerSegmentationReview.py
```

After restart, the module appears under *Modules → Katherine Feemster →
Segmentation Review*.

## Dependencies inside Slicer

Slicer 5.x ships with SimpleITK, numpy, scipy, and matplotlib in its
embedded Python, so no extra install is needed. If you want the exact
pipeline the CI runs, point the module at this repo:

```python
# in Slicer's Python Interactor:
sys.path.append("/path/to/medical-imaging-slicer-portfolio/scripts/python")
slicer.util.reloadScriptedModule("SlicerSegmentationReview")
```

## Design choices

- **Logic is separated from Widget.** The `SlicerSegmentationReviewLogic`
  class is pure-Python and importable outside Slicer — that's what makes
  it possible to unit-test the exact same code that runs in the Slicer UI.
- **Uses `sitkUtils.PullVolumeFromSlicer` / `PushVolumeToSlicer`.** Round-
  tripping through NumPy would lose spacing and orientation. `sitkUtils`
  preserves the ITK-space metadata.
- **Export via DICOMSegmentationPlugin.** This is the DCMQI-based exporter
  that's been in Slicer since 4.11; writing DICOM-SEG manually is a
  recipe for broken files that don't round-trip through OHIF or MIM.

## Files

- `scripts/slicer/SlicerSegmentationReview.py` — the module itself

## Related

This module is the UI surface for the pipelines in:

- [`multi-organ-ct-segmentation`](../multi-organ-ct-segmentation) — the
  segmentation logic it invokes
- [`brain-mri-lesion-segmentation`](../brain-mri-lesion-segmentation) —
  which uses the same review workflow for nnU-Net output
