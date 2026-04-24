"""
Multi-organ segmentation from a CT-like volume using classical SimpleITK
operators (thresholding, morphology, connected components).

This is the "baseline" pipeline a medical-imaging specialist would run
before handing volumes to nnU-Net / TotalSegmentator for AI segmentation.
The same API (in / out SimpleITK images) plugs into Slicer as a scripted
module — see scripts/slicer/slicer_segmentation_module.py.

Usage:
    python segmentation_pipeline.py        # runs on synthetic phantom
"""

from __future__ import annotations
import numpy as np
import SimpleITK as sitk

from synth_phantom import build_phantom


# --- Hounsfield windows (same conventions Slicer uses) ---
HU_LUNG = (-1000, -500)
HU_SOFT = (-100, 200)
HU_BONE = (250, 2000)


def threshold(img: sitk.Image, lo: float, hi: float) -> sitk.Image:
    return sitk.BinaryThreshold(img, lowerThreshold=lo, upperThreshold=hi,
                                insideValue=1, outsideValue=0)


def keep_largest(mask: sitk.Image, n: int = 1) -> sitk.Image:
    cc = sitk.ConnectedComponent(mask)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.BinaryThreshold(cc, 1, n, 1, 0)


def segment_lungs(ct: sitk.Image) -> sitk.Image:
    """Lungs: low-HU voxels inside the body envelope."""
    body = segment_body(ct)
    m = threshold(ct, -950, -450)
    m = sitk.And(m, body)
    m = sitk.BinaryMorphologicalOpening(m, [1, 1, 1])
    m = sitk.BinaryMorphologicalClosing(m, [2, 2, 2])
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(m), sortByObjectSize=True)
    return sitk.BinaryThreshold(cc, 1, 2, 1, 0)


def segment_bone(ct: sitk.Image) -> sitk.Image:
    m = threshold(ct, HU_BONE[0], HU_BONE[1])
    m = sitk.BinaryMorphologicalClosing(m, [2, 2, 2])
    return keep_largest(m, n=1)


def segment_liver_region_growing(
    ct: sitk.Image,
    seed_ijk: tuple[int, int, int] | None = None,
    hu_range: tuple[float, float] = (45.0, 110.0),
) -> sitk.Image:
    """Seeded confidence-connected region growing — the classical Slicer
    SegmentEditor 'Grow from seeds' approach before nnU-Net got good.

    If no seed is provided, one is auto-picked at the centroid of the
    hyper-dense hotspot near the expected liver location.
    """
    size = ct.GetSize()  # (x, y, z)
    if seed_ijk is None:
        # Auto-seed: centroid of the largest soft-tissue blob in the inferior
        # half of the volume (excluding bone, lungs, and body-wall).
        # This is roughly where a radiologist would click in Slicer.
        arr = sitk.GetArrayFromImage(ct)  # (z, y, x)
        z_half = arr.shape[0] // 2
        mask = (arr >= hu_range[0]) & (arr <= hu_range[1])
        mask[:z_half] = False  # inferior half only
        # erode a bit so we don't pick up body-wall voxels
        from scipy import ndimage as ndi
        mask = ndi.binary_erosion(mask, iterations=3)
        # largest connected component
        lbl, n = ndi.label(mask)
        if n:
            sizes = ndi.sum(mask, lbl, range(1, n + 1))
            biggest = 1 + int(np.argmax(sizes))
            zs, ys, xs = np.where(lbl == biggest)
            seed_ijk = (int(xs.mean()), int(ys.mean()), int(zs.mean()))
        else:
            seed_ijk = (size[0] // 2, size[1] // 2, int(size[2] * 0.7))

    # Smooth lightly first, then run confidence-connected.
    smoothed = sitk.CurvatureFlow(
        sitk.Cast(ct, sitk.sitkFloat32), timeStep=0.125, numberOfIterations=5)
    seg = sitk.ConfidenceConnected(
        smoothed, seedList=[seed_ijk],
        numberOfIterations=2, multiplier=2.0,
        initialNeighborhoodRadius=2, replaceValue=1,
    )
    seg = sitk.BinaryMorphologicalClosing(seg, [2, 2, 2])
    seg = sitk.BinaryFillhole(seg)
    # Clip to the plausible HU window.
    window = threshold(ct, hu_range[0], hu_range[1] + 10)  # +10 to capture lesion
    seg = sitk.And(seg, window)
    return keep_largest(seg, n=1)


def segment_soft_tissue(ct: sitk.Image, body_mask: sitk.Image) -> sitk.Image:
    """Liver via seeded region growing (closer to what Slicer actually does)."""
    return segment_liver_region_growing(ct)


def segment_body(ct: sitk.Image) -> sitk.Image:
    """Envelope of the patient. We keep all voxels with HU > -400 (soft/bone),
    take the largest component, then slice-by-slice convex-hull fill so
    internal air (lungs) is included."""
    m = threshold(ct, -400, 3000)
    m = sitk.BinaryMorphologicalClosing(m, [4, 4, 2])
    m = keep_largest(m, n=1)
    arr = sitk.GetArrayFromImage(m).astype(bool)
    from scipy import ndimage as ndi
    filled = np.zeros_like(arr)
    for z in range(arr.shape[0]):
        filled[z] = ndi.binary_fill_holes(arr[z])
    out = sitk.GetImageFromArray(filled.astype(np.uint8))
    out.CopyInformation(ct)
    return out


def dice(a: sitk.Image, b: sitk.Image) -> float:
    arr_a = sitk.GetArrayFromImage(a).astype(bool)
    arr_b = sitk.GetArrayFromImage(b).astype(bool)
    inter = np.logical_and(arr_a, arr_b).sum()
    denom = arr_a.sum() + arr_b.sum()
    return float(2.0 * inter / denom) if denom else 0.0


def hausdorff95(a: sitk.Image, b: sitk.Image) -> float:
    """Symmetric 95th-percentile Hausdorff distance in mm."""
    f = sitk.HausdorffDistanceImageFilter()
    try:
        f.Execute(a, b)
        return float(f.GetHausdorffDistance())
    except RuntimeError:
        return float("nan")


def run_pipeline(ct: sitk.Image) -> dict[str, sitk.Image]:
    body = segment_body(ct)
    lungs = segment_lungs(ct)
    bone = segment_bone(ct)
    liver = segment_soft_tissue(ct, body)
    return {"body": body, "lungs": lungs, "bone": bone, "liver": liver}


if __name__ == "__main__":
    vol, gt, ct, gt_img = build_phantom()
    pred = run_pipeline(ct)

    # ground-truth masks from the phantom
    gt_arr = sitk.GetArrayFromImage(gt_img)
    gt_masks = {
        "body": sitk.GetImageFromArray((gt_arr >= 1).astype(np.uint8)),
        "lungs": sitk.GetImageFromArray((gt_arr == 2).astype(np.uint8)),
        "bone": sitk.GetImageFromArray((gt_arr == 3).astype(np.uint8)),
        "liver": sitk.GetImageFromArray(((gt_arr == 4) | (gt_arr == 5)).astype(np.uint8)),
    }
    for m in gt_masks.values():
        m.CopyInformation(ct)

    print(f"{'structure':<10} {'Dice':>8} {'HD95 (mm)':>10}")
    print("-" * 32)
    for k in pred:
        d = dice(pred[k], gt_masks[k])
        h = hausdorff95(pred[k], gt_masks[k])
        print(f"{k:<10} {d:>8.3f} {h:>10.2f}")
