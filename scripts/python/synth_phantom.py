"""
Synthetic CT-like phantom generator.

Produces a 3D volume (numpy array + SimpleITK image with spacing) containing:
  - an outer soft-tissue ellipsoid (body)
  - two lungs (low-HU ellipsoids)
  - a spine (high-HU vertical cylinder)
  - a liver (soft-tissue sphere)
  - a "lesion" sphere inside the liver that scripts can learn to segment

The point is: no real patient data, but the geometry behaves like CT enough
to demonstrate the full pipeline (segmentation, registration, Dice, mosaic).
"""

from __future__ import annotations
import numpy as np
import SimpleITK as sitk


def _ellipsoid(shape, center, radii):
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    cz, cy, cx = center
    rz, ry, rx = radii
    return ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _cylinder_z(shape, center_yx, radius, z_range):
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    cy, cx = center_yx
    mask_r = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    mask_z = (zz >= z_range[0]) & (zz <= z_range[1])
    return mask_r & mask_z


def build_phantom(
    shape=(96, 160, 160),
    spacing=(2.0, 1.5, 1.5),     # mm, (z, y, x)
    seed=7,
):
    """Return (volume_hu, labels, sitk_image, sitk_labels)."""
    rng = np.random.default_rng(seed)
    vol = np.full(shape, -1000.0, dtype=np.float32)  # air

    Z, Y, X = shape
    cz, cy, cx = Z / 2, Y / 2, X / 2

    body = _ellipsoid(shape, (cz, cy, cx), (Z * 0.45, Y * 0.42, X * 0.38))
    vol[body] = 40.0  # soft tissue ~40 HU

    # Lungs occupy the upper (superior) portion of the thorax only.
    lung_r = (Z * 0.22, Y * 0.22, X * 0.14)
    lung_cz = cz - Z * 0.18  # shift lungs superior
    lung_L = _ellipsoid(shape, (lung_cz, cy - 4, cx - X * 0.18), lung_r)
    lung_R = _ellipsoid(shape, (lung_cz, cy - 4, cx + X * 0.18), lung_r)
    vol[lung_L | lung_R] = -820.0  # lung parenchyma

    # Spine: posterior midline.
    spine = _cylinder_z(shape, (cy + Y * 0.28, cx), radius=min(Y, X) * 0.055,
                        z_range=(int(Z * 0.05), int(Z * 0.95)))
    vol[spine] = 380.0  # cortical bone

    # Liver: inferior to lungs, right side (patient-left of image), anterior.
    liver_cz = cz + Z * 0.18
    liver = _ellipsoid(shape, (liver_cz, cy - Y * 0.04, cx + X * 0.12),
                       (Z * 0.14, Y * 0.18, X * 0.20))
    liver = liver & body & ~lung_L & ~lung_R & ~spine
    vol[liver] = 60.0

    # Focal lesion inside the liver, hyper-dense.
    lesion_center = (liver_cz - 1, cy - Y * 0.02, cx + X * 0.16)
    lesion = _ellipsoid(shape, lesion_center, (5.5, 6.5, 6.5)) & liver
    vol[lesion] = 95.0  # hyper-dense lesion

    # realistic noise (low-dose CT-ish, keeps HU windows separable)
    vol += rng.normal(0.0, 6.0, size=shape).astype(np.float32)

    # labels: 0 background, 1 body, 2 lung, 3 spine, 4 liver, 5 lesion
    labels = np.zeros(shape, dtype=np.uint8)
    labels[body] = 1
    labels[lung_L | lung_R] = 2
    labels[spine] = 3
    labels[liver] = 4
    labels[lesion] = 5

    img = sitk.GetImageFromArray(vol)
    img.SetSpacing(spacing[::-1])  # sitk uses (x, y, z)
    lbl = sitk.GetImageFromArray(labels)
    lbl.SetSpacing(spacing[::-1])
    lbl.CopyInformation(img)

    return vol, labels, img, lbl


if __name__ == "__main__":
    vol, lbl, img, lbl_img = build_phantom()
    print("phantom shape", vol.shape, "spacing", img.GetSpacing())
    print("label counts:", {int(k): int(v) for k, v in zip(*np.unique(lbl, return_counts=True))})
