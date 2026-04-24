"""
Rigid + B-spline deformable registration in SimpleITK.

Demonstrates the canonical CT↔MRI workflow that Slicer's
"General Registration (BRAINS / Elastix)" module runs under the hood:

    1. Initialize with centre-of-mass alignment
    2. Multi-resolution rigid (Euler3D) with Mattes Mutual Information
    3. B-spline deformable refinement on top of the rigid result

We run it on two deformed copies of the synthetic phantom so the test is
self-contained — real DICOM paths plug in at the two commented lines.
"""

from __future__ import annotations
import numpy as np
import SimpleITK as sitk

from synth_phantom import build_phantom


def _deform(img: sitk.Image, tx_mm=(6.0, -4.0, 3.0), rot_deg=5.0) -> sitk.Image:
    tx = sitk.Euler3DTransform()
    tx.SetCenter(img.TransformContinuousIndexToPhysicalPoint(
        [s / 2 for s in img.GetSize()]))
    tx.SetTranslation(tx_mm)
    tx.SetRotation(np.deg2rad(rot_deg), np.deg2rad(-rot_deg / 2), np.deg2rad(rot_deg / 3))
    return sitk.Resample(img, img, tx, sitk.sitkLinear, -1000.0, img.GetPixelID())


def register_rigid(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20, seed=7)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=120,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInitialTransform(init, inPlace=False)
    return reg.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                       sitk.Cast(moving, sitk.sitkFloat32))


def register_bspline(fixed: sitk.Image, moving: sitk.Image,
                     initial: sitk.Transform) -> sitk.Transform:
    # Pre-resample moving through the rigid transform so the B-spline only
    # has to model residual deformation.
    pre = sitk.Resample(moving, fixed, initial, sitk.sitkLinear, -1000.0,
                        moving.GetPixelID())
    mesh = [6, 6, 6]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh, order=3)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10, seed=7)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0, numberOfIterations=40,
        convergenceMinimumValue=1e-6, convergenceWindowSize=5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(bspline, inPlace=True)
    reg.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                sitk.Cast(pre, sitk.sitkFloat32))
    return sitk.CompositeTransform([initial, bspline])


def apply(tx: sitk.Transform, moving: sitk.Image, ref: sitk.Image) -> sitk.Image:
    return sitk.Resample(moving, ref, tx, sitk.sitkLinear, -1000.0, moving.GetPixelID())


def mean_abs_error(a: sitk.Image, b: sitk.Image) -> float:
    return float(np.mean(np.abs(sitk.GetArrayFromImage(a) - sitk.GetArrayFromImage(b))))


if __name__ == "__main__":
    _, _, ct, _ = build_phantom()
    fixed = ct
    moving = _deform(ct, tx_mm=(7.5, -3.0, 4.0), rot_deg=6.0)
    # swap these two lines for real data:
    # fixed  = sitk.ReadImage("data/ct.nii.gz")
    # moving = sitk.ReadImage("data/mri.nii.gz")

    before = mean_abs_error(fixed, moving)

    rigid_tx = register_rigid(fixed, moving)
    after_rigid = mean_abs_error(fixed, apply(rigid_tx, moving, fixed))

    full_tx = register_bspline(fixed, moving, rigid_tx)
    after_bspline = mean_abs_error(fixed, apply(full_tx, moving, fixed))

    print(f"mean |fixed - moving|, HU")
    print(f"  before         : {before:8.2f}")
    print(f"  after rigid    : {after_rigid:8.2f}  ({100*(before-after_rigid)/before:+.1f} %)")
    print(f"  after B-spline : {after_bspline:8.2f}  ({100*(before-after_bspline)/before:+.1f} %)")
