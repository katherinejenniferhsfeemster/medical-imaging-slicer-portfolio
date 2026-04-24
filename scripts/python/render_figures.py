"""
Regenerate every figure on the portfolio site from scratch.

Produces four PNGs into assets/renders/:
    1. slice_mosaic.png    — axial / coronal / sagittal triad of the phantom
    2. segmentation_overlay.png — ground truth vs. pipeline prediction
    3. volume_render.png   — maximum-intensity projection "3D" look
    4. dice_curve.png      — synthetic training curves (delegated)

All four ship in <docs/assets> and in <assets/renders>.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from synth_phantom import build_phantom
from segmentation_pipeline import run_pipeline
from dice_curve import plot as plot_dice


BG = "#FBFAF7"
INK = "#0F1A1F"
ACCENT = "#2E7A7B"
AMBER = "#D9A441"


def _hu_window(img, wl=40, ww=400):
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((img - lo) / (hi - lo), 0, 1)


def slice_mosaic(vol, labels, out: Path):
    wl_ww = [(-600, 1500), (40, 400), (40, 400)]
    titles = ["Lung window  WL=-600  WW=1500",
              "Soft-tissue  WL=40  WW=400",
              "Annotated overlay"]
    z = vol.shape[0] // 2
    y = vol.shape[1] // 2
    x = vol.shape[2] // 2

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11), dpi=150,
                             facecolor=BG)
    plane_names = ["Axial", "Coronal", "Sagittal"]
    for col, (wl, ww) in enumerate(wl_ww):
        for row, (plane, idx) in enumerate(zip(plane_names, [z, y, x])):
            ax = axes[row, col]
            if plane == "Axial":
                sl = vol[idx]
                lb = labels[idx]
            elif plane == "Coronal":
                sl = vol[:, idx]
                lb = labels[:, idx]
            else:
                sl = vol[:, :, idx]
                lb = labels[:, :, idx]
            img = _hu_window(sl, wl=wl, ww=ww)
            ax.imshow(img, cmap="gray", origin="lower", aspect="auto")
            if col == 2:  # overlay column
                overlay = np.zeros((*lb.shape, 4))
                colors = {2: (0.18, 0.48, 0.48, 0.35),  # lungs teal
                          3: (0.85, 0.65, 0.25, 0.55),  # bone amber
                          4: (0.12, 0.35, 0.36, 0.35),  # liver
                          5: (0.86, 0.20, 0.20, 0.80)}  # lesion red
                for v, c in colors.items():
                    overlay[lb == v] = c
                ax.imshow(overlay, origin="lower", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(plane, color=INK, fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(titles[col], color=INK, fontsize=11, fontweight="bold")
            for s in ax.spines.values():
                s.set_edgecolor("#E3DED3")

    fig.suptitle("Synthetic CT phantom — multi-plane review",
                 color=INK, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def segmentation_overlay(vol, gt, pred_liver_arr, out: Path):
    """Axial mosaic of 9 slices through the liver with GT and prediction."""
    # pick slices that actually contain liver
    liver_slices = np.where((gt == 4).any(axis=(1, 2)) | (gt == 5).any(axis=(1, 2)))[0]
    if len(liver_slices) == 0:
        liver_slices = np.array([vol.shape[0] // 2])
    idxs = np.linspace(liver_slices[0], liver_slices[-1], 9).astype(int)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=150, facecolor=BG)
    for ax, k in zip(axes.ravel(), idxs):
        img = _hu_window(vol[k], 40, 400)
        ax.imshow(img, cmap="gray", origin="lower")
        # GT liver in teal, lesion in amber
        liver_mask = (gt[k] == 4) | (gt[k] == 5)
        lesion_mask = gt[k] == 5
        pred_mask = pred_liver_arr[k].astype(bool)
        # GT outline
        ax.contour(liver_mask, levels=[0.5], colors=[ACCENT], linewidths=1.5)
        ax.contour(lesion_mask, levels=[0.5], colors=[AMBER], linewidths=1.5)
        # prediction filled
        ov = np.zeros((*pred_mask.shape, 4))
        ov[pred_mask] = (0.18, 0.48, 0.48, 0.25)
        ax.imshow(ov, origin="lower")
        ax.set_title(f"z = {k}", fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#E3DED3")

    fig.suptitle("Liver segmentation — ground truth (teal outline) vs. SimpleITK pipeline (teal fill)",
                 color=INK, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def volume_render(vol, labels, out: Path):
    """Simulated 3D volume render: MIP of bone + soft-tissue composite."""
    # MIP projections
    bone = np.clip((vol - 200) / 500, 0, 1)
    soft = np.clip((vol + 100) / 400, 0, 1)

    mip_axial = bone.max(axis=0) * 0.65 + soft.max(axis=0) * 0.35
    mip_coronal = bone.max(axis=1) * 0.65 + soft.max(axis=1) * 0.35
    mip_sagittal = bone.max(axis=2) * 0.65 + soft.max(axis=2) * 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=150, facecolor=BG)
    for ax, img, title in zip(
        axes, [mip_coronal, mip_sagittal, mip_axial],
        ["Coronal MIP", "Sagittal MIP", "Axial MIP"],
    ):
        ax.imshow(img, cmap="bone", origin="lower", aspect="auto")
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#E3DED3")
    fig.suptitle("Maximum-intensity projections — synthetic thoracoabdominal phantom",
                 color=INK, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main(out_dir: Path):
    vol, labels, ct, gt_img = build_phantom()
    preds = run_pipeline(ct)
    pred_liver = sitk.GetArrayFromImage(preds["liver"])

    slice_mosaic(vol, labels, out_dir / "slice_mosaic.png")
    segmentation_overlay(vol, labels, pred_liver, out_dir / "segmentation_overlay.png")
    volume_render(vol, labels, out_dir / "volume_render.png")
    plot_dice(out_dir / "dice_curve.png")

    print("rendered:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name:<28} {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main(Path("assets/renders"))
