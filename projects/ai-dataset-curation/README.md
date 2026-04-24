# AI dataset curation for 3D Slicer

The boring-but-critical half of medical-imaging work: taking a stack of
DICOMs from a hospital PACS and turning it into a reproducible,
de-identified, AI-ready dataset.

## Workflow

```
  PACS export
       │
       ▼
  ┌─────────────────────────┐
  │ DICOM de-identification │   PS3.15 Basic profile
  │ + deterministic SHA-256 │   private tags stripped
  └──────────┬──────────────┘   burned-in text flagged
             ▼
  ┌─────────────────────────┐
  │ BIDS conversion         │   sub-<id>/ses-<n>/anat/
  │ (dcm2niix → NIfTI)      │   participants.tsv generated
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │ Automated QC            │   motion, coverage, HU range,
  │ (SimpleITK + matplotlib)│   Z-score mosaic per subject
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │ Slicer review           │   radiologist approves each case
  │ Markups + SegmentEditor │   ground truth saved as DICOM-SEG
  └──────────┬──────────────┘
             ▼
       nnU-Net-ready
```

## De-identification

`scripts/python/deidentify_dicom.py` implements a conservative subset of
**DICOM PS3.15, Annex E, Basic Application Confidentiality Profile**:

- Removes Patient Name, Birth Date, Address, Phone, Referring Physician, etc.
- Strips *every* private tag — Siemens and GE both leak PHI through
  private tags.
- Regenerates Study / Series / SOP / Frame-of-Reference UIDs so the
  original identifiers can't be cross-referenced to the hospital PACS.
- Assigns a pseudonym derived from `sha256(salt + original_patient_id)[:12]`
  so the same patient lands under the same anonymous ID across modalities
  and visits.
- Flags (but does not automatically remove) burned-in pixel text — that
  case needs OCR + manual review and belongs in a separate step.

## QC

Every converted subject gets:

- A 9-slice axial mosaic saved as `qc/<subject_id>.png`.
- A one-line row in `qc/summary.tsv`: voxel size, volume, HU mean / std,
  motion score (via `scipy.ndimage.shift` + normalised mutual information
  against a template).

Anything three sigma off the cohort mean gets flagged for manual review
before entering the training set.

## Slicer integration

Once de-identified and converted, volumes load into Slicer via
*DICOM → Import*. The `SlicerSegmentationReview` module from
`projects/slicer-python-extension` provides the review UI.

## Files

- `scripts/python/deidentify_dicom.py` — PS3.15 Basic profile + self-test

## Running the self-test

```bash
python scripts/python/deidentify_dicom.py
# self-test OK — pseudonym: ANON-2DDC0A7A70C2
```

Or on a real directory:

```bash
python scripts/python/deidentify_dicom.py /path/to/raw /path/to/deid
```
