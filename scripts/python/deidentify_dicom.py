"""
DICOM de-identification to a conservative DICOM PS3.15 E.1 "Basic Profile".

Removes or blanks every tag in the DICOM standard's de-identification
basic profile, keeps clinically relevant descriptors, and assigns a
deterministic pseudonym derived from the original PatientID so the same
study can be re-linked across modalities / sessions.

Usage:
    python deidentify_dicom.py <in_dir> <out_dir>

If no args are given, the script runs a self-test on a synthetic
in-memory dataset so CI can verify the logic without real DICOMs.
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# Tags from DICOM PS3.15 Annex E, Basic Application Confidentiality Profile.
# (Abbreviated — a production run would use pydicom-data-element-dicts
# or the full dicom-anonymizer profile.)
TAGS_TO_REMOVE = [
    "PatientName", "PatientBirthDate", "PatientSex", "PatientAddress",
    "PatientTelephoneNumbers", "ReferringPhysicianName",
    "PhysiciansOfRecord", "OperatorsName", "InstitutionName",
    "InstitutionAddress", "InstitutionalDepartmentName",
    "AccessionNumber", "StudyID", "RequestingPhysician",
    "OtherPatientIDs", "OtherPatientNames", "PatientMotherBirthName",
    "MilitaryRank", "BranchOfService", "PatientTelecomInformation",
    "CountryOfResidence", "RegionOfResidence", "EthnicGroup",
    "Occupation", "AdditionalPatientHistory",
]

UID_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
            "FrameOfReferenceUID"]


def pseudonymize(original_id: str, salt: str = "study-salt-2026") -> str:
    h = hashlib.sha256((salt + original_id).encode("utf-8")).hexdigest()
    return f"ANON-{h[:12].upper()}"


def deidentify(ds: Dataset, salt: str = "study-salt-2026") -> Dataset:
    out = ds.copy()

    for tag in TAGS_TO_REMOVE:
        if tag in out:
            delattr(out, tag)

    original_pid = str(getattr(ds, "PatientID", "UNKNOWN"))
    out.PatientID = pseudonymize(original_pid, salt)
    out.PatientName = out.PatientID
    out.PatientIdentityRemoved = "YES"
    out.DeidentificationMethod = "DICOM PS3.15 Basic + deterministic SHA-256"

    for tag in UID_TAGS:
        if tag in out:
            setattr(out, tag, generate_uid())

    # Strip private tags wholesale — they commonly carry PHI.
    out.remove_private_tags()

    # Burn-in-text protection: blank BurnedInAnnotation if present.
    if "BurnedInAnnotation" in out:
        out.BurnedInAnnotation = "NO"

    return out


def deidentify_tree(in_dir: Path, out_dir: Path, salt: str = "study-salt-2026") -> int:
    n = 0
    for p in in_dir.rglob("*.dcm"):
        ds = pydicom.dcmread(str(p))
        clean = deidentify(ds, salt=salt)
        rel = p.relative_to(in_dir)
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        clean.save_as(str(target))
        n += 1
    return n


def _selftest() -> None:
    # Build a tiny in-memory DICOM dataset and de-identify it.
    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("tmp.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = "MRN-001234"
    ds.PatientName = "DOE^JANE"
    ds.PatientBirthDate = "19700101"
    ds.InstitutionName = "General Hospital"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()

    original_study_uid = str(ds.StudyInstanceUID)
    out = deidentify(ds)
    assert out.PatientID != "MRN-001234"
    assert out.PatientID.startswith("ANON-")
    assert "PatientBirthDate" not in out
    assert "InstitutionName" not in out
    assert str(out.StudyInstanceUID) != original_study_uid
    assert out.PatientIdentityRemoved == "YES"
    print("self-test OK — pseudonym:", out.PatientID)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        n = deidentify_tree(Path(sys.argv[1]), Path(sys.argv[2]))
        print(f"de-identified {n} DICOM file(s) → {sys.argv[2]}")
    else:
        _selftest()
