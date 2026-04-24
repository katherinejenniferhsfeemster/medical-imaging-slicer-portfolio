"""
3D Slicer scripted module — Segmentation Review

A compact Slicer extension that pulls in the same SimpleITK pipeline used in
`scripts/python/segmentation_pipeline.py`, so radiologists can:

    1. Load a DICOM series into a Volume node.
    2. Run the baseline organ pipeline (body / lungs / bone / liver).
    3. Merge the output into a Segmentation node with proper names and colors.
    4. Compute Dice & HD95 against any ground-truth segmentation already in the scene.
    5. Export the reviewed result as DICOM-SEG.

The module is structured to Slicer's standard
`ScriptedLoadableModule` pattern — it loads inside Slicer 5.x when dropped into
`<extensions>/SlicerSegmentationReview/SlicerSegmentationReview.py` and added
to the module paths in Settings → Modules.

Running outside Slicer (e.g. in CI) is explicitly guarded so this file is
importable by pytest without Slicer present.
"""

from __future__ import annotations

try:
    import slicer
    import qt
    import ctk
    import vtk
    from slicer.ScriptedLoadableModule import (
        ScriptedLoadableModule,
        ScriptedLoadableModuleWidget,
        ScriptedLoadableModuleLogic,
    )
    SLICER_AVAILABLE = True
except ImportError:  # running outside Slicer
    SLICER_AVAILABLE = False
    ScriptedLoadableModule = object
    ScriptedLoadableModuleWidget = object
    ScriptedLoadableModuleLogic = object


SEGMENT_COLORS = {
    "body":  (0.95, 0.88, 0.80),
    "lungs": (0.18, 0.48, 0.48),
    "bone":  (0.85, 0.65, 0.25),
    "liver": (0.36, 0.22, 0.12),
}


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------
class SlicerSegmentationReview(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Segmentation Review"
        parent.categories = ["Katherine Feemster"]
        parent.contributors = ["Katherine Feemster"]
        parent.helpText = (
            "Runs a baseline SimpleITK organ segmentation on the active volume, "
            "computes Dice / HD95 against a reference segmentation, and exports "
            "the approved result as DICOM-SEG."
        )
        parent.acknowledgementText = "MIT licensed. Built on 3D Slicer + SimpleITK."


# ---------------------------------------------------------------------------
# Logic (Slicer-independent parts live here)
# ---------------------------------------------------------------------------
class SlicerSegmentationReviewLogic(ScriptedLoadableModuleLogic):
    """Pure-Python logic, usable from CI and from the widget."""

    def run_pipeline_on_volume(self, input_volume_node):
        """Run the shared SimpleITK pipeline and return a dict of label arrays."""
        import sitkUtils
        from segmentation_pipeline import run_pipeline  # shared with CI

        sitk_image = sitkUtils.PullVolumeFromSlicer(input_volume_node)
        return run_pipeline(sitk_image)

    def push_masks_to_segmentation(self, masks, reference_volume_node, name="Baseline"):
        import sitkUtils
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", name)
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
            reference_volume_node)

        seg = segmentation_node.GetSegmentation()
        for organ, mask_image in masks.items():
            tmp_vol = sitkUtils.PushVolumeToSlicer(
                mask_image, name=f"__tmp_{organ}", className="vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                tmp_vol, segmentation_node)
            slicer.mrmlScene.RemoveNode(tmp_vol)
            segment_id = seg.GetNthSegmentID(seg.GetNumberOfSegments() - 1)
            segment = seg.GetSegment(segment_id)
            segment.SetName(organ)
            segment.SetColor(*SEGMENT_COLORS.get(organ, (0.8, 0.8, 0.8)))
        return segmentation_node

    def compute_metrics(self, predicted_node, reference_node):
        """Dice + HD95 via Slicer's SegmentComparison logic."""
        import SegmentStatistics
        import SegmentComparison
        comparison = SegmentComparison.SegmentComparisonLogic()
        return comparison.ComputeDiceStatistics(predicted_node, reference_node)

    def export_dicom_seg(self, segmentation_node, reference_volume_node, out_path):
        """Round-trip to DICOM-SEG using Slicer's DICOMSegmentationPlugin."""
        import DICOMSegmentationPlugin
        exporter = DICOMSegmentationPlugin.DICOMSegmentationPluginClass()
        exportables = exporter.examineForExport(segmentation_node)
        for e in exportables:
            e.directory = out_path
        exporter.export(exportables)


# ---------------------------------------------------------------------------
# Widget (only instantiated when running inside Slicer)
# ---------------------------------------------------------------------------
class SlicerSegmentationReviewWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        if not SLICER_AVAILABLE:
            return
        ScriptedLoadableModuleWidget.setup(self)

        box = ctk.ctkCollapsibleButton()
        box.text = "Segmentation Review"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Input CT:", self.volumeSelector)

        self.referenceSelector = slicer.qMRMLNodeComboBox()
        self.referenceSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.referenceSelector.noneEnabled = True
        self.referenceSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Reference (optional):", self.referenceSelector)

        self.runButton = qt.QPushButton("Run baseline pipeline")
        self.runButton.connect("clicked(bool)", self.onRun)
        form.addRow(self.runButton)

        self.exportButton = qt.QPushButton("Export DICOM-SEG…")
        self.exportButton.connect("clicked(bool)", self.onExport)
        form.addRow(self.exportButton)

        self.logic = SlicerSegmentationReviewLogic()
        self.outputNode = None

    def onRun(self):
        vol = self.volumeSelector.currentNode()
        if vol is None:
            slicer.util.errorDisplay("Select an input CT volume first.")
            return
        masks = self.logic.run_pipeline_on_volume(vol)
        self.outputNode = self.logic.push_masks_to_segmentation(masks, vol)
        ref = self.referenceSelector.currentNode()
        if ref is not None:
            stats = self.logic.compute_metrics(self.outputNode, ref)
            slicer.util.infoDisplay(f"Dice / HD95 vs reference:\n{stats}")

    def onExport(self):
        if self.outputNode is None:
            slicer.util.errorDisplay("Run the pipeline first.")
            return
        directory = qt.QFileDialog.getExistingDirectory(self.parent, "DICOM-SEG output")
        if directory:
            self.logic.export_dicom_seg(self.outputNode, self.volumeSelector.currentNode(), directory)
            slicer.util.infoDisplay(f"DICOM-SEG written to {directory}")
