# Methodological & Clinical Limitations

## 1. Academic & Non-Diagnostic Scope
This software system is strictly an academic image-processing research prototype developed for educational demonstration, algorithmic validation, and computer vision coursework. 

> [!CAUTION]
> It has **NOT** undergone medical device certification (such as FDA 510(k), CE mark, or equivalent regulatory clearance) and must **NEVER** be used as a standalone diagnostic tool, screening mechanism, or clinical decision support system for patient healthcare.

---

## 2. Technical and Algorithmic Limitations

### 2.1 Intensity Contrast Dependency
Mathematical Morphological Reconstruction operates purely on local intensity variations and spatial gradients. 
- In brain MRI, hyperintensity (brightness) or hypointensity (darkness) is sequence-dependent (e.g., T1-weighted, T2-weighted, FLAIR, T1-contrast-enhanced).
- The algorithm does not know whether an intensity peak represents neoplastic tumor tissue, vasogenic edema, subacute hemorrhage, arterial flow void, calcification, or normal anatomy.

### 2.2 Anatomical Mimics & Normal Physiological Hyperintensities
Multiple non-pathological intracranial structures present hyperintense profiles in brain MRI that can be detected as candidate abnormal regions unless skull-stripping or intracranial masking is pre-applied:
- Choroid plexus in the lateral ventricles.
- Subcutaneous scalp fat and diploic marrow space.
- Dural venous sinuses and prominent pial vasculature.
- Post-operative surgical resection margins or radiation necrosis scars.

### 2.3 Two-Dimensional Slice Limitation
- This prototype processes single 2D axial MRI slices independently.
- It does not reconstruct 3D volumetric connectivity across adjacent coronal, sagittal, or axial slices.
- Partial volume averaging effects (where voxel intensity represents a mixture of cerebrospinal fluid, tumor tissue, and white matter) can affect 2D boundary extraction.

### 2.4 Lack of Histopathological & Molecular Characterization
- Mathematical morphology extracts geometric attributes (area, perimeter, circularity, solidity).
- It cannot determine histological grade (e.g. WHO Grade I–IV glioma), molecular status (IDH mutation, 1p/19q codeletion, MGMT promoter methylation), or cellular proliferation index (Ki-67).
- A definitive brain tumor diagnosis requires comprehensive multi-parametric MRI, neuro-radiological review, and biopsy / histopathological confirmation.

---

## 3. Recommended Best Practices for Demonstration
1. When demonstrating on healthy control brain MRI scans (e.g. Sample 3), note that the algorithm correctly reports **zero candidate abnormal regions** when appropriate area and threshold cutoffs are selected.
2. Clearly distinguish between **candidate abnormal regions** and confirmed pathological diagnoses during presentations.
3. Validate against ground-truth reference masks only using empirical metrics (Dice / IoU) and never fabricate performance scores.
