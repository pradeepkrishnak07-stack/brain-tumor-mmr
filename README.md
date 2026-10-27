# Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-yellow.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Render](https://img.shields.io/badge/Render-Deploy%20Ready-46E3B7.svg)](https://render.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-49%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

A comprehensive medical image-processing research platform implementing **Mathematical Morphological Reconstruction (MMR)** for brain MRI analysis, geodesic filtering, candidate abnormal region extraction, morphometric measurement, and automated report generation.

> [!IMPORTANT]
> **ACADEMIC & EDUCATIONAL PROTOTYPE NOTICE**  
> This software is an algorithmic prototype developed exclusively for computer vision and medical image-processing research and educational demonstrations. It is **NOT** a certified clinical diagnostic tool and must **NEVER** be used as a substitute for professional neuro-radiological diagnosis or patient treatment. All segmented regions are strictly labeled **candidate abnormal regions**.

---

## Table of Contents
1. [Overview & Features](#1-overview--features)
2. [Mathematical Morphological Reconstruction (MMR) Methodology](#2-mathematical-morphological-reconstruction-mmr-methodology)
3. [Technology Stack](#3-technology-stack)
4. [Project Architecture & Repository Structure](#4-project-architecture--repository-structure)
5. [Local Installation and Execution](#5-local-installation-and-execution)
   - [Running the Streamlit Web Application (Default)](#running-the-streamlit-web-application)
   - [Running the FastAPI Backend (Microservice Option)](#running-the-fastapi-backend)
6. [Automated Testing Suite](#6-automated-testing-suite)
7. [GitHub Repository Setup & Synchronization](#7-github-repository-setup--synchronization)
8. [Deploying the Application on Render](#8-deploying-the-application-on-render)
   - [Method 1: One-Click Render Blueprint (render.yaml)](#method-1-one-click-render-blueprint-renderyaml)
   - [Method 2: Manual Web Service Setup via Render Dashboard](#method-2-manual-web-service-setup-via-render-dashboard)
9. [Render Free-Tier Behavior & Limitations](#9-render-free-tier-behavior--limitations)
10. [Troubleshooting Guide](#10-troubleshooting-guide)
11. [Methodological Limitations & Safety](#11-methodological-limitations--safety)
12. [Academic References](#12-academic-references)

---

## 1. Overview & Features

### Core Capabilities:
- **MRI Upload & Preprocessing:** Upload axial brain MRI scans (PNG, JPG, JPEG, TIFF) up to 25 MB. Automatic grayscale conversion, Gaussian noise suppression, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **Genuine Mathematical Morphological Reconstruction:** Strict geodesic dilation under continuous mask constraint (Vincent 1993 hybrid queue algorithm) preserving original lesion boundaries without geometric rounding.
- **Selective Residual Filtering:** White Top-Hat (WTH) by reconstruction, H-Dome peak isolation, and Closing by Reconstruction for hyperintense/hypointense focal extraction.
- **Adaptive Candidate Segmentation:** Multi-strategy thresholding (Otsu minimum variance, Triangle adaptive skew, Percentile cutoff, or Manual slider), connected component filtering by pixel area, and cavity hole filling.
- **Candidate Region Extraction & Morphometrics:** Calculates quantitative area, perimeter, centroid coordinates, bounding boxes, aspect ratio, circularity, solidity, extent, and mean/max radiometric intensities.
- **Interactive Visualizations:**
  - 8-stage image pipeline gallery displaying every mathematical phase.
  - Non-destructive annotated overlays (cyan boundary contours and yellow bounding boxes).
  - 1D intensity cross-sectional profile analyzer comparing original signal, reconstructed geodesic surface, and candidate residual peaks.
- **Empirical Ground-Truth Validation:** Dice Similarity Coefficient (DSC), IoU / Jaccard Index, Sensitivity (Recall), Specificity, and pixel-level contingency classification when reference ground truth is provided.
- **Export & Download Center:** Download annotated overlay (PNG), binary candidate mask (PNG), full morphometric feature table (CSV), and comprehensive markdown analysis reports.

---

## 2. Mathematical Morphological Reconstruction (MMR) Methodology

### Standard Morphological Opening vs. Morphological Reconstruction
In standard morphological opening $\gamma_B(f) = (f \ominus B) \oplus B$, an image is eroded by structuring element $B$ and then dilated. While this removes bright components smaller than $B$, it causes severe structural distortion: corners are rounded, edges shift inward, and fine boundary details are permanently lost.

In **Mathematical Morphological Reconstruction (MMR)**, the image is eroded to generate a marker $g = f \ominus B$, followed by iterative geodesic dilations restricted by the original image mask $f$:
$$\delta^{(1)}_f(g) = (g \oplus B) \wedge f$$
where $\wedge$ denotes the pointwise infimum (minimum). Geodesic dilation is iterated until mathematical convergence (idempotence):
$$R^\delta_f(g) = \lim_{k \to \infty} \delta^{(k)}_f(g)$$
Because the mask $f$ acts as a ceiling constraint, shapes larger than $B$ recover their **exact original boundaries with zero spatial distortion**.

### White Top-Hat by Reconstruction
Focal hyperintense candidate regions (such as enhancing lesions) are isolated via the morphological residual:
$$\text{WTH}(f) = f - R^\delta_f(f \ominus B)$$
Slowly varying background parenchyma is subtracted, isolating sharp focal contrast peaks.

```
 ┌───────────────────────┐
 │   Input Brain MRI     │ (2D/3D PNG, JPG, TIFF)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Grayscale Conversion  │ (Normalized to [0, 1] float32)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │     Preprocessing     │ (Gaussian Filter + CLAHE Contrast Enhancement)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │  Marker & Mask Setup  │ (Marker g = f ⊖ B; Mask = f; Condition: g ≤ f)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ MMR Geodesic Engine   │ (Iterative Geodesic Dilation: δ_f(g) until convergence)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Residual Map (WTH)    │ (Residual = Mask - Reconstructed)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Candidate Extraction  │ (Thresholding + Morphological Cleaning + Area Filter)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Quantitative Analysis │ (Area, BBox, Centroid, Perimeter, Circularity, Solidity)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │  Visualization & Rpt  │ (Multi-stage Gallery, Overlay, CSV, Markdown Report)
 └───────────────────────┘
```

---

## 3. Technology Stack

- **Language:** Python 3.11 / Python 3.14
- **Web UI Framework:** Streamlit (v1.35+)
- **Alternative Microservice API:** FastAPI + Uvicorn
- **Image Processing & Vision:**
  - `opencv-python-headless` (Server-safe computer vision without GUI/X11 dependencies)
  - `scikit-image` (Vincent priority-queue morphological reconstruction, label analysis, regionprops)
  - `scipy` (Scientific multidimensional filtering and ndimage operations)
  - `Pillow` (PIL image I/O and format decoding)
- **Data & Scientific Plotting:**
  - `numpy` (High-performance array operations)
  - `pandas` (Morphometric tables and CSV generation)
  - `matplotlib` (Headless `Agg` backend for intensity cross-sections)
- **Deployment Platform:** Render (Web Service / Blueprint), Git, GitHub

---

## 4. Project Architecture & Repository Structure

```
IDEA_FEST/
├── app.py                             # Main interactive Streamlit application (Render Web Entry Point)
├── requirements.txt                   # Production Python dependencies (includes opencv-python-headless)
├── render.yaml                        # Render Blueprint configuration for automated deployment
├── Procfile                           # Cloud platform process definition
├── .streamlit/
│   └── config.toml                    # Streamlit server configuration (headless, 0.0.0.0, max upload 25MB)
├── .gitignore                         # Excludes venv, cache, secrets, and private patient scans
├── src/                               # Core modular scientific image-processing libraries
│   ├── preprocessing.py               # Grayscale conversion, CLAHE, Gaussian denoise
│   ├── morphological_reconstruction.py# Geodesic dilation engine, marker generation, WTH
│   ├── segmentation.py                # Thresholding, component filtering, hole filling
│   ├── feature_extraction.py          # Morphological measurements (Area, Circularity, etc.)
│   ├── visualization.py               # Contours, bounding boxes, cross-sectional profiles
│   ├── evaluation.py                  # Dice, IoU, Sensitivity, Specificity against ground truth
│   ├── synthetic_data.py              # Medically styled synthetic MRI phantoms
│   └── report_generator.py            # Markdown report and CSV synthesis
├── sample_data/                       # Verified synthetic sample MRI phantoms & ground truth masks
├── tests/                             # Automated testing suite (Unit, integration, and UI tests)
│   ├── test_preprocessing.py
│   ├── test_reconstruction.py
│   ├── test_segmentation.py
│   ├── test_features.py
│   ├── test_evaluation.py
│   ├── test_report.py
│   ├── test_pipeline_integration.py
│   └── test_streamlit_app.py
├── backend/                           # Optional FastAPI REST API microservice
│   ├── app/                           # FastAPI routes, schemas, and engine
│   ├── tests/                         # API endpoint tests
│   └── requirements.txt
└── README.md                          # Comprehensive documentation
```

---

## 5. Local Installation and Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12+
- Git

### Installation Steps

1. Clone or open the repository:
   ```bash
   git clone https://github.com/pradeepkrishnak07-stack/brain-tumor-mmr.git
   cd brain-tumor-mmr
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit Web Application

To run the interactive UI locally:
```bash
streamlit run app.py
```
Streamlit will launch and display local and network access URLs:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Running the FastAPI Backend (Microservice Option)

If you wish to run the FastAPI REST server:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive OpenAPI / Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## 6. Automated Testing Suite

The repository contains **49 automated unit, integration, and AppTest tests**. Run the full test suite with:

```bash
pytest -v
```

Expected output:
```
backend/tests/test_api.py (8 passed)
tests/test_evaluation.py (5 passed)
tests/test_features.py (4 passed)
tests/test_pipeline_integration.py (4 passed)
tests/test_preprocessing.py (12 passed)
tests/test_reconstruction.py (6 passed)
tests/test_report.py (2 passed)
tests/test_segmentation.py (6 passed)
tests/test_streamlit_app.py (2 passed)
======================= 49 passed in 6.4s =======================
```

---

## 7. GitHub Repository Setup & Synchronization

The repository is connected to GitHub at:
`https://github.com/pradeepkrishnak07-stack/brain-tumor-mmr.git`

To commit and push updates to GitHub:

```bash
# 1. Check status of modified and untracked files
git status

# 2. Stage updated files
git add requirements.txt render.yaml Procfile .streamlit/config.toml .gitignore app.py src/ visualization.py tests/ README.md

# 3. Commit with a clear descriptive message
git commit -m "Configure Brain Tumor MMR Streamlit project for Render cloud deployment"

# 4. Push to GitHub main branch
git push origin main
```

---

## 8. Deploying the Application on Render

Render provides free hosting for Python web services. You can deploy this application using either **Method 1 (Blueprint with `render.yaml`)** or **Method 2 (Manual Setup)**.

### Method 1: One-Click Render Blueprint (render.yaml)

1. Go to [Render Dashboard](https://dashboard.render.com/) and sign in with your GitHub account.
2. Click **New +** at the top right, then select **Blueprint**.
3. Connect your repository: `pradeepkrishnak07-stack/brain-tumor-mmr`.
4. Render will automatically read the root `render.yaml` file:
   - **Service Name:** `brain-tumor-mmr`
   - **Runtime:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
   - **Health Check Path:** `/_stcore/health`
5. Click **Apply**.
6. Render will build the environment, install dependencies, and launch your application.

---

### Method 2: Manual Web Service Setup via Render Dashboard

If you prefer setting up the web service manually:

1. Sign in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** $\to$ **Web Service**.
3. Select **Build and deploy from a Git repository**.
4. Choose `pradeepkrishnak07-stack/brain-tumor-mmr` from your list of repositories.
5. Configure the service parameters exactly as follows:

| Setting | Value |
| :--- | :--- |
| **Name** | `brain-tumor-mmr` |
| **Region** | Select the region closest to you (e.g., Oregon, Ohio, Frankfurt, Singapore) |
| **Branch** | `main` |
| **Root Directory** | *(Leave blank / empty)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT` |
| **Instance Type** | `Free` |

6. Expand **Advanced** and configure:
   - **Health Check Path:** `/_stcore/health`
   - **Add Environment Variables:**
     - `PYTHON_VERSION` = `3.11.9`
     - `STREAMLIT_SERVER_HEADLESS` = `true`
     - `STREAMLIT_BROWSER_GATHER_USAGE_STATS` = `false`
     - `STREAMLIT_SERVER_ENABLE_CORS` = `false`
     - `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION` = `false`
     - `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` = `25`

7. Click **Create Web Service**.

8. **Monitor Deployment Logs:**
   Render displays live build logs. Look for:
   ```
   ==> Installing dependencies with 'pip install -r requirements.txt'...
   ==> Successfully installed streamlit opencv-python-headless scikit-image...
   ==> Starting service with 'streamlit run app.py --server.address 0.0.0.0 --server.port $PORT'...
   You can now view your Streamlit app in your browser.
   URL: http://0.0.0.0:10000
   ==> Your service is live 🎉
   ```

9. **Open Public URL:**
   Once the status turns **Live**, click the link under the service header (e.g., `https://brain-tumor-mmr-xxxx.onrender.com`).

---

## 9. Render Free-Tier Behavior & Limitations

- **Spin-down / Sleep Behavior:** Render free web services automatically spin down (sleep) after 15 minutes of inactivity. When a new visitor accesses your URL, Render wakes up the container. The initial spin-up typically takes **30–60 seconds**. Subsequent requests run immediately at full speed.
- **RAM Limits:** Free tier services have **512 MB of RAM**. The application has been engineered to be lightweight:
  - Memory footprint during idle: ~90 MB.
  - Peak processing memory for a $512 \times 512$ MRI slice: ~180 MB (well below the 512 MB ceiling).
- **Monthly Usage Allowance:** 750 free instance hours per month, sufficient for running 1 continuous free service 24/7.

---

## 10. Troubleshooting Guide

### 1. `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
- **Cause:** Using standard `opencv-python` which requires X11/GL graphics libraries absent in headless cloud containers.
- **Solution:** Our `requirements.txt` specifically uses `opencv-python-headless>=4.8.0`. Never mix `opencv-python` and `opencv-python-headless`.

### 2. Streamlit Web Service Fails with `Port binding error` or `Connection timed out`
- **Cause:** Hardcoding local port 8501 instead of Render's dynamic `$PORT` environment variable.
- **Solution:** Always use `--server.address 0.0.0.0 --server.port $PORT` in the start command.

### 3. Matplotlib GUI warning: `Starting a Matplotlib GUI outside of the main thread will likely fail`
- **Cause:** Matplotlib attempting to use a GUI backend (`TkAgg` or `Qt`) on a headless server.
- **Solution:** Handled in our codebase by configuring `matplotlib.use('Agg')` prior to importing `pyplot`.

### 4. `File Not Found: sample_data/sample_1_t1_hyperintense.png`
- **Cause:** Relative paths that fail when executed from varying working directories.
- **Solution:** Our `app.py` resolves paths relative to `os.path.dirname(os.path.abspath(__file__))` and automatically synthesizes sample phantoms if absent.

---

## 11. Methodological Limitations & Safety

1. **Intensity Non-Specificity:** High-intensity regions in MRI do not uniquely identify neoplastic tissue; normal anatomical structures (choroid plexus, fat, vasculature) or non-tumoral pathologies (ischemia, inflammation) can present hyperintense contrast.
2. **2D Axial Slices:** Single axial slice analysis does not account for 3D volumetric continuity across adjacent coronal or sagittal slices.
3. **No Histological Grading:** Morphology determines spatial geometry (area, perimeter, circularity); it cannot determine WHO tumor grade (I–IV) or molecular alterations (IDH mutation, MGMT promoter status).
4. **Non-Clinical Research Prototype:** This software has not received regulatory clearance (such as FDA 510(k) or CE mark) and must not be used for clinical decision-making. All segmented findings are strictly labeled **candidate abnormal regions**.

---

## 12. Academic References

1. **Vincent, L. (1993).** *Morphological grayscale reconstruction in image analysis: Applications and efficient algorithms.* IEEE Transactions on Image Processing, 2(2), 176–201. [DOI: 10.1109/83.217222](https://doi.org/10.1109/83.217222)
2. **Soille, P. (2003).** *Morphological Image Analysis: Principles and Applications (2nd ed.).* Springer-Verlag, Berlin.
3. **Serra, J. (1982).** *Image Analysis and Mathematical Morphology.* Academic Press, London.
4. **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing (4th Edition).* Pearson.
5. **Dice, L. R. (1945).** *Measures of the amount of ecologic association between species.* Ecology, 26(3), 297–302.

---

*Academic Medical Image Processing Platform — Mathematical Morphological Reconstruction (MMR).*
