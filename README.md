# Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)

[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8-purple.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4-38B2AC.svg)](https://tailwindcss.com/)
[![Netlify](https://img.shields.io/badge/Netlify-Deploy%20Ready-00C7B7.svg)](https://www.netlify.com/)
[![Tests](https://img.shields.io/badge/Tests-49%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

An academic medical image-processing web application implementing **Mathematical Morphological Reconstruction (MMR)** for brain MRI analysis, geodesic filtering, candidate abnormal region extraction, morphometric measurement, and automated report generation.

> [!IMPORTANT]
> **ACADEMIC & EDUCATIONAL PROTOTYPE NOTICE**  
> This software is an algorithmic prototype developed exclusively for computer vision and medical image-processing research and educational demonstrations. It is **NOT** a certified clinical diagnostic tool and must **NEVER** be used as a substitute for professional neuro-radiological diagnosis or patient treatment. All segmented regions are strictly labeled **candidate abnormal regions**.

---

## Table of Contents
1. [Netlify-Only Architecture (Zero External Backend Required)](#1-netlify-only-architecture)
2. [Project Features & Capabilities](#2-project-features--capabilities)
3. [Mathematical Morphological Reconstruction (MMR) Theory](#3-mathematical-morphological-reconstruction-mmr-theory)
4. [Technology Stack](#4-technology-stack)
5. [Repository Structure](#5-repository-structure)
6. [Deploying to Netlify (Step-by-Step)](#6-deploying-to-netlify-step-by-step)
7. [Local Development Setup](#7-local-development-setup)
8. [Automated Testing Suite](#8-automated-testing-suite)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Methodological Limitations & Safety](#10-methodological-limitations--safety)
11. [Academic References](#11-academic-references)

---

## 1. Netlify-Only Architecture

This project is engineered to deploy **exclusively on Netlify** without requiring any external cloud servers (such as Render, Railway, or Heroku):

- **Single Hosting Platform:** 100% of the application is hosted on Netlify's high-speed global CDN.
- **Client-Side MMR Engine ([`frontend/src/services/mmrEngine.ts`](file:///c:/Users/PK/Desktop/IDEA_FEST/frontend/src/services/mmrEngine.ts)):** The Mathematical Morphological Reconstruction algorithm, grayscale conversion, Gaussian filtering, White Top-Hat residual computation, Otsu thresholding, cavity hole-filling, and connected component morphometrics execute **directly in the user's browser using HTML5 Canvas and typed Float32Array arrays**.
- **Zero Cold Starts:** Because computation runs on the client device, there are no server sleep delays, cold starts, or monthly hosting bills.
- **Data Privacy:** MRI slices uploaded by the user never leave their local browser, ensuring zero risk of transmitting sensitive patient data over the internet.
- **Multi-Environment Support:** The codebase also includes standalone Python implementations ([`app.py`](file:///c:/Users/PK/Desktop/IDEA_FEST/app.py) for Streamlit and [`backend/app/main.py`](file:///c:/Users/PK/Desktop/IDEA_FEST/backend/app/main.py) for FastAPI) for researchers who wish to run Python experiments locally.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Netlify Global Edge CDN                               │
│                                                                                 │
│   • Single Page Application (React 19 + TypeScript 5 + Tailwind 4)              │
│   • Pre-bundled Sample MRI Phantoms (/sample_data/...)                          │
│   • SPA Redirect Rules (/* -> /index.html 200)                                  │
│   • Security Headers (DENY frame, nosniff, XSS protection)                      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Loads into browser
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  Client-Side In-Browser MMR WebEngine                           │
│                                                                                 │
│   1. Grayscale Conversion & Normalization (Luma Rec. 601)                       │
│   2. Gaussian Noise Filtering (Separable 1D Kernel)                            │
│   3. Dynamic Contrast Enhancement & CLAHE Simulation                           │
│   4. Morphological Erosion / Dilation Marker Generation (Disk, Square, etc.)   │
│   5. Vincent (1993) Geodesic Morphological Reconstruction by Dilation           │
│   6. White Top-Hat (WTH) Residual Extraction                                    │
│   7. Adaptive Otsu / Triangle Optimal Variance Thresholding                     │
│   8. Morphological Cavity Hole Filling (Border Inversion Flood-Fill)            │
│   9. 8-Connected Component Labeling & Morphometric Geometry Measurements        │
│  10. 8-Stage Image Pipeline Gallery & Recharts 1D Scanline Intensity Profile    │
│  11. Export Center: PNG Overlay, Binary Mask, CSV Metrics, Markdown Report      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Features & Capabilities

- **MRI Upload & Preprocessing:** Upload axial brain MRI scans (PNG, JPG, JPEG, TIFF) up to 25 MB with client-side format and size validation.
- **Pre-Configured Verified Phantoms:** Bundled with 4 sample MRI phantoms directly accessible on Netlify:
  1. *Sample 1: T1-Gd Hyperintense Lesion* (Parietal contrast-enhancing nodule with edema halo).
  2. *Sample 2: T2 Frontal Lesion* (Frontal hyperintensity).
  3. *Sample 3: Normal Control MRI* (Healthy slice verifying zero false-positive detection).
  4. *Sample 4: Geometric Calibration Standard* (Geometric shapes evaluating noise suppression).
- **Interactive Controls:**
  - Structuring element shapes: Disk, Square, Diamond, Octagon.
  - Footprint radius: 1 to 30 pixels.
  - Reconstruction strategy: Opening by Reconstruction, H-Dome, Closing by Reconstruction.
  - Thresholding method: Otsu, Triangle, Percentile cutoff, or Manual slider.
  - Minimum area filter: 10 to 500 pixels.
  - Cavity hole filling toggle.
- **Results & Visualizations:**
  - 4 Real-time Metric Cards: Slice Resolution, Computation Duration (ms), Candidate Region Count, Total Area (% of slice).
  - 8-Stage Image Pipeline Gallery: Original MRI, Preprocessed, MMR Marker (Seed), MMR Mask (Constraint), MMR Reconstructed, Normalized Residual (WTH), Binary Candidate Mask, Final Annotated Overlay.
  - Interactive 1D Intensity Cross-Section Profile (Recharts) comparing mask signal, reconstructed surface, and candidate residual peak.
  - Quantitative Morphometric Measurements Table (Area, Centroid X/Y, Bounding Box, Perimeter, Aspect Ratio, Circularity, Solidity, Extent, Mean/Max Intensity).
  - Honest Empty State: Clearly reports 0 candidate regions on healthy normal control slices with zero fabricated diagnosis.
  - Download Center: Download annotated overlay (PNG), candidate binary mask (PNG), feature measurements table (CSV), and comprehensive markdown analysis report.

---

## 3. Mathematical Morphological Reconstruction (MMR) Theory

### Standard Opening vs. Morphological Reconstruction
In standard morphological opening $\gamma_B(f) = (f \ominus B) \oplus B$, an image is eroded by structuring element $B$ and then dilated. While this eliminates small bright artifacts, it severely rounds corners and alters lesion boundaries.

In **Mathematical Morphological Reconstruction (MMR)**, the image is eroded to create a marker $g = f \ominus B$, and then repeatedly geodesically dilated under the continuous constraint of the original mask $f$:
$$\delta^{(1)}_f(g) = (g \oplus B) \wedge f$$
where $\wedge$ denotes the pointwise minimum. Reconstruction converges at idempotence:
$$R^\delta_f(g) = \lim_{k \to \infty} \delta^{(k)}_f(g)$$
**Advantage:** Shapes larger than $B$ are reconstructed with **zero geometric distortion and exact boundary preservation**.

### White Top-Hat Residual
Focal hyperintense candidate regions (such as enhancing lesions) are isolated via:
$$\text{WTH}(f) = f - R^\delta_f(f \ominus B)$$

---

## 4. Technology Stack

- **Hosting Platform:** Netlify (Static CDN, SPA Redirects, Security Headers)
- **Frontend Framework:** React 19 with TypeScript 5
- **Bundler & Build Tool:** Vite 8 (Rolldown engine)
- **Styling:** Tailwind CSS 4 + PostCSS
- **Vector Icons:** Lucide React
- **Data Visualizations:** Recharts 3 (Composed 1D intensity profiles)
- **Client-Side Image Processing:** Pure TypeScript / HTML5 Canvas / Float32Array typed arrays
- **Python Research Suite (Local):** FastAPI 0.115+, Streamlit 1.35+, `opencv-python-headless`, `scikit-image`, `numpy`, `scipy`

---

## 5. Repository Structure

```
IDEA_FEST/
├── netlify.toml                       # Netlify configuration (base: frontend, publish: dist, SPA redirects)
├── frontend/                          # React + TypeScript web application (Deploys to Netlify)
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── index.html
│   ├── public/
│   │   ├── favicon.svg
│   │   └── sample_data/               # Static MRI sample phantoms bundled for Netlify
│   └── src/
│       ├── App.tsx                    # Main application controller
│       ├── main.tsx                   # React root entry point
│       ├── types.ts                   # Type definitions
│       ├── services/
│       │   ├── mmrEngine.ts           # In-browser Mathematical Morphological Reconstruction engine
│       │   └── api.ts                 # Service dispatcher (defaults to in-browser MMR engine)
│       └── components/                # Modular UI views (Home, Analysis, Results, Methodology, About)
├── src/                               # Python scientific image-processing libraries
├── backend/                           # Optional FastAPI REST API microservice (for local Python use)
├── app.py                             # Optional standalone Streamlit application (for local Python use)
├── requirements.txt                   # Python dependencies (for local testing)
├── render.yaml                        # Optional Render Blueprint
├── .gitignore                         # Comprehensive ignore rules
└── README.md                          # Main project guide
```

---

## 6. Deploying to Netlify (Step-by-Step)

Because this application runs entirely in the browser, deploying to Netlify requires **zero environment variables** and **zero external servers**:

### Step 1: Push Code to GitHub
Ensure your repository is up to date on GitHub:
```bash
git push origin main
```
Repository: `https://github.com/pradeepkrishnak07-stack/brain-tumor-mmr.git`

### Step 2: Import into Netlify
1. Go to [app.netlify.com](https://app.netlify.com/) and sign in with GitHub.
2. Click **Add new site** $\to$ **Import an existing project**.
3. Choose **GitHub** and select your repository: `pradeepkrishnak07-stack/brain-tumor-mmr`.

### Step 3: Confirm Build Settings
Netlify automatically detects [`netlify.toml`](file:///c:/Users/PK/Desktop/IDEA_FEST/netlify.toml) from your repository root:
- **Base directory:** `frontend`
- **Build command:** `npm run build`
- **Publish directory:** `dist`
- *(Environment variables: Leave blank! No backend URL required).*

### Step 4: Deploy
Click **Deploy site**. Netlify will build the application in under 20 seconds and assign your public HTTPS URL (e.g., `https://brain-tumor-mmr.netlify.app`).

---

## 7. Local Development Setup

### Running the Frontend Locally

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open `http://localhost:5173` in your browser. The application will immediately display **"Netlify Standalone Engine (In-Browser MMR)"** in the top navigation bar.

---

## 8. Automated Testing Suite

### Testing Frontend Production Build
```bash
cd frontend
npm run build
```
Builds the production bundle into `frontend/dist` with zero errors.

### Testing Python Modules (Optional Local Research Suite)
```bash
pytest -v
```
Runs the 49 automated unit and integration tests across preprocessing, reconstruction invariants, and segmentation.

---

## 9. Troubleshooting Guide

### 1. 404 Error when refreshing pages
- **Solution:** Handled automatically by the SPA redirect rule in `netlify.toml`:
  ```toml
  [[redirects]]
    from = "/*"
    to = "/index.html"
    status = 200
  ```

### 2. Node Version Incompatibility
- **Solution:** Configured in `netlify.toml` with `NODE_VERSION = "22"` to match Vite 8 requirements.

### 3. Uploaded Image Exceeds Size
- **Solution:** The frontend enforces a 25 MB limit for memory safety. Please upload a standard axial MRI slice ($\le 25\text{ MB}$).

---

## 10. Methodological Limitations & Safety

1. **Intensity Non-Specificity:** High-intensity regions in MRI do not uniquely identify neoplastic tissue; normal anatomical structures (choroid plexus, fat, vasculature) or non-tumoral pathologies (ischemia, inflammation) can present hyperintense contrast.
2. **2D Axial Slices:** Single axial slice analysis does not account for 3D volumetric continuity across adjacent slices.
3. **No Histological Grading:** Morphology determines spatial geometry (area, perimeter, circularity); it cannot determine WHO tumor grade (I–IV) or molecular alterations (IDH mutation, MGMT promoter status).
4. **Non-Clinical Research Prototype:** This software has not received regulatory clearance (such as FDA 510(k) or CE mark) and must not be used for clinical decision-making. All segmented findings are strictly labeled **candidate abnormal regions**.

---

## 11. Academic References

1. **Vincent, L. (1993).** *Morphological grayscale reconstruction in image analysis: Applications and efficient algorithms.* IEEE Transactions on Image Processing, 2(2), 176–201. [DOI: 10.1109/83.217222](https://doi.org/10.1109/83.217222)
2. **Soille, P. (2003).** *Morphological Image Analysis: Principles and Applications (2nd ed.).* Springer-Verlag, Berlin.
3. **Serra, J. (1982).** *Image Analysis and Mathematical Morphology.* Academic Press, London.
4. **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing (4th Edition).* Pearson.
5. **Dice, L. R. (1945).** *Measures of the amount of ecologic association between species.* Ecology, 26(3), 297–302.

---

*Academic Medical Image Processing Platform — Mathematical Morphological Reconstruction (MMR).*
