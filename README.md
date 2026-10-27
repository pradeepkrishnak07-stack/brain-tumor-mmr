# Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)

[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8-purple.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4-38B2AC.svg)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-yellow.svg)](https://www.python.org/)
[![Netlify](https://img.shields.io/badge/Netlify-Ready-00C7B7.svg)](https://www.netlify.com/)
[![Render](https://img.shields.io/badge/Render-Deploy%20Ready-46E3B7.svg)](https://render.com/)
[![Tests](https://img.shields.io/badge/Tests-49%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

A full-stack, modular, and academically documented medical image-processing web application implementing **Mathematical Morphological Reconstruction (MMR)** for brain MRI analysis, geodesic filtering, candidate abnormal region extraction, morphometric feature computation, and downloadable reporting.

> [!IMPORTANT]
> **ACADEMIC & EDUCATIONAL PROTOTYPE NOTICE**  
> This software is an algorithmic prototype developed exclusively for computer vision and medical image-processing research and educational demonstrations. It is **NOT** a certified clinical diagnostic tool and must **NEVER** be used as a substitute for professional neuro-radiological diagnosis or patient treatment. All segmented regions are strictly labeled **candidate abnormal regions**.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Project Features & Pages](#2-project-features--pages)
3. [Technology Stack](#3-technology-stack)
4. [Mathematical Morphological Reconstruction (MMR) Theory](#4-mathematical-morphological-reconstruction-mmr-theory)
5. [Repository Structure](#5-repository-structure)
6. [Local Development Setup](#6-local-development-setup)
   - [Running the FastAPI Backend](#running-the-fastapi-backend)
   - [Running the React Frontend](#running-the-react-frontend)
   - [Running the Streamlit Standalone App](#running-the-streamlit-standalone-app)
7. [Automated Testing Suite](#7-automated-testing-suite)
8. [GitHub Setup & Synchronization](#8-github-setup--synchronization)
9. [Deploying the Python Backend on Render](#9-deploying-the-python-backend-on-render)
10. [Deploying the React Frontend on Netlify](#10-deploying-the-react-frontend-on-netlify)
11. [Connecting Netlify Frontend to Deployed Backend](#11-connecting-netlify-frontend-to-deployed-backend)
12. [Alternative: Deploying Standalone Streamlit App on Render](#12-alternative-deploying-standalone-streamlit-app-on-render)
13. [Troubleshooting Guide](#13-troubleshooting-guide)
14. [Methodological Limitations & Safety](#14-methodological-limitations--safety)
15. [Academic References](#15-academic-references)

---

## 1. Architecture Overview

The system uses a decoupled, production-ready client-server architecture:
- **Frontend (Client):** A responsive React 19 + TypeScript Single Page Application (SPA) built with Vite 8 and Tailwind CSS 4, deployed on **Netlify's global CDN**.
- **Backend (API):** A high-performance asynchronous Python FastAPI server running `opencv-python-headless`, `scikit-image`, and `NumPy`, deployed on **Render** (or Railway/Fly.io).
- **Standalone Alternative:** A self-contained Streamlit application (`app.py`) deployable directly to Render as an all-in-one web service.

```
┌────────────────────────────────┐         HTTPS JSON / Multipart         ┌────────────────────────────────┐
│   Netlify CDN (React / Vite)   │ ─────────────────────────────────────► │  Cloud Python (FastAPI Engine) │
│                                │                                        │                                │
│  • Home Dashboard              │ ◄───────────────────────────────────── │  • Grayscale & Normalization   │
│  • MRI Upload & Previews       │        Base64 Images & Morphometrics   │  • Gaussian Denoise & CLAHE    │
│  • Parameter Controls          │                                        │  • Geodesic MMR Engine (OBR)   │
│  • 8-Stage Gallery & Overlays  │                                        │  • White Top-Hat Residuals     │
│  • 1D Profile Charts (Recharts)│                                        │  • Otsu & Component Filtering  │
│  • CSV / Report Downloads      │                                        │  • Morphometric Measurements   │
└────────────────────────────────┘                                        └────────────────────────────────┘
```

---

## 2. Project Features & Pages

- **Home Dashboard (`/`):**
  - Project introduction to Mathematical Morphological Reconstruction.
  - "Start MRI Analysis" CTA button and "Explore Methodology" link.
  - Interactive highlights: Image Preprocessing, Morphological Reconstruction, Candidate Region Analysis.
  - Prominent academic medical disclaimer.
- **MRI Analysis (`/analysis`):**
  - Drag-and-drop file uploader supporting PNG, JPG, JPEG, and TIFF axial MRI slices up to 25 MB.
  - Client-side and server-side file format and size validation.
  - Pre-configured verified sample phantom selector (T1-Gd Hyperintense Lesion, T2 Frontal Lesion, Normal Control healthy slice, and Geometric Calibration standard).
  - Preprocessing controls: Gaussian noise sigma, CLAHE clip limit, and tile grid.
  - MMR Controls: Reconstruction strategy (Opening by Reconstruction, H-Dome, Closing by Reconstruction), Structuring Element geometry (Disk, Square, Diamond, Octagon), footprint radius, and H-prominence.
  - Segmentation Controls: Otsu, Triangle, Percentile, or Manual thresholding, minimum area filter, and cavity filling.
  - Execution button with animated spinner and backend connectivity detector.
- **Results (`/results`):**
  - 4 Real-time Metric Cards: Slice Resolution, Computation Duration (ms), Candidate Region Count, Total Area (% of slice).
  - 8-Stage Image Pipeline Gallery: Original MRI, Preprocessed (CLAHE), MMR Marker (Seed), MMR Mask (Constraint), MMR Reconstructed, Normalized Residual (WTH), Binary Candidate Mask, Final Annotated Overlay (Cyan contours + Yellow Bounding Box).
  - Interactive 1D Intensity Cross-Section Profile (Recharts) showing original mask profile vs reconstructed profile vs candidate peak residual along the lesion scanline.
  - Quantitative Morphometric Measurements Table (Area, Centroid X/Y, Bounding Box, Perimeter, Aspect Ratio, Circularity, Solidity, Extent, Mean/Max Intensity).
  - Honest Empty State for negative findings (e.g. Healthy Normal Control): clearly reports 0 candidate regions with zero fabricated diagnosis.
  - Download Center: Annotated Overlay (PNG), Binary Mask (PNG), Feature Table (CSV), Full Analysis Report (Markdown).
- **Methodology (`/methodology`):**
  - Mathematical theory of complete lattices and geodesic transformations.
  - Mathematical formulation of geodesic dilation $\delta^{(1)}_f(g) = (g \oplus B) \wedge f$.
  - Comparison: Standard morphological opening vs. MMR boundary preservation.
- **About (`/about`):**
  - Problem statement, research objectives, and technology stack.
  - Documented methodological limitations and clinical disclaimer.
  - Academic bibliography (Vincent 1993, Soille 2003, Serra 1982, Gonzalez & Woods 2018).

---

## 3. Technology Stack

### Frontend (Netlify)
- **Framework:** React 19 with TypeScript 5
- **Build Tool:** Vite 8 (Rolldown engine)
- **Styling:** Tailwind CSS 4 + PostCSS
- **Icons:** Lucide React
- **Charts:** Recharts 3 (Composed 1D intensity profiles)
- **Routing:** Client-side tab state with Netlify SPA redirect rule

### Backend (Render / Cloud)
- **Framework:** FastAPI 0.115+ with Uvicorn
- **Image Processing:**
  - `opencv-python-headless>=4.8.0` (Linux-safe OpenCV without GUI dependencies)
  - `scikit-image>=0.22.0` (Vincent hybrid priority-queue reconstruction algorithm)
  - `scipy>=1.10.0` (Multidimensional morphological filter operations)
  - `Pillow>=10.0.0` (Image I/O and format decoding)
- **Data & Plotting:** NumPy, Pandas, Matplotlib (`Agg` non-GUI headless backend)

---

## 4. Mathematical Morphological Reconstruction (MMR) Theory

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

## 5. Repository Structure

```
IDEA_FEST/
├── netlify.toml                       # Netlify configuration (base: frontend, publish: dist, SPA redirects)
├── render.yaml                        # Render Blueprint (defines both brain-tumor-mmr-api and brain-tumor-mmr)
├── requirements.txt                   # Root Python dependencies (with opencv-python-headless)
├── Procfile                           # Cloud process startup definition
├── app.py                             # Interactive Streamlit application
├── .streamlit/
│   └── config.toml                    # Streamlit server configuration
├── .gitignore                         # Comprehensive ignore rules
├── frontend/                          # React + TypeScript Vite frontend (Deploys to Netlify)
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── index.html
│   ├── .env.example                   # Template: VITE_API_BASE_URL
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── types.ts
│       ├── services/api.ts            # API service calling VITE_API_BASE_URL
│       └── components/                # Modular React views (Home, Analysis, Results, etc.)
├── backend/                           # FastAPI Python backend (Deploys to Render)
│   ├── requirements.txt
│   ├── render.yaml
│   ├── Procfile
│   ├── Dockerfile
│   └── app/
│       ├── main.py                    # FastAPI entry point (/health, /api/analyze, /api/samples)
│       ├── preprocessing.py
│       ├── morphological_reconstruction.py
│       ├── segmentation.py
│       ├── feature_extraction.py
│       ├── visualization.py
│       └── report_generator.py
├── src/                               # Shared scientific MMR modules
├── tests/                             # Automated test suite
└── sample_data/                       # Verified synthetic MRI phantoms
```

---

## 6. Local Development Setup

### Running the FastAPI Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. (Optional) Activate your virtual environment:
   ```bash
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the development API server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - Swagger Documentation: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

### Running the React Frontend

1. In a new terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Verify `frontend/.env` is configured (defaults to local backend):
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```
4. Launch the Vite development server:
   ```bash
   npm run dev
   ```
5. Open `http://localhost:5173` in your browser. The frontend will display a green **"Backend Connected"** status indicator.

### Running the Streamlit Standalone App

```bash
streamlit run app.py
```
Open `http://localhost:8501`.

---

## 7. Automated Testing Suite

The project includes **49 automated tests** covering backend API routes, image processing mathematical invariants, feature computation, and Streamlit execution:

```bash
pytest -v
```

To test the frontend build:
```bash
cd frontend
npm run build
```

---

## 8. GitHub Setup & Synchronization

The project is linked to GitHub at:
`https://github.com/pradeepkrishnak07-stack/brain-tumor-mmr.git`

To commit and push updates:
```bash
git add .
git commit -m "Configure Brain Tumor MMR for Netlify frontend and Render backend deployment"
git push origin main
```

---

## 9. Deploying the Python Backend on Render

Because Netlify specializes in static frontend hosting and cannot host persistent Python/OpenCV runtimes, the Python backend is deployed to **Render** as a free web service:

### Step 1: Create the Web Service on Render
1. Sign in to [Render Dashboard](https://dashboard.render.com/) with GitHub.
2. Click **New +** $\to$ **Web Service**.
3. Connect your repository: `pradeepkrishnak07-stack/brain-tumor-mmr`.
4. Configure the settings:
   - **Name:** `brain-tumor-mmr-api`
   - **Region:** Any region (e.g., Oregon, Ohio, Frankfurt)
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
5. Expand **Advanced** and configure:
   - **Health Check Path:** `/health`
   - **Environment Variables:**
     - `PYTHON_VERSION`: `3.11.9`
     - `CORS_ORIGINS`: `*` (or your Netlify site URL once created)
6. Click **Create Web Service**.
7. Once deployed, Render provides your public backend HTTPS URL:
   `https://brain-tumor-mmr-api.onrender.com`
8. Verify in your browser: `https://brain-tumor-mmr-api.onrender.com/health` should return `{"status":"healthy",...}`.

---

## 10. Deploying the React Frontend on Netlify

### Step 1: Import Project to Netlify
1. Sign in to [Netlify.com](https://www.netlify.com/) using your GitHub account.
2. Click **Add new site** $\to$ **Import an existing project**.
3. Select **GitHub** and authorize access to `pradeepkrishnak07-stack/brain-tumor-mmr`.

### Step 2: Configure Build Settings
Netlify will automatically detect [`netlify.toml`](file:///c:/Users/PK/Desktop/IDEA_FEST/netlify.toml):
- **Base directory:** `frontend`
- **Build command:** `npm run build`
- **Publish directory:** `dist` (or `frontend/dist`)

### Step 3: Add Production Environment Variable
Before deploying, click **Add environment variables**:
- **Key:** `VITE_API_BASE_URL`
- **Value:** Your deployed Render backend HTTPS URL (e.g. `https://brain-tumor-mmr-api.onrender.com`)

### Step 4: Deploy
Click **Deploy site**. Netlify will build the React application and provide a public URL (e.g. `https://brain-tumor-mmr.netlify.app`).

---

## 11. Connecting Netlify Frontend to Deployed Backend

1. In your **Netlify Dashboard**, navigate to **Site configuration** $\to$ **Environment variables**.
2. Ensure `VITE_API_BASE_URL` contains your Render backend URL without trailing slash:
   ```text
   VITE_API_BASE_URL = https://brain-tumor-mmr-api.onrender.com
   ```
3. Whenever you update `VITE_API_BASE_URL`, trigger a redeploy (**Deploys** $\to$ **Trigger deploy** $\to$ **Clear cache and deploy site**) so Vite embeds the variable during build.
4. Open your public Netlify URL in your browser:
   - Check the top navbar: it will display a green **"Backend Connected"** badge.
   - Choose a sample phantom or upload an axial MRI image and click **Execute MMR Analysis**.
   - The analysis runs on your cloud Python engine and displays the 8-stage gallery, 1D profile, and feature measurements table directly in your browser.

---

## 12. Alternative: Deploying Standalone Streamlit App on Render

If you prefer deploying the standalone Streamlit application (`app.py`) as a single-service website:
1. In Render, select **New +** $\to$ **Blueprint** and connect the repository.
2. Render reads `render.yaml` and launches `brain-tumor-mmr` with:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
3. Access the Streamlit UI directly on Render's assigned HTTPS URL.

---

## 13. Troubleshooting Guide

### 1. Frontend displays "Backend Server Not Detected" (Amber banner)
- **Cause:** Render's free tier spins down after 15 minutes of inactivity (cold start), or `VITE_API_BASE_URL` is misconfigured.
- **Solution:** 
  1. Open your backend URL directly (`https://your-api.onrender.com/health`) in a browser tab to wake up the Render container (takes 30–60 seconds).
  2. Verify that `VITE_API_BASE_URL` in Netlify matches your Render backend URL exactly (with `https://`).
  3. Redeploy the Netlify site after changing environment variables.

### 2. Netlify 404 on page refresh
- **Cause:** Missing Single Page Application redirect rule.
- **Solution:** Configured in `netlify.toml`:
  ```toml
  [[redirects]]
    from = "/*"
    to = "/index.html"
    status = 200
  ```

### 3. File upload fails with `413 Request Entity Too Large`
- **Cause:** Uploaded image exceeds 25 MB limit.
- **Solution:** Upload a standard axial MRI slice ($\le 25\text{ MB}$).

### 4. Backend fails on Render with `ImportError: libGL.so.1`
- **Cause:** Standard `opencv-python` requires native X11 libraries.
- **Solution:** `requirements.txt` specifically uses `opencv-python-headless>=4.8.0`.

---

## 14. Methodological Limitations & Safety

1. **Intensity Non-Specificity:** High-intensity regions in MRI do not uniquely indicate neoplastic tissue; normal anatomical structures (choroid plexus, fat, vasculature) or non-tumoral pathologies (ischemia, inflammation) can present hyperintense contrast.
2. **2D Axial Slices:** Single axial slice analysis does not account for 3D volumetric continuity across adjacent slices.
3. **No Histological Grading:** Morphology determines spatial geometry (area, perimeter, circularity); it cannot determine WHO tumor grade (I–IV) or molecular status (IDH mutation, MGMT promoter).
4. **Non-Clinical Research Prototype:** This software has not received regulatory clearance (such as FDA 510(k) or CE mark) and must not be used for clinical decision-making. All segmented findings are strictly labeled **candidate abnormal regions**.

---

## 15. Academic References

1. **Vincent, L. (1993).** *Morphological grayscale reconstruction in image analysis: Applications and efficient algorithms.* IEEE Transactions on Image Processing, 2(2), 176–201. [DOI: 10.1109/83.217222](https://doi.org/10.1109/83.217222)
2. **Soille, P. (2003).** *Morphological Image Analysis: Principles and Applications (2nd ed.).* Springer-Verlag, Berlin.
3. **Serra, J. (1982).** *Image Analysis and Mathematical Morphology.* Academic Press, London.
4. **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing (4th Edition).* Pearson.
5. **Dice, L. R. (1945).** *Measures of the amount of ecologic association between species.* Ecology, 26(3), 297–302.

---

*Academic Medical Image Processing Platform — Mathematical Morphological Reconstruction (MMR).*
