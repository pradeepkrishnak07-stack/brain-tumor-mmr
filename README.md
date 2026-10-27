# Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)

[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8-purple.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4-38B2AC.svg)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-yellow.svg)](https://www.python.org/)
[![Netlify](https://img.shields.io/badge/Netlify-Ready-00C7B7.svg)](https://www.netlify.com/)
[![Tests](https://img.shields.io/badge/Tests-47%20Passed-brightgreen.svg)]()

A full-stack, modular, and academically documented medical image-processing web application implementing **Mathematical Morphological Reconstruction (MMR)** for brain MRI analysis, geodesic filtering, candidate abnormal region identification, morphometric feature computation, and downloadable reporting.

> [!IMPORTANT]
> **ACADEMIC & EDUCATIONAL PROTOTYPE NOTICE**  
> This software is an algorithmic prototype developed exclusively for computer vision and medical image processing research and educational demonstrations. It is **NOT** a certified clinical diagnostic tool and must **NEVER** be used as a substitute for professional neuro-radiological diagnosis or patient treatment. All segmented findings are strictly labeled **candidate abnormal regions**.

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
7. [Automated Testing Suite](#7-automated-testing-suite)
8. [GitHub Repository Setup](#8-github-repository-setup)
9. [Backend Cloud Deployment (Render / Railway / Fly.io)](#9-backend-cloud-deployment)
10. [Frontend Netlify Deployment](#10-frontend-netlify-deployment)
11. [Connecting Frontend to Deployed Backend](#11-connecting-frontend-to-deployed-backend)
12. [Methodological Limitations & Safety](#12-methodological-limitations--safety)
13. [Academic References](#13-academic-references)

---

## 1. Architecture Overview

The system uses a decoupled, production-ready client-server architecture:
- **Frontend (Client):** A responsive React + TypeScript Single Page Application (SPA) built with Vite and Tailwind CSS, deployed globally on Netlify's high-speed CDN.
- **Backend (API):** A high-performance asynchronous Python FastAPI server running OpenCV, scikit-image, and NumPy, deployable to containerized Python hosts (Render, Railway, Fly.io, AWS, etc.).

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

- **Page 1: Home Dashboard (`/`):**
  - Project introduction to Mathematical Morphological Reconstruction.
  - "Start MRI Analysis" CTA button and "Explore Methodology" link.
  - Three feature highlight cards: Image Preprocessing, Morphological Reconstruction, Candidate Region Analysis.
  - "How It Works" 4-step execution workflow.
  - Prominent academic medical disclaimer.
- **Page 2: MRI Analysis (`/analysis`):**
  - Drag-and-drop file uploader supporting PNG, JPG, JPEG, and TIFF axial MRI slices.
  - Built-in verified sample phantom selector (T1-Gd Hyperintense Lesion, T2 Frontal Lesion, Normal Control healthy slice, and Geometric Calibration standard).
  - Configurable preprocessing controls: Gaussian noise sigma, CLAHE clip limit, and tile grid.
  - MMR Controls: Reconstruction strategy (Opening by Reconstruction, H-Dome, Closing by Reconstruction), Structuring Element geometry (Disk, Square, Diamond, Octagon), footprint radius, and H-prominence.
  - Segmentation Controls: Otsu, Triangle, Percentile, or Manual thresholding, minimum area filter, and cavity filling.
  - Execution button with animated spinner and backend connectivity detector.
- **Page 3: Results (`/results`):**
  - 4 Real-time Metric Cards: Slice Resolution, Computation Duration (ms), Candidate Region Count, Total Area (% of slice).
  - 8-Stage Image Pipeline Gallery: Original MRI, Preprocessed (CLAHE), MMR Marker (Seed), MMR Mask (Constraint), MMR Reconstructed, Normalized Residual (WTH), Binary Candidate Mask, Final Annotated Overlay (Cyan contours + Yellow Bounding Box).
  - Interactive 1D Intensity Cross-Section Profile (Recharts) showing original mask profile vs reconstructed profile vs candidate peak residual along the lesion scanline.
  - Full-featured Quantitative Morphometric Measurements Table (Area, Centroid X/Y, Bounding Box, Perimeter, Aspect Ratio, Circularity, Solidity, Extent, Mean/Max Intensity).
  - Honest Empty State for negative findings (e.g. Healthy Normal Control): clearly reports 0 candidate regions with zero fabricated diagnosis.
  - Download Center: Annotated Overlay (PNG), Binary Mask (PNG), Feature Table (CSV), Full Analysis Report (Markdown).
- **Page 4: Methodology (`/methodology`):**
  - Mathematical theory of complete lattices and geodesic transformations.
  - Comparison: Standard morphological opening $\gamma_B(f) = (f \ominus B) \oplus B$ (which distorts boundaries and rounds corners) vs. MMR $R^\delta_f(g)$ (which reconstructs retained shapes with perfect boundary fidelity).
  - Operational pipeline flowchart diagram.
  - Metric formulation reference table.
- **Page 5: About (`/about`):**
  - Problem statement, objectives, and proposed solution.
  - Full technology stack documentation.
  - Documented methodological and clinical limitations.
  - Academic bibliography (Vincent 1993, Soille 2003, Serra 1982, Gonzalez & Woods 2018).

---

## 3. Technology Stack

### Frontend
- **React 19** with **TypeScript**
- **Vite 8** modern bundler
- **Tailwind CSS 4** for styling
- **Lucide React** icons
- **Recharts** for cross-sectional profile charts
- **Netlify** for global CDN hosting and SPA routing

### Backend
- **Python 3.11+**
- **FastAPI** asynchronous web API framework
- **Uvicorn** ASGI production web server
- **scikit-image** geodesic morphological reconstruction
- **OpenCV (cv2)** image transformations, CLAHE, contours, and annotations
- **NumPy & SciPy** matrix operations
- **Pillow (PIL)** multi-format image decoding
- **Pandas** tabular feature extraction
- **Pytest** automated testing framework (47 passing tests)

---

## 4. Mathematical Morphological Reconstruction (MMR) Theory

### The Geodesic Dilation Recurrence
Given mask $f$ (reference constraint) and marker $g$ (seed) satisfying $g(x, y) \le f(x, y) \quad \forall (x, y)$:

$$\delta^{(1)}_f(g) = (g \oplus B) \wedge f$$

where $\oplus$ represents morphological dilation with structuring element $B$ and $\wedge$ represents the pointwise minimum operator:

$$(g \wedge f)(x, y) = \min\big(g(x, y), f(x, y)\big)$$

The geodesic dilation of size $k$ is recursively applied:
$$\delta^{(k)}_f(g) = \delta^{(1)}_f\left(\delta^{(k-1)}_f(g)\right)$$

Morphological reconstruction by dilation is the stationary convergence limit:
$$R^\delta_f(g) = \lim_{k \to \infty} \delta^{(k)}_f(g)$$

### Opening by Reconstruction (OBR) & White Top-Hat (WTH)
1. **Marker Synthesis:** $g = f \ominus B$ (erosion of mask by structuring element $B$).
2. **Reconstruction:** $R^\delta_f(f \ominus B)$ restores shapes $\ge B$ to their exact original boundaries without corner roundoff.
3. **Residual Extraction (WTH):**
   $$\text{WTH}(f) = f - R^\delta_f(f \ominus B)$$
   The residual selectively isolates focal hyperintense candidate anomalies (tumors/nodules) smaller than or equal to $B$ while completely suppressing slowly varying background brain parenchyma.

---

## 5. Repository Structure

```
brain-tumor-mmr/
├── frontend/                          # React + TypeScript + Vite SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.tsx             # Responsive header with backend health status
│   │   │   ├── HomeView.tsx           # Home dashboard & feature highlights
│   │   │   ├── AnalysisView.tsx       # File upload & parameter configuration
│   │   │   ├── ResultsView.tsx        # 8-stage gallery, 1D chart, measurements table
│   │   │   ├── MethodologyView.tsx    # Mathematical theory & flow diagram
│   │   │   └── AboutView.tsx          # Objectives, limitations & references
│   │   ├── services/
│   │   │   └── api.ts                 # Typed API client with fallback & health checking
│   │   ├── types.ts                   # TypeScript interfaces & data models
│   │   ├── App.tsx                    # Primary application router
│   │   ├── main.tsx                   # React DOM entry point
│   │   └── index.css                  # Tailwind styles
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .env.example
│   └── .env
├── backend/                           # Python FastAPI Service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI routes, CORS, analysis pipeline
│   │   ├── preprocessing.py           # Grayscale, normalization, Gaussian, CLAHE
│   │   ├── morphological_reconstruction.py # Geodesic reconstruction engine & residuals
│   │   ├── segmentation.py            # Residual thresholding, morphological cleanup
│   │   ├── feature_extraction.py      # Area, BBox, centroid, circularity, solidity
│   │   ├── visualization.py           # Overlays, galleries, PNG encoding
│   │   ├── synthetic_data.py          # Synthetic brain MRI phantoms with GT masks
│   │   ├── evaluation.py              # Dice, IoU, sensitivity, confusion matrix
│   │   └── report_generator.py        # Markdown & CSV analysis reports
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_api.py                # FastAPI endpoint integration tests
│   ├── sample_data/                   # Verified sample brain MRI phantoms
│   ├── requirements.txt               # Backend dependencies
│   ├── Dockerfile                     # Container configuration for cloud deployment
│   ├── Procfile                       # Process file for Heroku / Render / Railway
│   ├── render.yaml                    # 1-Click Render blueprint
│   └── README.md
├── tests/                             # Root unit test suite
│   ├── test_preprocessing.py
│   ├── test_reconstruction.py
│   ├── test_segmentation.py
│   ├── test_features.py
│   ├── test_evaluation.py
│   ├── test_report.py
│   └── test_pipeline_integration.py
├── sample_data/
├── docs/
│   ├── methodology.md
│   ├── limitations.md
│   └── references.md
├── netlify.toml                       # Netlify frontend build and SPA routing config
├── .gitignore                         # Git ignore configuration
├── .env.example                       # Root environment variables template
└── README.md                          # Main project guide
```

---

## 6. Local Development Setup

### Prerequisites
- **Python 3.11+**
- **Node.js 20+** and **npm**
- **Git**

### Running the FastAPI Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the FastAPI server with live reload:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. Verify the backend is operating:
   - Interactive Swagger API docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

---

### Running the React Frontend

1. In a separate terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Ensure `.env` is configured (defaults to local backend):
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

4. Start the Vite development server:
   ```bash
   npm run dev
   ```

5. Open your browser to `http://localhost:5173`. The application will automatically connect to your local Python backend.

---

## 7. Automated Testing Suite

The project includes **47 automated unit and integration tests** validating image decoding, mathematical invariants, geometric preservation, API routes, and error handling:

```powershell
python -m pytest tests backend/tests -v
```

Expected output:
```
tests/test_evaluation.py (5 tests passed)
tests/test_features.py (4 tests passed)
tests/test_pipeline_integration.py (4 tests passed)
tests/test_preprocessing.py (12 tests passed)
tests/test_reconstruction.py (6 tests passed)
tests/test_report.py (2 tests passed)
tests/test_segmentation.py (6 tests passed)
backend/tests/test_api.py (8 tests passed)

======================= 47 passed in 5.7s =======================
```

To test the frontend build:
```bash
cd frontend
npm run build
```

---

## 8. GitHub Repository Setup

To store your project on GitHub:

1. Initialize Git in the project root:
   ```powershell
   cd c:\Users\PK\Desktop\IDEA_FEST
   git init
   ```

2. Add all files (excluding files ignored by `.gitignore`):
   ```powershell
   git add .
   ```

3. Create the initial commit:
   ```powershell
   git commit -m "Initial commit: Brain Tumor Identification Using MMR (Full-Stack)"
   ```

4. Set default branch to `main`:
   ```powershell
   git branch -M main
   ```

5. Create a new empty repository on [GitHub](https://github.com/new) (e.g. `brain-tumor-mmr`).

6. Connect your local repository to GitHub:
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-mmr.git
   ```

7. Push your code to GitHub:
   ```powershell
   git push -u origin main
   ```

---

## 9. Backend Cloud Deployment

Because Netlify hosts static sites and serverless JavaScript/TypeScript functions (and cannot run a persistent Python/OpenCV/scikit-image service), the Python backend is deployed to a compatible cloud hosting provider.

### Option A: Deploy on Render (Recommended Free Tier)

1. Sign up or log in at [Render.com](https://render.com).
2. Click **New +** $\to$ **Web Service**.
3. Connect your GitHub repository `brain-tumor-mmr`.
4. Configure the service settings:
   - **Name:** `brain-tumor-mmr-api`
   - **Region:** Any region (e.g., Oregon or Frankfurt)
   - **Root Directory:** `backend`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   - `CORS_ORIGINS`: `*` (or your Netlify site URL once created)
   - `PYTHON_VERSION`: `3.11.9`
6. Click **Deploy Web Service**.
7. Once deployment finishes, Render will provide a public HTTPS URL:
   `https://brain-tumor-mmr-api.onrender.com`
8. Verify in your browser: `https://brain-tumor-mmr-api.onrender.com/health` should return `{"status":"healthy",...}`.

### Option B: Deploy on Railway
1. Sign in to [Railway.app](https://railway.app).
2. Click **New Project** $\to$ **Deploy from GitHub repo**.
3. Select `brain-tumor-mmr`.
4. Under Settings, set **Root Directory** to `backend`.
5. Under Variables, set `PORT=8000` and `CORS_ORIGINS=*`.
6. Railway automatically uses `backend/Procfile` or `backend/Dockerfile` and generates an HTTPS URL.

---

## 10. Frontend Netlify Deployment

1. Sign in to [Netlify.com](https://www.netlify.com).
2. Click **Add new site** $\to$ **Import an existing project**.
3. Choose **GitHub** and authorize access to your repository.
4. Select your `brain-tumor-mmr` repository.
5. Netlify will automatically detect `netlify.toml` from the root directory:
   - **Base directory:** `frontend`
   - **Build command:** `npm run build`
   - **Publish directory:** `dist`
6. Click **Environment variables** $\to$ **Add a variable**:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** Your deployed backend HTTPS URL (e.g. `https://brain-tumor-mmr-api.onrender.com`)
7. Click **Deploy site**.
8. Netlify will build the React application and assign a public URL (e.g. `https://brain-tumor-mmr.netlify.app`).

---

## 11. Connecting Frontend to Deployed Backend

To ensure MRI analysis works on your live Netlify website:

1. In your **Netlify Dashboard**, go to **Site configuration** $\to$ **Environment variables**.
2. Verify or update:
   ```
   VITE_API_BASE_URL = https://your-backend-service.onrender.com
   ```
3. Trigger a redeploy (**Deploys** $\to$ **Trigger deploy** $\to$ **Clear cache and deploy site**) so Vite embeds the production API URL during the build.
4. Open your public Netlify URL in your browser:
   - Look at the top-right header: it will display a green **"Backend Connected"** badge.
   - Select *Sample 1* or upload an axial MRI image and click **Execute MMR Analysis**.
   - The analysis will execute on your cloud backend and return the 8-stage gallery, 1D profile chart, and morphometric feature table directly to your browser!

---

## 12. Methodological Limitations & Safety

1. **Intensity Non-Specificity:** Morphological reconstruction operates on intensity gradients. In MRI, hyperintensity (brightness) is sequence-dependent (T1, T2, FLAIR) and is not unique to neoplasia. Normal tissues (choroid plexus, fat, vasculature) or non-tumoral pathologies (ischemia, infection, demyelinating lesions) can produce candidate peaks.
2. **2D Axial Slices:** Single axial slice analysis does not account for 3D volumetric continuity or partial volume averaging across adjacent coronal or sagittal slices.
3. **No Histological Grading:** Morphology determines spatial geometry (area, perimeter, circularity); it cannot determine WHO tumor grade (I–IV) or molecular alterations (IDH mutation, MGMT promoter status).
4. **Non-Clinical Research Prototype:** This software has not received regulatory clearance (such as FDA 510(k) or CE mark) and must not be used for clinical decision-making. All segmented findings are strictly labeled **candidate abnormal regions**.

---

## 13. Academic References

1. **Vincent, L. (1993).** *Morphological grayscale reconstruction in image analysis: Applications and efficient algorithms.* IEEE Transactions on Image Processing, 2(2), 176–201. [DOI: 10.1109/83.217222](https://doi.org/10.1109/83.217222)
2. **Soille, P. (2003).** *Morphological Image Analysis: Principles and Applications (2nd ed.).* Springer-Verlag, Berlin.
3. **Serra, J. (1982).** *Image Analysis and Mathematical Morphology.* Academic Press, London.
4. **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing (4th Edition).* Pearson.
5. **Dice, L. R. (1945).** *Measures of the amount of ecologic association between species.* Ecology, 26(3), 297–302.

---

*Academic Full-Stack Image Processing Research Project — Mathematical Morphological Reconstruction (MMR).*
