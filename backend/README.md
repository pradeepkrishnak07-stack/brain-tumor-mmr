# Brain Tumor MMR - Backend API

FastAPI service powering the Mathematical Morphological Reconstruction (MMR) Brain MRI Analysis application.

## Endpoints

- `GET /` — Service overview and documentation.
- `GET /health` — Health-check endpoint.
- `GET /api/samples` — List available verified sample MRI phantoms.
- `POST /api/analyze` — Multipart form upload for MRI slice analysis.
- `POST /api/analyze-sample/{sample_id}` — Analyze a built-in verified sample phantom.

## Local Execution

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Cloud Deployment Guide

### Deploy on Render (Recommended Free Tier)
1. Fork or push this repository to your GitHub account.
2. Sign in to [Render.com](https://render.com).
3. Click **New +** $\to$ **Web Service**.
4. Connect your GitHub repository.
5. Set:
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variable:
   - `CORS_ORIGINS`: `*` (or your Netlify site URL, e.g., `https://your-site.netlify.app`)
7. Click **Deploy Web Service**.
8. Once deployed, copy your public service HTTPS URL (e.g., `https://brain-tumor-mmr-api.onrender.com`).
9. Paste this URL into your Netlify frontend environment variable `VITE_API_BASE_URL`.
