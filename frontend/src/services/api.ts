import type { AnalysisParameters, AnalysisResult, SampleItem } from '../types';

// Normalized API base URL without trailing slash
const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const API_BASE_URL = rawBaseUrl.replace(/\/+$/, '');

export async function checkBackendHealth(): Promise<{ ok: boolean; message?: string; service?: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return { ok: true, service: data.service };
    }
    return { ok: false, message: `Server returned status ${res.status}` };
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return { ok: false, message: 'Backend connection timed out (cold start or unreachable).' };
    }
    return {
      ok: false,
      message: `Unable to connect to backend at ${API_BASE_URL}. Ensure the Python FastAPI server is running.`
    };
  }
}

export async function fetchSamplePhantoms(): Promise<SampleItem[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/samples`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.samples || [];
  } catch {
    // Return fallback sample definitions if backend is offline
    return [
      {
        id: 'sample_1',
        name: 'Sample 1: T1-Gd Hyperintense Lesion',
        description: 'Axial Brain MRI phantom with solitary parietal contrast-enhancing nodule and edema halo.',
        sequence: 'T1-Weighted Contrast Enhanced (T1-Gd)',
        expected_finding: 'Single candidate abnormal nodule in parietal cortex'
      },
      {
        id: 'sample_2',
        name: 'Sample 2: T2 Frontal Lesion',
        description: 'Axial Brain MRI phantom with prominent frontal hyperintensity.',
        sequence: 'T2 / FLAIR',
        expected_finding: 'Frontal lobe candidate hyperintense region'
      },
      {
        id: 'sample_3',
        name: 'Sample 3: Normal Control MRI (No Lesion)',
        description: 'Healthy normal control brain MRI slice containing zero lesions.',
        sequence: 'T1-Weighted Normal Control',
        expected_finding: 'Zero candidate regions detected (demonstrating false-positive rejection)'
      },
      {
        id: 'sample_4',
        name: 'Sample 4: Geometric Calibration Phantom',
        description: 'Synthetic calibration standard with known disks, squares, and impulsive noise.',
        sequence: 'Calibration Standard',
        expected_finding: 'Preservation of large shapes with complete suppression of small noise'
      }
    ];
  }
}

export async function runMriAnalysis(
  file: File,
  params: AnalysisParameters
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('enable_denoise', String(params.enable_denoise));
  formData.append('gaussian_sigma', String(params.gaussian_sigma));
  formData.append('enable_clahe', String(params.enable_clahe));
  formData.append('clahe_clip_limit', String(params.clahe_clip_limit));
  formData.append('clahe_tile_grid', String(params.clahe_tile_grid));
  formData.append('recon_method', params.recon_method);
  formData.append('selem_shape', params.selem_shape);
  formData.append('selem_size', String(params.selem_size));
  formData.append('h_param', String(params.h_param));
  formData.append('thresh_method', params.thresh_method);
  formData.append('manual_threshold', String(params.manual_threshold));
  formData.append('percentile_val', String(params.percentile_val));
  formData.append('min_area', String(params.min_area));
  formData.append('opening_radius', String(params.opening_radius));
  formData.append('closing_radius', String(params.closing_radius));
  formData.append('fill_holes', String(params.fill_holes));

  const res = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    let errMsg = `Analysis failed with status ${res.status}`;
    try {
      const errData = await res.json();
      if (errData.detail) errMsg = errData.detail;
    } catch {
      // ignore
    }
    throw new Error(errMsg);
  }

  return await res.json();
}

export async function runSampleAnalysis(
  sampleId: string,
  params: AnalysisParameters
): Promise<AnalysisResult> {
  const query = new URLSearchParams({
    enable_denoise: String(params.enable_denoise),
    gaussian_sigma: String(params.gaussian_sigma),
    enable_clahe: String(params.enable_clahe),
    clahe_clip_limit: String(params.clahe_clip_limit),
    clahe_tile_grid: String(params.clahe_tile_grid),
    recon_method: params.recon_method,
    selem_shape: params.selem_shape,
    selem_size: String(params.selem_size),
    h_param: String(params.h_param),
    thresh_method: params.thresh_method,
    manual_threshold: String(params.manual_threshold),
    percentile_val: String(params.percentile_val),
    min_area: String(params.min_area),
    opening_radius: String(params.opening_radius),
    closing_radius: String(params.closing_radius),
    fill_holes: String(params.fill_holes)
  });

  const res = await fetch(`${API_BASE_URL}/api/analyze-sample/${sampleId}?${query.toString()}`, {
    method: 'POST'
  });

  if (!res.ok) {
    let errMsg = `Sample analysis failed with status ${res.status}`;
    try {
      const errData = await res.json();
      if (errData.detail) errMsg = errData.detail;
    } catch {
      // ignore
    }
    throw new Error(errMsg);
  }

  return await res.json();
}
