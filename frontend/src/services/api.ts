import type { AnalysisParameters, AnalysisResult, SampleItem } from '../types';
import { executeBrowserMMR } from './mmrEngine';

// Normalized API base URL without trailing slash (if configured)
const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || '';
export const API_BASE_URL = rawBaseUrl.replace(/\/+$/, '');

export const SAMPLE_DATASETS: SampleItem[] = [
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

export const SAMPLE_PATH_MAP: Record<string, { path: string; filename: string }> = {
  sample_1: { path: '/sample_data/sample_1_t1_hyperintense.png', filename: 'sample_1_t1_hyperintense.png' },
  sample_2: { path: '/sample_data/sample_2_t2_frontal_lesion.png', filename: 'sample_2_t2_frontal_lesion.png' },
  sample_3: { path: '/sample_data/sample_3_normal_control.png', filename: 'sample_3_normal_control.png' },
  sample_4: { path: '/sample_data/sample_4_calibration_phantom.png', filename: 'sample_4_calibration_phantom.png' }
};

/**
 * Checks whether an external Python API is reachable.
 * In a Netlify-only deployment (no backend URL configured or backend unreachable),
 * the system returns `{ ok: true, isBrowserEngine: true }` indicating the browser MMR engine is ready.
 */
export async function checkBackendHealth(): Promise<{
  ok: boolean;
  isBrowserEngine: boolean;
  service: string;
  message?: string;
}> {
  if (!API_BASE_URL) {
    // Netlify-only mode: browser-native MMR engine is active
    return {
      ok: true,
      isBrowserEngine: true,
      service: 'Netlify Standalone Engine (In-Browser MMR)'
    };
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return {
        ok: true,
        isBrowserEngine: false,
        service: data.service || 'FastAPI Cloud API'
      };
    }
  } catch {
    // External API unreachable, fallback to browser MMR engine
  }

  return {
    ok: true,
    isBrowserEngine: true,
    service: 'Netlify Standalone Engine (In-Browser MMR)'
  };
}

export async function fetchSamplePhantoms(): Promise<SampleItem[]> {
  if (API_BASE_URL) {
    try {
      const res = await fetch(`${API_BASE_URL}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        if (data.samples && data.samples.length > 0) return data.samples;
      }
    } catch {
      // Fallback to static sample datasets
    }
  }
  return SAMPLE_DATASETS;
}

/**
 * Executes Mathematical Morphological Reconstruction (MMR) on an uploaded MRI file.
 * Prioritizes the client-side Web Engine for 100% Netlify-only standalone deployments.
 */
export async function runMriAnalysis(
  file: File,
  params: AnalysisParameters
): Promise<AnalysisResult> {
  // If no external backend configured, run directly in browser
  if (!API_BASE_URL) {
    return await executeBrowserMMR(file, file.name, params);
  }

  // If external backend configured, attempt remote analysis with browser fallback
  try {
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

    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('Remote backend unavailable, falling back to in-browser MMR engine.', err);
  }

  // Fallback to in-browser MMR engine
  return await executeBrowserMMR(file, file.name, params);
}

/**
 * Executes MMR analysis on a pre-configured sample MRI phantom.
 */
export async function runSampleAnalysis(
  sampleId: string,
  params: AnalysisParameters
): Promise<AnalysisResult> {
  // If no external backend configured, run in browser with bundled sample assets
  if (!API_BASE_URL) {
    const sampleInfo = SAMPLE_PATH_MAP[sampleId] || SAMPLE_PATH_MAP.sample_1;
    return await executeBrowserMMR(sampleInfo.path, sampleInfo.filename, params);
  }

  try {
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

    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('Remote backend unavailable, falling back to in-browser MMR engine for sample.', err);
  }

  const sampleInfo = SAMPLE_PATH_MAP[sampleId] || SAMPLE_PATH_MAP.sample_1;
  return await executeBrowserMMR(sampleInfo.path, sampleInfo.filename, params);
}
