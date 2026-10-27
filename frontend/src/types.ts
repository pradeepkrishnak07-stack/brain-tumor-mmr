export interface AnalysisParameters {
  enable_denoise: boolean;
  gaussian_sigma: number;
  enable_clahe: boolean;
  clahe_clip_limit: number;
  clahe_tile_grid: number;
  recon_method: 'opening_by_reconstruction' | 'h_dome' | 'closing_by_reconstruction' | 'regional_maxima_marker';
  selem_shape: 'disk' | 'square' | 'diamond' | 'octagon';
  selem_size: number;
  h_param: number;
  thresh_method: 'otsu' | 'triangle' | 'percentile' | 'manual';
  manual_threshold: number;
  percentile_val: number;
  min_area: number;
  opening_radius: number;
  closing_radius: number;
  fill_holes: boolean;
}

export interface RegionRecord {
  Region_ID: number;
  Area_Pixels: number;
  Centroid_X: number;
  Centroid_Y: number;
  BBox_YMin: number;
  BBox_XMin: number;
  BBox_YMax: number;
  BBox_XMax: number;
  Width: number;
  Height: number;
  Perimeter_Pixels: number;
  Aspect_Ratio: number;
  Circularity: number;
  Solidity: number;
  Extent: number;
  Mean_Intensity: number;
  Max_Intensity: number;
}

export interface IntensityProfile {
  slice_y: number;
  x_coords: number[];
  mask_profile: number[];
  reconstructed_profile: number[];
  residual_profile: number[];
}

export interface AnalysisResult {
  filename: string;
  dimensions: {
    width: number;
    height: number;
    channels: number;
  };
  processing_time_ms: number;
  parameters: Record<string, any>;
  results_summary: {
    candidate_regions_count: number;
    total_candidate_area_px: number;
    area_percentage: number;
    dominant_region_id: number | null;
    dominant_region_area: number;
    status_message: string;
  };
  regions: RegionRecord[];
  intensity_profile: IntensityProfile;
  images: {
    original: string;
    preprocessed: string;
    marker: string;
    mask: string;
    reconstructed: string;
    residual: string;
    candidate_mask: string;
    overlay: string;
  };
  report_markdown: string;
  disclaimer: string;
}

export interface SampleItem {
  id: string;
  name: string;
  description: string;
  sequence: string;
  expected_finding: string;
}
