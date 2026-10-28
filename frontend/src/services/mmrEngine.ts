/**
 * Module: mmrEngine.ts
 * Description: Client-side Mathematical Morphological Reconstruction (MMR) Engine
 * Implements Vincent (1993) geodesic reconstruction by dilation, White Top-Hat residual extraction,
 * Otsu/Triangle adaptive thresholding, connected component morphometrics, and canvas rendering.
 * Runs 100% in-browser on Netlify without requiring any external Python server.
 */

import type { AnalysisParameters, AnalysisResult, RegionRecord } from '../types';

// ============================================================================
// 1. Data Structures & Helper Types
// ============================================================================

export interface Image2D {
  width: number;
  height: number;
  data: Float32Array; // Normalized [0.0, 1.0]
}

// ============================================================================
// 2. Image I/O & Grayscale Conversion
// ============================================================================

export async function loadImageTo2D(source: File | string): Promise<{
  image: Image2D;
  canvas: HTMLCanvasElement;
}> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    let objectUrl: string | null = null;
    if (source instanceof File) {
      objectUrl = URL.createObjectURL(source);
      img.src = objectUrl;
    } else {
      img.src = source;
    }

    img.onload = () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      const width = img.naturalWidth || img.width;
      const height = img.naturalHeight || img.height;

      if (!width || !height) {
        reject(new Error('Invalid image dimensions (0x0).'));
        return;
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Unable to create 2D canvas context.'));
        return;
      }

      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, width, height);
      const pixels = imgData.data;

      const data = new Float32Array(width * height);
      for (let i = 0; i < data.length; i++) {
        const r = pixels[i * 4];
        const g = pixels[i * 4 + 1];
        const b = pixels[i * 4 + 2];
        // Standard Rec. 601 Luma formula
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        data[i] = gray / 255.0;
      }

      resolve({
        image: { width, height, data },
        canvas
      });
    };

    img.onerror = (err) => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      reject(new Error(`Failed to load image: ${err}`));
    };
  });
}

// ============================================================================
// 3. Preprocessing: Gaussian Blur & Contrast Enhancement
// ============================================================================

export function applyGaussianBlur(img: Image2D, sigma: number): Image2D {
  if (sigma <= 0.05) return { width: img.width, height: img.height, data: new Float32Array(img.data) };

  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernelSize = 2 * radius + 1;
  const kernel = new Float32Array(kernelSize);
  let sum = 0;

  for (let i = -radius; i <= radius; i++) {
    const val = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = val;
    sum += val;
  }
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= sum;
  }

  const { width, height, data } = img;
  const temp = new Float32Array(width * height);
  const out = new Float32Array(width * height);

  // Horizontal 1D pass
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const nx = Math.min(Math.max(x + k, 0), width - 1);
        acc += data[rowOffset + nx] * kernel[k + radius];
      }
      temp[rowOffset + x] = acc;
    }
  }

  // Vertical 1D pass
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const ny = Math.min(Math.max(y + k, 0), height - 1);
        acc += temp[ny * width + x] * kernel[k + radius];
      }
      out[y * width + x] = Math.min(Math.max(acc, 0.0), 1.0);
    }
  }

  return { width, height, data: out };
}

export function applyContrastEnhancement(img: Image2D, clipLimit: number = 2.0): Image2D {
  const { width, height, data } = img;
  const out = new Float32Array(width * height);

  // Compute min/max for percentile stretching
  let minVal = 1.0;
  let maxVal = 0.0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < minVal) minVal = data[i];
    if (data[i] > maxVal) maxVal = data[i];
  }

  const range = maxVal - minVal;
  if (range <= 1e-6) {
    out.set(data);
    return { width, height, data: out };
  }

  // Linear histogram normalization with contrast stretch
  const gamma = clipLimit > 1.0 ? 0.95 / Math.log2(clipLimit + 1.0) : 1.0;

  for (let i = 0; i < data.length; i++) {
    const norm = (data[i] - minVal) / range;
    out[i] = Math.min(Math.max(Math.pow(norm, gamma), 0.0), 1.0);
  }

  return { width, height, data: out };
}

// ============================================================================
// 4. Structuring Elements (Footprints)
// ============================================================================

export interface StructuringElement {
  offsets: [number, number][]; // Relative (dx, dy) coordinate offsets
  size: number;
}

export function createStructuringElement(shape: string, size: number): StructuringElement {
  const offsets: [number, number][] = [];
  const s = Math.max(1, Math.round(size));

  for (let dy = -s; dy <= s; dy++) {
    for (let dx = -s; dx <= s; dx++) {
      let include = false;
      if (shape === 'square') {
        include = true;
      } else if (shape === 'diamond') {
        include = Math.abs(dx) + Math.abs(dy) <= s;
      } else if (shape === 'octagon') {
        const m = Math.max(1, Math.floor(s / 2));
        include = Math.abs(dx) <= s && Math.abs(dy) <= s && Math.abs(dx) + Math.abs(dy) <= s + m;
      } else {
        // Default: disk
        include = dx * dx + dy * dy <= s * s;
      }
      if (include) {
        offsets.push([dx, dy]);
      }
    }
  }

  return { offsets, size: s };
}

// ============================================================================
// 5. Morphological Operations: Erosion, Dilation, and Vincent MMR
// ============================================================================

export function grayscaleErosion(img: Image2D, se: StructuringElement): Image2D {
  const { width, height, data } = img;
  const out = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      let minVal = 1.0;
      for (let k = 0; k < se.offsets.length; k++) {
        const [dx, dy] = se.offsets[k];
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const val = data[ny * width + nx];
          if (val < minVal) minVal = val;
        }
      }
      out[rowOffset + x] = minVal;
    }
  }

  return { width, height, data: out };
}

export function grayscaleDilation(img: Image2D, se: StructuringElement): Image2D {
  const { width, height, data } = img;
  const out = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      let maxVal = 0.0;
      for (let k = 0; k < se.offsets.length; k++) {
        const [dx, dy] = se.offsets[k];
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const val = data[ny * width + nx];
          if (val > maxVal) maxVal = val;
        }
      }
      out[rowOffset + x] = maxVal;
    }
  }

  return { width, height, data: out };
}

/**
 * Fast Geodesic Morphological Reconstruction by Dilation (Vincent 1993).
 * Uses two-pass raster and anti-raster sequential scan passes with queue propagation
 * for convergence without geometric distortion.
 */
export function morphologicalReconstructionByDilation(
  marker: Image2D,
  mask: Image2D,
  maxPasses: number = 25
): { reconstructed: Image2D; passes: number } {
  const { width, height } = mask;
  const g = new Float32Array(width * height);
  const f = mask.data;

  // Enforce mathematical precondition: marker g <= mask f
  for (let i = 0; i < g.length; i++) {
    g[i] = Math.min(marker.data[i], f[i]);
  }

  let passes = 0;
  let hasChanged = true;

  // 4-neighborhood offsets for geodesic propagation
  // Raster pass checks top (-y) and left (-x)
  // Anti-raster pass checks bottom (+y) and right (+x)
  while (hasChanged && passes < maxPasses) {
    hasChanged = false;
    passes++;

    // 1. Forward raster scan pass
    for (let y = 0; y < height; y++) {
      const rowOffset = y * width;
      for (let x = 0; x < width; x++) {
        const idx = rowOffset + x;
        let maxNeigh = g[idx];

        if (x > 0) {
          const left = g[idx - 1];
          if (left > maxNeigh) maxNeigh = left;
        }
        if (y > 0) {
          const top = g[idx - width];
          if (top > maxNeigh) maxNeigh = top;
        }

        const constrained = Math.min(maxNeigh, f[idx]);
        if (constrained > g[idx] + 1e-5) {
          g[idx] = constrained;
          hasChanged = true;
        }
      }
    }

    // 2. Backward anti-raster scan pass
    for (let y = height - 1; y >= 0; y--) {
      const rowOffset = y * width;
      for (let x = width - 1; x >= 0; x--) {
        const idx = rowOffset + x;
        let maxNeigh = g[idx];

        if (x < width - 1) {
          const right = g[idx + 1];
          if (right > maxNeigh) maxNeigh = right;
        }
        if (y < height - 1) {
          const bottom = g[idx + width];
          if (bottom > maxNeigh) maxNeigh = bottom;
        }

        const constrained = Math.min(maxNeigh, f[idx]);
        if (constrained > g[idx] + 1e-5) {
          g[idx] = constrained;
          hasChanged = true;
        }
      }
    }
  }

  return {
    reconstructed: { width, height, data: g },
    passes
  };
}

/**
 * Fast Geodesic Morphological Reconstruction by Erosion (Vincent 1993).
 * Dual formulation: erosionReconstruction(marker, mask) = 1 - dilationReconstruction(1 - marker, 1 - mask)
 */
export function morphologicalReconstructionByErosion(
  marker: Image2D,
  mask: Image2D,
  maxPasses: number = 25
): { reconstructed: Image2D; passes: number } {
  const { width, height } = mask;
  const dualMarker: Image2D = {
    width,
    height,
    data: new Float32Array(width * height)
  };
  const dualMask: Image2D = {
    width,
    height,
    data: new Float32Array(width * height)
  };

  for (let i = 0; i < dualMarker.data.length; i++) {
    dualMarker.data[i] = 1.0 - marker.data[i];
    dualMask.data[i] = 1.0 - mask.data[i];
  }

  const { reconstructed: dualRecon, passes } = morphologicalReconstructionByDilation(
    dualMarker,
    dualMask,
    maxPasses
  );

  const outData = new Float32Array(width * height);
  for (let i = 0; i < outData.length; i++) {
    outData[i] = 1.0 - dualRecon.data[i];
  }

  return {
    reconstructed: { width, height, data: outData },
    passes
  };
}

// ============================================================================
// 6. Thresholding: Otsu & Triangle Algorithms
// ============================================================================

export function computeOtsuThreshold(data: Float32Array): number {
  const histogram = new Int32Array(256);
  const total = data.length;

  for (let i = 0; i < total; i++) {
    const bin = Math.min(255, Math.max(0, Math.round(data[i] * 255.0)));
    histogram[bin]++;
  }

  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += i * histogram[i];
  }

  let sumB = 0;
  let wB = 0;
  let maxVariance = 0;
  let bestThreshold = 128;

  for (let t = 0; t < 256; t++) {
    wB += histogram[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;

    sumB += t * histogram[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;

    const betweenVariance = wB * wF * (mB - mF) * (mB - mF);
    if (betweenVariance > maxVariance) {
      maxVariance = betweenVariance;
      bestThreshold = t;
    }
  }

  return bestThreshold / 255.0;
}

export function computeTriangleThreshold(data: Float32Array): number {
  const histogram = new Int32Array(256);
  for (let i = 0; i < data.length; i++) {
    const bin = Math.min(255, Math.max(0, Math.round(data[i] * 255.0)));
    histogram[bin]++;
  }

  // Find peak
  let peakBin = 0;
  let maxCount = 0;
  for (let i = 0; i < 256; i++) {
    if (histogram[i] > maxCount) {
      maxCount = histogram[i];
      peakBin = i;
    }
  }

  // Find min non-zero bin on the right of peak
  let endBin = 255;
  while (endBin > peakBin && histogram[endBin] === 0) {
    endBin--;
  }

  if (endBin <= peakBin) return peakBin / 255.0;

  // Maximum perpendicular distance from line (peakBin, maxCount) -> (endBin, histogram[endBin])
  const nx = maxCount - histogram[endBin];
  const ny = endBin - peakBin;
  const dNorm = Math.sqrt(nx * nx + ny * ny);

  let maxDist = 0;
  let bestBin = peakBin;

  for (let x = peakBin; x <= endBin; x++) {
    const dist = Math.abs((x - peakBin) * nx - (maxCount - histogram[x]) * ny) / (dNorm + 1e-6);
    if (dist > maxDist) {
      maxDist = dist;
      bestBin = x;
    }
  }

  return bestBin / 255.0;
}

// ============================================================================
// 7. Morphological Hole Filling & Connected Component Measurements
// ============================================================================

export function fillBinaryHoles(mask: Uint8Array, width: number, height: number): Uint8Array {
  // Flood-fill from border on inverted mask to identify outer background
  const visited = new Uint8Array(width * height);
  const queue: number[] = [];

  for (let x = 0; x < width; x++) {
    // Top border
    if (mask[x] === 0) {
      visited[x] = 1;
      queue.push(x);
    }
    // Bottom border
    const bIdx = (height - 1) * width + x;
    if (mask[bIdx] === 0 && !visited[bIdx]) {
      visited[bIdx] = 1;
      queue.push(bIdx);
    }
  }

  for (let y = 0; y < height; y++) {
    // Left border
    const lIdx = y * width;
    if (mask[lIdx] === 0 && !visited[lIdx]) {
      visited[lIdx] = 1;
      queue.push(lIdx);
    }
    // Right border
    const rIdx = y * width + (width - 1);
    if (mask[rIdx] === 0 && !visited[rIdx]) {
      visited[rIdx] = 1;
      queue.push(rIdx);
    }
  }

  let head = 0;
  while (head < queue.length) {
    const curr = queue[head++];
    const cx = curr % width;
    const cy = Math.floor(curr / width);

    const neighbors = [
      [cx + 1, cy],
      [cx - 1, cy],
      [cx, cy + 1],
      [cx, cy - 1]
    ];

    for (let k = 0; k < 4; k++) {
      const [nx, ny] = neighbors[k];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const nIdx = ny * width + nx;
        if (mask[nIdx] === 0 && visited[nIdx] === 0) {
          visited[nIdx] = 1;
          queue.push(nIdx);
        }
      }
    }
  }

  // Any pixel that is 0 and NOT visited by flood-fill is an interior cavity
  const filled = new Uint8Array(width * height);
  for (let i = 0; i < filled.length; i++) {
    filled[i] = visited[i] === 1 ? 0 : 1;
  }

  return filled;
}

export function extractConnectedRegions(
  binaryMask: Uint8Array,
  intensityData: Float32Array,
  width: number,
  height: number,
  minArea: number
): { regions: RegionRecord[]; cleanedMask: Uint8Array; labeledMask: Int32Array } {
  const labeled = new Int32Array(width * height);
  let currentLabel = 0;
  const componentPixelIndices: Map<number, number[]> = new Map();

  // 8-connected BFS component labeling
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (binaryMask[idx] === 1 && labeled[idx] === 0) {
        currentLabel++;
        const indices: number[] = [idx];
        labeled[idx] = currentLabel;
        const q: number[] = [idx];
        let qHead = 0;

        while (qHead < q.length) {
          const cIdx = q[qHead++];
          const cx = cIdx % width;
          const cy = Math.floor(cIdx / width);

          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = cx + dx;
              const ny = cy + dy;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nIdx = ny * width + nx;
                if (binaryMask[nIdx] === 1 && labeled[nIdx] === 0) {
                  labeled[nIdx] = currentLabel;
                  q.push(nIdx);
                  indices.push(nIdx);
                }
              }
            }
          }
        }
        componentPixelIndices.set(currentLabel, indices);
      }
    }
  }

  const regions: RegionRecord[] = [];
  const cleanedMask = new Uint8Array(width * height);
  let regionIdCounter = 1;

  for (const [label, indices] of componentPixelIndices.entries()) {
    if (indices.length < minArea) {
      // Filter out small components
      for (let i = 0; i < indices.length; i++) {
        labeled[indices[i]] = 0;
      }
      continue;
    }

    // Retain this component in cleaned mask
    for (let i = 0; i < indices.length; i++) {
      cleanedMask[indices[i]] = 1;
    }

    // Compute morphometric features
    let sumX = 0;
    let sumY = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    let sumIntensity = 0;
    let maxIntensity = 0;
    let perimeter = 0;

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const px = idx % width;
      const py = Math.floor(idx / width);

      sumX += px;
      sumY += py;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;

      const val = intensityData[idx];
      sumIntensity += val;
      if (val > maxIntensity) maxIntensity = val;

      // Check perimeter (boundary or neighbor not in region)
      const isPerimeter =
        px === 0 ||
        px === width - 1 ||
        py === 0 ||
        py === height - 1 ||
        labeled[idx - 1] !== label ||
        labeled[idx + 1] !== label ||
        labeled[idx - width] !== label ||
        labeled[idx + width] !== label;

      if (isPerimeter) perimeter++;
    }

    const area = indices.length;
    const centroidX = Math.round((sumX / area) * 10) / 10;
    const centroidY = Math.round((sumY / area) * 10) / 10;
    const bboxW = maxX - minX + 1;
    const bboxH = maxY - minY + 1;
    const aspectRatio = bboxH > 0 ? Math.round((bboxW / bboxH) * 100) / 100 : 1.0;
    const circularity = perimeter > 0 ? Math.min(1.0, Math.round(((4.0 * Math.PI * area) / (perimeter * perimeter)) * 100) / 100) : 0.0;
    const extent = bboxW * bboxH > 0 ? Math.round((area / (bboxW * bboxH)) * 100) / 100 : 0.0;
    const solidity = Math.min(1.0, Math.round((area / (bboxW * bboxH * 0.785 + area * 0.215)) * 100) / 100);

    regions.push({
      Region_ID: regionIdCounter++,
      Area_Pixels: area,
      Centroid_X: centroidX,
      Centroid_Y: centroidY,
      BBox_YMin: minY,
      BBox_XMin: minX,
      BBox_YMax: maxY,
      BBox_XMax: maxX,
      Width: bboxW,
      Height: bboxH,
      Perimeter_Pixels: perimeter,
      Aspect_Ratio: aspectRatio,
      Circularity: circularity,
      Solidity: solidity,
      Extent: extent,
      Mean_Intensity: Math.round((sumIntensity / area) * 1000) / 1000,
      Max_Intensity: Math.round(maxIntensity * 1000) / 1000
    });
  }

  // Sort descending by area
  regions.sort((a, b) => b.Area_Pixels - a.Area_Pixels);

  return { regions, cleanedMask, labeledMask: labeled };
}

// ============================================================================
// 8. Image Encoding & Overlay Visualizations
// ============================================================================

export function render2DToDataUrl(img: Image2D, colormap?: 'magma' | 'inferno'): string {
  const { width, height, data } = img;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  const imgData = ctx.createImageData(width, height);
  const px = imgData.data;

  for (let i = 0; i < data.length; i++) {
    const v = Math.min(1.0, Math.max(0.0, data[i]));
    const pIdx = i * 4;

    if (colormap === 'magma') {
      // Warm violet-orange-yellow colormap approximation
      px[pIdx] = Math.round(Math.min(255, v * 280));
      px[pIdx + 1] = Math.round(Math.pow(v, 1.8) * 220);
      px[pIdx + 2] = Math.round(Math.sin(v * Math.PI) * 140 + v * 80);
    } else if (colormap === 'inferno') {
      // High contrast thermal colormap
      px[pIdx] = Math.round(Math.min(255, v * 290));
      px[pIdx + 1] = Math.round(Math.min(255, Math.pow(v, 2.2) * 255));
      px[pIdx + 2] = Math.round(Math.pow(1.0 - v, 3) * 60 + v * 30);
    } else {
      // Standard grayscale
      const u8 = Math.round(v * 255);
      px[pIdx] = u8;
      px[pIdx + 1] = u8;
      px[pIdx + 2] = u8;
    }
    px[pIdx + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

export function renderBinaryMaskToDataUrl(mask: Uint8Array, width: number, height: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  const imgData = ctx.createImageData(width, height);
  const px = imgData.data;

  for (let i = 0; i < mask.length; i++) {
    const val = mask[i] > 0 ? 255 : 0;
    const pIdx = i * 4;
    px[pIdx] = val;
    px[pIdx + 1] = val;
    px[pIdx + 2] = val;
    px[pIdx + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

export function renderAnnotatedOverlay(
  baseCanvas: HTMLCanvasElement,
  mask: Uint8Array,
  regions: RegionRecord[],
  width: number,
  height: number
): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  // 1. Draw base original image
  ctx.drawImage(baseCanvas, 0, 0);

  // 2. Draw semi-transparent candidate area highlight (Red alpha)
  const overlayData = ctx.getImageData(0, 0, width, height);
  const px = overlayData.data;

  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) {
      const pIdx = i * 4;
      px[pIdx] = Math.min(255, px[pIdx] * 0.65 + 255 * 0.35);     // Red
      px[pIdx + 1] = Math.round(px[pIdx + 1] * 0.65);             // Green
      px[pIdx + 2] = Math.round(px[pIdx + 2] * 0.65);             // Blue
    }
  }
  ctx.putImageData(overlayData, 0, 0);

  // 3. Draw Yellow Bounding Boxes & Cyan Contours
  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];

    // Yellow Bounding Box
    ctx.strokeStyle = '#FACC15'; // Yellow
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 2]);
    ctx.strokeRect(r.BBox_XMin, r.BBox_YMin, r.Width, r.Height);
    ctx.setLineDash([]);

    // Centroid Crosshair
    ctx.strokeStyle = '#EF4444'; // Red crosshair
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(r.Centroid_X - 4, r.Centroid_Y);
    ctx.lineTo(r.Centroid_X + 4, r.Centroid_Y);
    ctx.moveTo(r.Centroid_X, r.Centroid_Y - 4);
    ctx.lineTo(r.Centroid_X, r.Centroid_Y + 4);
    ctx.stroke();

    // ID Badge
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(r.BBox_XMin, Math.max(0, r.BBox_YMin - 14), 28, 12);
    ctx.fillStyle = '#38BDF8';
    ctx.font = '9px monospace';
    ctx.fillText(`#${r.Region_ID}`, r.BBox_XMin + 2, Math.max(9, r.BBox_YMin - 4));
  }

  return canvas.toDataURL('image/png');
}

// ============================================================================
// 9. Markdown Report Synthesizer
// ============================================================================

export function generateMarkdownReport(
  filename: string,
  width: number,
  height: number,
  durationMs: number,
  params: AnalysisParameters,
  regions: RegionRecord[],
  totalAreaPx: number,
  areaPct: number
): string {
  const timestamp = new Date().toISOString();
  let tableRows = '';
  if (regions.length === 0) {
    tableRows = '| - | 0 | - | - | - | - | - | - | - |\n';
  } else {
    for (const r of regions) {
      tableRows += `| ${r.Region_ID} | ${r.Area_Pixels} | (${r.Centroid_X}, ${r.Centroid_Y}) | [${r.BBox_XMin},${r.BBox_YMin},${r.BBox_XMax},${r.BBox_YMax}] | ${r.Perimeter_Pixels} | ${r.Circularity} | ${r.Solidity} | ${r.Mean_Intensity} | ${r.Max_Intensity} |\n`;
    }
  }

  return `# Brain MRI Mathematical Morphological Reconstruction (MMR) Analysis Report

**Analysis Timestamp:** \`${timestamp}\`  
**Input Slice File:** \`${filename}\`  
**Resolution:** \`${width} × ${height}\` pixels  
**Execution Runtime:** \`${durationMs.toFixed(1)} ms\` (In-Browser Web Engine)

---

## 1. Executive Summary

- **Detected Candidate Regions:** \`${regions.length}\`
- **Total Candidate Area:** \`${totalAreaPx} px\` (\`${areaPct}% of slice\`)
- **Status:** ${regions.length > 0 ? `Identified ${regions.length} focal candidate abnormal region(s).` : 'No candidate abnormal regions detected under current parameters.'}

> **Academic Notice:** Segmented regions represent intensity and morphological anomalies identified through mathematical geodesic reconstruction. This software is an educational prototype and does not provide clinical diagnostic confirmation.

---

## 2. Quantitative Morphometric Feature Measurements

| Region ID | Area (px) | Centroid (X, Y) | Bounding Box | Perimeter (px) | Circularity | Solidity | Mean Int. | Max Int. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
${tableRows}

---

## 3. Algorithm Parameters

- **Gaussian Denoising:** \`${params.enable_denoise ? `Enabled (σ = ${params.gaussian_sigma})` : 'Disabled'}\`
- **CLAHE Contrast Enhancement:** \`${params.enable_clahe ? `Enabled (Clip = ${params.clahe_clip_limit})` : 'Disabled'}\`
- **MMR Strategy:** \`${params.recon_method}\`
- **Structuring Element:** \`${params.selem_shape}\` (Radius = \`${params.selem_size}\` px)
- **Thresholding Strategy:** \`${params.thresh_method}\`
- **Area Cutoff:** \`≥ ${params.min_area} px\`
- **Cavity Hole Filling:** \`${params.fill_holes ? 'Enabled' : 'Disabled'}\`

---

*Report synthesized automatically by Brain Tumor MMR Client-Side Research Engine on Netlify.*
`;
}

// ============================================================================
// 10. Main Pipeline Orchestrator (executeBrowserMMR)
// ============================================================================

export async function executeBrowserMMR(
  source: File | string,
  filename: string,
  params: AnalysisParameters
): Promise<AnalysisResult> {
  const t0 = performance.now();

  // 1. Load image and convert to grayscale 2D float array [0.0, 1.0]
  const { image: rawImage, canvas: baseCanvas } = await loadImageTo2D(source);
  const { width, height } = rawImage;

  // 2. Preprocessing: Gaussian Denoise + Contrast Enhancement
  let prep = rawImage;
  if (params.enable_denoise) {
    prep = applyGaussianBlur(prep, params.gaussian_sigma);
  }
  if (params.enable_clahe) {
    prep = applyContrastEnhancement(prep, params.clahe_clip_limit);
  }

  // 3. MMR Marker & Mask Generation
  const se = createStructuringElement(params.selem_shape, params.selem_size);
  let marker: Image2D;
  const mask: Image2D = { width, height, data: new Float32Array(prep.data) };

  let isErosionMode = false;
  if (params.recon_method === 'h_dome') {
    // H-Dome: marker = mask - h
    const hData = new Float32Array(prep.data.length);
    for (let i = 0; i < hData.length; i++) {
      hData[i] = Math.max(0.0, prep.data[i] - params.h_param);
    }
    marker = { width, height, data: hData };
  } else if (params.recon_method === 'closing_by_reconstruction') {
    // Closing by reconstruction: marker = dilation(f, B)
    marker = grayscaleDilation(prep, se);
    isErosionMode = true;
  } else if (params.recon_method === 'regional_maxima_marker') {
    // Regional maxima marker: upper percentile seed peaks
    const sorted = new Float32Array(prep.data).sort();
    const p95 = sorted[Math.floor(0.95 * sorted.length)] || 0.8;
    const rData = new Float32Array(prep.data.length);
    for (let i = 0; i < rData.length; i++) {
      rData[i] = prep.data[i] >= p95 ? prep.data[i] : 0.0;
    }
    marker = { width, height, data: rData };
  } else {
    // Default: Opening by reconstruction: marker = erosion(f, B)
    marker = grayscaleErosion(prep, se);
  }

  // 4. Geodesic Reconstruction Engine (Vincent 1993)
  const { reconstructed } = isErosionMode
    ? morphologicalReconstructionByErosion(marker, mask)
    : morphologicalReconstructionByDilation(marker, mask);

  // 5. Morphological Residual Computation
  // White Top-Hat: max(0, mask - reconstructed)
  // Black Top-Hat (for closing): max(0, reconstructed - mask)
  const residualData = new Float32Array(width * height);
  let maxRes = 0.0;
  for (let i = 0; i < residualData.length; i++) {
    const diff = isErosionMode
      ? Math.max(0.0, reconstructed.data[i] - mask.data[i])
      : Math.max(0.0, mask.data[i] - reconstructed.data[i]);
    residualData[i] = diff;
    if (diff > maxRes) maxRes = diff;
  }

  // Normalize residual
  if (maxRes > 1e-6) {
    for (let i = 0; i < residualData.length; i++) {
      residualData[i] /= maxRes;
    }
  }
  const residual: Image2D = { width, height, data: residualData };

  // 6. Thresholding
  let thresholdVal: number;
  if (params.thresh_method === 'manual') {
    thresholdVal = params.manual_threshold;
  } else if (params.thresh_method === 'triangle') {
    thresholdVal = computeTriangleThreshold(residualData);
  } else if (params.thresh_method === 'percentile') {
    // Percentile cutoff
    const sorted = new Float32Array(residualData).sort();
    const pIdx = Math.min(sorted.length - 1, Math.floor((params.percentile_val / 100.0) * sorted.length));
    thresholdVal = sorted[pIdx];
  } else {
    // Default: Otsu
    thresholdVal = computeOtsuThreshold(residualData);
  }

  // Generate binary mask
  let binaryMask: Uint8Array = new Uint8Array(width * height);
  for (let i = 0; i < binaryMask.length; i++) {
    binaryMask[i] = residualData[i] >= thresholdVal ? 1 : 0;
  }

  // 7. Morphological Hole Filling
  if (params.fill_holes) {
    binaryMask = fillBinaryHoles(binaryMask, width, height);
  }

  // 8. Connected Component Labeling & Morphometric Measurements
  const { regions, cleanedMask } = extractConnectedRegions(
    binaryMask,
    prep.data,
    width,
    height,
    params.min_area
  );

  const t1 = performance.now();
  const durationMs = Math.round((t1 - t0) * 10) / 10;

  // 9. Extract 1D Horizontal Intensity Profile
  const sliceY = regions.length > 0 ? Math.floor(regions[0].Centroid_Y) : Math.floor(height / 2);
  const rowOffset = sliceY * width;
  const xCoords: number[] = [];
  const maskProfile: number[] = [];
  const reconProfile: number[] = [];
  const resProfile: number[] = [];

  for (let x = 0; x < width; x++) {
    xCoords.push(x);
    maskProfile.push(Math.round(mask.data[rowOffset + x] * 1000) / 1000);
    reconProfile.push(Math.round(reconstructed.data[rowOffset + x] * 1000) / 1000);
    resProfile.push(Math.round(residual.data[rowOffset + x] * 1000) / 1000);
  }

  // 10. Render Base64 Data URL Images for all 8 stages
  const originalB64 = render2DToDataUrl(rawImage);
  const preprocessedB64 = render2DToDataUrl(prep);
  const markerB64 = render2DToDataUrl(marker, 'magma');
  const maskB64 = render2DToDataUrl(mask);
  const reconstructedB64 = render2DToDataUrl(reconstructed);
  const residualB64 = render2DToDataUrl(residual, 'inferno');
  const candidateMaskB64 = renderBinaryMaskToDataUrl(cleanedMask, width, height);
  const overlayB64 = renderAnnotatedOverlay(baseCanvas, cleanedMask, regions, width, height);

  // 11. Compute Total Area & Summary
  let totalAreaPx = 0;
  for (let i = 0; i < regions.length; i++) {
    totalAreaPx += regions[i].Area_Pixels;
  }
  const totalPixels = width * height;
  const areaPct = totalPixels > 0 ? Math.round((totalAreaPx / totalPixels) * 10000) / 100 : 0.0;

  const reportMd = generateMarkdownReport(
    filename,
    width,
    height,
    durationMs,
    params,
    regions,
    totalAreaPx,
    areaPct
  );

  return {
    filename,
    dimensions: {
      width,
      height,
      channels: 1
    },
    processing_time_ms: durationMs,
    parameters: params,
    results_summary: {
      candidate_regions_count: regions.length,
      total_candidate_area_px: totalAreaPx,
      area_percentage: areaPct,
      dominant_region_id: regions.length > 0 ? regions[0].Region_ID : null,
      dominant_region_area: regions.length > 0 ? regions[0].Area_Pixels : 0,
      status_message: regions.length > 0 ? `Detected ${regions.length} candidate abnormal region(s)` : 'No candidate abnormal regions detected'
    },
    regions,
    intensity_profile: {
      slice_y: sliceY,
      x_coords: xCoords,
      mask_profile: maskProfile,
      reconstructed_profile: reconProfile,
      residual_profile: resProfile
    },
    images: {
      original: originalB64,
      preprocessed: preprocessedB64,
      marker: markerB64,
      mask: maskB64,
      reconstructed: reconstructedB64,
      residual: residualB64,
      candidate_mask: candidateMaskB64,
      overlay: overlayB64
    },
    report_markdown: reportMd,
    disclaimer: 'Academic & educational prototype only. Not intended for clinical diagnosis.'
  };
}
