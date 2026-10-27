import React from 'react';
import {
  Workflow,
  ArrowDown,
  BookOpen
} from 'lucide-react';

export const MethodologyView: React.FC = () => {
  const steps = [
    {
      num: '01',
      title: 'MRI Upload & Input Validation',
      desc: 'Validates input file dimensions, channel formats (2D grayscale, 3D RGB, 4D RGBA), and non-empty byte streams. Rejects corrupt or non-finite inputs.'
    },
    {
      num: '02',
      title: 'Grayscale Conversion & Normalization',
      desc: 'Transforms arbitrary color channels into a single 8-bit luminance matrix, followed by floating-point min-max normalization to [0.0, 1.0].'
    },
    {
      num: '03',
      title: 'Noise Reduction & CLAHE Enhancement',
      desc: 'Applies Gaussian filtering (σ) to suppress high-frequency acquisition noise, followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) for local parenchymal contrast.'
    },
    {
      num: '04',
      title: 'Marker and Mask Image Creation',
      desc: 'Constructs mathematically rigorous marker seed g and constraint mask f. For Opening by Reconstruction, marker g = f ⊖ B (erosion of mask by structuring element B), ensuring invariant g ≤ f.'
    },
    {
      num: '05',
      title: 'Mathematical Morphological Reconstruction',
      desc: 'Repeatedly executes geodesic dilations δ_f(g) = (g ⊕ B) ∧ f until convergence limit R^δ_f(g). Perfectly preserves exact object shapes larger than B without boundary rounding.'
    },
    {
      num: '06',
      title: 'Residual Map Extraction (White Top-Hat)',
      desc: 'Calculates the difference between original preprocessed mask and reconstructed image: WTH(f) = f - R^δ_f(f ⊖ B). Selectively isolates focal hyperintense candidate peaks.'
    },
    {
      num: '07',
      title: 'Candidate Region Segmentation & Cleanup',
      desc: 'Applies automated Otsu, Triangle, or percentile thresholding, morphological closing to bridge intra-lesion micro-cavities, and connected-component size filtering to discard noise.'
    },
    {
      num: '08',
      title: 'Quantitative Morphometrics & Reporting',
      desc: 'Calculates pixel area, centroid coordinates, bounding box, perimeter, aspect ratio, circularity (isoperimetric quotient), and solidity. Generates downloadable overlays, CSVs, and reports.'
    }
  ];

  return (
    <div className="space-y-10 py-4 max-w-5xl mx-auto">
      {/* Title */}
      <div>
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300 text-xs font-semibold tracking-wide mb-3">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Algorithmic Architecture</span>
        </div>
        <h1 className="text-3xl font-bold text-white tracking-tight">Mathematical Methodology</h1>
        <p className="text-slate-400 text-sm mt-1">
          Complete mathematical derivation and operational pipeline of Mathematical Morphological Reconstruction.
        </p>
      </div>

      {/* Concept Comparison Box */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
          <div className="text-xs font-bold text-red-400 uppercase tracking-wider">Conventional Standard Opening</div>
          <h3 className="text-base font-bold text-white">γ_B(f) = (f ⊖ B) ⊕ B</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            In standard morphological opening, the image is eroded and then dilated by structuring element B.
            While it eliminates bright features smaller than B, it severely rounds corners, degrades object geometry,
            and displaces true structural boundaries.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900 border border-cyan-800/60 space-y-3">
          <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider">Morphological Reconstruction (MMR)</div>
          <h3 className="text-base font-bold text-white">R^δ_f(g) = lim k→∞ δ^(k)_f(g)</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            In MMR, the image is eroded into a marker seed g = f ⊖ B and then iteratively dilated under the continuous
            constraint of mask f: δ_f(g) = (g ⊕ B) ∧ f. Reconstruction stops exactly at object boundaries, preserving
            shapes larger than B with <strong>zero edge distortion or corner rounding</strong>.
          </p>
        </div>
      </div>

      {/* Pipeline Diagram Flow */}
      <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 sm:p-8 space-y-6">
        <h2 className="text-lg font-bold text-white flex items-center space-x-2">
          <Workflow className="w-5 h-5 text-cyan-400" />
          <span>Complete Execution Flow Diagram</span>
        </h2>

        <div className="space-y-4">
          {steps.map((st, i) => (
            <div key={st.num} className="relative">
              <div className="flex items-start space-x-4 p-4 rounded-xl bg-slate-800/40 border border-slate-800 hover:border-slate-700 transition-colors">
                <div className="w-9 h-9 rounded-lg bg-cyan-950 text-cyan-400 flex items-center justify-center font-mono font-bold text-sm flex-shrink-0 border border-cyan-800/60">
                  {st.num}
                </div>
                <div className="space-y-1">
                  <h4 className="font-bold text-white text-sm">{st.title}</h4>
                  <p className="text-xs text-slate-400 leading-relaxed">{st.desc}</p>
                </div>
              </div>
              {i < steps.length - 1 && (
                <div className="flex justify-center py-1">
                  <ArrowDown className="w-4 h-4 text-slate-600" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Morphometric Feature Definitions */}
      <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 sm:p-8 space-y-4">
        <h2 className="text-lg font-bold text-white">Extracted Morphometric Feature Formulations</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-800/80 text-cyan-400 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-2.5 px-3">Metric</th>
                <th className="py-2.5 px-3">Formula</th>
                <th className="py-2.5 px-3">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono">
              <tr>
                <td className="py-2.5 px-3 font-bold text-white">Area</td>
                <td className="py-2.5 px-3 text-cyan-300">A = ∑ 1</td>
                <td className="py-2.5 px-3 font-sans text-slate-400">Total pixel count belonging to the candidate component.</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-white">Centroid</td>
                <td className="py-2.5 px-3 text-cyan-300">(cx, cy) = (∑x/A, ∑y/A)</td>
                <td className="py-2.5 px-3 font-sans text-slate-400">Spatial center of mass in 2D image coordinates.</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-white">Circularity</td>
                <td className="py-2.5 px-3 text-cyan-300">C = (4π · A) / P²</td>
                <td className="py-2.5 px-3 font-sans text-slate-400">Isoperimetric quotient (1.0 = perfect circle; lower for irregular borders).</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-white">Aspect Ratio</td>
                <td className="py-2.5 px-3 text-cyan-300">AR = Width / Height</td>
                <td className="py-2.5 px-3 font-sans text-slate-400">Geometric elongation ratio along cardinal axes.</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-white">Solidity</td>
                <td className="py-2.5 px-3 text-cyan-300">S = A / A_convex_hull</td>
                <td className="py-2.5 px-3 font-sans text-slate-400">Ratio of region area to its convex hull area.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
