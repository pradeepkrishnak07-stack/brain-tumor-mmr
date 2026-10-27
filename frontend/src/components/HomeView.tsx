import React from 'react';
import {
  Brain,
  Sliders,
  Layers,
  Sparkles,
  ArrowRight,
  ShieldAlert,
  Microscope,
  Cpu
} from 'lucide-react';

interface HomeViewProps {
  onStartAnalysis: () => void;
  onExploreMethodology: () => void;
}

export const HomeView: React.FC<HomeViewProps> = ({
  onStartAnalysis,
  onExploreMethodology
}) => {
  return (
    <div className="space-y-12 py-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-900 via-slate-900/90 to-cyan-950/40 border border-slate-800 p-8 sm:p-12 shadow-2xl">
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-3xl space-y-6 relative z-10">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300 text-xs font-semibold tracking-wide">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Academic Computer Vision & Medical Image Processing</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight leading-tight">
            Brain Tumor Identification Using{' '}
            <span className="bg-gradient-to-r from-cyan-400 via-sky-300 to-teal-300 bg-clip-text text-transparent">
              Mathematical Morphological Reconstruction
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            A transparent, mathematically authentic image-processing research platform that applies
            geodesic morphological transformations under mask constraints to isolate candidate abnormal
            regions in Brain MRI slices without boundary distortion or black-box opacity.
          </p>

          <div className="flex flex-wrap gap-4 pt-2">
            <button
              onClick={onStartAnalysis}
              className="inline-flex items-center space-x-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-sky-600 hover:from-cyan-400 hover:to-sky-500 text-slate-950 font-bold text-sm sm:text-base shadow-lg shadow-cyan-500/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <Brain className="w-5 h-5 text-slate-950" />
              <span>Start MRI Analysis</span>
              <ArrowRight className="w-4 h-4 text-slate-950" />
            </button>

            <button
              onClick={onExploreMethodology}
              className="inline-flex items-center space-x-2 px-5 py-3.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold text-sm sm:text-base transition-colors"
            >
              <span>Explore Methodology</span>
            </button>
          </div>
        </div>
      </div>

      {/* Mandatory Medical Disclaimer Banner */}
      <div className="rounded-2xl bg-red-950/20 border-l-4 border-red-500 p-5 text-red-200/90 text-sm flex items-start space-x-3.5">
        <ShieldAlert className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-red-300">ACADEMIC & NON-DIAGNOSTIC NOTICE:</span> This application is an
          educational prototype developed for computer vision and medical image processing research. It is
          <strong> NOT </strong> a clinically certified medical diagnostic device and must <strong> NEVER </strong> be used
          for medical diagnosis, patient screening, or clinical management. Extracted regions represent mathematical
          <strong> candidate abnormal regions </strong> based on intensity contrast and morphology.
        </div>
      </div>

      {/* Three Core Feature Cards */}
      <div>
        <div className="text-center max-w-2xl mx-auto mb-8">
          <h2 className="text-2xl font-bold text-white tracking-tight">Core Image-Processing Pipeline</h2>
          <p className="text-slate-400 text-sm mt-1">
            Engineered with scikit-image, OpenCV, and NumPy using genuine mathematical morphology.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1 */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4 hover:border-slate-700 transition-colors">
            <div className="w-12 h-12 rounded-xl bg-sky-950 text-sky-400 flex items-center justify-center border border-sky-800/60">
              <Sliders className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">1. Medical Preprocessing</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Standardizes raw MRI slices with 8-bit grayscale mapping, intensity normalization, configurable Gaussian
              noise suppression, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
            </p>
            <ul className="text-xs text-slate-400 space-y-1.5 pt-2 border-t border-slate-800/60">
              <li>• Denoising for Rician & Gaussian MRI noise</li>
              <li>• CLAHE local contrast enhancement</li>
              <li>• Float32 normalized intensity range [0, 1]</li>
            </ul>
          </div>

          {/* Card 2 */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4 hover:border-cyan-800/60 transition-colors">
            <div className="w-12 h-12 rounded-xl bg-cyan-950 text-cyan-400 flex items-center justify-center border border-cyan-800/60">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">2. Morphological Reconstruction</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Applies genuine geodesic dilation under mask constraints: <code className="text-cyan-300">δ_f(g) = (g ⊕ B) ∧ f</code>.
              Preserves exact object contours without the rounding or boundary degradation caused by standard opening.
            </p>
            <ul className="text-xs text-slate-400 space-y-1.5 pt-2 border-t border-slate-800/60">
              <li>• Opening by Reconstruction (OBR)</li>
              <li>• White Top-Hat residual extraction</li>
              <li>• Multi-shape SEs (disk, square, diamond)</li>
            </ul>
          </div>

          {/* Card 3 */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4 hover:border-slate-700 transition-colors">
            <div className="w-12 h-12 rounded-xl bg-teal-950 text-teal-400 flex items-center justify-center border border-teal-800/60">
              <Microscope className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">3. Candidate Region Analysis</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Extracts candidate abnormal components through documented thresholding (Otsu, Triangle, Percentile),
              morphological opening/closing, and connected-component size filtering. Computes geometric morphometrics.
            </p>
            <ul className="text-xs text-slate-400 space-y-1.5 pt-2 border-t border-slate-800/60">
              <li>• Pixel Area, Perimeter, and Centroid (X, Y)</li>
              <li>• Circularity, Solidity, and Aspect Ratio</li>
              <li>• Honest handling of normal control MRIs (0 regions)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* How It Works Step-by-Step */}
      <div className="rounded-3xl bg-slate-900/50 border border-slate-800 p-8">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <span>Algorithm Execution Workflow</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
            <div className="text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">Step 1</div>
            <div className="font-semibold text-white text-sm">Upload & Preprocess</div>
            <p className="text-xs text-slate-400 mt-1">Grayscale conversion, Gaussian noise smoothing, and CLAHE contrast enhancement.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
            <div className="text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">Step 2</div>
            <div className="font-semibold text-white text-sm">Marker & Mask Setup</div>
            <p className="text-xs text-slate-400 mt-1">Erosion seed g = f ⊖ B constrained strictly by original mask f (g ≤ f).</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
            <div className="text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">Step 3</div>
            <div className="font-semibold text-white text-sm">Geodesic MMR Engine</div>
            <p className="text-xs text-slate-400 mt-1">Iterative geodesic dilation until convergence; White Top-Hat residual extraction.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
            <div className="text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">Step 4</div>
            <div className="font-semibold text-white text-sm">Segment & Measure</div>
            <p className="text-xs text-slate-400 mt-1">Otsu thresholding, connected-component size filter, and morphometric feature calculation.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
