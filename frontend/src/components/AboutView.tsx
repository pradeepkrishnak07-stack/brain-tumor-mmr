import React from 'react';
import {
  Info,
  Code2,
  Bookmark,
  Target,
  AlertTriangle,
  Lightbulb
} from 'lucide-react';

export const AboutView: React.FC = () => {
  return (
    <div className="space-y-10 py-4 max-w-5xl mx-auto">
      {/* Title */}
      <div>
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300 text-xs font-semibold tracking-wide mb-3">
          <Info className="w-3.5 h-3.5" />
          <span>Research Background & Specifications</span>
        </div>
        <h1 className="text-3xl font-bold text-white tracking-tight">About the Project</h1>
        <p className="text-slate-400 text-sm mt-1">
          Academic engineering project on non-linear morphological image processing for brain MRI analysis.
        </p>
      </div>

      {/* Problem Statement & Solution */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
          <div className="flex items-center space-x-2 text-amber-400 text-xs font-bold uppercase tracking-wider">
            <Target className="w-4 h-4" />
            <span>Problem Statement</span>
          </div>
          <h3 className="text-base font-bold text-white">Boundary Degradation & Black-Box Opacity</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Conventional linear filters (e.g. Gaussian blur) soften fine lesion boundaries, while standard morphological
            opening severely rounds corners and alters lesion geometry. Conversely, deep-learning models act as opaque black
            boxes that lack geometric interpretability and require massive training sets.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900 border border-cyan-800/60 space-y-3">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider">
            <Lightbulb className="w-4 h-4" />
            <span>Proposed Solution</span>
          </div>
          <h3 className="text-base font-bold text-white">Geodesic Morphological Reconstruction</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Mathematical Morphological Reconstruction (MMR) solves this by repeatedly applying geodesic dilations under
            the exact constraint of the input mask. Shapes larger than the structuring element are reconstructed with zero
            boundary distortion, allowing clean White Top-Hat residual extraction of candidate abnormal regions.
          </p>
        </div>
      </div>

      {/* Technologies Used Grid */}
      <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 sm:p-8 space-y-6">
        <h2 className="text-lg font-bold text-white flex items-center space-x-2">
          <Code2 className="w-5 h-5 text-cyan-400" />
          <span>Technology Stack</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-800 space-y-2">
            <div className="font-bold text-cyan-400 uppercase tracking-wider text-[11px]">Frontend Web Layer</div>
            <ul className="space-y-1.5 text-slate-300">
              <li>• React 18+ with TypeScript</li>
              <li>• Vite build system</li>
              <li>• Tailwind CSS responsive UI</li>
              <li>• Lucide React icons</li>
              <li>• Recharts interactive data viz</li>
              <li>• Netlify global CDN deployment</li>
            </ul>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-800 space-y-2">
            <div className="font-bold text-emerald-400 uppercase tracking-wider text-[11px]">Backend API Layer</div>
            <ul className="space-y-1.5 text-slate-300">
              <li>• Python 3.11+</li>
              <li>• FastAPI asynchronous web framework</li>
              <li>• Uvicorn ASGI production server</li>
              <li>• Multi-origin CORS support</li>
              <li>• Base64 Data URI streaming</li>
              <li>• Docker & Procfile cloud deployable</li>
            </ul>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-800 space-y-2">
            <div className="font-bold text-amber-400 uppercase tracking-wider text-[11px]">Image-Processing Core</div>
            <ul className="space-y-1.5 text-slate-300">
              <li>• scikit-image geodesic reconstruction</li>
              <li>• OpenCV (cv2) CLAHE & filters</li>
              <li>• NumPy numerical matrix algebra</li>
              <li>• Pillow cross-format decoding</li>
              <li>• Pandas tabular feature extraction</li>
              <li>• Pytest automated test suite (47 tests)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Limitations & Ethical Boundaries */}
      <div className="rounded-3xl bg-slate-900 border border-red-900/40 p-6 sm:p-8 space-y-4">
        <div className="flex items-center space-x-2 text-red-400 font-bold text-base">
          <AlertTriangle className="w-5 h-5" />
          <span>Documented Methodological & Clinical Limitations</span>
        </div>

        <div className="space-y-2.5 text-xs text-slate-300 leading-relaxed">
          <p>
            <strong>1. Intensity Non-Specificity:</strong> Morphological reconstruction operates on intensity gradients.
            In MRI, hyperintensity (brightness) is sequence-dependent (T1, T2, FLAIR) and is not unique to neoplasms.
            Normal tissues such as choroid plexus, fat, vasculature, or non-tumoral pathologies (ischemia, infection)
            can produce candidate peaks.
          </p>
          <p>
            <strong>2. 2D Slice Processing:</strong> This prototype analyzes single 2D axial MRI slices and does not
            compute 3D inter-slice volumetric connectivity or partial volume averaging across adjacent slices.
          </p>
          <p>
            <strong>3. Lack of Histopathological Grading:</strong> Morphology extracts geometric attributes (area,
            perimeter, circularity); it cannot determine WHO tumor grade (I–IV) or molecular alterations (IDH mutation, MGMT).
          </p>
          <p>
            <strong>4. Academic Research Only:</strong> This software is an educational prototype and does not have clinical
            regulatory clearance (FDA 510(k), CE mark). All output regions are strictly labeled <em>candidate abnormal regions</em>.
          </p>
        </div>
      </div>

      {/* Academic References */}
      <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 sm:p-8 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center space-x-2">
          <Bookmark className="w-5 h-5 text-cyan-400" />
          <span>Academic References & Bibliography</span>
        </h2>

        <div className="space-y-3 text-xs text-slate-300">
          <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-800">
            <strong>Vincent, L. (1993).</strong> <em>Morphological grayscale reconstruction in image analysis: Applications and efficient algorithms.</em> IEEE Transactions on Image Processing, 2(2), 176–201.
          </div>
          <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-800">
            <strong>Soille, P. (2003).</strong> <em>Morphological Image Analysis: Principles and Applications (2nd ed.).</em> Springer-Verlag, Berlin.
          </div>
          <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-800">
            <strong>Serra, J. (1982).</strong> <em>Image Analysis and Mathematical Morphology.</em> Academic Press, London.
          </div>
          <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-800">
            <strong>Gonzalez, R. C., & Woods, R. E. (2018).</strong> <em>Digital Image Processing (4th Edition).</em> Pearson Education.
          </div>
        </div>
      </div>
    </div>
  );
};
