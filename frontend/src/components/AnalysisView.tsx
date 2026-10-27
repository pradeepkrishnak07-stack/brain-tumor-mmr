import React, { useState, useRef } from 'react';
import {
  Upload,
  Sliders,
  Play,
  RotateCcw,
  AlertCircle,
  Server
} from 'lucide-react';
import type { AnalysisParameters, SampleItem } from '../types';
import { API_BASE_URL } from '../services/api';

interface AnalysisViewProps {
  onRunUploadAnalysis: (file: File, params: AnalysisParameters) => Promise<void>;
  onRunSampleAnalysis: (sampleId: string, params: AnalysisParameters) => Promise<void>;
  samples: SampleItem[];
  isLoading: boolean;
  backendOnline: boolean | null;
  errorMessage: string | null;
}

export const AnalysisView: React.FC<AnalysisViewProps> = ({
  onRunUploadAnalysis,
  onRunSampleAnalysis,
  samples,
  isLoading,
  backendOnline,
  errorMessage
}) => {
  // Source mode: 'upload' | 'sample'
  const [sourceMode, setSourceMode] = useState<'upload' | 'sample'>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('sample_1');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Algorithm Parameters with sensible defaults
  const [params, setParams] = useState<AnalysisParameters>({
    enable_denoise: true,
    gaussian_sigma: 1.0,
    enable_clahe: true,
    clahe_clip_limit: 2.5,
    clahe_tile_grid: 8,
    recon_method: 'opening_by_reconstruction',
    selem_shape: 'disk',
    selem_size: 8,
    h_param: 0.15,
    thresh_method: 'otsu',
    manual_threshold: 0.35,
    percentile_val: 92.0,
    min_area: 80,
    opening_radius: 2,
    closing_radius: 3,
    fill_holes: true
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async () => {
    if (sourceMode === 'upload') {
      if (!selectedFile) return;
      await onRunUploadAnalysis(selectedFile, params);
    } else {
      await onRunSampleAnalysis(selectedSampleId, params);
    }
  };

  return (
    <div className="space-y-8 py-4">
      {/* Title & Description */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">MRI Analysis & Parameter Setup</h1>
        <p className="text-slate-400 text-sm mt-1">
          Upload an axial brain MRI slice or choose a verified sample phantom, configure morphological parameters, and execute the MMR pipeline.
        </p>
      </div>

      {/* Backend Offline Notification Banner */}
      {backendOnline === false && (
        <div className="rounded-xl bg-amber-950/40 border border-amber-800/80 p-4 text-amber-200 text-sm flex items-start space-x-3">
          <Server className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-bold text-amber-300">Backend Server Not Detected</div>
            <p className="text-xs text-amber-200/80 leading-relaxed">
              The frontend is currently unable to reach the FastAPI backend at <code className="bg-amber-900/60 px-1 py-0.5 rounded text-amber-100">{API_BASE_URL}</code>.
            </p>
            <div className="text-xs text-amber-300/90 pt-1">
              <strong>Local Development:</strong> Start the backend using <code className="bg-slate-900 text-cyan-300 px-1.5 py-0.5 rounded">cd backend && uvicorn app.main:app --port 8000</code>.
              <br />
              <strong>Deployed Website:</strong> Set the <code className="bg-slate-900 text-cyan-300 px-1.5 py-0.5 rounded">VITE_API_BASE_URL</code> environment variable in Netlify to your deployed backend HTTPS URL (e.g. on Render/Railway).
            </div>
          </div>
        </div>
      )}

      {/* Error Message Banner */}
      {errorMessage && (
        <div className="rounded-xl bg-red-950/40 border border-red-800 p-4 text-red-200 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-red-300">Analysis Error:</span> {errorMessage}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Selection & Preview (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Source Selector Mode */}
          <div className="flex rounded-xl bg-slate-900 p-1 border border-slate-800">
            <button
              onClick={() => setSourceMode('upload')}
              className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all ${
                sourceMode === 'upload'
                  ? 'bg-cyan-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Upload Brain MRI
            </button>
            <button
              onClick={() => setSourceMode('sample')}
              className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all ${
                sourceMode === 'sample'
                  ? 'bg-cyan-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Verified Sample Phantoms
            </button>
          </div>

          {/* Upload Dropzone */}
          {sourceMode === 'upload' ? (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
                selectedFile
                  ? 'border-cyan-500/60 bg-cyan-950/10'
                  : 'border-slate-700 hover:border-slate-500 bg-slate-900/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="flex flex-col items-center justify-center space-y-3">
                <div className="w-12 h-12 rounded-xl bg-slate-800 text-cyan-400 flex items-center justify-center">
                  <Upload className="w-6 h-6" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-white">
                    {selectedFile ? selectedFile.name : 'Click or drag and drop an axial MRI slice'}
                  </div>
                  <p className="text-xs text-slate-400 mt-1">Supports PNG, JPG, JPEG, and TIFF</p>
                </div>
              </div>
            </div>
          ) : (
            /* Sample Selector */
            <div className="space-y-3">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Select Pre-configured MRI Dataset
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {samples.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => setSelectedSampleId(s.id)}
                    className={`p-3.5 rounded-xl border text-left cursor-pointer transition-all ${
                      selectedSampleId === s.id
                        ? 'border-cyan-500 bg-cyan-950/30 ring-1 ring-cyan-500/40'
                        : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
                    }`}
                  >
                    <div className="text-sm font-bold text-white">{s.name}</div>
                    <div className="text-xs text-cyan-400 font-medium mt-0.5">{s.sequence}</div>
                    <p className="text-xs text-slate-400 mt-1.5 line-clamp-2">{s.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Live Preview Box */}
          {sourceMode === 'upload' && previewUrl && (
            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-4 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-white">Image Preview</span>
                <button
                  onClick={handleReset}
                  className="text-red-400 hover:text-red-300 font-medium flex items-center space-x-1"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Clear</span>
                </button>
              </div>
              <div className="flex items-center justify-center bg-black/60 rounded-xl overflow-hidden p-2 max-h-72">
                <img src={previewUrl} alt="Preview" className="max-h-64 object-contain rounded-lg" />
              </div>
            </div>
          )}

          {/* Action Button */}
          <div>
            <button
              onClick={handleSubmit}
              disabled={isLoading || (sourceMode === 'upload' && !selectedFile)}
              className={`w-full py-4 rounded-xl font-bold text-base flex items-center justify-center space-x-2 shadow-lg transition-all ${
                isLoading || (sourceMode === 'upload' && !selectedFile)
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-gradient-to-r from-cyan-500 to-sky-600 hover:from-cyan-400 hover:to-sky-500 text-slate-950 shadow-cyan-500/25 active:scale-[0.99]'
              }`}
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Running MMR Pipeline...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 fill-slate-950" />
                  <span>Execute MMR Analysis</span>
                </>
              )}
            </button>
            <p className="text-center text-xs text-slate-500 mt-2">
              Analyzed purely via Mathematical Morphological Reconstruction. Zero diagnostic fabrication.
            </p>
          </div>
        </div>

        {/* Right Column: Parameters Configuration Panel (5 Cols) */}
        <div className="lg:col-span-5 space-y-5">
          <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-5 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-white text-base flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <span>Algorithm Parameters</span>
              </h3>
              <span className="text-[11px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded">Configurable</span>
            </div>

            {/* 1. Preprocessing Section */}
            <div className="space-y-3">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">1. Preprocessing</span>

              <div className="flex items-center justify-between">
                <label className="text-xs text-slate-300 font-medium">Gaussian Denoising</label>
                <input
                  type="checkbox"
                  checked={params.enable_denoise}
                  onChange={(e) => setParams({ ...params, enable_denoise: e.target.checked })}
                  className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
                />
              </div>

              {params.enable_denoise && (
                <div>
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>Gaussian Sigma (σ)</span>
                    <span className="text-white font-mono">{params.gaussian_sigma.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="3.0"
                    step="0.1"
                    value={params.gaussian_sigma}
                    onChange={(e) => setParams({ ...params, gaussian_sigma: parseFloat(e.target.value) })}
                    className="w-full accent-cyan-500"
                  />
                </div>
              )}

              <div className="flex items-center justify-between pt-1">
                <label className="text-xs text-slate-300 font-medium">CLAHE Contrast Equalization</label>
                <input
                  type="checkbox"
                  checked={params.enable_clahe}
                  onChange={(e) => setParams({ ...params, enable_clahe: e.target.checked })}
                  className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
                />
              </div>

              {params.enable_clahe && (
                <div>
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>CLAHE Clip Limit</span>
                    <span className="text-white font-mono">{params.clahe_clip_limit.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="5.0"
                    step="0.25"
                    value={params.clahe_clip_limit}
                    onChange={(e) => setParams({ ...params, clahe_clip_limit: parseFloat(e.target.value) })}
                    className="w-full accent-cyan-500"
                  />
                </div>
              )}
            </div>

            {/* 2. Morphological Reconstruction Section */}
            <div className="space-y-3 pt-3 border-t border-slate-800">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">2. MMR Strategy</span>

              <div>
                <label className="text-xs text-slate-300 font-medium block mb-1">Reconstruction Method</label>
                <select
                  value={params.recon_method}
                  onChange={(e) => setParams({ ...params, recon_method: e.target.value as any })}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="opening_by_reconstruction">Opening by Reconstruction (WTH)</option>
                  <option value="h_dome">H-Dome Reconstruction (Regional Peaks)</option>
                  <option value="closing_by_reconstruction">Closing by Reconstruction (BTH)</option>
                  <option value="regional_maxima_marker">Regional Maxima Marker</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Structuring Element</label>
                  <select
                    value={params.selem_shape}
                    onChange={(e) => setParams({ ...params, selem_shape: e.target.value as any })}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none"
                  >
                    <option value="disk">Disk</option>
                    <option value="square">Square</option>
                    <option value="diamond">Diamond</option>
                    <option value="octagon">Octagon</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">SE Radius (px)</label>
                  <input
                    type="number"
                    min="1"
                    max="30"
                    value={params.selem_size}
                    onChange={(e) => setParams({ ...params, selem_size: parseInt(e.target.value) || 1 })}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono"
                  />
                </div>
              </div>

              {params.recon_method === 'h_dome' && (
                <div>
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>H-Dome Prominence Cutoff (h)</span>
                    <span className="text-white font-mono">{params.h_param.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.02"
                    max="0.50"
                    step="0.01"
                    value={params.h_param}
                    onChange={(e) => setParams({ ...params, h_param: parseFloat(e.target.value) })}
                    className="w-full accent-cyan-500"
                  />
                </div>
              )}
            </div>

            {/* 3. Candidate Segmentation Section */}
            <div className="space-y-3 pt-3 border-t border-slate-800">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">3. Candidate Segmentation</span>

              <div>
                <label className="text-xs text-slate-300 font-medium block mb-1">Threshold Strategy</label>
                <select
                  value={params.thresh_method}
                  onChange={(e) => setParams({ ...params, thresh_method: e.target.value as any })}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="otsu">Otsu (Minimum Variance)</option>
                  <option value="triangle">Triangle (Adaptive)</option>
                  <option value="percentile">Percentile Cutoff</option>
                  <option value="manual">Manual Cutoff Slider</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Minimum Component Area (pixels)</span>
                  <span className="text-white font-mono">{params.min_area} px</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="500"
                  step="10"
                  value={params.min_area}
                  onChange={(e) => setParams({ ...params, min_area: parseInt(e.target.value) })}
                  className="w-full accent-cyan-500"
                />
              </div>

              <div className="flex items-center justify-between pt-1">
                <label className="text-xs text-slate-300 font-medium">Fill Internal Micro-Cavities</label>
                <input
                  type="checkbox"
                  checked={params.fill_holes}
                  onChange={(e) => setParams({ ...params, fill_holes: e.target.checked })}
                  className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
