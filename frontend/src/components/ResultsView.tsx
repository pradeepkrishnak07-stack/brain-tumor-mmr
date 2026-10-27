import React, { useState } from 'react';
import {
  Download,
  CheckCircle2,
  Maximize2,
  Table as TableIcon,
  Activity
} from 'lucide-react';
import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
  Area,
  ComposedChart
} from 'recharts';
import type { AnalysisResult } from '../types';

interface ResultsViewProps {
  result: AnalysisResult;
  onNewAnalysis: () => void;
}

export const ResultsView: React.FC<ResultsViewProps> = ({
  result,
  onNewAnalysis
}) => {
  const [activeImageKey, setActiveImageKey] = useState<keyof AnalysisResult['images']>('overlay');
  const [modalImage, setModalImage] = useState<{ title: string; src: string } | null>(null);

  const imagesConfig: { key: keyof AnalysisResult['images']; label: string; desc: string }[] = [
    { key: 'original', label: '1. Original MRI', desc: 'Raw input slice' },
    { key: 'preprocessed', label: '2. Preprocessed', desc: 'CLAHE & Gaussian filter' },
    { key: 'marker', label: '3. MMR Marker (Seed)', desc: 'Erosion g = f ⊖ B' },
    { key: 'mask', label: '4. MMR Mask (Constraint)', desc: 'Upper bound constraint' },
    { key: 'reconstructed', label: '5. MMR Reconstruction', desc: 'Stationary geodesic limit' },
    { key: 'residual', label: '6. Residual Map (WTH)', desc: 'Mask - Reconstructed' },
    { key: 'candidate_mask', label: '7. Candidate Mask', desc: 'Thresholded & cleaned' },
    { key: 'overlay', label: '8. Annotated Overlay', desc: 'Contours (cyan) & BBox' }
  ];

  // Prepare chart data for Recharts from 1D horizontal intensity profile
  const chartData = result.intensity_profile.x_coords.map((x, idx) => ({
    x,
    Mask: result.intensity_profile.mask_profile[idx] ?? 0,
    Reconstructed: result.intensity_profile.reconstructed_profile[idx] ?? 0,
    Residual: result.intensity_profile.residual_profile[idx] ?? 0
  }));

  // Download helper for Base64 Data URIs
  const downloadDataUri = (dataUri: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUri;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Download helper for Text / CSV / Markdown
  const downloadTextFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const downloadCsv = () => {
    if (result.regions.length === 0) return;
    const headers = Object.keys(result.regions[0]).join(',');
    const rows = result.regions.map(r => Object.values(r).join(',')).join('\n');
    downloadTextFile(`${headers}\n${rows}`, `${result.filename}_measurements.csv`, 'text/csv');
  };

  const isZeroRegions = result.results_summary.candidate_regions_count === 0;

  return (
    <div className="space-y-8 py-4">
      {/* Header and Quick Stats */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Analysis Results</h1>
            <span className="text-xs bg-slate-800 text-cyan-400 px-2.5 py-0.5 rounded-full font-mono border border-slate-700">
              {result.filename}
            </span>
          </div>
          <p className="text-slate-400 text-xs sm:text-sm mt-1">
            Mathematical Morphological Reconstruction completed in{' '}
            <strong className="text-emerald-400 font-mono">{result.processing_time_ms} ms</strong>.
          </p>
        </div>

        <button
          onClick={onNewAnalysis}
          className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-semibold border border-slate-700 transition-colors"
        >
          ← New Analysis
        </button>
      </div>

      {/* 4 Summary Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Slice Resolution</div>
          <div className="text-2xl font-bold text-cyan-400 mt-1 font-mono">
            {result.dimensions.width} × {result.dimensions.height}
          </div>
          <div className="text-xs text-slate-500 mt-0.5">2D Axial Plane</div>
        </div>

        <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Computation Latency</div>
          <div className="text-2xl font-bold text-emerald-400 mt-1 font-mono">
            {result.processing_time_ms} ms
          </div>
          <div className="text-xs text-slate-500 mt-0.5">Priority-Queue Geodesic Algorithm</div>
        </div>

        <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Candidate Regions</div>
          <div className={`text-2xl font-bold mt-1 font-mono ${isZeroRegions ? 'text-slate-400' : 'text-red-400'}`}>
            {result.results_summary.candidate_regions_count}
          </div>
          <div className="text-xs text-slate-500 mt-0.5">
            {isZeroRegions ? 'Normal / Negative Finding' : 'Filtered Component(s)'}
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Total Candidate Area</div>
          <div className="text-2xl font-bold text-amber-400 mt-1 font-mono">
            {result.results_summary.total_candidate_area_px} px
          </div>
          <div className="text-xs text-slate-500 mt-0.5">
            {result.results_summary.area_percentage}% of total slice area
          </div>
        </div>
      </div>

      {/* Primary Visual Comparison & Stage Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Large Stage Display (8 cols) */}
        <div className="lg:col-span-8 rounded-3xl bg-slate-900 border border-slate-800 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">Active Stage View</span>
              <h3 className="text-lg font-bold text-white">
                {imagesConfig.find(i => i.key === activeImageKey)?.label}
              </h3>
            </div>

            <button
              onClick={() => setModalImage({
                title: imagesConfig.find(i => i.key === activeImageKey)?.label || '',
                src: result.images[activeImageKey]
              })}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-medium flex items-center space-x-1.5 border border-slate-700"
            >
              <Maximize2 className="w-3.5 h-3.5" />
              <span>Full View</span>
            </button>
          </div>

          <div className="flex items-center justify-center bg-black/70 rounded-2xl p-4 min-h-[380px] max-h-[500px] overflow-hidden">
            <img
              src={result.images[activeImageKey]}
              alt={activeImageKey}
              className="max-h-[460px] max-w-full object-contain rounded-xl shadow-2xl"
            />
          </div>

          {/* Quick Stage Switcher Thumbnails */}
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 pt-2">
            {imagesConfig.map((img) => (
              <button
                key={img.key}
                onClick={() => setActiveImageKey(img.key)}
                className={`flex flex-col items-center p-1.5 rounded-xl border transition-all text-center ${
                  activeImageKey === img.key
                    ? 'border-cyan-500 bg-cyan-950/40 ring-1 ring-cyan-500'
                    : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
                }`}
              >
                <img
                  src={result.images[img.key]}
                  alt={img.label}
                  className="w-12 h-12 object-cover rounded-lg bg-black/50"
                />
                <span className="text-[10px] text-slate-300 font-medium truncate w-full mt-1">
                  {img.label.split('.')[0]}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Side-by-Side Reference & Export Links (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Reference: Original Input vs Final Overlay */}
          <div className="rounded-3xl bg-slate-900 border border-slate-800 p-5 space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Side-by-Side Comparison</h4>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1 text-center">
                <span className="text-[11px] text-slate-400">Original MRI</span>
                <img
                  src={result.images.original}
                  alt="Original"
                  className="w-full h-32 object-contain bg-black/60 rounded-xl p-1 border border-slate-800"
                />
              </div>
              <div className="space-y-1 text-center">
                <span className="text-[11px] text-cyan-400 font-semibold">MMR Overlay</span>
                <img
                  src={result.images.overlay}
                  alt="Overlay"
                  className="w-full h-32 object-contain bg-black/60 rounded-xl p-1 border border-cyan-800/60"
                />
              </div>
            </div>
          </div>

          {/* Download Center Card */}
          <div className="rounded-3xl bg-slate-900 border border-slate-800 p-5 space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
              <Download className="w-4 h-4 text-cyan-400" />
              <span>Download Outputs</span>
            </h4>

            <div className="space-y-2">
              <button
                onClick={() => downloadDataUri(result.images.overlay, `${result.filename}_mmr_overlay.png`)}
                className="w-full py-2.5 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-white flex items-center justify-between border border-slate-700 transition-colors"
              >
                <span>Annotated Overlay (PNG)</span>
                <Download className="w-3.5 h-3.5 text-cyan-400" />
              </button>

              <button
                onClick={() => downloadDataUri(result.images.candidate_mask, `${result.filename}_candidate_mask.png`)}
                className="w-full py-2.5 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-white flex items-center justify-between border border-slate-700 transition-colors"
              >
                <span>Binary Mask (PNG)</span>
                <Download className="w-3.5 h-3.5 text-cyan-400" />
              </button>

              <button
                onClick={downloadCsv}
                disabled={isZeroRegions}
                className={`w-full py-2.5 px-3 rounded-xl text-xs font-semibold flex items-center justify-between border transition-colors ${
                  isZeroRegions
                    ? 'bg-slate-900 text-slate-600 border-slate-800 cursor-not-allowed'
                    : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-700'
                }`}
              >
                <span>Morphometric Features (CSV)</span>
                <Download className="w-3.5 h-3.5 text-cyan-400" />
              </button>

              <button
                onClick={() => downloadTextFile(result.report_markdown, `${result.filename}_analysis_report.md`, 'text/markdown')}
                className="w-full py-2.5 px-3 rounded-xl bg-cyan-950/60 hover:bg-cyan-900/60 text-xs font-semibold text-cyan-200 flex items-center justify-between border border-cyan-800/80 transition-colors"
              >
                <span>Full Analysis Report (Markdown)</span>
                <Download className="w-3.5 h-3.5 text-cyan-400" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Cross-Sectional 1D Intensity Profile (Recharts) */}
      <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Cross-Sectional Intensity Profile (Row y = {result.intensity_profile.slice_y})</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Visualizes how Morphological Reconstruction preserves tissue boundaries while leveling narrow peaks smaller than the structuring element.
            </p>
          </div>
          <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/80 px-2.5 py-1 rounded-full border border-cyan-800/60">
            Horizontal Scanline
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="x" stroke="#64748B" fontSize={11} />
              <YAxis domain={[0, 1]} stroke="#64748B" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '0.75rem', fontSize: '12px' }}
                labelStyle={{ color: '#94A3B8' }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
              <Area type="monotone" dataKey="Residual" fill="#EF4444" stroke="#EF4444" fillOpacity={0.25} name="Residual (Candidate Peak)" />
              <Line type="monotone" dataKey="Mask" stroke="#00D2FF" strokeWidth={1.8} dot={false} name="Mask Profile (Preprocessed)" />
              <Line type="monotone" dataKey="Reconstructed" stroke="#F59E0B" strokeWidth={1.8} strokeDasharray="4 4" dot={false} name="MMR Reconstructed" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Morphometric Feature Table */}
      <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <TableIcon className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white">Quantitative Morphometric Features</h3>
          </div>
          <span className="text-xs text-slate-400">
            {result.regions.length} Connected Component{result.regions.length === 1 ? '' : 's'}
          </span>
        </div>

        {isZeroRegions ? (
          <div className="py-8 text-center space-y-2">
            <CheckCircle2 className="w-10 h-10 text-slate-500 mx-auto" />
            <div className="text-sm font-semibold text-slate-300">
              No candidate abnormal regions detected under the selected parameters.
            </div>
            <p className="text-xs text-slate-500 max-w-md mx-auto">
              This is consistent with healthy control tissue or sub-threshold contrast gradients.
              The algorithm does not invent or fabricate tumor coordinates.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-800/60 text-slate-300 font-semibold uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="py-2.5 px-3">ID</th>
                  <th className="py-2.5 px-3">Area (px)</th>
                  <th className="py-2.5 px-3">Centroid (X, Y)</th>
                  <th className="py-2.5 px-3">BBox (YMin, XMin, YMax, XMax)</th>
                  <th className="py-2.5 px-3">Perimeter</th>
                  <th className="py-2.5 px-3">Aspect Ratio</th>
                  <th className="py-2.5 px-3">Circularity</th>
                  <th className="py-2.5 px-3">Solidity</th>
                  <th className="py-2.5 px-3">Extent</th>
                  <th className="py-2.5 px-3">Mean Intensity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300 font-mono">
                {result.regions.map((r) => (
                  <tr key={r.Region_ID} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2.5 px-3 font-bold text-cyan-400">#{r.Region_ID}</td>
                    <td className="py-2.5 px-3 font-bold text-white">{r.Area_Pixels}</td>
                    <td className="py-2.5 px-3">({r.Centroid_X}, {r.Centroid_Y})</td>
                    <td className="py-2.5 px-3">[{r.BBox_YMin}, {r.BBox_XMin}, {r.BBox_YMax}, {r.BBox_XMax}]</td>
                    <td className="py-2.5 px-3">{r.Perimeter_Pixels}</td>
                    <td className="py-2.5 px-3">{r.Aspect_Ratio}</td>
                    <td className="py-2.5 px-3 text-amber-300">{r.Circularity}</td>
                    <td className="py-2.5 px-3">{r.Solidity}</td>
                    <td className="py-2.5 px-3">{r.Extent}</td>
                    <td className="py-2.5 px-3 text-emerald-400">{r.Mean_Intensity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Full-View Image Modal */}
      {modalImage && (
        <div
          onClick={() => setModalImage(null)}
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="max-w-4xl w-full bg-slate-900 border border-slate-800 rounded-3xl p-5 space-y-3"
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h4 className="font-bold text-white text-sm">{modalImage.title}</h4>
              <button
                onClick={() => setModalImage(null)}
                className="text-slate-400 hover:text-white text-xs px-2 py-1 rounded bg-slate-800"
              >
                Close (ESC)
              </button>
            </div>
            <div className="flex items-center justify-center bg-black/80 rounded-2xl p-2 max-h-[75vh] overflow-hidden">
              <img src={modalImage.src} alt={modalImage.title} className="max-h-[70vh] object-contain rounded-xl" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
