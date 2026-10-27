import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { HomeView } from './components/HomeView';
import { AnalysisView } from './components/AnalysisView';
import { ResultsView } from './components/ResultsView';
import { MethodologyView } from './components/MethodologyView';
import { AboutView } from './components/AboutView';
import {
  checkBackendHealth,
  fetchSamplePhantoms,
  runMriAnalysis,
  runSampleAnalysis,
  API_BASE_URL
} from './services/api';
import type { AnalysisParameters, AnalysisResult, SampleItem } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'analysis' | 'results' | 'methodology' | 'about'>('home');
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Check health and load samples on mount
  useEffect(() => {
    const init = async () => {
      const health = await checkBackendHealth();
      setBackendOnline(health.ok);

      const sampleList = await fetchSamplePhantoms();
      setSamples(sampleList);
    };

    init();

    // Periodic health re-check every 30 seconds
    const interval = setInterval(async () => {
      const h = await checkBackendHealth();
      setBackendOnline(h.ok);
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleRunUploadAnalysis = async (file: File, params: AnalysisParameters) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const data = await runMriAnalysis(file, params);
      setResult(data);
      setActiveTab('results');
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred during analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunSampleAnalysis = async (sampleId: string, params: AnalysisParameters) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const data = await runSampleAnalysis(sampleId, params);
      setResult(data);
      setActiveTab('results');
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred during analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendOnline={backendOnline}
        hasResults={result !== null}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'home' && (
          <HomeView
            onStartAnalysis={() => setActiveTab('analysis')}
            onExploreMethodology={() => setActiveTab('methodology')}
          />
        )}

        {activeTab === 'analysis' && (
          <AnalysisView
            onRunUploadAnalysis={handleRunUploadAnalysis}
            onRunSampleAnalysis={handleRunSampleAnalysis}
            samples={samples}
            isLoading={isLoading}
            backendOnline={backendOnline}
            errorMessage={errorMessage}
          />
        )}

        {activeTab === 'results' && result && (
          <ResultsView
            result={result}
            onNewAnalysis={() => setActiveTab('analysis')}
          />
        )}

        {activeTab === 'methodology' && <MethodologyView />}

        {activeTab === 'about' && <AboutView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#080B10] py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-400">Brain Tumor MMR</span>
            <span>•</span>
            <span>Academic Image-Processing Research Prototype</span>
          </div>

          <div className="flex items-center space-x-4">
            <span className="text-[11px] font-mono text-slate-400">
              API: {API_BASE_URL}
            </span>
            <span>•</span>
            <span className="text-slate-400">Not for clinical use</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
