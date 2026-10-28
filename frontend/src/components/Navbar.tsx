import React from 'react';
import { Brain, AlertCircle } from 'lucide-react';

interface NavbarProps {
  activeTab: 'home' | 'analysis' | 'results' | 'methodology' | 'about';
  setActiveTab: (tab: 'home' | 'analysis' | 'results' | 'methodology' | 'about') => void;
  backendOnline: boolean | null;
  serviceName?: string;
  hasResults: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  backendOnline,
  serviceName,
  hasResults
}) => {
  return (
    <header className="sticky top-0 z-50 bg-[#0B0F17]/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo / Title */}
          <div
            onClick={() => setActiveTab('home')}
            className="flex items-center space-x-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-sky-400 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-white tracking-tight">Brain Tumor MMR</span>
                <span className="text-[10px] font-semibold uppercase tracking-wider bg-cyan-950 text-cyan-400 px-2 py-0.5 rounded-full border border-cyan-800/60">
                  Academic Prototype
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">Mathematical Morphological Reconstruction</p>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => setActiveTab('home')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'home'
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              Home
            </button>
            <button
              onClick={() => setActiveTab('analysis')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'analysis'
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              Analysis
            </button>
            <button
              onClick={() => setActiveTab('results')}
              disabled={!hasResults}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1 ${
                !hasResults
                  ? 'text-slate-600 cursor-not-allowed'
                  : activeTab === 'results'
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <span>Results</span>
              {hasResults && (
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse ml-1" />
              )}
            </button>
            <button
              onClick={() => setActiveTab('methodology')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'methodology'
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              Methodology
            </button>
            <button
              onClick={() => setActiveTab('about')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'about'
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              About
            </button>
          </nav>

          {/* Backend Health Badge */}
          <div className="hidden lg:flex items-center space-x-2 pl-3 border-l border-slate-800">
            {backendOnline === true && (
              <div className="flex items-center space-x-1.5 text-xs text-emerald-400 bg-emerald-950/60 px-2.5 py-1 rounded-full border border-emerald-800/60">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="font-medium">{serviceName || 'Engine Ready'}</span>
              </div>
            )}
            {backendOnline === false && (
              <div className="flex items-center space-x-1.5 text-xs text-amber-400 bg-amber-950/60 px-2.5 py-1 rounded-full border border-amber-800/60" title="Backend offline or starting up">
                <AlertCircle className="w-3.5 h-3.5" />
                <span className="font-medium">Engine Offline</span>
              </div>
            )}
            {backendOnline === null && (
              <div className="text-xs text-slate-400 animate-pulse">Initializing...</div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
