import React from 'react';
// 1. Import the logo from your assets folder
import tabiyaLogo from '../assets/tabiya.png'; 


export const Layout = ({ children, currentView, setView }) => {
  return (
    <div className="min-h-screen bg-tabiya-mint font-sans">
      <nav className="bg-white border-b border-slate-100 p-4 sticky top-0 z-50 shadow-sm">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          
          {/* LOGO & WORDMARK GROUP */}
          <div className="flex items-center gap-2 group cursor-pointer">
            <img 
              src={tabiyaLogo} 
              alt="Tabiya Logo" 
              className="h-9 w-auto object-contain"
            />
            <span className="text-2xl font-bold tracking-tighter text-tabiya-navy lowercase">
              tabiya
            </span>
          </div>

          {/* PERSONA SWITCHER */}
          <div className="flex bg-slate-100 p-1 rounded-full border border-slate-200 shadow-inner">
            {['jobseeker', 'employer', 'policy', 'config'].map((mode) => (
              <button
                key={mode}
                onClick={() => setView(mode)}
                className={`px-5 py-2 rounded-full text-xs font-black capitalize transition-all duration-300 ${
                  currentView === mode 
                    ? 'bg-tabiya-navy text-white shadow-lg' 
                    : 'text-slate-500 hover:text-tabiya-navy'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>

          {/* ACTION BUTTON */}
          <button className="hidden md:block bg-tabiya-green text-tabiya-navy px-6 py-2 rounded-full font-bold text-sm hover:shadow-md transition-all active:scale-95">
            Get Started
          </button>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto py-12 px-4">
        {children}
      </main>
    </div>
  );
};