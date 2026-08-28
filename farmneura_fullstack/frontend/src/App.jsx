import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Overview from './pages/Overview';
import PlotMonitoring from './pages/PlotMonitoring';
import Registry from './pages/Registry';
import Auth from './pages/Auth';

export default function App() {
  const [lang, setLang] = useState('🇲🇾 Bahasa Melayu');
  const [selectedFarmId, setSelectedFarmId] = useState('');
  const [selectedPlotId, setSelectedPlotId] = useState('');
  const [authInitialTab, setAuthInitialTab] = useState('login');

  // Auth State initialized from localStorage
  const [user, setUser] = useState(() => {
    try {
      const savedUser = localStorage.getItem('farmneura_user');
      return savedUser ? JSON.parse(savedUser) : null;
    } catch {
      return null;
    }
  });

  // Default view: if user logged in -> '📋 Overview', else -> 'landing'
  const [activeTab, setActiveTab] = useState(() => {
    try {
      const savedUser = localStorage.getItem('farmneura_user');
      return savedUser ? '📋 Overview' : 'landing';
    } catch {
      return 'landing';
    }
  });

  const handleOpenAuth = (mode = 'login') => {
    setAuthInitialTab(mode);
    setActiveTab('🔑 Auth');
  };

  const handleSelectAction = (farmId, plotId) => {
    setSelectedFarmId(farmId);
    setSelectedPlotId(plotId);
    setActiveTab('📷 Plot Monitoring');
  };

  const handleAuthSuccess = (userData) => {
    setUser(userData);
    setActiveTab('📋 Overview');
  };

  const handleLogout = () => {
    localStorage.removeItem('farmneura_user');
    localStorage.removeItem('farmneura_token');
    setUser(null);
    setActiveTab('landing');
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={(tabKey) => {
          if (!user && tabKey !== 'landing' && tabKey !== '🔑 Auth') {
            handleOpenAuth('login');
          } else {
            setActiveTab(tabKey);
          }
        }} 
        lang={lang}
        setLang={setLang}
        user={user}
        onLogout={handleLogout}
      />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-6">
        {/* Unauthenticated Landing Page */}
        {(!user && activeTab === 'landing') && (
          <Landing 
            onOpenAuth={handleOpenAuth} 
            lang={lang} 
          />
        )}

        {/* Authentication Page */}
        {activeTab === '🔑 Auth' && (
          <Auth 
            onAuthSuccess={handleAuthSuccess} 
            lang={lang} 
            initialTab={authInitialTab}
          />
        )}

        {/* Protected Authenticated Pages */}
        {(user && activeTab === '📋 Overview') && (
          <Overview 
            onSelectAction={handleSelectAction} 
            lang={lang} 
            user={user}
          />
        )}

        {(user && activeTab === '📷 Plot Monitoring') && (
          <PlotMonitoring 
            selectedFarmId={selectedFarmId}
            selectedPlotId={selectedPlotId}
            onSelectFarmPlot={(fId, pId) => {
              setSelectedFarmId(fId);
              setSelectedPlotId(pId);
            }}
            lang={lang}
            setLang={setLang}
          />
        )}

        {(user && activeTab === '⚙️ Registry & Management') && (
          <Registry lang={lang} />
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 py-4 text-center text-xs text-gray-500">
        FarmNeura v2 • Mobile Precision Agriculture Assistant
      </footer>
    </div>
  );
}
