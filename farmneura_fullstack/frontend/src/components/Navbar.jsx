import React, { useState } from 'react';
import { Sprout, Globe, Menu, X, User as UserIcon, LogOut, KeyRound } from 'lucide-react';
import { translations } from '../utils/translations';

export default function Navbar({ activeTab, setActiveTab, lang, setLang, user, onLogout }) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const t = translations[lang] || translations["🇲🇾 Bahasa Melayu"];

  const navItems = [
    { key: '📋 Overview', label: t.overviewTab },
    { key: '📷 Plot Monitoring', label: t.monitoringTab },
    { key: '⚙️ Registry & Management', label: t.registryTab }
  ];

  return (
    <header className="bg-gradient-to-r from-farmGreen-800 via-farmGreen-500 to-warmOrange-500 text-white shadow-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Brand Logo */}
        <div 
          onClick={() => setActiveTab(user ? '📋 Overview' : 'landing')}
          className="flex items-center space-x-2 cursor-pointer group"
        >
          <Sprout className="w-7 h-7 text-green-300 flex-shrink-0 group-hover:scale-110 transition" />
          <div>
            <h1 className="text-lg md:text-xl font-bold leading-tight">FarmNeura v2</h1>
            <p className="text-[10px] md:text-xs text-green-100 font-light">Precision Agriculture Assistant</p>
          </div>
        </div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-3">
          {/* Pages displayed ONLY when user is logged in */}
          {user && (
            <nav className="flex space-x-1">
              {navItems.map((item) => (
                <button
                  key={item.key}
                  onClick={() => setActiveTab(item.key)}
                  className={`px-3 py-1.5 rounded-lg text-xs md:text-sm font-medium transition ${
                    activeTab === item.key
                      ? 'bg-white text-farmGreen-800 shadow font-semibold'
                      : 'text-green-50 hover:bg-white/10'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          )}

          {/* Language Selector */}
          <div className="flex items-center space-x-1 bg-white/10 px-2 py-1 rounded-lg border border-white/20">
            <Globe className="w-4 h-4 text-green-200" />
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value)}
              className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer"
            >
              <option value="🇲🇾 Bahasa Melayu" className="text-gray-900">🇲🇾 Melayu</option>
              <option value="🇬🇧 English" className="text-gray-900">🇬🇧 English</option>
            </select>
          </div>

          {/* User Auth Control */}
          {user ? (
            <div className="flex items-center space-x-2 bg-white/15 pl-3 pr-1 py-1 rounded-xl border border-white/25">
              <div className="flex items-center space-x-1.5 text-xs font-bold text-white">
                <UserIcon className="w-3.5 h-3.5 text-green-200" />
                <span className="max-w-[100px] truncate">{user.full_name}</span>
              </div>
              <button
                onClick={onLogout}
                title={t.btnLogout}
                className="p-1 rounded-lg bg-red-500/80 hover:bg-red-600 text-white transition focus:outline-none flex items-center space-x-1 px-2"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span className="text-[11px] font-bold">{t.btnLogout}</span>
              </button>
            </div>
          ) : (
            <button
              onClick={() => setActiveTab('🔑 Auth')}
              className={`px-3.5 py-1.5 rounded-xl text-xs md:text-sm font-extrabold transition flex items-center space-x-1.5 shadow-md ${
                activeTab === '🔑 Auth'
                  ? 'bg-white text-farmGreen-800 shadow-lg'
                  : 'bg-white/20 hover:bg-white/30 text-white border border-white/30'
              }`}
            >
              <KeyRound className="w-4 h-4 text-green-200" />
              <span>{t.authTab}</span>
            </button>
          )}
        </div>

        {/* Mobile Header Controls */}
        <div className="flex md:hidden items-center space-x-2">
          {/* Mobile Language Dropdown */}
          <div className="flex items-center space-x-1 bg-white/15 px-2 py-1 rounded-lg border border-white/25">
            <Globe className="w-3.5 h-3.5 text-green-200" />
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value)}
              className="bg-transparent text-[11px] font-bold text-white focus:outline-none cursor-pointer"
            >
              <option value="🇲🇾 Bahasa Melayu" className="text-gray-900">🇲🇾 MY</option>
              <option value="🇬🇧 English" className="text-gray-900">🇬🇧 EN</option>
            </select>
          </div>

          {/* Mobile Auth Button (When Not Logged In) */}
          {!user && (
            <button
              onClick={() => setActiveTab('🔑 Auth')}
              className="px-2.5 py-1 rounded-lg bg-white/20 hover:bg-white/30 text-white text-[11px] font-extrabold border border-white/30 transition flex items-center space-x-1"
            >
              <KeyRound className="w-3.5 h-3.5" />
              <span>{t.tabLogin}</span>
            </button>
          )}

          {/* Mobile Hamburger Toggle (Only when logged in) */}
          {user && (
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-1.5 rounded-lg bg-white/15 hover:bg-white/25 border border-white/25 transition focus:outline-none"
              aria-label="Toggle Mobile Menu"
            >
              {isMobileMenuOpen ? (
                <X className="w-5 h-5 text-white" />
              ) : (
                <Menu className="w-5 h-5 text-white" />
              )}
            </button>
          )}
        </div>
      </div>

      {/* Mobile Dropdown Menu Drawer (Only when logged in) */}
      {isMobileMenuOpen && user && (
        <div className="md:hidden border-t border-white/20 bg-farmGreen-900/95 backdrop-blur-md px-4 py-3 space-y-1.5 shadow-xl animate-fadeIn">
          {navItems.map((item) => (
            <button
              key={item.key}
              onClick={() => {
                setActiveTab(item.key);
                setIsMobileMenuOpen(false);
              }}
              className={`w-full text-left px-4 py-2.5 rounded-xl text-xs font-bold transition flex items-center justify-between ${
                activeTab === item.key
                  ? 'bg-white text-farmGreen-800 shadow-md font-extrabold'
                  : 'text-white/90 hover:bg-white/10'
              }`}
            >
              <span>{item.label}</span>
              {activeTab === item.key && <span className="text-farmGreen-700 font-extrabold text-sm">✓</span>}
            </button>
          ))}

          {/* Mobile Auth Drawer Item */}
          <div className="pt-2 border-t border-white/15 flex items-center justify-between px-2">
            <div className="flex items-center space-x-2 text-xs font-bold text-white">
              <UserIcon className="w-4 h-4 text-green-300" />
              <span>{user.full_name}</span>
            </div>
            <button
              onClick={() => {
                onLogout();
                setIsMobileMenuOpen(false);
              }}
              className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-bold transition flex items-center space-x-1"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>{t.btnLogout}</span>
            </button>
          </div>
        </div>
      )}
    </header>
  );
}
