import React from 'react';
import { Sprout, Bot, Wifi, LayoutGrid, Zap, ShieldCheck, UserPlus, LogIn } from 'lucide-react';
import { translations } from '../utils/translations';

export default function Landing({ onOpenAuth, lang }) {
  const t = translations[lang] || translations["Bahasa Melayu"];

  return (
    <div className="space-y-10 py-2">
      {/* Hero Section Card */}
      <div className="relative bg-gradient-to-r from-emerald-900 via-emerald-800 to-amber-700 rounded-3xl p-6 md:p-10 text-white shadow-2xl overflow-hidden border border-emerald-700">
        {/* Background Glow Overlay */}
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-emerald-400/20 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -left-24 w-80 h-80 bg-amber-400/20 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl mx-auto text-center space-y-6">
          {/* Tagline Badge */}
          <div className="inline-flex items-center space-x-2 bg-black/25 backdrop-blur-md px-4 py-1.5 rounded-full border border-white/20 shadow-xs">
            <Sprout className="w-4 h-4 text-emerald-300 animate-pulse" />
            <span className="text-xs font-extrabold uppercase tracking-wider text-emerald-100">{t.landingTagline}</span>
          </div>

          {/* Main Hero Headline */}
          <h1 className="text-2xl md:text-4xl lg:text-5xl font-black leading-tight tracking-tight text-white drop-shadow-lg">
            {t.landingHeroTitle}
          </h1>

          <p className="text-xs md:text-sm text-emerald-100 font-medium leading-relaxed max-w-2xl mx-auto drop-shadow-xs">
            {t.landingHeroSubtitle}
          </p>

          {/* Hero Action Buttons */}
          <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
            <button
              onClick={() => onOpenAuth('signup')}
              className="w-full sm:w-auto px-7 py-3.5 rounded-2xl bg-white text-emerald-950 font-black text-xs md:text-sm shadow-2xl hover:bg-emerald-50 transform hover:-translate-y-0.5 transition flex items-center justify-center space-x-2"
            >
              <UserPlus className="w-4 h-4 text-emerald-700" />
              <span>{t.btnGetStarted}</span>
            </button>

            <button
              onClick={() => onOpenAuth('login')}
              className="w-full sm:w-auto px-7 py-3.5 rounded-2xl bg-emerald-950/60 hover:bg-emerald-950/80 text-white font-extrabold text-xs md:text-sm border border-emerald-400/30 backdrop-blur-md shadow-xl transition flex items-center justify-center space-x-2"
            >
              <LogIn className="w-4 h-4 text-emerald-300" />
              <span>{t.btnLoginHero}</span>
            </button>
          </div>

          {/* Privacy & Safety Guarantee */}
          <div className="pt-3 flex items-center justify-center space-x-2 text-[11px] text-emerald-200 font-medium">
            <ShieldCheck className="w-4 h-4 text-emerald-300" />
            <span>Akaun & Data Ladang Anda Dilindungi 100% Secara Peribadi</span>
          </div>
        </div>
      </div>

      {/* Highlights Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-white border border-gray-200 rounded-2xl p-4 text-center shadow-xs">
          <div className="text-2xl font-black text-emerald-700">99.2%</div>
          <div className="text-[11px] font-bold text-gray-500 uppercase mt-0.5">Ketepatan AI Vision</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-2xl p-4 text-center shadow-xs">
          <div className="text-2xl font-black text-emerald-700">YOLOv8</div>
          <div className="text-[11px] font-bold text-gray-500 uppercase mt-0.5">ONNX Engine</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-2xl p-4 text-center shadow-xs">
          <div className="text-2xl font-black text-amber-600">Masa Nyata</div>
          <div className="text-[11px] font-bold text-gray-500 uppercase mt-0.5">Aliran Telemetri IoT</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-2xl p-4 text-center shadow-xs">
          <div className="text-2xl font-black text-amber-600">LLM Fusion</div>
          <div className="text-[11px] font-bold text-gray-500 uppercase mt-0.5">Cadangan Agronomis</div>
        </div>
      </div>

      {/* Core Features Grid */}
      <div className="space-y-4">
        <div className="text-center max-w-xl mx-auto space-y-1">
          <h2 className="text-lg md:text-xl font-extrabold text-gray-900">Ciri-Ciri Utama FarmNeura v2</h2>
          <p className="text-xs text-gray-500 font-medium">Platform pintar yang memodenkan pengurusan tanaman pertanian jitu anda.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Feature 1 */}
          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-xs hover:shadow-md transition space-y-2">
            <div className="w-10 h-10 rounded-xl bg-emerald-100 flex items-center justify-center text-emerald-800 font-bold">
              <Bot className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-gray-900 text-sm">{t.featureAiTitle}</h3>
            <p className="text-xs text-gray-600 leading-relaxed font-normal">{t.featureAiDesc}</p>
          </div>

          {/* Feature 2 */}
          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-xs hover:shadow-md transition space-y-2">
            <div className="w-10 h-10 rounded-xl bg-amber-100 flex items-center justify-center text-amber-800 font-bold">
              <Wifi className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-gray-900 text-sm">{t.featureIotTitle}</h3>
            <p className="text-xs text-gray-600 leading-relaxed font-normal">{t.featureIotDesc}</p>
          </div>

          {/* Feature 3 */}
          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-xs hover:shadow-md transition space-y-2">
            <div className="w-10 h-10 rounded-xl bg-green-100 flex items-center justify-center text-green-800 font-bold">
              <LayoutGrid className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-gray-900 text-sm">{t.featureManagementTitle}</h3>
            <p className="text-xs text-gray-600 leading-relaxed font-normal">{t.featureManagementDesc}</p>
          </div>

          {/* Feature 4 */}
          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-xs hover:shadow-md transition space-y-2">
            <div className="w-10 h-10 rounded-xl bg-orange-100 flex items-center justify-center text-orange-800 font-bold">
              <Zap className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-gray-900 text-sm">{t.featureQuickScanTitle}</h3>
            <p className="text-xs text-gray-600 leading-relaxed font-normal">{t.featureQuickScanDesc}</p>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="bg-gray-100/90 border border-gray-200 rounded-3xl p-6 md:p-8 space-y-6">
        <div className="text-center max-w-md mx-auto">
          <h2 className="text-lg md:text-xl font-extrabold text-gray-900">{t.howItWorksTitle}</h2>
          <p className="text-xs text-gray-500 mt-0.5">3 langkah mudah untuk mula mengurus ladang anda.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-2xl p-4 border border-gray-200 text-center space-y-2 shadow-xs">
            <div className="w-8 h-8 rounded-full bg-emerald-700 text-white font-black text-xs flex items-center justify-center mx-auto">1</div>
            <h4 className="font-extrabold text-xs text-gray-900">{t.step1Title}</h4>
            <p className="text-[11px] text-gray-600 leading-relaxed">{t.step1Desc}</p>
          </div>

          <div className="bg-white rounded-2xl p-4 border border-gray-200 text-center space-y-2 shadow-xs">
            <div className="w-8 h-8 rounded-full bg-emerald-700 text-white font-black text-xs flex items-center justify-center mx-auto">2</div>
            <h4 className="font-extrabold text-xs text-gray-900">{t.step2Title}</h4>
            <p className="text-[11px] text-gray-600 leading-relaxed">{t.step2Desc}</p>
          </div>

          <div className="bg-white rounded-2xl p-4 border border-gray-200 text-center space-y-2 shadow-xs">
            <div className="w-8 h-8 rounded-full bg-emerald-700 text-white font-black text-xs flex items-center justify-center mx-auto">3</div>
            <h4 className="font-extrabold text-xs text-gray-900">{t.step3Title}</h4>
            <p className="text-[11px] text-gray-600 leading-relaxed">{t.step3Desc}</p>
          </div>
        </div>
      </div>

      {/* Bottom CTA Card */}
      <div className="bg-gradient-to-r from-emerald-900 via-emerald-800 to-amber-700 rounded-3xl p-6 md:p-8 text-white text-center space-y-4 shadow-xl border border-emerald-700">
        <h2 className="text-xl md:text-2xl font-black text-white drop-shadow-md">{t.ctaBottomTitle}</h2>
        <p className="text-xs text-emerald-100 max-w-lg mx-auto font-medium leading-relaxed drop-shadow-xs">
          {t.ctaBottomDesc}
        </p>

        <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
          <button
            onClick={() => onOpenAuth('signup')}
            className="w-full sm:w-auto px-6 py-3 rounded-xl bg-white text-emerald-950 font-black text-xs shadow-lg hover:bg-emerald-50 transition flex items-center justify-center space-x-1.5"
          >
            <UserPlus className="w-4 h-4 text-emerald-700" />
            <span>{t.tabSignup}</span>
          </button>

          <button
            onClick={() => onOpenAuth('login')}
            className="w-full sm:w-auto px-6 py-3 rounded-xl bg-emerald-950/60 hover:bg-emerald-950/80 text-white border border-emerald-400/30 font-extrabold text-xs transition flex items-center justify-center space-x-1.5"
          >
            <LogIn className="w-4 h-4 text-emerald-300" />
            <span>{t.tabLogin}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
