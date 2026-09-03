import React, { useEffect, useState } from 'react';
import { fetchOverviewSummary } from '../services/api';
import { Leaf, AlertTriangle, Camera, MapPin } from 'lucide-react';
import { translations } from '../utils/translations';

export default function Overview({ onSelectAction, lang, user }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const t = translations[lang] || translations["Bahasa Melayu"];

  useEffect(() => {
    loadSummary();
  }, [user?.id]);

  const loadSummary = async () => {
    try {
      setLoading(true);
      const res = await fetchOverviewSummary(user?.id);
      setData(res);
    } catch (err) {
      console.error("Error loading summary:", err);
    } finally {
      setLoading(false);
    }
  };

  const currentHour = new Date().getHours();
  const greeting = currentHour < 12 ? t.greetingMorning : (currentHour < 18 ? t.greetingAfternoon : t.greetingEvening);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-emerald-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Greeting Header */}
      <div>
        <h2 className="text-xl sm:text-2xl font-black text-gray-900">{greeting}, {user?.full_name || 'Petani'} 👋</h2>
        <p className="text-xs sm:text-sm text-gray-500 mt-0.5">{t.overviewSubtitle}</p>
      </div>

      {/* Metric Cards Summary Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4">
        {/* Card 1: Overall Crop Health */}
        <div className="bg-white border border-gray-200 rounded-2xl p-3.5 sm:p-4 shadow-xs flex flex-col justify-between min-h-[90px]">
          <div className="flex items-center space-x-1.5 text-[11px] sm:text-xs font-bold text-gray-500">
            <Leaf className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
            <span className="truncate">{t.metricOverallHealth}</span>
          </div>
          <div className="text-xl sm:text-2xl font-black text-emerald-700 mt-2">
            {data?.overall_health_pct || 100}% <span className="text-xs font-bold text-emerald-600">Sihat</span>
          </div>
        </div>

        {/* Card 2: Needs Attention */}
        <div className="bg-white border border-gray-200 rounded-2xl p-3.5 sm:p-4 shadow-xs flex flex-col justify-between min-h-[90px]">
          <div className="flex items-center space-x-1.5 text-[11px] sm:text-xs font-bold text-orange-600">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
            <span className="truncate">{t.metricNeedsAttention}</span>
          </div>
          <div className="text-xl sm:text-2xl font-black text-orange-600 mt-2">
            {data?.needs_attention_plots || 0} <span className="text-xs font-bold text-orange-500">Plot</span>
          </div>
        </div>

        {/* Card 3: Needs Photo Update */}
        <div className="bg-white border border-gray-200 rounded-2xl p-3.5 sm:p-4 shadow-xs flex flex-col justify-between min-h-[90px]">
          <div className="flex items-center space-x-1.5 text-[11px] sm:text-xs font-bold text-blue-600">
            <Camera className="w-3.5 h-3.5 flex-shrink-0" />
            <span className="truncate">{t.metricNeedsPhoto}</span>
          </div>
          <div className="text-xl sm:text-2xl font-black text-gray-900 mt-2">
            {data?.needs_photo_plots || 0} <span className="text-xs font-bold text-gray-500">Plot</span>
          </div>
        </div>

        {/* Card 4: Active Plots */}
        <div className="bg-white border border-gray-200 rounded-2xl p-3.5 sm:p-4 shadow-xs flex flex-col justify-between min-h-[90px]">
          <div className="flex items-center space-x-1.5 text-[11px] sm:text-xs font-bold text-gray-500">
            <MapPin className="w-3.5 h-3.5 text-emerald-700 flex-shrink-0" />
            <span className="truncate">{t.metricActivePlots}</span>
          </div>
          <div className="text-xl sm:text-2xl font-black text-gray-900 mt-2">
            {data?.active_plots || 0} <span className="text-xs font-bold text-gray-500">Plot</span>
          </div>
        </div>

        {/* Card 5: Total Harvest Yield (KG) */}
        <div className="bg-white border border-emerald-200 bg-emerald-50/30 rounded-2xl p-3.5 sm:p-4 shadow-xs flex flex-col justify-between min-h-[90px]">
          <div className="flex items-center space-x-1.5 text-[11px] sm:text-xs font-bold text-emerald-800">
            <span className="text-sm">🌾</span>
            <span className="truncate">Jumlah Tuaian (KG)</span>
          </div>
          <div className="text-xl sm:text-2xl font-black text-emerald-800 mt-2">
            {data?.total_harvest_kg || 0} <span className="text-xs font-bold text-emerald-600">kg</span>
          </div>
        </div>

        {/* Card 6: Estimated Revenue (RM) */}
        <div className="bg-white border border-amber-200 bg-amber-50/30 rounded-2xl p-3.5 sm:p-4 shadow-xs flex flex-col justify-between min-h-[90px]">
          <div className="flex items-center space-x-1.5 text-[11px] sm:text-xs font-bold text-amber-800">
            <span className="text-sm">💰</span>
            <span className="truncate">Anggaran Hasil RM</span>
          </div>
          <div className="text-xl sm:text-2xl font-black text-amber-900 mt-2">
            RM {(data?.total_harvest_revenue_myr || 0).toFixed(2)}
          </div>
        </div>
      </div>


      {/* Today's Action List */}
      <div className="space-y-3">
        <div>
          <h3 className="text-base sm:text-lg font-black text-gray-900">{t.actionListTitle}</h3>
          <p className="text-xs text-gray-500">{t.actionListSubtitle}</p>
        </div>

        {data?.today_action_list?.length === 0 ? (
          <div className="bg-white p-6 rounded-2xl border border-gray-200 text-center text-gray-500 text-xs font-medium shadow-xs">
            Tiada plot berdaftar lagi. Sila ke menu Pendaftaran & Pengurusan untuk menambah ladang dan plot anda.
          </div>
        ) : (
          <div className="space-y-3">
            {data?.today_action_list?.map((item) => (
              <div 
                key={item.plot_id} 
                className="bg-white border border-gray-200 rounded-2xl p-4 shadow-xs hover:shadow-md transition space-y-3"
              >
                {/* Header: Plot & Farm Location */}
                <div className="flex items-center justify-between border-b border-gray-100 pb-2.5">
                  <div className="font-black text-gray-900 text-sm flex items-center space-x-1.5">
                    <MapPin className="w-4 h-4 text-emerald-700 flex-shrink-0" />
                    <span>{item.plot_name}</span>
                    <span className="text-xs text-gray-500 font-semibold">({item.farm_name})</span>
                  </div>
                </div>

                {/* Body: Status Badge & Action Button */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-0.5">
                  <div className="flex-1">
                    {item.is_attention ? (
                      <span className="bg-orange-50 text-orange-700 border border-orange-200 px-3 py-1.5 rounded-xl text-xs font-bold inline-flex items-center gap-1.5">
                        {t.statusAttention}
                      </span>
                    ) : item.is_overdue ? (
                      <span className="bg-blue-50 text-blue-700 border border-blue-200 px-3 py-1.5 rounded-xl text-xs font-bold inline-flex items-center gap-1.5">
                        {t.statusOverdue} ({item.days_overdue} hari)
                      </span>
                    ) : (
                      <span className="bg-emerald-50 text-emerald-800 border border-emerald-200 px-3 py-1.5 rounded-xl text-xs font-bold inline-flex items-center gap-1.5">
                        {t.statusOptimal}
                      </span>
                    )}
                  </div>

                  <button
                    onClick={() => onSelectAction(item.farm_id, item.plot_id)}
                    className={`w-full sm:w-auto px-5 py-2.5 rounded-xl text-xs font-black transition flex items-center justify-center space-x-1.5 shadow-sm ${
                      item.is_attention || item.is_overdue
                        ? 'bg-emerald-700 hover:bg-emerald-800 text-white'
                        : 'bg-gray-100 hover:bg-gray-200 text-gray-800'
                    }`}
                  >
                    <span>{item.btn_label}</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
