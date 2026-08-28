import React, { useEffect, useState } from 'react';
import { fetchOverviewSummary } from '../services/api';
import { Leaf, AlertTriangle, Camera, MapPin } from 'lucide-react';
import { translations } from '../utils/translations';

export default function Overview({ onSelectAction, lang }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const t = translations[lang] || translations["🇲🇾 Bahasa Melayu"];

  useEffect(() => {
    loadSummary();
  }, []);

  const loadSummary = async () => {
    try {
      setLoading(true);
      const res = await fetchOverviewSummary();
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
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-farmGreen-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Greeting Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">{greeting} 👋</h2>
        <p className="text-sm text-gray-500 mt-1">{t.overviewSubtitle}</p>
      </div>

      {/* 2x2 Metric Cards Summary Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Card 1: Overall Crop Health */}
        <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center space-x-2 text-xs font-semibold text-gray-500 mb-1">
            <Leaf className="w-4 h-4 text-farmGreen-500" />
            <span>{t.metricOverallHealth}</span>
          </div>
          <div className="text-2xl font-bold text-farmGreen-500">
            {data?.overall_health_pct || 100}% Good 🍃
          </div>
        </div>

        {/* Card 2: Needs Attention */}
        <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center space-x-2 mb-1">
            <span className="bg-warmOrange-500 text-white text-xs px-2 py-0.5 rounded-full font-semibold flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> {t.metricNeedsAttention}
            </span>
          </div>
          <div className="text-2xl font-bold text-warmOrange-500">
            {data?.needs_attention_plots || 0} Plots
          </div>
        </div>

        {/* Card 3: Needs Photo Update */}
        <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center space-x-2 text-xs font-semibold text-gray-500 mb-1">
            <Camera className="w-4 h-4 text-blue-600" />
            <span>{t.metricNeedsPhoto}</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {data?.needs_photo_plots || 0} Plots
          </div>
        </div>

        {/* Card 4: Active Plots */}
        <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center space-x-2 text-xs font-semibold text-gray-500 mb-1">
            <MapPin className="w-4 h-4 text-gray-600" />
            <span>{t.metricActivePlots}</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {data?.active_plots || 0}
          </div>
        </div>
      </div>

      {/* Today's Action List */}
      <div>
        <h3 className="text-lg font-bold text-gray-900 mb-1">{t.actionListTitle}</h3>
        <p className="text-xs text-gray-500 mb-3">{t.actionListSubtitle}</p>

        {data?.today_action_list?.length === 0 ? (
          <div className="bg-white p-6 rounded-xl border border-gray-200 text-center text-gray-500 text-sm">
            No plots registered yet. Go to Registry to add farms and plots.
          </div>
        ) : (
          <div className="space-y-3">
            {data?.today_action_list?.map((item) => (
              <div 
                key={item.plot_id} 
                className="bg-white border border-gray-200 rounded-xl p-4 flex items-center justify-between shadow-sm"
              >
                <div>
                  <div className="font-bold text-gray-900 text-base flex items-center space-x-2">
                    <MapPin className="w-4 h-4 text-farmGreen-500" />
                    <span>{item.plot_name}</span>
                    <span className="text-xs text-gray-400 font-normal">({item.farm_name})</span>
                  </div>
                  <div className="mt-2">
                    {item.is_attention ? (
                      <span className="bg-orange-50 text-warmOrange-500 border border-orange-200 px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1">
                        {t.statusAttention}
                      </span>
                    ) : item.is_overdue ? (
                      <span className="bg-blue-50 text-blue-700 border border-blue-200 px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1">
                        {t.statusOverdue} ({item.days_overdue} days)
                      </span>
                    ) : (
                      <span className="bg-green-50 text-green-700 border border-green-200 px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1">
                        {t.statusOptimal}
                      </span>
                    )}
                  </div>
                </div>

                <button
                  onClick={() => onSelectAction(item.farm_id, item.plot_id)}
                  className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center space-x-1 ${
                    item.is_attention || item.is_overdue
                      ? 'bg-farmGreen-500 hover:bg-farmGreen-700 text-white shadow'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <span>{item.is_attention ? t.btnTakeAction : (item.is_overdue ? t.btnScanNow : t.btnInspect)}</span>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

