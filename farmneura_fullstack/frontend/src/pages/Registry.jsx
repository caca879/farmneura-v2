import React, { useEffect, useState } from 'react';
import { fetchFarms, createFarm, deleteFarm, fetchPlots, createPlot, deletePlot, fetchCrops, createCrop, deleteCrop } from '../services/api';
import { Trash2, MapPin, Building, Sprout } from 'lucide-react';
import { translations } from '../utils/translations';

export default function Registry({ lang, user }) {
  const t = translations[lang] || translations["🇲🇾 Bahasa Melayu"];

  const [activeTab, setActiveTab] = useState('farm'); // 'farm' | 'plot' | 'crop'

  const [farms, setFarms] = useState([]);
  const [plots, setPlots] = useState([]);
  const [crops, setCrops] = useState([]);

  // Farm Form
  const [farmName, setFarmName] = useState('');
  const [farmLoc, setFarmLoc] = useState('');
  const [farmSize, setFarmSize] = useState(1000);

  // Plot Form
  const [selectedFarmId, setSelectedFarmId] = useState('');
  const [plotName, setPlotName] = useState('');
  const [plotSize, setPlotSize] = useState(1000);
  const [cycleStart, setCycleStart] = useState(new Date().toISOString().split('T')[0]);
  const [cycleEnd, setCycleEnd] = useState(new Date().toISOString().split('T')[0]);
  const [costBudget, setCostBudget] = useState(0);
  const [plotNotes, setPlotNotes] = useState('');

  // Crop (Intercropping) Form
  const [cropFarmId, setCropFarmId] = useState('');
  const [cropPlotId, setCropPlotId] = useState('');
  const [cropName, setCropName] = useState('');
  const [cropVariety, setCropVariety] = useState('');
  const [cropPlantingDate, setCropPlantingDate] = useState(new Date().toISOString().split('T')[0]);
  const [harvestDays, setHarvestDays] = useState(50);

  useEffect(() => {
    loadData();
  }, [user?.id]);

  const loadData = async () => {
    try {
      const farmData = await fetchFarms(user?.id);
      setFarms(farmData);
      if (farmData.length > 0) {
        setSelectedFarmId(farmData[0].id);
        setCropFarmId(farmData[0].id);
      }

      const plotData = await fetchPlots();
      setPlots(plotData);
      if (plotData.length > 0) setCropPlotId(plotData[0].id);

      const cropData = await fetchCrops();
      setCrops(cropData);
    } catch (err) {
      console.error(err);
    }
  };

  const filteredPlotsForCrop = cropFarmId
    ? plots.filter(p => p.farm_id === cropFarmId)
    : plots;

  const handleCreateFarm = async (e) => {
    e.preventDefault();
    if (!farmName.trim()) return alert("Farm Name is required.");

    try {
      await createFarm({
        name: farmName,
        location: farmLoc,
        size_sq_ft: parseFloat(farmSize),
        user_id: user?.id
      });
      setFarmName('');
      setFarmLoc('');
      loadData();
    } catch (err) {
      alert("Error creating farm: " + (err.response?.data?.detail || err.message));
    }
  };


  const handleDeleteFarm = async (id) => {
    if (confirm("Deleting this farm will remove all associated plots. Continue?")) {
      try {
        await deleteFarm(id);
        loadData();
      } catch (err) {
        console.error(err);
      }
    }
  };

  const handleCreatePlot = async (e) => {
    e.preventDefault();
    if (!plotName.trim() || !selectedFarmId) return alert("Plot Name and Farm are required.");

    try {
      await createPlot({
        farm_id: selectedFarmId,
        name: plotName,
        size_sq_ft: parseFloat(plotSize),
        cycle_start_date: cycleStart,
        cycle_end_date: cycleEnd,
        cost_budget_myr: parseFloat(costBudget),
        notes: plotNotes
      });
      setPlotName('');
      setPlotNotes('');
      loadData();
    } catch (err) {
      alert("Error creating plot: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleDeletePlot = async (id) => {
    if (confirm("Delete this plot?")) {
      try {
        await deletePlot(id);
        loadData();
      } catch (err) {
        console.error(err);
      }
    }
  };

  const handleCreateCrop = async (e) => {
    e.preventDefault();
    if (!cropName.trim() || !cropPlotId) return alert("Crop Name and Target Plot are required.");

    try {
      await createCrop({
        plot_id: cropPlotId,
        name: cropName,
        variety: cropVariety,
        planting_date: cropPlantingDate,
        harvest_target_days: parseInt(harvestDays)
      });
      setCropName('');
      setCropVariety('');
      loadData();
    } catch (err) {
      alert("Error creating crop: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleDeleteCrop = async (id) => {
    if (confirm("Delete this crop registration?")) {
      try {
        await deleteCrop(id);
        loadData();
      } catch (err) {
        console.error(err);
      }
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">{t.registryTitle}</h2>
        <p className="text-sm text-gray-500 mt-1">{t.registrySubtitle}</p>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex border-b border-gray-200">
        <button
          onClick={() => setActiveTab('farm')}
          className={`py-3 px-6 text-sm font-bold border-b-2 transition ${
            activeTab === 'farm' ? 'border-farmGreen-500 text-farmGreen-500' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabRegFarm}
        </button>
        <button
          onClick={() => setActiveTab('plot')}
          className={`py-3 px-6 text-sm font-bold border-b-2 transition ${
            activeTab === 'plot' ? 'border-farmGreen-500 text-farmGreen-500' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabRegPlot}
        </button>
        <button
          onClick={() => setActiveTab('crop')}
          className={`py-3 px-6 text-sm font-bold border-b-2 transition ${
            activeTab === 'crop' ? 'border-farmGreen-500 text-farmGreen-500' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabRegCrop}
        </button>
      </div>

      {/* TAB 1: REGISTER FARM */}
      {activeTab === 'farm' && (
        <div className="space-y-6">
          <form onSubmit={handleCreateFarm} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-gray-900 text-base">{t.regFarmHeader}</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.farmName}</label>
                <input
                  type="text"
                  placeholder="e.g. Farm C - Rawang"
                  value={farmName}
                  onChange={(e) => setFarmName(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.farmLoc}</label>
                <input
                  type="text"
                  placeholder="e.g. Selangor, Malaysia"
                  value={farmLoc}
                  onChange={(e) => setFarmLoc(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.farmSize}</label>
                <input
                  type="number"
                  value={farmSize}
                  onChange={(e) => setFarmSize(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <button type="submit" className="bg-farmGreen-500 hover:bg-farmGreen-700 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow transition">
              {t.btnRegFarm}
            </button>
          </form>

          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
            <h4 className="font-bold text-gray-900 text-sm mb-3">{t.registeredFarms} ({farms.length})</h4>
            <div className="divide-y divide-gray-100">
              {farms.map((f) => (
                <div key={f.id} className="py-3 flex items-center justify-between">
                  <div>
                    <div className="font-bold text-gray-900 text-sm">{f.name}</div>
                    <div className="text-xs text-gray-500">{f.location || 'No location set'} • {f.size_sq_ft} sq ft</div>
                  </div>
                  <button onClick={() => handleDeleteFarm(f.id)} className="text-gray-400 hover:text-red-500 p-1">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: REGISTER PLOT */}
      {activeTab === 'plot' && (
        <div className="space-y-6">
          <form onSubmit={handleCreatePlot} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-gray-900 text-base">{t.regPlotHeader}</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectParentFarm}</label>
                <select
                  value={selectedFarmId}
                  onChange={(e) => setSelectedFarmId(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold text-gray-800"
                >
                  {farms.map(f => (
                    <option key={f.id} value={f.id}>{f.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.plotName}</label>
                <input
                  type="text"
                  placeholder="e.g. Plot 4"
                  value={plotName}
                  onChange={(e) => setPlotName(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cycleStart}</label>
                <input
                  type="date"
                  value={cycleStart}
                  onChange={(e) => setCycleStart(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cycleEnd}</label>
                <input
                  type="date"
                  value={cycleEnd}
                  onChange={(e) => setCycleEnd(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.costBudget}</label>
                <input
                  type="number"
                  value={costBudget}
                  onChange={(e) => setCostBudget(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-gray-700 mb-1">{t.plotNotes}</label>
              <textarea
                placeholder="Soil type, irrigation row, microclimate notes, etc."
                value={plotNotes}
                onChange={(e) => setPlotNotes(e.target.value)}
                className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm h-20"
              />
            </div>

            <button type="submit" className="bg-farmGreen-500 hover:bg-farmGreen-700 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow transition">
              {t.btnRegPlot}
            </button>
          </form>

          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
            <h4 className="font-bold text-gray-900 text-sm mb-3">{t.registeredPlots} ({plots.length})</h4>
            <div className="divide-y divide-gray-100">
              {plots.map((p) => {
                const farmObj = farms.find(f => f.id === p.farm_id);
                return (
                  <div key={p.id} className="py-3 flex items-center justify-between">
                    <div>
                      <div className="font-bold text-gray-900 text-sm">{p.name} <span className="text-xs font-normal text-gray-500">({farmObj?.name || 'Farm'})</span></div>
                      <div className="text-xs text-gray-500">Cycle: {p.cycle_start_date} to {p.cycle_end_date} • {p.size_sq_ft} sq ft</div>
                    </div>
                    <button onClick={() => handleDeletePlot(p.id)} className="text-gray-400 hover:text-red-500 p-1">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: REGISTER CROPS (INTERCROPPING) */}
      {activeTab === 'crop' && (
        <div className="space-y-6">
          <form onSubmit={handleCreateCrop} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-gray-900 text-base">{t.regCropHeader}</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectTargetFarm}</label>
                <select
                  value={cropFarmId}
                  onChange={(e) => {
                    const fId = e.target.value;
                    setCropFarmId(fId);
                    const matchingPlots = plots.filter(p => p.farm_id === fId);
                    if (matchingPlots.length > 0) setCropPlotId(matchingPlots[0].id);
                  }}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold text-gray-800"
                >
                  {farms.map(f => (
                    <option key={f.id} value={f.id}>{f.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectTargetPlot}</label>
                <select
                  value={cropPlotId}
                  onChange={(e) => setCropPlotId(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold text-gray-800"
                >
                  {filteredPlotsForCrop.map(p => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cropName}</label>
                <input
                  type="text"
                  placeholder="e.g. Pakchoy / Bendi / Yardlong Bean"
                  value={cropName}
                  onChange={(e) => setCropName(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cropVariety}</label>
                <input
                  type="text"
                  placeholder="e.g. Green Fortune F1"
                  value={cropVariety}
                  onChange={(e) => setCropVariety(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.harvestDays}</label>
                <input
                  type="number"
                  value={harvestDays}
                  onChange={(e) => setHarvestDays(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <button type="submit" className="bg-farmGreen-500 hover:bg-farmGreen-700 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow transition">
              {t.btnRegCrop}
            </button>
          </form>

          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
            <h4 className="font-bold text-gray-900 text-sm mb-3">{t.registeredCrops} ({crops.length})</h4>
            <div className="divide-y divide-gray-100">
              {crops.map((c) => {
                const targetPlot = plots.find(p => p.id === c.plot_id);
                const targetFarm = targetPlot ? farms.find(f => f.id === targetPlot.farm_id) : null;
                return (
                  <div key={c.id} className="py-3 flex items-center justify-between">
                    <div>
                      <div className="font-bold text-gray-900 text-sm">🌿 {c.name} {c.variety ? `(${c.variety})` : ''}</div>
                      <div className="text-xs text-gray-500">
                        📍 Farm: <strong>{targetFarm?.name || 'Farm'}</strong> ➔ Plot: <strong>{targetPlot?.name || 'Plot'}</strong> • Target Cycle: {c.harvest_target_days} days
                      </div>
                    </div>
                    <button onClick={() => handleDeleteCrop(c.id)} className="text-gray-400 hover:text-red-500 p-1">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );

}
