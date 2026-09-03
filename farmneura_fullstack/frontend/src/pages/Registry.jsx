import React, { useEffect, useState } from 'react';
import { 
  fetchFarms, createFarm, deleteFarm, 
  fetchPlots, createPlot, deletePlot, 
  fetchCrops, createCrop, deleteCrop,
  fetchHarvests, createHarvest, deleteHarvest 
} from '../services/api';
import { Trash2, MapPin, Building, Sprout, DollarSign } from 'lucide-react';
import { translations } from '../utils/translations';

export default function Registry({ lang, user }) {
  const t = translations[lang] || translations["🇲🇾 Bahasa Melayu"];

  const [activeTab, setActiveTab] = useState('farm'); // 'farm' | 'plot' | 'crop' | 'harvest'

  const [farms, setFarms] = useState([]);
  const [plots, setPlots] = useState([]);
  const [crops, setCrops] = useState([]);
  const [harvests, setHarvests] = useState([]);

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

  // Harvest Record Form
  const [harvestPlotId, setHarvestPlotId] = useState('');
  const [harvestWeightKg, setHarvestWeightKg] = useState('');
  const [harvestPricePerKg, setHarvestPricePerKg] = useState('8.00');
  const [harvestDate, setHarvestDate] = useState(new Date().toISOString().split('T')[0]);
  const [harvestNotes, setHarvestNotes] = useState('');

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
      } else {
        setSelectedFarmId('');
        setCropFarmId('');
      }

      const plotData = await fetchPlots(null, user?.id);
      setPlots(plotData);
      if (plotData.length > 0) {
        setCropPlotId(plotData[0].id);
        setHarvestPlotId(plotData[0].id);
      } else {
        setCropPlotId('');
        setHarvestPlotId('');
      }

      const cropData = await fetchCrops(null, user?.id);
      setCrops(cropData);

      const harvestData = await fetchHarvests(null, user?.id);
      setHarvests(harvestData);
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
    if (!selectedFarmId) return alert("Select a parent farm.");
    if (!plotName.trim()) return alert("Plot Name is required.");

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
    if (!cropPlotId) return alert("Select a parent plot.");
    if (!cropName.trim()) return alert("Crop Name is required.");

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

  const handleCreateHarvest = async (e) => {
    e.preventDefault();
    if (!harvestPlotId) return alert("Sila pilih plot sasaran.");
    if (!harvestWeightKg || parseFloat(harvestWeightKg) <= 0) return alert("Sila masukkan berat tuaian (KG).");
    if (!harvestPricePerKg || parseFloat(harvestPricePerKg) < 0) return alert("Sila masukkan anggaran harga 1 KG (RM).");

    try {
      await createHarvest({
        plot_id: harvestPlotId,
        user_id: user?.id,
        yield_weight_kg: parseFloat(harvestWeightKg),
        price_per_kg_myr: parseFloat(harvestPricePerKg),
        harvest_date: harvestDate,
        notes: harvestNotes
      });
      setHarvestWeightKg('');
      setHarvestNotes('');
      loadData();
      alert("✅ " + (lang.includes('Melayu') ? "Rekod tuaian berjaya disimpan!" : "Harvest record saved successfully!"));
    } catch (err) {
      alert("Error saving harvest: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleDeleteHarvest = async (id) => {
    if (confirm(lang.includes('Melayu') ? "Padam rekod tuaian ini?" : "Delete this harvest record?")) {
      try {
        await deleteHarvest(id);
        loadData();
      } catch (err) {
        console.error(err);
      }
    }
  };

  const calculatedRevenue = (parseFloat(harvestWeightKg || 0) * parseFloat(harvestPricePerKg || 0)).toFixed(2);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">{t.registryTitle}</h2>
        <p className="text-sm text-gray-500 mt-1">{t.registrySubtitle}</p>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex border-b border-gray-200 overflow-x-auto">
        <button
          onClick={() => setActiveTab('farm')}
          className={`py-3 px-5 text-xs sm:text-sm font-bold border-b-2 transition whitespace-nowrap ${
            activeTab === 'farm' ? 'border-farmGreen-500 text-farmGreen-500' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabRegFarm}
        </button>
        <button
          onClick={() => setActiveTab('plot')}
          className={`py-3 px-5 text-xs sm:text-sm font-bold border-b-2 transition whitespace-nowrap ${
            activeTab === 'plot' ? 'border-farmGreen-500 text-farmGreen-500' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabRegPlot}
        </button>
        <button
          onClick={() => setActiveTab('crop')}
          className={`py-3 px-5 text-xs sm:text-sm font-bold border-b-2 transition whitespace-nowrap ${
            activeTab === 'crop' ? 'border-farmGreen-500 text-farmGreen-500' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabRegCrop}
        </button>
        <button
          onClick={() => setActiveTab('harvest')}
          className={`py-3 px-5 text-xs sm:text-sm font-bold border-b-2 transition whitespace-nowrap ${
            activeTab === 'harvest' ? 'border-farmGreen-500 text-farmGreen-500' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabRegHarvest}
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
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.farmLocation}</label>
                <input
                  type="text"
                  placeholder="e.g. Rawang, Selangor"
                  value={farmLoc}
                  onChange={(e) => setFarmLoc(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.farmAreaSqFt}</label>
                <input
                  type="number"
                  value={farmSize}
                  onChange={(e) => setFarmSize(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <button
              type="submit"
              className="bg-farmGreen-500 hover:bg-farmGreen-700 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow transition"
            >
              + {t.btnAddFarm}
            </button>
          </form>

          {/* Existing Farms List */}
          <div className="space-y-3">
            <h3 className="font-bold text-gray-900 text-base">{t.existingFarmsList} ({farms.length})</h3>
            {farms.length === 0 ? (
              <div className="bg-white p-6 rounded-xl border border-gray-200 text-center text-gray-500 text-sm">
                {lang.includes('Melayu') ? "Tiada ladang berdaftar. Gunakan borang di atas untuk menambah ladang pertama anda." : "No registered farms found. Use the form above to add your first farm."}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {farms.map(f => (
                  <div key={f.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                    <div>
                      <div className="font-bold text-gray-900 flex items-center space-x-2">
                        <Building className="w-4 h-4 text-farmGreen-500" />
                        <span>{f.name}</span>
                      </div>
                      <div className="text-xs text-gray-500 mt-1 flex items-center space-x-3">
                        <span>📍 {f.location || 'N/A'}</span>
                        <span>📐 {f.size_sq_ft} {t.cardSqFt}</span>
                      </div>
                    </div>
                    <button
                      onClick={() => handleDeleteFarm(f.id)}
                      className="text-gray-400 hover:text-red-500 p-2"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
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
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold"
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
                  placeholder="e.g. Plot A1 - Bendera"
                  value={plotName}
                  onChange={(e) => setPlotName(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.plotSizeSqFt}</label>
                <input
                  type="number"
                  value={plotSize}
                  onChange={(e) => setPlotSize(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cycleStartDate}</label>
                <input
                  type="date"
                  value={cycleStart}
                  onChange={(e) => setCycleStart(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cycleEndDate}</label>
                <input
                  type="date"
                  value={cycleEnd}
                  onChange={(e) => setCycleEnd(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.costBudgetMyr}</label>
                <input
                  type="number"
                  placeholder="e.g. 500"
                  value={costBudget}
                  onChange={(e) => setCostBudget(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-gray-700 mb-1">{t.notesLabel}</label>
              <input
                type="text"
                placeholder="Optional notes or soil info"
                value={plotNotes}
                onChange={(e) => setPlotNotes(e.target.value)}
                className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
              />
            </div>

            <button
              type="submit"
              className="bg-farmGreen-500 hover:bg-farmGreen-700 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow transition"
            >
              + {t.btnAddPlot}
            </button>
          </form>

          {/* Existing Plots List */}
          <div className="space-y-3">
            <h3 className="font-bold text-gray-900 text-base">{t.existingPlotsList} ({plots.length})</h3>
            {plots.length === 0 ? (
              <div className="bg-white p-6 rounded-xl border border-gray-200 text-center text-gray-500 text-sm">
                {lang.includes('Melayu') ? "Tiada plot berdaftar. Gunakan borang di atas untuk menambah plot pertama anda." : "No registered plots found. Use the form above to add your first plot."}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {plots.map(p => {
                  const parentFarm = farms.find(f => f.id === p.farm_id);
                  return (
                    <div key={p.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                      <div>
                        <div className="font-bold text-gray-900 flex items-center space-x-2">
                          <MapPin className="w-4 h-4 text-farmGreen-500" />
                          <span>{p.name}</span>
                          <span className="text-xs text-gray-400">({parentFarm?.name || 'Unknown Farm'})</span>
                        </div>
                        <div className="text-xs text-gray-500 mt-1 space-y-0.5">
                          <div>📅 {t.cardCycleLabel}: {p.cycle_start_date} {t.cardTo} {p.cycle_end_date}</div>
                          <div>💰 {t.cardBudgetLabel}: RM {p.cost_budget_myr} | 📐 {p.size_sq_ft} {t.cardSqFt}</div>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeletePlot(p.id)}
                        className="text-gray-400 hover:text-red-500 p-2"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: REGISTER CROP */}
      {activeTab === 'crop' && (
        <div className="space-y-6">
          <form onSubmit={handleCreateCrop} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-gray-900 text-base">{t.regCropHeader}</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectTargetFarm}</label>
                <select
                  value={cropFarmId}
                  onChange={(e) => setCropFarmId(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold"
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
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold"
                >
                  {filteredPlotsForCrop.map(p => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cropNameLabel}</label>
                <input
                  type="text"
                  placeholder="e.g. Bendi / Terung / Tomato"
                  value={cropName}
                  onChange={(e) => setCropName(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.cropVarietyLabel}</label>
                <input
                  type="text"
                  placeholder="e.g. F1 Hybrid 803"
                  value={cropVariety}
                  onChange={(e) => setCropVariety(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.plantingDateLabel}</label>
                <input
                  type="date"
                  value={cropPlantingDate}
                  onChange={(e) => setCropPlantingDate(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.harvestTargetDaysLabel}</label>
                <input
                  type="number"
                  value={harvestDays}
                  onChange={(e) => setHarvestDays(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <button
              type="submit"
              className="bg-farmGreen-500 hover:bg-farmGreen-700 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow transition"
            >
              + {t.btnAddCrop}
            </button>
          </form>

          {/* Existing Registered Crops List */}
          <div className="space-y-3">
            <h3 className="font-bold text-gray-900 text-base">{t.registeredCropsList} ({crops.length})</h3>
            {crops.length === 0 ? (
              <div className="bg-white p-6 rounded-xl border border-gray-200 text-center text-gray-500 text-sm">
                {lang.includes('Melayu') ? "Tiada tanaman berdaftar. Gunakan borang di atas untuk menambah varieti tanaman pertama anda." : "No registered crops found. Use the form above to add intercropping records."}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {crops.map(c => {
                  const parentPlot = plots.find(p => p.id === c.plot_id);
                  const parentFarm = farms.find(f => f.id === parentPlot?.farm_id);
                  return (
                    <div key={c.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                      <div>
                        <div className="font-bold text-gray-900 flex items-center space-x-2">
                          <Sprout className="w-4 h-4 text-emerald-600" />
                          <span>{c.name}</span>
                          <span className="text-xs text-gray-500 font-normal">({c.variety || 'Standard'})</span>
                        </div>
                        <div className="text-xs text-gray-500 mt-1 space-y-0.5">
                          <div>📍 Plot: <strong>{parentPlot?.name || 'N/A'}</strong> ({parentFarm?.name || 'N/A'})</div>
                          <div>🌱 {t.cardPlantedLabel}: {c.planting_date} | ⏱️ {t.cardHarvestCycleLabel}: ~{c.harvest_target_days} {t.cardDays}</div>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteCrop(c.id)}
                        className="text-gray-400 hover:text-red-500 p-2"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 4: HARVEST RECORDS & REVENUE ESTIMATION */}
      {activeTab === 'harvest' && (
        <div className="space-y-6">
          <form onSubmit={handleCreateHarvest} className="bg-white border border-emerald-200 bg-emerald-50/20 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-emerald-900 text-base flex items-center gap-2">
              <span>🌾</span> {t.regHarvestHeader}
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectTargetPlotHarvest.toUpperCase()}</label>
                <select
                  value={harvestPlotId}
                  onChange={(e) => setHarvestPlotId(e.target.value)}
                  className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold"
                >
                  {plots.map(p => {
                    const f = farms.find(farm => farm.id === p.farm_id);
                    return (
                      <option key={p.id} value={p.id}>{p.name} ({f?.name || 'Farm'})</option>
                    );
                  })}
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.harvestDateLabel.toUpperCase()}</label>
                <input
                  type="date"
                  value={harvestDate}
                  onChange={(e) => setHarvestDate(e.target.value)}
                  className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.harvestWeightLabel.toUpperCase()}</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 15.5"
                  value={harvestWeightKg}
                  onChange={(e) => setHarvestWeightKg(e.target.value)}
                  className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-sm font-bold text-emerald-700"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.harvestPriceLabel.toUpperCase()}</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 8.00"
                  value={harvestPricePerKg}
                  onChange={(e) => setHarvestPricePerKg(e.target.value)}
                  className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-sm font-bold text-amber-700"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-500 mb-1">{t.totalRevenueLabel.toUpperCase()}</label>
                <div className="w-full bg-amber-100/70 border border-amber-300 rounded-xl px-3 py-2 text-sm font-black text-amber-900">
                  RM {calculatedRevenue}
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.harvestNotesLabel.toUpperCase()}</label>
                <input
                  type="text"
                  placeholder="e.g. Gred A Bendi Kualiti Tinggi"
                  value={harvestNotes}
                  onChange={(e) => setHarvestNotes(e.target.value)}
                  className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-sm"
                />
              </div>
            </div>

            <button
              type="submit"
              className="bg-emerald-700 hover:bg-emerald-800 text-white font-bold text-xs px-6 py-2.5 rounded-xl shadow transition"
            >
              {t.btnSaveHarvest}
            </button>
          </form>

          {/* Existing Harvest Records List */}
          <div className="space-y-3">
            <h3 className="font-bold text-gray-900 text-base">{t.harvestListTitle} ({harvests.length})</h3>
            {harvests.length === 0 ? (
              <div className="bg-white p-6 rounded-xl border border-gray-200 text-center text-gray-500 text-sm">
                {t.noHarvestsMsg}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {harvests.map(h => {
                  const parentPlot = plots.find(p => p.id === h.plot_id);
                  const parentFarm = farms.find(f => f.id === parentPlot?.farm_id);
                  return (
                    <div key={h.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
                      <div>
                        <div className="font-black text-gray-900 text-sm flex items-center space-x-2">
                          <span>🌾 {h.yield_weight_kg} kg</span>
                          <span className="bg-amber-100 text-amber-900 text-xs px-2 py-0.5 rounded-full font-bold">
                            RM {h.total_revenue_myr.toFixed(2)}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 mt-1 space-y-0.5">
                          <div>📍 Plot: <strong>{parentPlot?.name || 'N/A'}</strong> ({parentFarm?.name || 'N/A'})</div>
                          <div>📅 {t.harvestDateLabel}: {h.harvest_date} | 💵 {t.cardPricePerKg}: RM {h.price_per_kg_myr.toFixed(2)}</div>
                          {h.notes && <div className="italic text-gray-600">📝 {h.notes}</div>}
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteHarvest(h.id)}
                        className="text-gray-400 hover:text-red-500 p-2"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
