import React, { useEffect, useState, useRef } from 'react';
import { fetchFarms, fetchPlots, fetchCrops, fetchTelemetry, submitInspection, saveQuickScan, fetchInspectionHistory, deleteInspection } from '../services/api';
import { Camera, RefreshCw, Cpu, Activity, Leaf, Trash2, Calendar, AlertCircle, CheckCircle, Sprout, Image as ImageIcon, Zap } from 'lucide-react';
import { translations, localizeDiagnosis } from '../utils/translations';

const renderFormattedText = (text) => {
  if (!text) return null;
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index} className="font-bold text-gray-900">{part.slice(2, -2)}</strong>;
    }
    return part;
  });
};

export default function PlotMonitoring({ selectedFarmId, selectedPlotId, onSelectFarmPlot, lang, setLang, user }) {
  const t = translations[lang] || translations["Bahasa Melayu"];
  const isEnglish = Boolean(lang?.includes('English'));
  const languageChoice = isEnglish ? 'English' : 'Bahasa Melayu';

  const cameraInputRef = useRef(null);
  const galleryInputRef = useRef(null);

  const [farms, setFarms] = useState([]);
  const [activeFarmId, setActiveFarmId] = useState(selectedFarmId || '');
  const [plots, setPlots] = useState([]);
  const [activePlotId, setActivePlotId] = useState(selectedPlotId || '');
  const [crops, setCrops] = useState([]);
  const [activeCropId, setActiveCropId] = useState('');

  // Diagnostic & Model Options
  const [modelPref, setModelPref] = useState('Auto-Detect');
  const [llmProvider, setLlmProvider] = useState('Auto');
  const [fieldNotes, setFieldNotes] = useState('');


  // Sub Tab: 'record' vs 'quick_scan' vs 'history'
  const [activeTab, setActiveTab] = useState('record');

  // Image Upload & Inspection State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);

  // Quick Scan Post-Diagnosis Save Form
  const [saveFarmId, setSaveFarmId] = useState('');
  const [savePlots, setSavePlots] = useState([]);
  const [savePlotId, setSavePlotId] = useState('');
  const [saveCrops, setSaveCrops] = useState([]);
  const [saveCropId, setSaveCropId] = useState('');
  const [savingQuickScan, setSavingQuickScan] = useState(false);

  // Live Telemetry & History
  const [telemetry, setTelemetry] = useState(null);
  const [syncingTelemetry, setSyncingTelemetry] = useState(false);
  const [history, setHistory] = useState([]);
  const [modalImage, setModalImage] = useState(null);

  useEffect(() => {
    loadFarms();
  }, []);

  useEffect(() => {
    if (activeFarmId) {
      loadPlots(activeFarmId);
    }
  }, [activeFarmId]);

  useEffect(() => {
    if (activePlotId) {
      loadTelemetryData(activePlotId);
      loadHistoryData(activePlotId);
      loadCrops(activePlotId);
    }
  }, [activePlotId]);

  const loadCrops = async (pId) => {
    try {
      const data = await fetchCrops(pId, user?.id);
      setCrops(data);
      if (data.length > 0) {
        setActiveCropId(data[0].id);
        autoSetModelPref(data[0].name);
      } else {
        setActiveCropId('');
      }
    } catch (err) {
      console.error(err);
    }
  };


  const autoSetModelPref = (cropName) => {
    if (!cropName) return;
    const nameLower = cropName.toLowerCase();
    if (nameLower.includes('okra') || nameLower.includes('bendi')) {
      setModelPref('Okra');
    } else if (nameLower.includes('tomato')) {
      setModelPref('Tomato');
    }
  };


  const loadFarms = async () => {
    try {
      const data = await fetchFarms(user?.id);
      setFarms(data);
      if (data.length > 0) {
        if (!data.some(f => f.id === activeFarmId)) {
          setActiveFarmId(data[0].id);
        }
      } else {
        setActiveFarmId('');
        setPlots([]);
        setActivePlotId('');
      }
    } catch (err) {
      console.error(err);
    }
  };


  const loadPlots = async (fId) => {
    try {
      const data = await fetchPlots(fId, user?.id);
      setPlots(data);
      if (data.length > 0) {
        if (!data.some(p => p.id === activePlotId)) {
          setActivePlotId(data[0].id);
        }
      } else {
        setActivePlotId('');
      }
    } catch (err) {
      console.error(err);
    }
  };


  const loadTelemetryData = async (pId) => {
    try {
      setSyncingTelemetry(true);
      const data = await fetchTelemetry(pId);
      setTelemetry(data);
    } catch (err) {
      console.error(err);
    } finally {
      setSyncingTelemetry(false);
    }
  };

  const loadHistoryData = async (pId) => {
    try {
      const data = await fetchInspectionHistory(pId);
      setHistory(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (saveFarmId) {
      loadSavePlots(saveFarmId);
    }
  }, [saveFarmId]);

  useEffect(() => {
    if (savePlotId) {
      loadSaveCrops(savePlotId);
    }
  }, [savePlotId]);

  const loadSavePlots = async (fId) => {
    try {
      const data = await fetchPlots(fId, user?.id);
      setSavePlots(data);
      if (data.length > 0) {
        setSavePlotId(data[0].id);
      } else {
        setSavePlotId('');
      }
    } catch (err) {
      console.error(err);
    }
  };

  const loadSaveCrops = async (pId) => {
    try {
      const data = await fetchCrops(pId, user?.id);
      setSaveCrops(data);
      if (data.length > 0) {
        setSaveCropId(data[0].id);
      } else {
        setSaveCropId('');
      }
    } catch (err) {
      console.error(err);
    }
  };


  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setCurrentResult(null);
    }
  };

  const handleDiagnose = async () => {
    if (!selectedFile) return;
    if (activeTab === 'record' && !activePlotId) return;

    try {
      setDiagnosing(true);
      const formData = new FormData();
      if (activeTab === 'record') {
        formData.append('plot_id', activePlotId);
        if (activeCropId) formData.append('crop_id', activeCropId);
      } else {
        formData.append('plot_id', 'quick_scan');
      }
      formData.append('model_preference', modelPref);
      formData.append('language_choice', languageChoice);
      formData.append('llm_provider', llmProvider);
      if (fieldNotes) formData.append('field_notes', fieldNotes);

      formData.append('file', selectedFile);

      const result = await submitInspection(formData);
      setCurrentResult(result);
      if (activeTab === 'record') {
        loadHistoryData(activePlotId);
      } else {
        if (farms.length > 0) {
          setSaveFarmId(farms[0].id);
        }
      }
    } catch (err) {
      console.error(err);
      alert("Error processing inspection. Please try again.");
    } finally {
      setDiagnosing(false);
    }
  };

  const handleSaveQuickScan = async () => {
    if (!savePlotId || !currentResult) return;
    try {
      setSavingQuickScan(true);
      await saveQuickScan({
        plot_id: savePlotId,
        crop_id: saveCropId || null,
        image_url: currentResult.image_url,
        leaf_count: currentResult.leaf_count,
        diagnosis: currentResult.diagnosis,
        intervention: currentResult.intervention,
        field_notes: fieldNotes || null,
      });

      alert("Quick Scan record successfully saved to Plot History Log!");
      setActiveFarmId(saveFarmId);
      setActivePlotId(savePlotId);
      loadHistoryData(savePlotId);
      setActiveTab('history');
    } catch (err) {
      console.error(err);
      alert("Error saving Quick Scan record. Please try again.");
    } finally {
      setSavingQuickScan(false);
    }
  };

  const handleSavePlotRecord = async () => {
    if (!activePlotId || !currentResult) return;
    try {
      setSavingQuickScan(true);
      await saveQuickScan({
        plot_id: activePlotId,
        crop_id: activeCropId || null,
        image_url: currentResult.image_url,
        leaf_count: currentResult.leaf_count,
        diagnosis: currentResult.diagnosis,
        intervention: currentResult.intervention,
        field_notes: fieldNotes || null,
      });

      alert("Record successfully saved to Plot History Log!");
      loadHistoryData(activePlotId);
      setActiveTab('history');
    } catch (err) {
      console.error(err);
      alert("Error saving record to Plot History Log. Please try again.");
    } finally {
      setSavingQuickScan(false);
    }
  };



  const handleDeleteHistory = async (id) => {
    if (confirm("Are you sure you want to delete this log?")) {
      try {
        await deleteInspection(id);
        loadHistoryData(activePlotId);
      } catch (err) {
        console.error(err);
      }
    }
  };

  const selectedPlotObj = plots.find(p => p.id === activePlotId);

  return (
    <div className="space-y-6">
      {/* Target Farm, Plot & Crop Intercropping Selectors */}
      <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs font-bold text-gray-700 uppercase tracking-wider mb-1">{t.targetFarmLabel}</label>
          <select
            value={activeFarmId}
            onChange={(e) => {
              setActiveFarmId(e.target.value);
              onSelectFarmPlot(e.target.value, '');
            }}
            className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold text-gray-900 focus:ring-2 focus:ring-farmGreen-500 focus:outline-none"
          >
            {farms.length === 0 ? (
              <option value="">{lang?.includes('Melayu') ? 'Tiada Ladang Berdaftar' : 'No Registered Farms'}</option>
            ) : (
              farms.map(f => (
                <option key={f.id} value={f.id}>{f.name}</option>
              ))
            )}
          </select>
        </div>

        <div>
          <label className="block text-xs font-bold text-gray-700 uppercase tracking-wider mb-1">{t.targetPlotLabel}</label>
          <select
            value={activePlotId}
            onChange={(e) => {
              setActivePlotId(e.target.value);
              onSelectFarmPlot(activeFarmId, e.target.value);
            }}
            className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold text-gray-900 focus:ring-2 focus:ring-farmGreen-500 focus:outline-none"
          >
            {plots.length === 0 ? (
              <option value="">{lang?.includes('Melayu') ? 'Tiada Plot Berdaftar' : 'No Registered Plots'}</option>
            ) : (
              plots.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))
            )}

          </select>
        </div>

        <div>
          <label className="block text-xs font-bold text-gray-700 uppercase tracking-wider mb-1">{t.targetCropLabel}</label>
          <select
            value={activeCropId}
            onChange={(e) => {
              setActiveCropId(e.target.value);
              const selectedCrop = crops.find(c => c.id === e.target.value);
              if (selectedCrop) autoSetModelPref(selectedCrop.name);
            }}
            className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-sm font-semibold text-gray-900 focus:ring-2 focus:ring-farmGreen-500 focus:outline-none"
          >
            {crops.length === 0 ? (
              <option value="">{t.allCropsGeneral}</option>
            ) : (
              crops.map(c => (
                <option key={c.id} value={c.id}>{c.name} {c.variety ? `(${c.variety})` : ''}</option>
              ))
            )}

          </select>
        </div>
      </div>


      {/* Sub Tabs: Record vs Quick Scan vs History */}
      <div className="flex border-b border-gray-200 flex-wrap">
        <button
          onClick={() => {
            setActiveTab('record');
            setCurrentResult(null);
          }}
          className={`py-3 px-6 text-sm font-bold border-b-2 transition ${
            activeTab === 'record'
              ? 'border-farmGreen-500 text-farmGreen-500'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabTakeRecord}
        </button>

        <button
          onClick={() => {
            setActiveTab('quick_scan');
            setCurrentResult(null);
            if (farms.length > 0) setSaveFarmId(farms[0].id);
          }}
          className={`py-3 px-6 text-sm font-bold border-b-2 transition ${
            activeTab === 'quick_scan'
              ? 'border-amber-500 text-amber-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <span>{t.tabQuickScan}</span>
        </button>


        <button
          onClick={() => setActiveTab('history')}
          className={`py-3 px-6 text-sm font-bold border-b-2 transition ${
            activeTab === 'history'
              ? 'border-farmGreen-500 text-farmGreen-500'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.tabHistoryLog} ({history.length})
        </button>
      </div>

      {activeTab === 'quick_scan' && (
        <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-xl shadow-xs text-xs text-amber-900 font-medium leading-relaxed">
          {t.quickScanBanner}
        </div>
      )}

      {activeTab === 'record' && (
        /* Cloud IoT Telemetry Stream Card */
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-2xl p-4 shadow-sm mb-6">

            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Cpu className="w-5 h-5 text-farmGreen-500" />
                <h4 className="font-bold text-gray-900 text-sm">{t.iotStreamTitle}</h4>
              </div>
              <button
                onClick={() => loadTelemetryData(activePlotId)}
                disabled={syncingTelemetry}
                className="bg-white hover:bg-gray-50 border border-gray-300 px-3 py-1 rounded-xl text-xs font-semibold text-gray-700 shadow-sm flex items-center space-x-1"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${syncingTelemetry ? 'animate-spin' : ''}`} />
                <span>{t.btnSyncIot}</span>
              </button>
            </div>

            {telemetry && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-white p-3 rounded-xl border border-green-100 shadow-xs">
                  <span className="text-xs text-gray-500 font-medium">{t.soilMoisture}</span>
                  <div className="text-lg font-bold text-gray-900 mt-0.5">{telemetry.soil_moisture}%</div>
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${telemetry.soil_moisture < 40 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                    {telemetry.soil_moisture < 40 ? 'DEFICIT' : 'OPTIMAL'}
                  </span>
                </div>

                <div className="bg-white p-3 rounded-xl border border-green-100 shadow-xs">
                  <span className="text-xs text-gray-500 font-medium">{t.airTemp}</span>
                  <div className="text-lg font-bold text-gray-900 mt-0.5">{telemetry.air_temp}°C</div>
                </div>

                <div className="bg-white p-3 rounded-xl border border-green-100 shadow-xs">
                  <span className="text-xs text-gray-500 font-medium">{t.soilEc}</span>
                  <div className="text-lg font-bold text-gray-900 mt-0.5">{telemetry.soil_ec} mS/cm</div>
                </div>

                <div className="bg-white p-3 rounded-xl border border-green-100 shadow-xs">
                  <span className="text-xs text-gray-500 font-medium">{t.soilPh}</span>
                  <div className="text-lg font-bold text-gray-900 mt-0.5">{telemetry.soil_ph}</div>
                </div>
              </div>
            )}
            <p className="text-[11px] text-gray-500 mt-2 font-mono">{telemetry?.server_status} | Sync: {telemetry?.timestamp}</p>
          </div>
      )}

      {(activeTab === 'record' || activeTab === 'quick_scan') && (
        <div className="space-y-6">
          {/* Photo Capture / Upload Card */}

          <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-gray-900 text-base">{t.captureTitle}</h3>
            
            {/* Hidden Mobile File Inputs */}
            <input
              type="file"
              accept="image/*"
              capture="environment"
              ref={cameraInputRef}
              onChange={handleFileChange}
              className="hidden"
            />

            <input
              type="file"
              accept="image/*"
              ref={galleryInputRef}
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Mobile Dual Action Buttons */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => cameraInputRef.current?.click()}
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-4 rounded-xl shadow transition text-xs flex items-center justify-center space-x-2"
              >
                <Camera className="w-4 h-4" />
                <span>{t.btnTakeLivePhoto}</span>
              </button>

              <button
                type="button"
                onClick={() => galleryInputRef.current?.click()}
                className="w-full bg-gray-100 hover:bg-gray-200 text-gray-800 border border-gray-300 font-bold py-3 px-4 rounded-xl shadow-xs transition text-xs flex items-center justify-center space-x-2"
              >
                <ImageIcon className="w-4 h-4 text-gray-600" />
                <span>{t.btnChooseGallery}</span>
              </button>
            </div>

            {selectedFile && (
              <div className="text-xs font-semibold text-gray-600 bg-gray-50 px-3 py-1.5 rounded-lg border border-gray-200 flex items-center space-x-1">
                <span>{t.photoSelected}</span>
                <span className="font-bold text-gray-900 truncate">{selectedFile.name}</span>
              </div>
            )}

            {previewUrl && (
              <div className="mt-3 bg-black/5 rounded-xl p-2 border border-gray-200">
                <img 
                  src={previewUrl} 
                  alt="Canopy Preview" 
                  onClick={() => setModalImage(previewUrl)}
                  className="w-full h-auto max-h-[550px] object-contain rounded-lg cursor-pointer hover:opacity-95 transition" 
                />
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">{t.aiModelLabel}</label>
                <select
                  value={modelPref}
                  onChange={(e) => setModelPref(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-xs font-semibold text-gray-800"
                >
                  <option value="Auto-Detect">Auto-Detect (YOLOv8)</option>
                  <option value="Okra Pod">Okra Pod Ripeness (3 Classes)</option>
                  <option value="Okra Leaf">Okra Leaf Disease (3 Classes)</option>
                  <option value="Tomato">Tomato Disease (8 Classes)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-700 mb-1">
                  {isEnglish ? 'LLM Provider / Model' : 'Penyedia Model LLM AI'}
                </label>
                <select
                  value={llmProvider}
                  onChange={(e) => setLlmProvider(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-xs font-semibold text-gray-800"
                >
                  <option value="Auto">⚡ Auto (Groq ➔ Gemini ➔ OpenAI)</option>
                  <option value="Groq">🤖 Groq AI (Llama 3.3 70B)</option>
                  <option value="Gemini">✨ Google Gemini AI (Gemini 1.5/2.0)</option>
                  <option value="OpenAI">🟢 OpenAI (GPT-4o-mini)</option>
                </select>
              </div>
            </div>


            <button
              onClick={handleDiagnose}
              disabled={!selectedFile || diagnosing}
              className={`w-full py-3 px-4 rounded-xl font-bold text-xs sm:text-sm shadow transition flex items-center justify-center space-x-2 text-center ${
                !selectedFile || diagnosing
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  : 'bg-farmGreen-500 hover:bg-farmGreen-700 text-white'
              }`}
            >
              {diagnosing ? (
                <div className="flex items-center justify-center space-x-2">
                  <RefreshCw className="w-4 h-4 animate-spin flex-shrink-0 self-center" />
                  <span className="leading-snug">{t.btnProcessing}</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Activity className="w-4 h-4 flex-shrink-0 self-center" />
                  <span className="leading-snug">{t.btnDiagnose}</span>
                </div>
              )}
            </button>
          </div>

          {/* Results Display */}
          {currentResult && (
            <div className="space-y-4">
              {/* Annotated Bounding Box Image Card */}
              <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm space-y-2">
                <div className="text-xs font-bold text-gray-700 flex items-center justify-between gap-2">
                  <span>{t.annotatedDetectionsTitle}</span>
                  <span className="whitespace-nowrap flex-shrink-0 bg-emerald-100 text-emerald-800 text-[11px] px-2.5 py-1 rounded-full font-bold border border-emerald-200 shadow-xs flex items-center justify-center">
                    YOLOv8 ONNX
                  </span>
                </div>
                <div className="bg-black/5 rounded-xl p-2 border border-gray-200">

                  <img 
                    key={currentResult.id || currentResult.image_url}
                    src={`${currentResult.image_url}?t=${Date.now()}`} 
                    alt="Annotated Detections" 
                    onClick={() => setModalImage(`${currentResult.image_url}?t=${Date.now()}`)}
                    className="w-full h-auto max-h-[600px] object-contain rounded-lg cursor-pointer hover:opacity-95 transition"
                  />

                </div>
              </div>


              {/* Metric Card */}
              <div className="bg-white border border-gray-200 rounded-2xl p-4 text-center shadow-sm">
                <div className="text-3xl font-extrabold text-farmGreen-500">{currentResult.leaf_count}</div>
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mt-1 flex items-center justify-center space-x-1">
                  {currentResult.diagnosis?.toLowerCase().includes('okra pod') || currentResult.diagnosis?.toLowerCase().includes('buah bendi') || modelPref.includes('Pod') ? (
                    <span>{t.totalOkraPods}</span>
                  ) : (
                    <span>{t.totalLeaves}</span>
                  )}
                </div>
              </div>

              {/* Diagnosis Card */}
              <div className="bg-white border-l-4 border-amber-500 rounded-xl p-4 shadow-sm">
                <h4 className="font-bold text-gray-900 text-sm mb-1">{t.cropDiagnosisTitle}</h4>
                <div className="text-xs text-gray-700 leading-relaxed">{renderFormattedText(localizeDiagnosis(currentResult.diagnosis, isEnglish))}</div>
              </div>

              {/* LLM Intervention Card */}
              <div className="bg-amber-50 border-l-4 border-warmOrange-500 rounded-xl p-4 shadow-sm">
                <h4 className="font-bold text-warmOrange-500 text-sm mb-2">{t.interventionTitle}</h4>
                <div className="text-xs text-gray-800 whitespace-pre-line leading-relaxed">{renderFormattedText(currentResult.intervention)}</div>
              </div>


              {/* Field Notes & Save Button */}
              {activeTab === 'record' ? (
                <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm space-y-3">
                  <label className="block text-xs font-bold text-gray-700">{t.fieldNotesLabel}</label>
                  <textarea
                    placeholder={t.fieldNotesPlaceholder}
                    value={fieldNotes}
                    onChange={(e) => setFieldNotes(e.target.value)}
                    className="w-full bg-gray-50 border border-gray-300 rounded-xl px-3 py-2 text-xs h-20"
                  />

                  <button
                    onClick={handleSavePlotRecord}
                    disabled={savingQuickScan}
                    className={`w-full font-bold py-3 rounded-xl shadow transition text-sm flex items-center justify-center space-x-2 ${
                      savingQuickScan ? 'bg-gray-300 text-gray-500 cursor-not-allowed' : 'bg-farmGreen-500 hover:bg-farmGreen-700 text-white'
                    }`}
                  >
                    <span>{savingQuickScan ? (languageChoice.includes('Melayu') ? 'Menyimpan...' : 'Saving...') : t.btnSaveRecord}</span>
                  </button>

                </div>
              ) : (
                /* Quick Scan Post-Diagnosis Save Card */
                <div className="bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-300 rounded-2xl p-5 shadow-sm space-y-4">
                  <h3 className="font-bold text-gray-900 text-sm flex items-center space-x-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    <span>{t.quickScanSaveHeader}</span>
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectTargetFarmSave}</label>
                      <select
                        value={saveFarmId}
                        onChange={(e) => setSaveFarmId(e.target.value)}
                        className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-xs font-semibold text-gray-900"
                      >
                        {farms.map(f => (
                          <option key={f.id} value={f.id}>{f.name}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectTargetPlotSave}</label>
                      <select
                        value={savePlotId}
                        onChange={(e) => setSavePlotId(e.target.value)}
                        className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-xs font-semibold text-gray-900"
                      >
                        {savePlots.map(p => (
                          <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-bold text-gray-700 mb-1">{t.selectTargetCropSave}</label>
                      <select
                        value={saveCropId}
                        onChange={(e) => setSaveCropId(e.target.value)}
                        className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-xs font-semibold text-gray-900"
                      >
                        {saveCrops.length === 0 ? (
                          <option value="">{t.allCropsGeneral}</option>
                        ) : (
                          saveCrops.map(c => (
                            <option key={c.id} value={c.id}>{c.name} {c.variety ? `(${c.variety})` : ''}</option>
                          ))
                        )}

                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-bold text-gray-700 mb-1">{t.fieldNotesLabel}</label>
                    <textarea
                      placeholder={t.fieldNotesPlaceholder}
                      value={fieldNotes}
                      onChange={(e) => setFieldNotes(e.target.value)}
                      className="w-full bg-white border border-gray-300 rounded-xl px-3 py-2 text-xs h-20"
                    />
                  </div>

                  <button
                    onClick={handleSaveQuickScan}
                    disabled={!savePlotId || savingQuickScan}
                    className="w-full bg-farmGreen-500 hover:bg-farmGreen-700 text-white font-bold py-3 rounded-xl shadow transition text-sm flex items-center justify-center space-x-2"
                  >
                    {savingQuickScan ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <span>{t.btnSaveQuickScan}</span>
                    )}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'history' && (
        /* History Log Tab */
        <div className="space-y-4">
          <h3 className="font-bold text-gray-900 text-base">{t.historyLogHeader} {selectedPlotObj?.name}</h3>

          
          {history.length === 0 ? (
            <div className="bg-white p-8 rounded-xl border border-dashed border-gray-300 text-center text-gray-500 text-sm">
              {t.noHistory}
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((rec) => {
                const imgSrc = rec.image_url?.startsWith('http') 
                  ? rec.image_url 
                  : `${import.meta.env.VITE_API_BASE_URL ? import.meta.env.VITE_API_BASE_URL.replace('/api/v1', '') : ''}${rec.image_url}`;

                return (
                  <div key={rec.id} className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-3">
                    <div className="flex items-center justify-between border-b border-gray-100 pb-2 flex-wrap gap-1">
                      <div className="flex items-center space-x-2 text-xs font-bold text-gray-800 flex-wrap gap-1">
                        <Calendar className="w-4 h-4 text-farmGreen-500" />
                        <span>{rec.created_at}</span>
                        <span className="bg-green-100 text-farmGreen-700 px-2 py-0.5 rounded-full font-medium">
                          {rec.stage_name} ({t.dayLabel} {rec.cycle_day})
                        </span>
                        {(() => {
                          const cropMatch = crops.find(c => c.id === rec.crop_id);
                          if (cropMatch) {
                            return (
                              <span className="bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-bold flex items-center space-x-1">
                                <span>{cropMatch.name}</span>
                              </span>
                            );
                          }
                          return null;
                        })()}
                      </div>

                      <button
                        onClick={() => handleDeleteHistory(rec.id)}
                        className="text-gray-400 hover:text-red-500 p-1"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    {rec.interval_tracking && (
                      <div className="bg-green-50 border-l-4 border-green-500 p-2.5 rounded text-xs text-green-800 font-medium">
                        {rec.interval_tracking}
                      </div>
                    )}

                    <div className="flex space-x-3">
                      <img 
                        src={imgSrc} 
                        alt="Canopy" 
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = 'https://images.unsplash.com/photo-1592417817098-8f3d6eb247a5?w=200&auto=format&fit=crop&q=80';
                        }}
                        onClick={() => setModalImage(imgSrc)}
                        className="w-24 h-24 object-cover rounded-lg border border-gray-200 flex-shrink-0 cursor-pointer hover:opacity-85 transition" 
                      />
                      <div className="space-y-1 text-xs">
                        <div className="font-semibold text-gray-900">
                          {rec.diagnosis?.toLowerCase().includes('okra pod') || rec.diagnosis?.toLowerCase().includes('buah bendi') || rec.diagnosis?.toLowerCase().includes('pod') ? (
                            <span>{t.okraPodsDetectedLabel} {rec.leaf_count}</span>
                          ) : (
                            <span>{t.leavesDetectedLabel} {rec.leaf_count}</span>
                          )}
                        </div>
                        <div className="text-gray-700"><strong>{t.diagnosisLabel}</strong> {renderFormattedText(localizeDiagnosis(rec.diagnosis, isEnglish))}</div>

                        <div className="text-gray-800 whitespace-pre-line leading-relaxed mt-1">
                          <strong>{t.interventionLabel}</strong> {renderFormattedText(rec.intervention)}
                        </div>
                        {rec.field_notes && (
                          <div className="text-gray-600 italic bg-gray-50 p-2 rounded border border-gray-200 mt-1">
                            <strong>{t.fieldNotesHistoryLabel}</strong> {rec.field_notes}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}


      {/* Fullscreen Image Lightbox Modal */}
      {modalImage && (
        <div 
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex flex-col items-center justify-center p-4"
          onClick={() => setModalImage(null)}
        >
          <div className="relative max-w-5xl w-full max-h-[90vh] flex flex-col items-center" onClick={(e) => e.stopPropagation()}>
            <button 
              onClick={() => setModalImage(null)}
              className="absolute -top-10 right-0 bg-white/20 hover:bg-white/40 text-white rounded-full px-3 py-1 text-xs font-bold shadow transition"
            >
              {t.closeLabel}
            </button>
            <img 
              src={modalImage} 
              alt="Enlarged View" 
              className="max-w-full max-h-[85vh] object-contain rounded-xl shadow-2xl border border-white/20"
            />
            <p className="text-white/80 text-xs mt-3 font-semibold">{t.enlargeHint}</p>
          </div>
        </div>
      )}
    </div>
  );
}

