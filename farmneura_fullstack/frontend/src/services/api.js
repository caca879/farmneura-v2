import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';


export const fetchOverviewSummary = async (userId = null) => {
  const url = userId ? `${API_BASE}/overview/summary?user_id=${userId}` : `${API_BASE}/overview/summary`;
  const res = await axios.get(url);
  return res.data;
};


export const signupUser = async (userData) => {
  const res = await axios.post(`${API_BASE}/auth/signup`, userData);
  return res.data;
};

export const loginUser = async (credentials) => {
  const res = await axios.post(`${API_BASE}/auth/login`, credentials);
  return res.data;
};


export const fetchFarms = async (userId = null) => {
  const url = userId ? `${API_BASE}/farms?user_id=${userId}` : `${API_BASE}/farms`;
  const res = await axios.get(url);
  return res.data;
};


export const createFarm = async (farmData) => {
  const res = await axios.post(`${API_BASE}/farms`, farmData);
  return res.data;
};

export const deleteFarm = async (farmId) => {
  const res = await axios.delete(`${API_BASE}/farms/${farmId}`);
  return res.data;
};

export const fetchPlots = async (farmId = null, userId = null) => {
  const params = [];
  if (farmId) params.push(`farm_id=${farmId}`);
  if (userId) params.push(`user_id=${userId}`);
  const queryStr = params.length > 0 ? `?${params.join('&')}` : '';
  const res = await axios.get(`${API_BASE}/plots${queryStr}`);
  return res.data;
};

export const createPlot = async (plotData) => {
  const res = await axios.post(`${API_BASE}/plots`, plotData);
  return res.data;
};

export const deletePlot = async (plotId) => {
  const res = await axios.delete(`${API_BASE}/plots/${plotId}`);
  return res.data;
};

export const fetchCrops = async (plotId = null, userId = null) => {
  const params = [];
  if (plotId) params.push(`plot_id=${plotId}`);
  if (userId) params.push(`user_id=${userId}`);
  const queryStr = params.length > 0 ? `?${params.join('&')}` : '';
  const res = await axios.get(`${API_BASE}/crops${queryStr}`);
  return res.data;
};


export const createCrop = async (cropData) => {
  const res = await axios.post(`${API_BASE}/crops`, cropData);
  return res.data;
};

export const deleteCrop = async (cropId) => {
  const res = await axios.delete(`${API_BASE}/crops/${cropId}`);
  return res.data;
};


export const fetchTelemetry = async (plotId) => {
  const res = await axios.get(`${API_BASE}/telemetry/${plotId}`);
  return res.data;
};

export const submitInspection = async (formData) => {
  const res = await axios.post(`${API_BASE}/inspections/diagnose`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
};

export const saveQuickScan = async (saveData) => {
  const res = await axios.post(`${API_BASE}/inspections/save-quick-scan`, saveData);
  return res.data;
};


export const fetchInspectionHistory = async (plotId) => {
  const res = await axios.get(`${API_BASE}/inspections/history/${plotId}`);
  return res.data;
};

export const deleteInspection = async (inspectionId) => {
  const res = await axios.delete(`${API_BASE}/inspections/${inspectionId}`);
  return res.data;
};

export const fetchHarvests = async (plotId = null, userId = null) => {
  const params = [];
  if (plotId) params.push(`plot_id=${plotId}`);
  if (userId) params.push(`user_id=${userId}`);
  const queryStr = params.length > 0 ? `?${params.join('&')}` : '';
  const res = await axios.get(`${API_BASE}/harvests${queryStr}`);
  return res.data;
};

export const createHarvest = async (harvestData) => {
  const res = await axios.post(`${API_BASE}/harvests`, harvestData);
  return res.data;
};

export const deleteHarvest = async (harvestId) => {
  const res = await axios.delete(`${API_BASE}/harvests/${harvestId}`);
  return res.data;
};

