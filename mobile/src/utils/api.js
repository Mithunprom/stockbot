// REST API client for StockBot backend
import { API_BASE } from './config';

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

export const api = {
  health: () => fetchJSON('/health'),
  signals: (limit = 50) => fetchJSON(`/signals?limit=${limit}`),
  signalsActionable: () => fetchJSON('/signals/actionable'),
  positionsDetail: () => fetchJSON('/positions/detail'),
  portfolioSummary: () => fetchJSON('/portfolio/summary'),
  trades: (limit = 50) => fetchJSON(`/trades?limit=${limit}`),
  status: () => fetchJSON('/status'),
  riskReport: () => fetchJSON('/reports/risk'),
};
