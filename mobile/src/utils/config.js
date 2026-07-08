// Backend API configuration
// Set USE_LOCAL = true when developing at home with backend running locally
// Set USE_LOCAL = false to always use Railway (works from anywhere)
const USE_LOCAL = false;

const LOCAL_URL = 'http://10.0.0.128:8000';
const RAILWAY_URL = 'https://stockbot-production-cbde.up.railway.app';

export const API_BASE = USE_LOCAL ? LOCAL_URL : RAILWAY_URL;
export const WS_URL = API_BASE.replace('http', 'ws') + '/ws/dashboard';
