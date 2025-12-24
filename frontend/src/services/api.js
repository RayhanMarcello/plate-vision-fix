/**
 * PlateVision API Service
 */
import axios from 'axios';

// Use environment variable for production, fallback to relative path for development
const API_BASE = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api` 
  : '/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Upload image for plate detection
 */
export const uploadImage = async (file, saveToDb = true) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post(`/detect/upload?save_to_db=${saveToDb}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

/**
 * Get list of detections with pagination and filters
 */
export const getDetections = async (params = {}) => {
  const {
    page = 1,
    pageSize = 20,
    search,
    sourceType,
    isValid,
    startDate,
    endDate,
  } = params;
  
  const queryParams = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  
  if (search) queryParams.append('search', search);
  if (sourceType) queryParams.append('source_type', sourceType);
  if (isValid !== undefined) queryParams.append('is_valid', isValid.toString());
  if (startDate) queryParams.append('start_date', startDate);
  if (endDate) queryParams.append('end_date', endDate);
  
  const response = await api.get(`/detections?${queryParams.toString()}`);
  return response.data;
};

/**
 * Get single detection by ID
 */
export const getDetection = async (id) => {
  const response = await api.get(`/detections/${id}`);
  return response.data;
};

/**
 * Delete detection by ID
 */
export const deleteDetection = async (id) => {
  const response = await api.delete(`/detections/${id}`);
  return response.data;
};

/**
 * Get detection statistics
 */
export const getStatistics = async () => {
  const response = await api.get('/statistics');
  return response.data;
};

/**
 * Health check
 */
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
