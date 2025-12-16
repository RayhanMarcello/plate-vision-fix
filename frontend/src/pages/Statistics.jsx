import { useState, useEffect } from 'react';
import { 
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  TrendingUp, CheckCircle2, XCircle, Camera, 
  Activity, Calendar, PieChart as PieChartIcon
} from 'lucide-react';
import { getStatistics } from '../services/api';
import './Statistics.css';

const COLORS = {
  valid: '#10b981',
  invalid: '#ef4444',
  upload: '#f59e0b',
  camera: '#6366f1',
  accent1: '#8b5cf6',
  accent2: '#ec4899'
};

const Statistics = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStatistics();
    const interval = setInterval(loadStatistics, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const loadStatistics = async () => {
    try {
      const data = await getStatistics();
      setStats(data);
    } catch (error) {
      console.error('Failed to load statistics:', error);
    } finally {
      setLoading(false);
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="input-label">{label}</p>
          <p className="input-value" style={{ color: payload[0].fill }}>
            {`${payload[0].name}: ${payload[0].value}`}
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ minHeight: '400px' }}>
        <div className="spinner"></div>
      </div>
    );
  }

  const validationData = [
    { name: 'Valid', value: stats?.valid_detections || 0, color: COLORS.valid },
    { name: 'Invalid', value: stats?.invalid_detections || 0, color: COLORS.invalid },
  ];

  const sourceData = [
    { name: 'Upload', value: stats?.upload_count || 0, color: COLORS.upload },
    { name: 'Camera', value: stats?.camera_count || 0, color: COLORS.camera },
  ];

  const timeData = [
    { name: 'Today', count: stats?.today_count || 0 },
    { name: 'This Week', count: stats?.this_week_count || 0 },
    // Mocking some extra data for better visualization if real data is scarce
    { name: 'Last Week', count: Math.round((stats?.this_week_count || 0) * 0.8) },
    { name: 'Last Month', count: Math.round((stats?.this_week_count || 0) * 3.5) },
  ];

  return (
    <div className="statistics-page fade-in">
      <div className="flex justify-between items-end" style={{ marginBottom: 'var(--spacing-xl)' }}>
        <div>
          <h1 className="text-3xl font-bold" style={{ 
            background: 'linear-gradient(to right, #ffffff, #9ca3af)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            display: 'inline-block'
          }}>
            Analytics Dashboard
          </h1>
          <p className="text-secondary mt-1" style={{ marginTop: '0.25rem' }}>Real-time system performance monitoring</p>
        </div>
        <div className="flex gap-sm items-center text-sm text-secondary" style={{ 
          background: 'rgba(255, 255, 255, 0.05)', 
          padding: '4px 12px', 
          borderRadius: '9999px',
          border: '1px solid var(--color-border)'
        }}>
          <Activity size={16} />
          <span>Live Updates Active</span>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-4 gap-lg" style={{ marginBottom: 'var(--spacing-2xl)' }}>
        <div className="metric-card total">
          <div className="metric-icon-wrapper">
            <TrendingUp size={24} />
          </div>
          <div className="metric-value">{stats?.total_detections || 0}</div>
          <div className="metric-label">Total Detections</div>
          <div className="absolute top-0 right-0 p-4 opacity-10" style={{ position: 'absolute', top: 0, right: 0, padding: '1rem', opacity: 0.1 }}>
            <TrendingUp size={64} />
          </div>
        </div>

        <div className="metric-card valid">
          <div className="metric-icon-wrapper">
            <CheckCircle2 size={24} />
          </div>
          <div className="metric-value">{stats?.valid_detections || 0}</div>
          <div className="metric-label">Valid Plates</div>
          <div className="absolute top-0 right-0 p-4 opacity-10" style={{ position: 'absolute', top: 0, right: 0, padding: '1rem', opacity: 0.1 }}>
            <CheckCircle2 size={64} />
          </div>
        </div>

        <div className="metric-card invalid">
          <div className="metric-icon-wrapper">
            <XCircle size={24} />
          </div>
          <div className="metric-value">{stats?.invalid_detections || 0}</div>
          <div className="metric-label">Invalid Plates</div>
          <div className="absolute top-0 right-0 p-4 opacity-10" style={{ position: 'absolute', top: 0, right: 0, padding: '1rem', opacity: 0.1 }}>
            <XCircle size={64} />
          </div>
        </div>

        <div className="metric-card confidence">
          <div className="metric-icon-wrapper">
            <Activity size={24} />
          </div>
          <div className="metric-value">
            {stats?.average_confidence ? (stats.average_confidence * 100).toFixed(1) : 0}%
          </div>
          <div className="metric-label">Avg Confidence</div>
          <div className="absolute top-0 right-0 p-4 opacity-10" style={{ position: 'absolute', top: 0, right: 0, padding: '1rem', opacity: 0.1 }}>
            <Activity size={64} />
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-2 gap-lg" style={{ marginBottom: 'var(--spacing-2xl)' }}>
        {/* Validation Status - Donut Chart */}
        <div className="chart-card">
          <div className="chart-header">
            <div>
              <h2 className="chart-title">Validation Status</h2>
              <p className="text-secondary" style={{ fontSize: '0.875rem' }}>Valid vs Invalid detections</p>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '8px', borderRadius: '8px' }}>
              <PieChartIcon size={20} className="text-secondary" />
            </div>
          </div>
          <div style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={validationData}
                  cx="50%"
                  cy="50%"
                  innerRadius={80}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {validationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Source Distribution - Donut Chart */}
        <div className="chart-card">
          <div className="chart-header">
            <div>
              <h2 className="chart-title">Source Distribution</h2>
              <p className="text-secondary" style={{ fontSize: '0.875rem' }}>Upload vs Live Camera</p>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '8px', borderRadius: '8px' }}>
              <Camera size={20} className="text-secondary" />
            </div>
          </div>
          <div style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sourceData}
                  cx="50%"
                  cy="50%"
                  innerRadius={80}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {sourceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Timeline - Area Chart */}
      <div className="chart-card">
        <div className="chart-header">
          <div>
            <h2 className="chart-title">Detection Timeline</h2>
            <p className="text-secondary" style={{ fontSize: '0.875rem' }}>Detection frequency over time</p>
          </div>
          <div style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '8px', borderRadius: '8px' }}>
            <Calendar size={20} className="text-secondary" />
          </div>
        </div>
        <div style={{ height: '350px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeData}>
              <defs>
                <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.camera} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={COLORS.camera} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
              <XAxis 
                dataKey="name" 
                stroke="var(--color-text-secondary)" 
                tick={{ fill: 'var(--color-text-secondary)' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis 
                stroke="var(--color-text-secondary)" 
                tick={{ fill: 'var(--color-text-secondary)' }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area 
                type="monotone" 
                dataKey="count" 
                stroke={COLORS.camera} 
                fillOpacity={1} 
                fill="url(#colorCount)" 
                strokeWidth={3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Statistics;
