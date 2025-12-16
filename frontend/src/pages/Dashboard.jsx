import { useState, useEffect } from 'react';
import { Camera, Activity, CheckCircle2, XCircle } from 'lucide-react';
import { getStatistics, getDetections } from '../services/api';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [recentDetections, setRecentDetections] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [statsData, detectionsData] = await Promise.all([
        getStatistics(),
        getDetections({ page: 1, pageSize: 5 })
      ]);
      setStats(statsData);
      setRecentDetections(detectionsData.items);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ minHeight: '400px' }}>
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="fade-in">
      <div className="flex justify-between items-center" style={{ marginBottom: 'var(--spacing-xl)' }}>
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-secondary">Overview of PlateVision system</p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-4" style={{ marginBottom: 'var(--spacing-xl)' }}>
        <div className="card stat-card">
          <div className="flex justify-between items-center">
            <Camera size={24} style={{ color: 'var(--color-accent-primary)' }} />
          </div>
          <div className="stat-value">{stats?.total_detections || 0}</div>
          <div className="stat-label">Total Detections</div>
        </div>

        <div className="card stat-card">
          <div className="flex justify-between items-center">
            <CheckCircle2 size={24} style={{ color: 'var(--color-success)' }} />
          </div>
          <div className="stat-value">{stats?.valid_detections || 0}</div>
          <div className="stat-label">Valid Plates</div>
        </div>

        <div className="card stat-card">
          <div className="flex justify-between items-center">
            <XCircle size={24} style={{ color: 'var(--color-error)' }} />
          </div>
          <div className="stat-value">{stats?.invalid_detections || 0}</div>
          <div className="stat-label">Invalid Plates</div>
        </div>

        <div className="card stat-card">
          <div className="flex justify-between items-center">
            <Activity size={24} style={{ color: 'var(--color-info)' }} />
          </div>
          <div className="stat-value">{stats?.today_count || 0}</div>
          <div className="stat-label">Today's Detections</div>
        </div>
      </div>

      {/* Recent Detections */}
      <div className="card">
        <h2 className="text-xl font-semibold" style={{ marginBottom: 'var(--spacing-lg)' }}>
          Recent Detections
        </h2>
        
        {recentDetections.length === 0 ? (
          <div className="text-center text-muted" style={{ padding: 'var(--spacing-xl)' }}>
            <Camera size={48} style={{ margin: '0 auto var(--spacing-md)' }} />
            <p>No detections yet. Upload an image or start the camera.</p>
          </div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Plate Number</th>
                  <th>Confidence</th>
                  <th>Source</th>
                  <th>Status</th>
                  <th>Detected At</th>
                </tr>
              </thead>
              <tbody>
                {recentDetections.map((detection) => (
                  <tr key={detection.id}>
                    <td>
                      <span className="font-semibold">{detection.plate_number}</span>
                    </td>
                    <td>
                      <div style={{ width: '100px', background: 'var(--color-bg-tertiary)', height: '8px', borderRadius: 'var(--radius-sm)' }}>
                        <div style={{
                          width: `${detection.confidence * 100}%`,
                          height: '100%',
                          background: 'var(--color-accent-gradient)',
                          borderRadius: 'var(--radius-sm)'
                        }}></div>
                      </div>
                      <span className="text-sm text-muted">{(detection.confidence * 100).toFixed(1)}%</span>
                    </td>
                    <td>
                      <span className={`badge ${detection.source_type === 'camera' ? 'badge-info' : 'badge-warning'}`}>
                        {detection.source_type}
                      </span>
                    </td>
                    <td>
                      {detection.is_valid ? (
                        <span className="badge badge-success">Valid</span>
                      ) : (
                        <span className="badge badge-error">Invalid</span>
                      )}
                    </td>
                    <td className="text-sm text-secondary">
                      {new Date(detection.detected_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
