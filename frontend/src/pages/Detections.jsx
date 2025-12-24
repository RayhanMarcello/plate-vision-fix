import { useState, useEffect } from 'react';
import { Search, Trash2, Filter, ChevronLeft, ChevronRight } from 'lucide-react';
import { getDetections, deleteDetection } from '../services/api';

const Detections = () => {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  
  // Filters
  const [search, setSearch] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [validFilter, setValidFilter] = useState('');

  useEffect(() => {
    loadDetections();
  }, [page, search, sourceFilter, validFilter]);

  const loadDetections = async () => {
    setLoading(true);
    try {
      const params = {
        page,
        pageSize: 20,
      };
      
      if (search) params.search = search;
      if (sourceFilter) params.sourceType = sourceFilter;
      if (validFilter !== '') params.isValid = validFilter === 'true';
      
      const data = await getDetections(params);
      setDetections(data.items);
      setTotalPages(data.total_pages);
      setTotal(data.total);
    } catch (error) {
      console.error('Failed to load detections:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to delete this detection?')) return;
    
    try {
      await deleteDetection(id);
      loadDetections();
    } catch (error) {
      alert('Failed to delete detection');
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    setPage(1);
  };

  return (
    <div className="fade-in">
      <div style={{ marginBottom: 'var(--spacing-xl)' }}>
        <h1 className="text-2xl font-bold">Detections</h1>
        <p className="text-secondary">Manage and search detection records</p>
      </div>

      <div className="card" style={{ marginBottom: 'var(--spacing-lg)' }}>
        <div className="flex gap-md" style={{ marginBottom: 'var(--spacing-md)' }}>
          <form onSubmit={handleSearch} className="flex gap-sm" style={{ flex: 1 }}>
            <div className="input-group" style={{ flex: 1, marginBottom: 0 }}>
              <div style={{ position: 'relative' }}>
                <input
                  type="text"
                  className="input"
                  placeholder="Search by plate number..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={{ paddingLeft: '40px' }}
                />
                <Search 
                  size={18} 
                  style={{ 
                    position: 'absolute', 
                    left: 'var(--spacing-md)', 
                    top: '50%', 
                    transform: 'translateY(-50%)',
                    color: 'var(--color-text-muted)'
                  }} 
                />
              </div>
            </div>
            <button type="submit" className="btn btn-primary">
              <Search size={18} />
              Search
            </button>
          </form>
        </div>

        <div className="flex gap-md">
          <div className="input-group" style={{ marginBottom: 0 }}>
            <label className="input-label">
              <Filter size={14} style={{ display: 'inline', marginRight: '4px' }} />
              Source
            </label>
            <select 
              className="input"
              value={sourceFilter}
              onChange={(e) => { setSourceFilter(e.target.value); setPage(1); }}
            >
              <option value="">All Sources</option>
              <option value="upload">Upload</option>
              <option value="camera">Camera</option>
            </select>
          </div>

          <div className="input-group" style={{ marginBottom: 0 }}>
            <label className="input-label">
              <Filter size={14} style={{ display: 'inline', marginRight: '4px' }} />
              Validation
            </label>
            <select 
              className="input"
              value={validFilter}
              onChange={(e) => { setValidFilter(e.target.value); setPage(1); }}
            >
              <option value="">All</option>
              <option value="true">Valid</option>
              <option value="false">Invalid</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="flex justify-between items-center" style={{ marginBottom: 'var(--spacing-lg)' }}>
          <div className="text-sm text-secondary">
            Total: <span className="font-semibold">{total}</span> detections
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center" style={{ padding: 'var(--spacing-2xl)' }}>
            <div className="spinner"></div>
          </div>
        ) : detections.length === 0 ? (
          <div className="text-center text-muted" style={{ padding: 'var(--spacing-2xl)' }}>
            <p>No detections found</p>
          </div>
        ) : (
          <>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Plate Number</th>
                    <th>Image</th>
                    <th>Raw OCR</th>
                    <th>Confidence</th>
                    <th>Source</th>
                    <th>Status</th>
                    <th>Detected At</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {detections.map((detection) => (
                    <tr key={detection.id}>
                      <td className="text-sm text-muted">#{detection.id}</td>
                      <td>
                        <span className="font-semibold" style={{ fontFamily: 'monospace' }}>
                          {detection.plate_number}
                        </span>
                      </td>
                      <td>
                        {detection.image_data ? (
                          <img 
                            src={detection.image_data} 
                            alt={detection.plate_number} 
                            style={{ 
                              height: '40px', 
                              width: 'auto', 
                              borderRadius: '4px',
                              objectFit: 'contain',
                              border: '1px solid var(--color-border)'
                            }} 
                            onClick={() => window.open(detection.image_data, '_blank')}
                            title="Click to view full size"
                            className="cursor-pointer hover:opacity-80 transition-opacity"
                          />
                        ) : (
                          <span className="text-xs text-muted">No image</span>
                        )}
                      </td>
                      <td className="text-sm text-muted">{detection.raw_ocr_text || '-'}</td>
                      <td>
                        <div style={{ minWidth: '100px' }}>
                          <div style={{ 
                            width: '100%', 
                            background: 'var(--color-bg-tertiary)', 
                            height: '6px', 
                            borderRadius: 'var(--radius-sm)',
                            marginBottom: '4px'
                          }}>
                            <div style={{
                              width: `${detection.confidence * 100}%`,
                              height: '100%',
                              background: 'var(--color-accent-gradient)',
                              borderRadius: 'var(--radius-sm)'
                            }}></div>
                          </div>
                          <span className="text-sm">{(detection.confidence * 100).toFixed(1)}%</span>
                        </div>
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
                        {new Date(detection.detected_at).toLocaleString('id-ID')}
                      </td>
                      <td>
                        <button
                          onClick={() => handleDelete(detection.id)}
                          className="btn btn-error"
                          style={{ padding: 'var(--spacing-xs) var(--spacing-sm)' }}
                        >
                          <Trash2 size={16} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex justify-between items-center" style={{ marginTop: 'var(--spacing-lg)', paddingTop: 'var(--spacing-lg)', borderTop: '1px solid var(--color-border)' }}>
              <div className="text-sm text-secondary">
                Page {page} of {totalPages}
              </div>
              
              <div className="flex gap-sm">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="btn btn-secondary"
                >
                  <ChevronLeft size={18} />
                  Previous
                </button>
                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="btn btn-secondary"
                >
                  Next
                  <ChevronRight size={18} />
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Detections;
