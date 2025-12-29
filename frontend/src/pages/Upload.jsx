import { useState, useRef, useEffect } from 'react';
import { Upload as UploadIcon, Image as ImageIcon, CheckCircle2, AlertCircle, X } from 'lucide-react';
import { uploadImage } from '../services/api';
import './Upload.css';

// Module-level cache to persist state across navigation
let cachedState = {
  selectedFile: null,
  preview: null,
  results: null,
  error: null
};

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(cachedState.selectedFile);
  const [preview, setPreview] = useState(cachedState.preview);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(cachedState.results);
  const [error, setError] = useState(cachedState.error);
  const fileInputRef = useRef(null);

  // Sync state to cache
  useEffect(() => {
    cachedState = {
      selectedFile,
      preview,
      results,
      error
    };
  }, [selectedFile, preview, results, error]);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setResults(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setError(null);
      setResults(null);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    
    setUploading(true);
    setError(null);
    
    try {
      const detections = await uploadImage(selectedFile, true);
      setResults(detections);
      
      if (detections.length === 0) {
        setError('No license plates detected in the image');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="upload-page fade-in">
      <div style={{ marginBottom: 'var(--spacing-xl)' }}>
        <h1 className="text-2xl font-bold">Upload Image</h1>
        <p className="text-secondary">Upload a vehicle image for plate detection</p>
      </div>

      <div className="upload-layout">
        <div className="card upload-section">
          {!preview ? (
            <div
              className="dropzone"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
            >
              <ImageIcon size={64} className="dropzone-icon" />
              <h3 className="text-lg font-semibold">Drop image here or click to browse</h3>
              <p className="text-sm text-muted">Supports: JPG, PNG, JPEG</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </div>
          ) : (
            <div className="preview-container">
              <button onClick={handleReset} className="btn-close">
                <X size={20} />
              </button>
              <img src={preview} alt="Preview" className="preview-image" />
              
              <div className="preview-actions">
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="btn btn-primary btn-lg"
                >
                  {uploading ? (
                    <>
                      <div className="spinner" style={{ width: '20px', height: '20px' }}></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <UploadIcon size={20} />
                      Detect Plate
                    </>
                  )}
                </button>
                
                <button onClick={handleReset} className="btn btn-secondary">
                  Choose Another
                </button>
              </div>
            </div>
          )}
        </div>

        {(results || error) && (
          <div className="card results-section">
            <h2 className="text-xl font-semibold" style={{ marginBottom: 'var(--spacing-lg)' }}>
              Detection Results
            </h2>

            {error && (
              <div className="alert-error">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            {results && results.length > 0 && (
              <div className="results-list">
                {results.map((result, index) => (
                  <div key={index} className="result-item">
                    <div className="flex justify-between items-start" style={{ marginBottom: 'var(--spacing-md)' }}>
                      <div>
                        <div className="result-plate">{result.plate_number}</div>
                        {result.raw_ocr_text !== result.plate_number && (
                          <div className="text-sm text-muted">
                            Raw: {result.raw_ocr_text}
                          </div>
                        )}
                      </div>
                      
                      {result.is_valid ? (
                        <span className="badge badge-success">
                          <CheckCircle2 size={14} />
                          Valid
                        </span>
                      ) : (
                        <span className="badge badge-error">
                          <AlertCircle size={14} />
                          Invalid
                        </span>
                      )}
                    </div>

                    {result.cropped_image_path && (
                      <img 
                        src={result.cropped_image_path}
                        alt={result.plate_number}
                        className="result-plate-image"
                      />
                    )}

                    <div className="result-details">
                      <div className="result-detail">
                        <span className="text-sm text-muted">Confidence:</span>
                        <div className="flex gap-sm items-center">
                          <div className="confidence-bar-small">
                            <div 
                              style={{
                                width: `${result.confidence * 100}%`,
                                height: '100%',
                                background: 'var(--color-accent-gradient)',
                                borderRadius: 'var(--radius-sm)'
                              }}
                            ></div>
                          </div>
                          <span className="text-sm font-semibold">
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;
