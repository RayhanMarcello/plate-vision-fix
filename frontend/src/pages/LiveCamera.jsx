import { useEffect, useRef, useState } from 'react';
import { Camera, Video, VideoOff, AlertCircle } from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import './LiveCamera.css';

const LiveCamera = () => {
  const {
    isConnected,
    isStreaming,
    currentFrame,
    latestDetection,
    error,
    startCamera,
    stopCamera,
  } = useWebSocket();
  
  const [detections, setDetections] = useState([]);
  const canvasRef = useRef(null);

  // Update detections when new detection arrives
  useEffect(() => {
    if (latestDetection) {
      setDetections(prev => [latestDetection, ...prev.slice(0, 9)]);
    }
  }, [latestDetection]);

  // Draw frame on canvas
  useEffect(() => {
    if (currentFrame && canvasRef.current) {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
      };
      img.src = `data:image/jpeg;base64,${currentFrame}`;
    }
  }, [currentFrame]);

  // CLEANUP: Stop camera when component unmounts (navigating away)
  useEffect(() => {
    return () => {
      // This cleanup runs when navigating away from the page
      console.log('LiveCamera unmounting - stopping camera...');
      stopCamera();
    };
  }, [stopCamera]);

  const handleToggleCamera = () => {
    if (isStreaming) {
      stopCamera();
    } else {
      startCamera();
      setDetections([]);
    }
  };

  return (
    <div className="camera-page fade-in">
      <div className="flex justify-between items-center" style={{ marginBottom: 'var(--spacing-xl)' }}>
        <div>
          <h1 className="text-2xl font-bold">Live Camera</h1>
          <p className="text-secondary">Real-time Plate detection</p>
        </div>
        
        <div className="flex gap-md items-center">
          <div className="flex gap-sm items-center">
            <div className={`status-indicator ${isConnected ? 'active' : ''}`}></div>
            <span className="text-sm text-secondary">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          <button 
            onClick={handleToggleCamera}
            disabled={!isConnected}
            className={`btn btn-lg ${isStreaming ? 'btn-error' : 'btn-primary'}`}
          >
            {isStreaming ? (
              <>
                <VideoOff size={20} />
                Stop Camera
              </>
            ) : (
              <>
                <Video size={20} />
                Start Camera
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert-error" style={{ marginBottom: 'var(--spacing-lg)' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div className="camera-layout">
        <div className="camera-view-container">
          <div className="card camera-view">
            {!isStreaming && (
              <div className="camera-placeholder">
                <Camera size={64} />
                <p className="text-lg">Camera is not active</p>
                <p className="text-sm text-muted">Click "Start Camera" to begin detection</p>
              </div>
            )}
            <canvas ref={canvasRef} className={`camera-canvas ${!isStreaming ? 'hidden' : ''}`}></canvas>
          </div>
        </div>

        <div className="detections-sidebar">
          <div className="card">
            <h2 className="text-lg font-semibold" style={{ marginBottom: 'var(--spacing-md)' }}>
              Live Detections
            </h2>
            
            {detections.length === 0 ? (
              <div className="text-center text-muted" style={{ padding: 'var(--spacing-lg)' }}>
                <p className="text-sm">No detections yet</p>
              </div>
            ) : (
              <div className="detections-list">
                {detections.map((detection, index) => (
                  <div key={`${index}-${detection.detected_at}-${detection.plate_number}`} className="detection-item">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="plate-number">{detection.plate_number}</div>
                        <div className="text-sm text-muted">
                          {new Date(detection.detected_at).toLocaleTimeString()}
                        </div>
                      </div>
                      <div>
                        {detection.is_valid ? (
                          <span className="badge badge-success">Valid</span>
                        ) : (
                          <span className="badge badge-error">Invalid</span>
                        )}
                      </div>
                    </div>
                    
                    {detection.image_path && (
                      <img 
                        src={detection.image_path} 
                        alt={detection.plate_number}
                        className="detection-image"
                      />
                    )}
                    
                    <div className="flex gap-sm items-center">
                      <div className="confidence-bar">
                        <div 
                          className="confidence-fill"
                          style={{ width: `${detection.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-muted">
                        {(detection.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveCamera;
