/**
 * Live Camera Component - Browser-Based Camera Capture
 * Uses browser's MediaDevices API to capture video and sends frames to backend for detection
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, Video, VideoOff, AlertCircle, RefreshCw } from 'lucide-react';
import './LiveCamera.css';

const API_BASE = '/api';

const LiveCameraBrowser = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // Start browser camera
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment' // Prefer rear camera on mobile
        },
        audio: false
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setCameraReady(true);
          setIsStreaming(true);
        };
      }
    } catch (err) {
      console.error('Camera error:', err);
      if (err.name === 'NotAllowedError') {
        setError('Camera access denied. Please allow camera access in your browser settings.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found on this device.');
      } else {
        setError(`Failed to access camera: ${err.message}`);
      }
    }
  }, []);

  // Stop browser camera
  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setCameraReady(false);
    setIsStreaming(false);
    setIsProcessing(false);
  }, []);

  // Capture frame and send to backend for detection
  const captureAndDetect = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0);

    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      setIsProcessing(true);

      try {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const response = await fetch(`${API_BASE}/detect/frame`, {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const result = await response.json();
          
          if (result.detections && result.detections.length > 0) {
            // Add new detections to list
            setDetections(prev => {
              const newDetections = result.detections.map(d => ({
                ...d,
                detected_at: new Date().toISOString()
              }));
              return [...newDetections, ...prev].slice(0, 10);
            });
          }
        }
      } catch (err) {
        console.error('Detection error:', err);
      } finally {
        setIsProcessing(false);
      }
    }, 'image/jpeg', 0.8);
  }, [isProcessing]);

  // Start detection loop
  const startDetection = useCallback(() => {
    if (intervalRef.current) return;
    
    // Run detection every 500ms
    intervalRef.current = setInterval(() => {
      captureAndDetect();
    }, 500);
  }, [captureAndDetect]);

  // Stop detection loop
  const stopDetection = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Handle toggle camera
  const handleToggleCamera = () => {
    if (isStreaming) {
      stopDetection();
      stopCamera();
    } else {
      startCamera();
    }
  };

  // Start detection when camera is ready
  useEffect(() => {
    if (cameraReady && isStreaming) {
      startDetection();
    }
    return () => {
      stopDetection();
    };
  }, [cameraReady, isStreaming, startDetection, stopDetection]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetection();
      stopCamera();
    };
  }, [stopCamera, stopDetection]);

  return (
    <div className="camera-page fade-in">
      <div className="flex justify-between items-center" style={{ marginBottom: 'var(--spacing-xl)' }}>
        <div>
          <h1 className="text-2xl font-bold">Live Camera</h1>
          <p className="text-secondary">Real-time Plate detection (Browser Camera)</p>
        </div>
        
        <div className="flex gap-md items-center">
          <div className="flex gap-sm items-center">
            <div className={`status-indicator ${isStreaming ? 'active' : ''}`}></div>
            <span className="text-sm text-secondary">
              {isStreaming ? (isProcessing ? 'Processing...' : 'Active') : 'Inactive'}
            </span>
          </div>
          
          <button 
            onClick={handleToggleCamera}
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
            <video 
              ref={videoRef} 
              className={`camera-video ${!isStreaming ? 'hidden' : ''}`}
              autoPlay 
              playsInline 
              muted
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            
            {isProcessing && isStreaming && (
              <div className="processing-indicator">
                <RefreshCw size={20} className="spin" />
                <span>Detecting...</span>
              </div>
            )}
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
                  <div key={`${index}-${detection.plate_number}`} className="detection-item">
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
                    
                    {detection.image_data && (
                      <img 
                        src={detection.image_data} 
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

export default LiveCameraBrowser;
