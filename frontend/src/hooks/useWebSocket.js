/**
 * Custom hook for WebSocket camera connection
 */
import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = `ws://${window.location.hostname}:${window.location.port}/ws/camera`;

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [latestDetection, setLatestDetection] = useState(null);
  const [error, setError] = useState(null);
  
  const ws = useRef(null);
  const reconnectTimeout = useRef(null);
  const isStreamingRef = useRef(false);  // Track streaming state for cleanup
  
  // Keep streaming ref in sync with state
  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);
  
  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(WS_URL);
      
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
      };
      
      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'camera:frame':
              setCurrentFrame(message.data);
              break;
              
            case 'camera:status':
              if (message.status === 'started') {
                setIsStreaming(true);
              } else if (message.status === 'stopped') {
                setIsStreaming(false);
              }
              break;
              
            case 'detection:new':
              setLatestDetection(message.data);
              break;
              
            case 'error':
              setError(message.message);
              break;
              
            default:
              break;
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection error');
      };
      
      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setIsStreaming(false);
        
        // Auto-reconnect after 3 seconds
        reconnectTimeout.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 3000);
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError('Failed to create WebSocket connection');
    }
  }, []);
  
  const stopCamera = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'camera:stop' }));
      setIsStreaming(false);
    }
  }, []);
  
  const disconnect = useCallback(() => {
    // Clear reconnect timeout
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    
    // Stop camera before disconnecting if it's streaming
    if (ws.current && ws.current.readyState === WebSocket.OPEN && isStreamingRef.current) {
      console.log('Stopping camera before disconnect...');
      ws.current.send(JSON.stringify({ type: 'camera:stop' }));
    }
    
    // Close WebSocket connection
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    
    setIsConnected(false);
    setIsStreaming(false);
    setCurrentFrame(null);
  }, []);
  
  const startCamera = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'camera:start' }));
    }
  }, []);
  
  useEffect(() => {
    connect();
    
    // Cleanup function - runs when component unmounts or navigates away
    return () => {
      console.log('Cleaning up WebSocket connection...');
      disconnect();
    };
  }, [connect, disconnect]);
  
  return {
    isConnected,
    isStreaming,
    currentFrame,
    latestDetection,
    error,
    startCamera,
    stopCamera,
    disconnect,  // Expose disconnect for manual cleanup if needed
  };
};
