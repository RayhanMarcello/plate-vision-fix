import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard';
import LiveCamera from './pages/LiveCamera';
import Upload from './pages/Upload';
import Detections from './pages/Detections';
import Statistics from './pages/Statistics';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/camera" element={<LiveCamera />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/detections" element={<Detections />} />
          <Route path="/statistics" element={<Statistics />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
