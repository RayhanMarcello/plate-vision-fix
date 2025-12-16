import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Camera, 
  Upload, 
  Database, 
  BarChart3 
} from 'lucide-react';
import './Layout.css';

const Layout = ({ children }) => {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Camera size={32} className="logo-icon" />
            <div>
              <h1 className="logo-title">PlateVision</h1>
              <p className="logo-subtitle">LPR System</p>
            </div>
          </div>
        </div>
        
        <nav className="sidebar-nav">
          <NavLink to="/" className="nav-link" end>
            <LayoutDashboard size={20} />
            <span>Dashboard</span>
          </NavLink>
          
          <NavLink to="/camera" className="nav-link">
            <Camera size={20} />
            <span>Live Camera</span>
          </NavLink>
          
          <NavLink to="/upload" className="nav-link">
            <Upload size={20} />
            <span>Upload Image</span>
          </NavLink>
          
          <NavLink to="/detections" className="nav-link">
            <Database size={20} />
            <span>Detections</span>
          </NavLink>
          
          <NavLink to="/statistics" className="nav-link">
            <BarChart3 size={20} />
            <span>Statistics</span>
          </NavLink>
        </nav>
        
        <div className="sidebar-footer">
          <div className="system-status">
            <div className="status-indicator active"></div>
            <span className="text-sm text-secondary">System Active</span>
          </div>
        </div>
      </aside>
      
      <main className="main-content">
        {children}
      </main>
    </div>
  );
};

export default Layout;
