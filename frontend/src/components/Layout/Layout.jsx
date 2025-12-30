import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Camera, 
  Upload, 
  Database, 
  BarChart3,
  Menu,
  X
} from 'lucide-react';
import { useState } from 'react';
import './Layout.css';

const Layout = ({ children }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const isMobile = window.innerWidth <= 768;

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);
  const closeSidebar = () => setIsSidebarOpen(false);

  return (
    <div className="layout">
      {/* Mobile Toggle Button */}
      <button 
        className="mobile-menu-btn" 
        onClick={toggleSidebar}
        aria-label="Toggle Menu"
      >
        <Menu size={24} />
      </button>

      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div className="sidebar-overlay" onClick={closeSidebar} />
      )}

      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <Camera size={32} className="logo-icon" />
            <div>
              <h1 className="logo-title">PlateVision</h1>
              <p className="logo-subtitle">LPR System</p>
            </div>
          </div>
          {/* Mobile Close Button inside Sidebar */}
          <button className="sidebar-close-btn" onClick={closeSidebar}>
            <X size={20} />
          </button>
        </div>
        
        <nav className="sidebar-nav">
          <NavLink to="/" className="nav-link" end onClick={closeSidebar}>
            <LayoutDashboard size={20} />
            <span>Dashboard</span>
          </NavLink>
          
          <NavLink to="/camera" className="nav-link" onClick={closeSidebar}>
            <Camera size={20} />
            <span>Live Camera</span>
          </NavLink>
          
          <NavLink to="/upload" className="nav-link" onClick={closeSidebar}>
            <Upload size={20} />
            <span>Upload Image</span>
          </NavLink>
          
          <NavLink to="/detections" className="nav-link" onClick={closeSidebar}>
            <Database size={20} />
            <span>Detections</span>
          </NavLink>
          
          <NavLink to="/statistics" className="nav-link" onClick={closeSidebar}>
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
