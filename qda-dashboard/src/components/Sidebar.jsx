import React from 'react';
import { NavLink } from 'react-router-dom';

const Sidebar = () => {
  const navItems = [
    { path: '/', label: 'Overview', icon: '📊', section: 'Main' },
    { path: '/metrics', label: 'Performance Metrics', icon: '🎯', section: 'Analysis' },
    { path: '/confusion-matrix', label: 'Confusion Matrix', icon: '🔲', section: 'Analysis' },
    { path: '/tuning', label: 'Hyperparameter Tuning', icon: '⚡', section: 'Optimization' },
    { path: '/features', label: 'Permission Profiling', icon: '🔍', section: 'Features' },
    { path: '/predict', label: 'Malware Simulator', icon: '🧪', section: 'Interactive' },
    { path: '/research', label: 'Research & Artifacts', icon: '📄', section: 'Documentation' },
  ];

  const sections = ['Main', 'Analysis', 'Optimization', 'Features', 'Interactive', 'Documentation'];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <NavLink to="/" className="sidebar-logo">
          <div className="logo-icon">Q</div>
          <div className="logo-text">
            <span className="logo-title">QDA Engine</span>
            <span className="logo-subtitle">Group 1 • Malware Profiler</span>
          </div>
        </NavLink>
      </div>

      <nav className="sidebar-nav">
        {sections.map((section) => {
          const items = navItems.filter((item) => item.section === section);
          if (items.length === 0) return null;
          return (
            <React.Fragment key={section}>
              <div className="nav-section-title">{section}</div>
              {items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    isActive ? 'nav-link active' : 'nav-link'
                  }
                  end={item.path === '/'}
                >
                  <span className="nav-icon">{item.icon}</span>
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </React.Fragment>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-badge">
          QDA Model: reg=0.1
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
