import React, { useState } from 'react';
import { selectedFeatures } from '../data/experimentData';

const categorizeFeature = (feature) => {
  if (feature.toLowerCase().includes('permission') || feature.startsWith('ANDROID.permission')) {
    return 'Permission';
  }
  if (feature.includes('intent.action') || feature.includes('ACTION_')) {
    return 'Intent Action';
  }
  if (feature.includes('ContentResolver') || feature.includes('SmsManager') || feature.includes('Context:')) {
    return 'API Call';
  }
  if (feature.includes('JAVA_REFLECTION') || feature.includes('remount') || feature.includes('HAP_BIND')) {
    return 'Privilege / Reflection';
  }
  if (feature.includes('SERVICE') || feature.includes('service') || feature.includes('PROVIDER')) {
    return 'Component / Service';
  }
  return 'Behavioral / Signature';
};

const FeaturesPage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');

  const categories = ['All', 'Permission', 'Intent Action', 'API Call', 'Privilege / Reflection', 'Component / Service', 'Behavioral / Signature'];

  const filteredFeatures = selectedFeatures
    .map((name, index) => ({
      rank: index + 1,
      name,
      category: categorizeFeature(name),
    }))
    .filter((f) => {
      const matchesSearch = f.name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCat = activeCategory === 'All' || f.category === activeCategory;
      return matchesSearch && matchesCat;
    });

  return (
    <div className="features-page animate-in">
      <div className="page-header">
        <h1>Permission Profiling & Feature Vector Selection</h1>
        <p>Top 50 high-discriminant static features isolated from 50,621 dimensions via Mutual Information ranking.</p>
      </div>

      {/* Feature Summary KPI */}
      <div className="metrics-grid" style={{ marginBottom: '24px' }}>
        <div className="metric-card blue">
          <div className="metric-label">Original Feature Space</div>
          <div className="metric-value">50,621</div>
          <div className="metric-change neutral">MalDroid-2020 static vector</div>
        </div>
        <div className="metric-card emerald">
          <div className="metric-label">Selected Dimensions</div>
          <div className="metric-value">50</div>
          <div className="metric-change positive">Dimensionality reduced 99.9%</div>
        </div>
        <div className="metric-card purple">
          <div className="metric-label">Selection Method</div>
          <div className="metric-value" style={{ fontSize: '24px' }}>Mutual Info</div>
          <div className="metric-change positive">I(X; Y) non-linear capture</div>
        </div>
        <div className="metric-card cyan">
          <div className="metric-label">Covariance Matrix Size</div>
          <div className="metric-value" style={{ fontSize: '24px' }}>50 × 50</div>
          <div className="metric-change neutral">Per class (5 classes total)</div>
        </div>
      </div>

      {/* Filtering Bar */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ flex: '1', minWidth: '260px' }}>
            <input
              type="text"
              placeholder="Search features (e.g. SmsManager, VIBRATE, BOOT_COMPLETED)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 16px',
                borderRadius: '8px',
                background: 'rgba(0, 0, 0, 0.25)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                fontSize: '13px',
                outline: 'none',
              }}
            />
          </div>

          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '20px',
                  border: '1px solid',
                  borderColor: activeCategory === cat ? 'var(--accent-blue)' : 'var(--border-subtle)',
                  background: activeCategory === cat ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                  color: activeCategory === cat ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  fontSize: '12px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <span className="icon">🏷️</span> Ranked Feature Directory ({filteredFeatures.length} of 50 displayed)
          </div>
          <span className="status-badge best">I(X; Y) Ranked</span>
        </div>

        <div className="feature-grid">
          {filteredFeatures.map((f) => (
            <div key={f.rank} className="feature-tag">
              <span className="rank">#{f.rank}</span>
              <span style={{ marginRight: '8px' }}>{f.name}</span>
              <span
                style={{
                  fontSize: '10px',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  background:
                    f.category === 'Permission'
                      ? 'rgba(16, 185, 129, 0.15)'
                      : f.category === 'API Call'
                      ? 'rgba(59, 130, 246, 0.15)'
                      : f.category === 'Intent Action'
                      ? 'rgba(245, 158, 11, 0.15)'
                      : 'rgba(139, 92, 246, 0.15)',
                  color:
                    f.category === 'Permission'
                      ? '#34d399'
                      : f.category === 'API Call'
                      ? '#60a5fa'
                      : f.category === 'Intent Action'
                      ? '#fbbf24'
                      : '#c084fc',
                }}
              >
                {f.category}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FeaturesPage;
